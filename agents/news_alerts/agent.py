"""NewsAlertAgent — monitors RSS feeds and produces alerts for new articles.

Runs every 30 minutes. Compares against previous run's titles to detect new
articles. Alerts have a 4-hour TTL and low priority (1).

Also injects headlines into the command center's memory system so Jarvis has
proactive awareness of current events during voice conversations.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

try:
    from jarvis_log_client import JarvisLogger
except ImportError:
    import logging

    class JarvisLogger:  # noqa: E303
        def __init__(self, **kw: str) -> None:
            self._log = logging.getLogger(kw.get("service", __name__))
        def info(self, msg: str, **kw: object) -> None: self._log.info(msg)
        def warning(self, msg: str, **kw: object) -> None: self._log.warning(msg)
        def error(self, msg: str, **kw: object) -> None: self._log.error(msg)
        def debug(self, msg: str, **kw: object) -> None: self._log.debug(msg)

from jarvis_command_sdk import IJarvisAgent, AgentSchedule, Alert, JarvisSecret
from jarvis_command_sdk import IJarvisSecret

logger = JarvisLogger(service="jarvis-node")

REFRESH_INTERVAL_SECONDS = 1800  # 30 minutes
ALERT_TTL_HOURS = 4
MEMORY_TTL_HOURS = 24


class NewsAlertAgent(IJarvisAgent):
    """Background agent that monitors RSS feeds for new headlines."""

    def __init__(self) -> None:
        self._previous_titles: set[str] = set()
        self._current_articles: List[Dict[str, Any]] = []
        self._alerts: List[Alert] = []

    @property
    def name(self) -> str:
        return "news_alerts"

    @property
    def description(self) -> str:
        return "Monitors RSS news feeds and generates alerts for new headlines"

    @property
    def schedule(self) -> AgentSchedule:
        return AgentSchedule(
            interval_seconds=REFRESH_INTERVAL_SECONDS,
            run_on_startup=True,
        )

    @property
    def required_secrets(self) -> List[IJarvisSecret]:
        return [
            JarvisSecret(
                key="NOTIFICATION_FILTER",
                description=(
                    "Free-text instructions for when to send news alerts "
                    "(e.g. 'only sports news', 'tech and finance only'). "
                    "Leave blank to receive alerts for every new article."
                ),
                scope="integration",
                value_type="string",
                required=False,
                is_sensitive=False,
                friendly_name="Notification Filter",
            ),
        ]

    @property
    def include_in_context(self) -> bool:
        return False

    async def run(self) -> None:
        """Fetch news, detect new articles, and inject into CC memory."""
        try:
            # Import from this package's command
            try:
                from commands.get_news.command import NewsCommand, _DEFAULT_FEEDS
            except ImportError:
                from commands.custom_commands.get_news.command import NewsCommand, _DEFAULT_FEEDS

            cmd = NewsCommand()
            feed_urls = list(_DEFAULT_FEEDS.get("general", []))
            articles = cmd._fetch_articles(feed_urls)

            current_titles = {a["title"].strip().lower() for a in articles if a.get("title")}

            # On first run, just seed — don't alert on everything
            if not self._previous_titles:
                self._previous_titles = current_titles
                self._current_articles = articles
                logger.info("News agent seeded", article_count=len(articles))
                self._alerts = []
                # Still inject into memory on first run
                self._inject_memories(articles)
                return

            # Detect new articles
            new_titles = current_titles - self._previous_titles
            new_articles = [
                a for a in articles
                if a.get("title", "").strip().lower() in new_titles
            ]
            self._alerts = []

            if new_articles:
                # One LLM call for the whole batch: the model filters AND
                # composes the notifications, grouping related stories where
                # appropriate. With a filter set, this replaces N per-article
                # calls; without, we short-circuit to one notification per
                # article (preserves the "tell me every new headline" UX
                # without burning a model call to confirm it).
                filter_text = self._read_filter()
                if filter_text:
                    notifications = self._compose_notifications(
                        filter_text, new_articles
                    )
                else:
                    notifications = [
                        {
                            "title": f"News: {a.get('title', '')[:80]}",
                            "body": a.get("summary", "")[:200] or a.get("title", ""),
                        }
                        for a in new_articles
                    ]

                for n in notifications:
                    self._push_notification(n["title"], n["body"])
                    self._alerts.append(Alert(
                        source_agent=self.name,
                        title=n["title"],
                        summary=n["body"],
                        priority=1,
                    ))

                logger.info(
                    "News agent processed new articles",
                    new_articles=len(new_articles),
                    notifications=len(self._alerts),
                )

            self._previous_titles = current_titles
            self._current_articles = articles

            # Inject all current headlines into CC memory
            self._inject_memories(articles)

        except Exception as e:
            logger.error("News agent run failed", error=str(e))
            self._alerts = []

    def _inject_memories(self, articles: List[Dict[str, Any]]) -> None:
        """Push article headlines into CC memory system for proactive context."""
        try:
            from clients.rest_client import RestClient
        except ImportError:
            logger.debug("RestClient not available — skipping memory injection")
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        memories = []

        for article in articles:
            title = article.get("title", "").strip()
            if not title:
                continue

            source = article.get("source", "")
            summary = article.get("summary", "")

            content = title
            if summary:
                clean = summary[:200].strip()
                if clean and clean != title:
                    content = f"{title} — {clean}"

            title_hash = hashlib.md5(title.lower().encode()).hexdigest()[:8]

            memories.append({
                "content": content,
                "category": "news",
                "key": f"news:general:{today}:{title_hash}",
                "ttl_hours": MEMORY_TTL_HOURS,
                "source": f"news-agent:{source}",
            })

        if memories:
            result = RestClient.inject_memories(memories)
            if result:
                logger.info(
                    "News agent injected memories",
                    count=result.get("injected", 0) + result.get("updated", 0),
                )

    def _read_filter(self) -> str:
        """Return the user's NOTIFICATION_FILTER value (empty string if unset)."""
        try:
            from services.secret_service import get_secret_value
            return (get_secret_value("NOTIFICATION_FILTER", "integration") or "").strip()
        except ImportError:
            return ""
        except Exception as e:
            logger.warning("Failed to read NOTIFICATION_FILTER", error=str(e))
            return ""

    def _compose_notifications(
        self, filter_text: str, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Ask the LLM to filter + compose notifications for a batch.

        One LLM call for the whole batch. The model decides which articles
        match the user's rule AND writes the notification copy (title + body),
        grouping related stories where appropriate. Returns a list of
        ``{"title": ..., "body": ...}`` dicts ready to push.

        Fail-open contract: on any LLM/parse error we fall back to one
        notification per input article with raw title/summary copy — the
        user keeps getting headlines, just without the filter applied.
        """
        import json as _json
        import re

        try:
            from services.node_llm_client import ask_llm
        except ImportError:
            ask_llm = None  # type: ignore[assignment]

        def _fallback() -> List[Dict[str, str]]:
            return [
                {
                    "title": f"News: {a.get('title', '')[:80]}",
                    "body": (a.get("summary", "") or a.get("title", ""))[:200],
                }
                for a in articles
            ]

        if ask_llm is None or not articles:
            return _fallback() if articles else []

        article_lines = []
        for i, a in enumerate(articles, start=1):
            article_lines.append(
                f"{i}. {a.get('title', '').strip()}\n"
                f"   {a.get('summary', '').strip()[:300]}\n"
                f"   Source: {a.get('source', '').strip()}"
            )

        prompt = (
            "You are a personal news notification curator. The user has set "
            "this rule for when they want to be notified:\n\n"
            f"{filter_text}\n\n"
            f"Here are {len(articles)} new articles from today's feeds:\n\n"
            + "\n\n".join(article_lines) +
            "\n\nCompose the notifications the user should receive based on "
            "their rule. Skip articles that don't match. Group closely "
            "related stories into one notification when it improves clarity. "
            "Each notification has a short title (under 80 chars) and a one- "
            "to two-sentence body. If nothing matches, return an empty array.\n\n"
            "Respond with ONLY a JSON array, no other text, in this exact "
            'shape: [{"title": "...", "body": "..."}, ...]'
        )

        raw = ask_llm(prompt) or ""
        # Strip <think>...</think> blocks (thinking-mode models)
        cleaned = re.sub(
            r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        # Strip markdown code fences if the model added them
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        # Find the first JSON array — models sometimes prepend a sentence
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            parsed = _json.loads(cleaned)
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON array")
            result: List[Dict[str, str]] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                body = str(item.get("body", "")).strip()
                if title:
                    result.append({"title": title[:200], "body": (body or title)[:500]})
            return result
        except Exception as e:
            logger.warning(
                "News compose parse failed; failing open",
                error=str(e),
                raw=cleaned[:200],
            )
            return _fallback()

    def _push_notification(self, title: str, body: str) -> None:
        """Send a household-wide push notification via CC's relay endpoint.

        Caller owns the title/body copy verbatim — no auto-prefix, since
        notifications composed by the LLM already include any "News:" /
        category framing the user wants.
        """
        try:
            from clients.rest_client import RestClient
            from utils.service_discovery import get_command_center_url
        except ImportError:
            return

        cc_url = get_command_center_url()
        if not cc_url:
            return

        try:
            RestClient.post(
                f"{cc_url.rstrip('/')}/api/v0/node/push-notification",
                data={
                    "title": title[:200],
                    "body": (body or title)[:500],
                    "priority": "default",
                    "category": "news",
                },
                timeout=5,
            )
        except Exception as e:
            logger.warning("News push failed", title=title[:80], error=str(e))

    def get_context_data(self) -> Dict[str, Any]:
        return {}

    def get_alerts(self) -> List[Alert]:
        return list(self._alerts)
