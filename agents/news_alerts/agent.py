"""NewsAlertAgent — monitors RSS feeds and produces alerts for new articles.

Runs every 30 minutes. Compares against previous run's titles to detect new
articles. Alerts have a 4-hour TTL and low priority (1).

Also injects headlines into the command center's memory system so Jarvis has
proactive awareness of current events during voice conversations.
"""

import hashlib
from datetime import datetime, timezone
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
                filter_text = self._read_filter()

                # Filter is a HARD constraint when set. Two-stage to keep
                # composition deterministic and the LLM honest: stage 1 picks
                # matching articles (indices only), stage 2 builds the
                # notification copy locally without LLM creativity drift.
                if filter_text:
                    matched = self._filter_articles(filter_text, new_articles)
                else:
                    matched = new_articles

                # One notification per run regardless of headline count.
                if matched:
                    title, body, summary = self._compose_aggregate(matched)
                    self._push_notification(title, body)
                    self._alerts.append(Alert(
                        source_agent=self.name,
                        title=title,
                        summary=summary,
                        priority=1,
                    ))

                logger.info(
                    "News agent processed new articles",
                    new_articles=len(new_articles),
                    matched=len(matched),
                    notifications_sent=1 if matched else 0,
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

    def _filter_articles(
        self, filter_text: str, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Ask the LLM which articles match the user's rule. Returns subset.

        Single LLM call returning only matching article numbers — keeps the
        decision deterministic to verify and prevents the model from
        rewriting/embellishing headlines or smuggling in non-matching items
        through composition.

        Fail-CLOSED contract: when a filter is set and we can't reliably
        determine matches (LLM unreachable, malformed output, no parseable
        indices), we return [] rather than spamming irrelevant alerts. The
        next run will retry. The user explicitly asked to be filtered — we
        respect that even when the model is broken.
        """
        import json as _json
        import re

        try:
            from services.node_llm_client import ask_llm
        except ImportError:
            logger.warning("News filter: ask_llm unavailable; skipping run (fail-closed)")
            return []

        if not articles:
            return []

        article_lines = []
        for i, a in enumerate(articles, start=1):
            article_lines.append(
                f"{i}. {a.get('title', '').strip()}\n"
                f"   {a.get('summary', '').strip()[:300]}"
            )

        system = (
            "You are a strict news filter. Your only job is to identify which "
            "articles match the user's rule. Treat the rule as a HARD "
            "constraint:\n"
            "- Match only articles that CLEARLY and DIRECTLY satisfy the rule.\n"
            "- Tangential, adjacent, or 'kind of related' articles do NOT match.\n"
            "- When in doubt, SKIP the article. The cost of a false negative "
            "(missing one match) is much lower than a false positive (sending "
            "an irrelevant alert the user explicitly asked not to receive).\n"
            "- Do NOT rewrite, summarize, or compose anything. Output ONLY a "
            "JSON array of the matching article numbers."
        )

        prompt = (
            f'The user\'s rule:\n"""\n{filter_text}\n"""\n\n'
            f"Articles ({len(articles)} total):\n\n"
            + "\n\n".join(article_lines) +
            "\n\nReturn the numbers of articles that match the rule, as a JSON "
            'array. Example: [1, 4, 7]. If nothing matches, return: []. '
            "Output ONLY the array — no prose, no code fences, no explanation."
        )

        raw = ask_llm(prompt, system=system) or ""
        if not raw:
            logger.warning(
                "News filter: empty LLM response; skipping run (fail-closed)",
                article_count=len(articles),
            )
            return []

        # Strip <think>...</think> blocks (thinking-mode models)
        cleaned = re.sub(
            r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        # Strip markdown code fences if the model added them
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        # Find the first JSON array
        match = re.search(r"\[[^\[\]]*\]", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

        try:
            parsed = _json.loads(cleaned)
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON array of indices")
        except Exception as e:
            logger.warning(
                "News filter: parse failed; skipping run (fail-closed)",
                error=str(e),
                raw=raw[:200],
            )
            return []

        matched: List[Dict[str, Any]] = []
        for idx in parsed:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if 1 <= i <= len(articles):
                matched.append(articles[i - 1])

        logger.info(
            "News filter applied",
            input_articles=len(articles),
            matched=len(matched),
        )
        return matched

    def _compose_aggregate(
        self, articles: List[Dict[str, Any]]
    ) -> tuple[str, str, str]:
        """Build (push_title, push_body, tts_summary) from N matching articles.

        Deterministic, no LLM. Always emits exactly one notification per call.
        Body uses bullet-prefixed newline-separated headlines (renders well
        in both push notifications and the inbox). TTS summary uses periods
        for natural sentence pacing when the alert is read aloud.
        """
        n = len(articles)
        if n == 1:
            a = articles[0]
            title = f"News: {a.get('title', '').strip()[:80]}"
            body_text = (a.get("summary", "") or a.get("title", "")).strip()[:500]
            return title, body_text, body_text

        # 2+ matched articles → one aggregated notification.
        titles = [a.get("title", "").strip() for a in articles if a.get("title")]
        title = f"{n} news headlines"

        # Push/inbox body: bullet list. Truncate to a soft cap so push
        # notifications don't get cut off mid-headline.
        bullets: List[str] = []
        BODY_CAP = 500
        used = 0
        for t in titles:
            line = f"• {t}"
            if used + len(line) + 1 > BODY_CAP:
                remaining = len(titles) - len(bullets)
                if remaining > 0:
                    bullets.append(f"…and {remaining} more")
                break
            bullets.append(line)
            used += len(line) + 1
        body = "\n".join(bullets)

        # TTS summary: period-separated for sentence pacing. No bullets.
        summary = f"{n} new headlines: " + ". ".join(titles) + "."
        return title, body, summary[:500]

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
