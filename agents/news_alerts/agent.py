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

from jarvis_command_sdk import IJarvisAgent, AgentSchedule, Alert
from jarvis_command_sdk import IJarvisSecret

logger = JarvisLogger(service="jarvis-node")

REFRESH_INTERVAL_SECONDS = 60  # TEMP: 60s for testing (prod: 1800)
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
        return []

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
            now = datetime.now(timezone.utc)
            self._alerts = []

            for article in articles:
                title = article.get("title", "")
                if title.strip().lower() in new_titles:
                    self._alerts.append(Alert(
                        source_agent=self.name,
                        title=title,
                        summary=article.get("summary", "")[:200],
                        created_at=now,
                        expires_at=now + timedelta(hours=ALERT_TTL_HOURS),
                        priority=1,
                    ))

            self._previous_titles = current_titles
            self._current_articles = articles

            if self._alerts:
                logger.info("News agent found new articles", count=len(self._alerts))

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

    def get_context_data(self) -> Dict[str, Any]:
        return {}

    def get_alerts(self) -> List[Alert]:
        return list(self._alerts)
