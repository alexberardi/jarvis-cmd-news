"""NewsCommand — fetch RSS news headlines.

Returns structured article data for the LLM to compose into a spoken response,
or for use as a step in a briefing routine.
"""

import calendar
from typing import Any, Dict, List

import feedparser

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

from jarvis_command_sdk import (
    CommandExample,
    CommandResponse,
    FastPathPattern,
    IJarvisCommand,
    JarvisPackage,
    JarvisParameter,
    IJarvisParameter,
    IJarvisSecret,
    JarvisSecret,
    PreRouteResult,
    RequestInformation,
)

# Spoken word number → int. The LLM normally handles this; for pre-route we
# only support common small counts so "top three headlines" routes deterministically.
_SPOKEN_NUMBERS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Secrets arrive via the SDK's execute() wrapper — see run() below.

logger = JarvisLogger(service="jarvis-node")

_DEFAULT_FEEDS: Dict[str, List[str]] = {
    "general": [
        "https://feeds.apnews.com/rss/apf-topnews",
        "https://feeds.bbci.co.uk/news/rss.xml",
    ],
    "tech": [
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    ],
    "sports": [
        "https://www.espn.com/espn/rss/ncb/news",
        "https://feeds.apnews.com/rss/apf-sports",
    ],
    "business": [
        "https://feeds.reuters.com/reuters/businessNews",
    ],
    "science": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
    ],
    "health": [
        "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
    ],
}

_CATEGORIES = list(_DEFAULT_FEEDS.keys())


class NewsCommand(IJarvisCommand):

    @property
    def command_name(self) -> str:
        return "get_news"

    @property
    def description(self) -> str:
        return "Get the latest news headlines by category. Supports general, tech, sports, business, science, and health."

    @property
    def keywords(self) -> List[str]:
        return ["news", "headlines", "briefing news", "what's happening", "current events"]

    @property
    def parameters(self) -> List[IJarvisParameter]:
        return [
            JarvisParameter(
                "category",
                "string",
                required=False,
                description="News category to fetch.",
                enum_values=_CATEGORIES,
            ),
            JarvisParameter(
                "count",
                "int",
                required=False,
                description="Number of headlines to return (default 5).",
            ),
        ]

    @property
    def required_secrets(self) -> List[IJarvisSecret]:
        return [
            JarvisSecret(
                key="NEWS_RSS_FEEDS",
                description="Comma-separated custom RSS feed URLs (optional, merged with built-in feeds).",
                scope="integration",
                value_type="string",
                required=False,
                is_sensitive=False,
                friendly_name="Custom RSS Feeds",
            ),
        ]

    @property
    def required_packages(self) -> List[JarvisPackage]:
        return [JarvisPackage("feedparser")]

    def generate_prompt_examples(self) -> List[CommandExample]:
        return [
            CommandExample(
                voice_command="What's in the news?",
                expected_parameters={},
                is_primary=True,
            ),
            CommandExample(
                voice_command="Give me tech headlines",
                expected_parameters={"category": "tech"},
            ),
            CommandExample(
                voice_command="Any sports news?",
                expected_parameters={"category": "sports"},
            ),
            CommandExample(
                voice_command="Top 3 headlines",
                expected_parameters={"count": 3},
            ),
        ]

    def generate_adapter_examples(self) -> List[CommandExample]:
        return self.generate_prompt_examples()

    # ------------------------------------------------------------------
    # Fast-path patterns — bypass the LLM for the common shapes.
    # ------------------------------------------------------------------

    @property
    def fast_path_patterns(self) -> List[FastPathPattern]:
        # Build a regex that captures category and/or count. Categories are
        # a fixed enum, so the alternation is safe. The order is descending
        # specificity: most-specific (count + category) first, then count
        # alone, then category alone, then bare news request.
        cat_group = r"(?P<category>" + "|".join(_CATEGORIES) + r")"
        return [
            FastPathPattern(
                id="get_news.top_n_category",
                description="Bypass LLM for 'top N <category> headlines/news'",
                example="top 3 tech headlines",
                regex=(
                    r"^\s*(?:top\s+|give\s+me\s+(?:the\s+)?top\s+|read\s+(?:me\s+)?(?:the\s+)?top\s+)"
                    r"(?P<count>\d+|" + "|".join(_SPOKEN_NUMBERS) + r")\s+"
                    + cat_group + r"\s+(?:news|headlines?|stories)"
                    r"\s*[?.!]*$"
                ),
                handler="_fp_count_category",
            ),
            FastPathPattern(
                id="get_news.top_n",
                description="Bypass LLM for 'top N headlines/news' (no category)",
                example="top 3 headlines",
                regex=(
                    r"^\s*(?:top\s+|give\s+me\s+(?:the\s+)?top\s+|read\s+(?:me\s+)?(?:the\s+)?top\s+)"
                    r"(?P<count>\d+|" + "|".join(_SPOKEN_NUMBERS) + r")\s+"
                    r"(?:headlines?|news|stories)"
                    r"\s*[?.!]*$"
                ),
                handler="_fp_count_only",
            ),
            FastPathPattern(
                id="get_news.category",
                description="Bypass LLM for category-only news ('any tech news', 'give me sports headlines')",
                example="any tech news",
                regex=(
                    r"^\s*(?:any\s+|give\s+me\s+(?:the\s+|some\s+)?|tell\s+me\s+(?:the\s+|some\s+)?|read\s+(?:me\s+)?(?:the\s+|some\s+)?|what'?s\s+(?:the\s+|new\s+(?:in\s+)?)?|i\s+want\s+|show\s+me\s+(?:the\s+|some\s+)?)?"
                    + cat_group + r"\s+(?:news|headlines?|stories|update)"
                    r"\s*[?.!]*$"
                ),
                handler="_fp_category",
            ),
            FastPathPattern(
                id="get_news.bare",
                description="Bypass LLM for bare news requests ('what's in the news')",
                example="what's in the news",
                regex=(
                    r"^\s*(?:"
                    r"what'?s\s+(?:in\s+the\s+|new\s+in\s+(?:the\s+)?|the\s+)?news"
                    r"|news(?:\s+(?:please|update|briefing))?"
                    r"|give\s+me\s+(?:the\s+|some\s+)?(?:headlines?|news)"
                    r"|tell\s+me\s+(?:the\s+)?(?:headlines?|news)"
                    r"|read\s+(?:me\s+)?(?:the\s+)?(?:headlines?|news)"
                    r"|i\s+want\s+(?:the\s+|some\s+)?news"
                    r"|what'?s\s+happening"
                    r"|current\s+events"
                    r"|headlines?"
                    r")\s*[?.!]*$"
                ),
                handler="_fp_bare",
            ),
        ]

    @staticmethod
    def _parse_count(token: str) -> int | None:
        token = token.lower().strip()
        if token.isdigit():
            return int(token)
        return _SPOKEN_NUMBERS.get(token)

    def _fp_count_category(self, match, voice_command: str) -> PreRouteResult | None:
        count = self._parse_count(match.group("count"))
        if count is None or count <= 0:
            return None
        return PreRouteResult(arguments={
            "category": match.group("category").lower(),
            "count": count,
        })

    def _fp_count_only(self, match, voice_command: str) -> PreRouteResult | None:
        count = self._parse_count(match.group("count"))
        if count is None or count <= 0:
            return None
        return PreRouteResult(arguments={"count": count})

    def _fp_category(self, match, voice_command: str) -> PreRouteResult | None:
        return PreRouteResult(arguments={"category": match.group("category").lower()})

    def _fp_bare(self, match, voice_command: str) -> PreRouteResult | None:
        return PreRouteResult(arguments={})

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        request_info: RequestInformation,
        *,
        secrets: Dict[str, str] | None = None,
        **kwargs: Any,
    ) -> CommandResponse:
        category: str = kwargs.get("category", "general")
        count: int = kwargs.get("count", 5)

        feed_urls = list(_DEFAULT_FEEDS.get(category, _DEFAULT_FEEDS["general"]))

        # Merge custom feeds from secret (host passes them via the SDK wrapper)
        custom = (secrets or {}).get("NEWS_RSS_FEEDS")
        if custom:
            for url in custom.split(","):
                url = url.strip()
                if url:
                    feed_urls.append(url)

        articles = self._fetch_articles(feed_urls)

        if not articles:
            return CommandResponse.error_response(
                error_details="No news sources available. All feeds failed or returned no articles.",
            )

        articles = articles[:count]

        return CommandResponse.success_response(
            context_data={
                "category": category,
                "count": len(articles),
                "articles": articles,
            },
            wait_for_input=False,
        )

    # ------------------------------------------------------------------
    # Feed fetching
    # ------------------------------------------------------------------

    def _fetch_articles(self, feed_urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch and merge articles from multiple RSS feeds."""
        all_articles: List[Dict[str, Any]] = []
        seen_titles: set[str] = set()

        for url in feed_urls:
            try:
                parsed = feedparser.parse(url)
                source = getattr(parsed.feed, "title", url)

                for entry in parsed.entries:
                    title = getattr(entry, "title", None)
                    if not title:
                        continue

                    # Deduplicate by exact title
                    title_key = title.strip().lower()
                    if title_key in seen_titles:
                        continue
                    seen_titles.add(title_key)

                    published_parsed = getattr(entry, "published_parsed", None)
                    published_ts = calendar.timegm(published_parsed) if published_parsed else 0

                    all_articles.append({
                        "title": title,
                        "summary": getattr(entry, "summary", ""),
                        "source": source,
                        "published": published_parsed,
                        "_sort_ts": published_ts,
                    })

            except Exception as e:
                logger.warning("Failed to fetch RSS feed", url=url, error=str(e))

        # Sort newest first
        all_articles.sort(key=lambda a: a["_sort_ts"], reverse=True)

        # Format published date and remove sort key
        for article in all_articles:
            pp = article.pop("_sort_ts")
            pub = article.pop("published", None)
            if pub:
                try:
                    from datetime import datetime
                    article["published"] = datetime(*pub[:6]).strftime("%Y-%m-%d")
                except (TypeError, ValueError):
                    article["published"] = None
            else:
                article["published"] = None

        return all_articles
