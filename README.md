# News Headlines for Jarvis

Get the latest news headlines by category via RSS feeds.

## Components

| Type | Name | Description |
|------|------|-------------|
| Command | `get_news` | "What's in the news?", "Give me tech headlines", "Top 3 headlines" |

## Install

```bash
jarvis pantry install --url https://github.com/alexberardi/jarvis-cmd-news
```

Or from a local checkout:

```bash
jarvis pantry install --local /path/to/jarvis-cmd-news
```

## Categories

- general (default)
- tech
- sports
- business
- science
- health

## Secrets

| Key | Required | Description |
|-----|----------|-------------|
| `NEWS_RSS_FEEDS` | No | Comma-separated custom RSS feed URLs (merged with built-in feeds) |

## Structure

```
jarvis_package.yaml
commands/
  get_news/command.py
```

## License

MIT
