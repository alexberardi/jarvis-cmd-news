"""Pre-route tests for the get_news command.

Loaded via importlib so the tests run inside the jarvis-node-setup venv
where jarvis_command_sdk is installed; the cmd package itself has no test
infrastructure of its own.
"""

import importlib.util
import os

import pytest


def _load_command():
    here = os.path.dirname(os.path.abspath(__file__))
    cmd_path = os.path.join(here, "..", "commands", "get_news", "command.py")
    spec = importlib.util.spec_from_file_location("get_news_cmd_under_test", cmd_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NewsCommand


@pytest.fixture
def cmd():
    return _load_command()()


class TestPreRouteBare:
    @pytest.mark.parametrize("phrase", [
        "what's in the news",
        "what's the news",
        "news",
        "news please",
        "news update",
        "headlines",
        "give me the news",
        "give me headlines",
        "tell me the news",
        "read me the headlines",
        "what's happening",
        "current events",
    ])
    def test_bare(self, cmd, phrase):
        result = cmd.pre_route(phrase)
        assert result is not None
        assert result.arguments == {}


class TestPreRouteCategory:
    @pytest.mark.parametrize("phrase,category", [
        ("give me tech headlines", "tech"),
        ("any sports news", "sports"),
        ("tech news", "tech"),
        ("business news", "business"),
        ("science news", "science"),
        ("health news", "health"),
        ("show me tech headlines", "tech"),
        ("what's the tech news", "tech"),
    ])
    def test_category(self, cmd, phrase, category):
        result = cmd.pre_route(phrase)
        assert result is not None
        assert result.arguments == {"category": category}


class TestPreRouteTopN:
    @pytest.mark.parametrize("phrase,expected", [
        ("top 3 headlines", {"count": 3}),
        ("top 5 headlines", {"count": 5}),
        ("top three headlines", {"count": 3}),
        ("give me the top 10 headlines", {"count": 10}),
        ("top 5 tech headlines", {"category": "tech", "count": 5}),
        ("top three sports headlines", {"category": "sports", "count": 3}),
    ])
    def test_top_n(self, cmd, phrase, expected):
        result = cmd.pre_route(phrase)
        assert result is not None
        assert result.arguments == expected


class TestPreRouteNoMatch:
    @pytest.mark.parametrize("phrase", [
        "tell me a joke",
        "what time is it",
        "turn on the news",        # control_device shape
        "search news for tesla",   # search_web shape
        "",
    ])
    def test_returns_none(self, cmd, phrase):
        assert cmd.pre_route(phrase) is None


class TestFastPathPatterns:
    def test_ids_stable(self, cmd):
        ids = {p.id for p in cmd.fast_path_patterns}
        assert ids == {
            "get_news.top_n_category",
            "get_news.top_n",
            "get_news.category",
            "get_news.bare",
        }
