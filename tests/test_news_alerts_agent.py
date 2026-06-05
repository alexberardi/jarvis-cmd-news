"""Tests for NewsAlertAgent — aggregation + filter compliance.

Run inside the jarvis-node-setup venv where jarvis_command_sdk is installed;
this package has no test infrastructure of its own.
"""

import importlib.util
import os
import sys
import types
from typing import Any, Dict, List

import pytest


def _load_agent_module():
    here = os.path.dirname(os.path.abspath(__file__))
    agent_path = os.path.join(here, "..", "agents", "news_alerts", "agent.py")
    spec = importlib.util.spec_from_file_location("news_alerts_under_test", agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def agent_module():
    return _load_agent_module()


@pytest.fixture
def agent(agent_module):
    return agent_module.NewsAlertAgent()


def _article(title: str, summary: str = "", source: str = "test") -> Dict[str, Any]:
    return {"title": title, "summary": summary, "source": source}


# ─── _compose_aggregate ────────────────────────────────────────────────

class TestComposeAggregate:
    def test_single_article(self, agent):
        articles = [_article("Lakers win in OT", "Final score 112-110")]
        title, body, summary = agent._compose_aggregate(articles)
        assert title == "News: Lakers win in OT"
        assert body == "Final score 112-110"
        assert summary == "Final score 112-110"

    def test_single_article_no_summary_falls_back_to_title(self, agent):
        articles = [_article("Lakers win in OT", "")]
        title, body, summary = agent._compose_aggregate(articles)
        assert body == "Lakers win in OT"

    def test_multiple_articles_aggregate(self, agent):
        articles = [
            _article("Headline A"),
            _article("Headline B"),
            _article("Headline C"),
        ]
        title, body, summary = agent._compose_aggregate(articles)
        assert title == "3 news headlines"
        assert body == "• Headline A\n• Headline B\n• Headline C"
        # TTS summary has no bullets and uses periods.
        assert summary == "3 new headlines: Headline A. Headline B. Headline C."

    def test_long_title_truncated_in_single_path(self, agent):
        long_title = "x" * 120
        articles = [_article(long_title, "body")]
        title, _body, _summary = agent._compose_aggregate(articles)
        assert title.startswith("News: ")
        # "News: " is 6 chars + max 80 of title = 86
        assert len(title) <= 86

    def test_body_caps_and_emits_overflow_marker(self, agent):
        # 40 long-ish titles will blow past the 500-char body cap.
        articles = [_article("Long headline number {0}".format(i)) for i in range(40)]
        _title, body, _summary = agent._compose_aggregate(articles)
        assert "…and " in body
        assert "more" in body
        assert len(body) <= 600  # rough — cap + overflow marker

    def test_emits_one_notification_per_call(self, agent):
        # The signature itself encodes "single notification" — the contract is
        # that this method always returns one (title, body, summary) triple
        # regardless of input size.
        for n in (1, 2, 10, 50):
            articles = [_article(f"t{i}") for i in range(n)]
            result = agent._compose_aggregate(articles)
            assert isinstance(result, tuple) and len(result) == 3


# ─── _filter_articles ──────────────────────────────────────────────────

def _install_fake_node_llm_client(monkeypatch, ask_llm_impl):
    """Make `from services.node_llm_client import ask_llm` resolve to our fake."""
    services_mod = sys.modules.get("services") or types.ModuleType("services")
    monkeypatch.setitem(sys.modules, "services", services_mod)
    node_mod = types.ModuleType("services.node_llm_client")
    node_mod.ask_llm = ask_llm_impl  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "services.node_llm_client", node_mod)
    services_mod.node_llm_client = node_mod  # type: ignore[attr-defined]


class TestFilterArticles:
    def test_no_articles_returns_empty(self, agent):
        assert agent._filter_articles("sports", []) == []

    def test_llm_unavailable_fails_closed(self, agent, monkeypatch):
        # Make the import inside _filter_articles raise.
        monkeypatch.setitem(sys.modules, "services.node_llm_client", None)
        result = agent._filter_articles("sports", [_article("Lakers win")])
        # ImportError path → returns [] (fail-closed).
        assert result == []

    def test_llm_returns_matching_indices(self, agent, monkeypatch):
        articles = [
            _article("Lakers win in OT"),
            _article("Senate passes budget"),
            _article("Warriors clinch playoff spot"),
        ]
        _install_fake_node_llm_client(monkeypatch, lambda *a, **kw: "[1, 3]")
        matched = agent._filter_articles("only sports news", articles)
        assert matched == [articles[0], articles[2]]

    def test_llm_returns_empty_array(self, agent, monkeypatch):
        articles = [_article("Senate passes budget")]
        _install_fake_node_llm_client(monkeypatch, lambda *a, **kw: "[]")
        assert agent._filter_articles("only sports news", articles) == []

    def test_llm_returns_none_fails_closed(self, agent, monkeypatch):
        articles = [_article("a")]
        _install_fake_node_llm_client(monkeypatch, lambda *a, **kw: None)
        assert agent._filter_articles("sports", articles) == []

    def test_llm_returns_garbage_fails_closed(self, agent, monkeypatch):
        articles = [_article("a"), _article("b")]
        _install_fake_node_llm_client(monkeypatch, lambda *a, **kw: "I think articles 1 and 2 are sports.")
        # No parseable JSON array → fail-closed.
        assert agent._filter_articles("sports", articles) == []

    def test_llm_strips_think_block_and_code_fence(self, agent, monkeypatch):
        articles = [_article("a"), _article("b"), _article("c")]
        raw = "<think>let me reason</think>\n```json\n[2]\n```"
        _install_fake_node_llm_client(monkeypatch, lambda *a, **kw: raw)
        matched = agent._filter_articles("rule", articles)
        assert matched == [articles[1]]

    def test_llm_out_of_range_indices_dropped(self, agent, monkeypatch):
        articles = [_article("a"), _article("b")]
        _install_fake_node_llm_client(monkeypatch, lambda *a, **kw: "[1, 5, 99, 0, -1]")
        matched = agent._filter_articles("rule", articles)
        # Only index 1 (1-based) is in range.
        assert matched == [articles[0]]

    def test_llm_non_numeric_indices_dropped(self, agent, monkeypatch):
        articles = [_article("a"), _article("b")]
        _install_fake_node_llm_client(monkeypatch, lambda *a, **kw: '["foo", 2]')
        matched = agent._filter_articles("rule", articles)
        assert matched == [articles[1]]

    def test_prompt_contains_strict_guidance(self, agent, monkeypatch):
        captured: Dict[str, Any] = {}

        def fake_ask(prompt: str, *, system=None, **kw):
            captured["prompt"] = prompt
            captured["system"] = system
            return "[]"

        _install_fake_node_llm_client(monkeypatch, fake_ask)
        agent._filter_articles("only AI news", [_article("x")])

        # System prompt enforces the constraint.
        assert "HARD" in captured["system"] or "hard" in captured["system"].lower()
        assert "skip" in captured["system"].lower()
        assert "false positive" in captured["system"].lower()
        # User's rule is visible in the user prompt.
        assert "only AI news" in captured["prompt"]
