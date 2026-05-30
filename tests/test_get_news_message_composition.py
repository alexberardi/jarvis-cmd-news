"""Verify NewsCommand.run() returns a spoken `message` on the pre-route fast path.

The wrapper in command_execution_service falls through to LLM if a
pre-routed command's run() doesn't set context_data["message"], so the
fast-path savings are wasted unless we compose a message locally.
"""

import importlib.util
import os
import sys
import types

import pytest


def _load_command():
    here = os.path.dirname(os.path.abspath(__file__))
    cmd_path = os.path.join(here, "..", "commands", "get_news", "command.py")
    spec = importlib.util.spec_from_file_location("get_news_msg_test", cmd_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cmd_module():
    return _load_command()


def _stub_articles(n: int):
    return [
        {"title": f"Headline {i + 1}", "summary": f"Summary {i + 1}", "source": "src", "published": "2026-05-30"}
        for i in range(n)
    ]


def test_message_composed_when_pre_routed(cmd_module, monkeypatch):
    cmd = cmd_module.NewsCommand()
    # Mock _fetch_articles so we don't hit real RSS feeds
    monkeypatch.setattr(cmd, "_fetch_articles", lambda urls: _stub_articles(3))

    from core.request_information import RequestInformation
    req = RequestInformation(
        voice_command="news",
        conversation_id="c",
        is_validation_response=False,
        is_pre_routed=True,
    )
    resp = cmd.run(req)
    assert resp.context_data.get("message"), "expected pre-routed run() to compose a message"
    assert "Headline 1" in resp.context_data["message"]


def test_no_message_when_not_pre_routed(cmd_module, monkeypatch):
    cmd = cmd_module.NewsCommand()
    monkeypatch.setattr(cmd, "_fetch_articles", lambda urls: _stub_articles(2))

    from core.request_information import RequestInformation
    req = RequestInformation(
        voice_command="news",
        conversation_id="c",
        is_validation_response=False,
        is_pre_routed=False,
    )
    resp = cmd.run(req)
    # On the LLM path, no message — CC will compose from structured articles
    assert resp.context_data.get("message") is None


def test_compose_no_articles(cmd_module):
    msg = cmd_module._compose_news_message([], "general")
    assert "couldn't find" in msg.lower() or "no" in msg.lower()


def test_compose_with_category(cmd_module):
    msg = cmd_module._compose_news_message(_stub_articles(3), "tech")
    assert "tech" in msg.lower()
