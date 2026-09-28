import asyncio
import types

import pytest

from agents.agent_orchestrator import ResponseComposer
from core.intent_recognizer import IntentRecognizer
from core.llm_client import role_model
from evaluation.evaluator import LLMJudge
from tooling.tool_manager import ToolManager

ROLE_ENV = ["ZHIYING_INTENT_MODEL", "ZHIYING_REWRITE_MODEL", "ZHIYING_RERANK_MODEL",
            "ZHIYING_COMPOSER_MODEL", "ZHIYING_MEMORY_MODEL", "ZHIYING_JUDGE_MODEL"]


@pytest.fixture(autouse=True)
def clean_role_env(monkeypatch):
    for name in ROLE_ENV:
        monkeypatch.delenv(name, raising=False)


def test_role_model_falls_back_to_default_and_rejects_unknown_roles(monkeypatch):
    assert role_model("intent", "base") == "base"
    monkeypatch.setenv("ZHIYING_INTENT_MODEL", "  small  ")
    assert role_model("intent", "base") == "small"
    with pytest.raises(ValueError):
        role_model("planner", "base")


def test_each_component_reads_its_own_model(monkeypatch):
    monkeypatch.setenv("ZHIYING_INTENT_MODEL", "flash-intent")
    monkeypatch.setenv("ZHIYING_REWRITE_MODEL", "flash-rewrite")
    monkeypatch.setenv("ZHIYING_RERANK_MODEL", "flash-rerank")
    monkeypatch.setenv("ZHIYING_COMPOSER_MODEL", "plus-composer")
    monkeypatch.setenv("ZHIYING_JUDGE_MODEL", "other-family-judge")

    manager = ToolManager(api_key="k", model="base")
    assert IntentRecognizer(api_key="k", model="base").model == "flash-intent"
    assert (manager._model, manager._rewrite_model, manager._rerank_model) == ("base", "flash-rewrite", "flash-rerank")
    assert ResponseComposer(object(), "base")._model == "plus-composer"
    assert LLMJudge(object(), "base")._model == "other-family-judge"


def test_rerank_request_is_sent_with_the_rerank_model(monkeypatch):
    monkeypatch.setenv("ZHIYING_RERANK_MODEL", "flash-rerank")
    manager = ToolManager(api_key="k", model="base")
    sent = []

    class Messages:
        async def create(self, **kwargs):
            sent.append(kwargs["model"])
            return types.SimpleNamespace(content=[{"type": "text", "text": "[1, 0]"}])

    manager._client = types.SimpleNamespace(messages=Messages())
    ranked = asyncio.run(manager._rerank("q", [{"title": "a"}, {"title": "b"}], top_k=2))

    assert sent == ["flash-rerank"]
    assert [item["title"] for item in ranked] == ["b", "a"]
