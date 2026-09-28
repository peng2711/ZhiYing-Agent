import asyncio

from tooling.tool_manager import ToolManager, ToolResult


def test_parallel_recall_deduplicates_same_version_chunk_and_keeps_best_score():
    manager = ToolManager.__new__(ToolManager)

    async def rewrite_query(query, n=3):
        return ["query-a", "query-b"]

    async def call(name, params, context=None, use_cache=True):
        score = 0.7 if params["query"] == "query-a" else 0.9
        return ToolResult(success=True, tool_name=name, data=[{
            "source_id": "refund-policy", "version": "2.0", "chunk": 0,
            "content": "十五天退款", "score": score,
        }])

    async def rerank(query, items, top_k):
        return items[:top_k]

    manager.rewrite_query = rewrite_query
    manager.call = call
    manager._rerank = rerank

    result = asyncio.run(manager.search_with_rewrite("knowledge_search", "退款", top_k=5))
    assert len(result.data) == 1
    assert result.data[0]["score"] == 0.9


def _policy_manager(scores, fallback=False):
    """call() 返回给定分数的候选；记录 rewrite_query / _rerank 是否被调用。"""
    manager = ToolManager.__new__(ToolManager)
    manager.calls = {"rewrite": 0, "rerank": 0, "call": 0}

    async def rewrite_query(query, n=3):
        manager.calls["rewrite"] += 1
        return [query, query + "？"]

    async def call(name, params, context=None, use_cache=True):
        manager.calls["call"] += 1
        data = [{"source_id": f"doc{i}", "version": "1.0", "chunk": 0, "title": f"标题{i}",
                 "content": f"正文{i}", "score": score} for i, score in enumerate(scores)]
        return ToolResult(success=True, tool_name=name, data=data, fallback_used=fallback)

    async def rerank(query, items, top_k):
        manager.calls["rerank"] += 1
        return list(reversed(items))[:top_k]

    manager.rewrite_query = rewrite_query
    manager.call = call
    manager._rerank = rerank
    return manager


def test_auto_rerank_only_when_top_documents_are_close():
    close = _policy_manager([0.71, 0.70, 0.50])
    result = asyncio.run(close.search("kb", "退款", top_k=2, rerank="auto", rerank_margin=0.03))
    assert close.calls["rerank"] == 1 and result.reranked is True

    apart = _policy_manager([0.85, 0.70, 0.50])
    result = asyncio.run(apart.search("kb", "退款", top_k=2, rerank="auto", rerank_margin=0.03))
    assert apart.calls["rerank"] == 0 and result.reranked is False
    assert [item["source_id"] for item in result.data] == ["doc0", "doc1"]


def test_rerank_policies_always_and_never():
    always = _policy_manager([0.85, 0.70])
    asyncio.run(always.search("kb", "q", rerank="always"))
    never = _policy_manager([0.71, 0.70])
    asyncio.run(never.search("kb", "q", rerank="never"))
    assert (always.calls["rerank"], never.calls["rerank"]) == (1, 0)


def test_rewrite_is_off_by_default_and_rewrite_path_still_works():
    plain = _policy_manager([0.9, 0.5])
    asyncio.run(plain.search("kb", "q"))
    assert plain.calls == {"rewrite": 0, "rerank": 0, "call": 1}

    full = _policy_manager([0.9, 0.5])
    asyncio.run(full.search_with_rewrite("kb", "q"))
    assert full.calls == {"rewrite": 1, "rerank": 1, "call": 2}


def test_degraded_results_are_not_reranked_and_fallback_is_reported():
    manager = _policy_manager([0.0, 0.0], fallback=True)
    result = asyncio.run(manager.search("kb", "q", rerank="always"))
    assert manager.calls["rerank"] == 0
    assert result.fallback_used is True


def test_rerank_snippet_shows_content_not_metadata():
    item = {"source_id": "refund-quality", "version": "1.0", "updated_at": "2026-01-01",
            "document_name": "质量问题退款", "section": "质量问题退款", "status": "active",
            "title": "质量问题退款", "content": "商品存在质量问题（破损、功能故障）时，签收后 30 天内可以申请退款。"}
    snippet = ToolManager._rerank_snippet(item)
    assert snippet.startswith("【质量问题退款】商品存在质量问题")
    assert "updated_at" not in snippet and "2026-01-01" not in snippet


def test_rerank_keeps_candidates_the_model_left_out():
    import types

    manager = ToolManager.__new__(ToolManager)
    manager._model = "m"

    class Messages:
        async def create(self, **kwargs):
            return types.SimpleNamespace(content=[{"type": "text", "text": "[2, 2, 0]"}])

    manager._client = types.SimpleNamespace(messages=Messages())
    items = [{"title": str(i), "content": str(i)} for i in range(4)]
    ranked = asyncio.run(manager._rerank("q", items, top_k=4))
    assert [item["title"] for item in ranked] == ["2", "0", "1", "3"]


def test_agent_rag_tool_reads_policy_from_env(monkeypatch):
    from agents.tools import build_shared_rag_tools

    seen = {}

    class Manager:
        async def search(self, tool_name, query, top_k=5, **policy):
            seen.update(policy)
            return ToolResult(success=True, tool_name=tool_name, data=[])

    monkeypatch.setenv("ZHIYING_RAG_QUERY_REWRITE", "true")
    monkeypatch.setenv("ZHIYING_RAG_RERANK", "never")
    monkeypatch.setenv("ZHIYING_RAG_RERANK_MARGIN", "0.05")
    tool = build_shared_rag_tools(Manager())["search_knowledge_base"]
    asyncio.run(tool.handler(type("Req", (), {"message": "退款"})(), {"query": "退款"}))
    assert seen == {"rewrite": True, "rerank": "never", "rerank_margin": 0.05}

    monkeypatch.delenv("ZHIYING_RAG_QUERY_REWRITE")
    monkeypatch.delenv("ZHIYING_RAG_RERANK")
    monkeypatch.delenv("ZHIYING_RAG_RERANK_MARGIN")
    asyncio.run(tool.handler(type("Req", (), {"message": "退款"})(), {"query": "退款"}))
    assert seen == {"rewrite": False, "rerank": "auto", "rerank_margin": 0.03}
