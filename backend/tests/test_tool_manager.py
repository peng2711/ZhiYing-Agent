import asyncio

from mcp.tool_manager import MCPToolManager, ToolResult


def test_parallel_recall_deduplicates_same_version_chunk_and_keeps_best_score():
    manager = MCPToolManager.__new__(MCPToolManager)

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
