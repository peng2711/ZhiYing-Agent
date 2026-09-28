"""
知识库 MCP Server —— 用 Model Context Protocol 对外暴露知识库的只读能力。

任何 MCP Client（Claude Desktop、Claude Code、本项目的 tooling.mcp_client）都能通过
tools/list 发现这里的工具，再用 tools/call 调用，不需要了解 ChromaDB 或本项目的代码。

暴露范围刻意收窄：
  - 只有知识库检索和版本查询，全部是只读工具（readOnlyHint=True）。
  - 订单、退款、工单这些业务工具不走 MCP。MCP Server 拿不到调用方的登录身份，
    暴露它们等于绕过 api/main.py 的签名校验和订单归属检查。

启动方式：
  python -m tooling.mcp_server                                   # stdio，本地客户端拉起子进程
  python -m tooling.mcp_server --transport streamable-http       # HTTP，默认 127.0.0.1:8765/mcp

HTTP 模式没有接入鉴权，只应绑定本机地址；知识库内容是对外政策，不含用户数据。
"""

import argparse
import logging
import os
import pathlib
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SERVER_NAME = "zhiying-knowledge"
MAX_QUERY_CHARS = 2000
MAX_TOP_K = 10

class SearchOutput(BaseModel):
    """检索结果；声明成模型后，tools/list 会带上 outputSchema，客户端能拿到结构化结果。"""
    query: str
    results: List[Dict[str, Any]]


class VersionsOutput(BaseModel):
    versions: List[Dict[str, Any]]


_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def build_server(kb: Any, host: str = "127.0.0.1", port: int = 8765) -> FastMCP:
    """基于一个 KnowledgeBase 实例构建 MCP Server；测试时可以传入替身对象。"""
    server = FastMCP(
        SERVER_NAME,
        instructions="知应客服知识库：检索退款、发票、物流、登录排障等政策文档。只读。",
        host=host,
        port=port,
    )

    @server.tool(
        title="检索知识库",
        description=(
            "按语义检索知应客服知识库，返回最相关的政策片段及其来源、版本和相似度。"
            "只返回当前生效的版本。"
        ),
        annotations=_READ_ONLY,
    )
    async def search_knowledge_base(query: str, top_k: int = 5) -> SearchOutput:
        query = (query or "").strip()
        if not query:
            raise ValueError("query 不能为空")
        if len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query 不能超过 {MAX_QUERY_CHARS} 字")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k 必须在 1 到 {MAX_TOP_K} 之间")
        results = await kb.search_async(query, top_k=top_k)
        return SearchOutput(query=query, results=results)

    @server.tool(
        title="列出知识库文档版本",
        description="列出知识库中的文档及其版本、状态和生效日期；可按 source_id 过滤。",
        annotations=_READ_ONLY,
    )
    async def list_knowledge_versions(source_id: Optional[str] = None) -> VersionsOutput:
        return VersionsOutput(versions=await kb.list_versions_async(source_id or None))

    return server


def _default_kb() -> Any:
    """按与 api/main.py 相同的环境变量连接知识库。"""
    from tooling.embeddings import (
        collection_name_for, configured_embedding, configured_embedding_function, legacy_collection_for,
    )
    from tooling.knowledge_base import KnowledgeBase

    root = pathlib.Path(__file__).resolve().parent.parent
    embedding_model = configured_embedding()
    return KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "localhost").strip() or "localhost",
        chroma_port=int(os.getenv("CHROMA_PORT", "8001")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", str(root / "data" / "chroma")),
        embedding_function=configured_embedding_function(),
        collection_name=collection_name_for(embedding_model),
        migrate_from_collection=legacy_collection_for(embedding_model),
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="知应知识库 MCP Server")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    # stdio 模式下 stdout 是协议通道，日志只能写 stderr（logging 默认即 stderr）。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_server(_default_kb(), host=args.host, port=args.port).run(transport=args.transport)


if __name__ == "__main__":
    main()
