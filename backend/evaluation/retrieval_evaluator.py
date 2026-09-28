"""
检索评测：在隔离的临时知识库上量化 RAG 的召回质量。

指标（按文档 source_id 计算，同一文档的多个 chunk 只算一次）：
  - Hit@K：前 K 个结果里至少有一个正确文档的查询占比
  - Recall@K：前 K 个结果覆盖了多少比例的正确文档（有的查询有多个正确文档）
  - MRR@10：第一个正确文档排名的倒数，取平均

检索模式，用来拆开看每个环节的贡献：
  - vector：纯向量检索
  - rewrite：LLM 查询改写 → 多路召回 → 按最高分合并
  - rerank：向量召回 10 条 → 全部交给 LLM 重排
  - auto-rerank：向量召回 10 条，前两个文档分数接近时才重排（Agent 检索工具的默认策略）
  - rewrite+rerank：search_with_rewrite 的完整链路（改写 + 每个子查询召回 5 条 + 重排）
后四种需要 LLM（读取与服务相同的 LLM_API_KEY / LLM_MODEL / LLM_BASE_URL 配置）。

用法：
  python -m evaluation.retrieval_evaluator --embedding default bge-small-zh
  python -m evaluation.retrieval_evaluator --embedding bge-small-zh --modes vector rewrite rerank rewrite+rerank
"""
import argparse
import asyncio
import json
import logging
import os
import pathlib
import statistics
import tempfile
import time
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Set

import chromadb

from core.usage import track_request_usage
from tooling.embeddings import SUPPORTED_EMBEDDINGS, collection_name_for, get_embedding_function
from tooling.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

DATASET_DIR = pathlib.Path(__file__).resolve().parent / "datasets"
KS = (1, 3, 5)
MRR_DEPTH = 10
LLM_MODES = ("rewrite", "rerank", "auto-rerank", "rewrite+rerank")
MODES = ("vector", *LLM_MODES)

Searcher = Callable[[str], Awaitable[List[Dict[str, Any]]]]


def load_dataset(
    corpus_path: Optional[pathlib.Path] = None, cases_path: Optional[pathlib.Path] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    corpus = json.loads((corpus_path or DATASET_DIR / "retrieval_corpus.json").read_text(encoding="utf-8"))
    cases = json.loads((cases_path or DATASET_DIR / "retrieval_cases.json").read_text(encoding="utf-8"))
    return corpus["documents"], cases["cases"]


def ranked_sources(results: Iterable[Dict[str, Any]]) -> List[str]:
    """按出现顺序去重：同一文档的多个 chunk 只保留排名最靠前的一次。"""
    seen: List[str] = []
    for item in results:
        source = item.get("source_id")
        if source and source not in seen:
            seen.append(source)
    return seen


def score_query(ranked: Sequence[str], relevant: Set[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for k in KS:
        top = set(ranked[:k])
        scores[f"hit@{k}"] = float(bool(top & relevant))
        scores[f"recall@{k}"] = len(top & relevant) / len(relevant)
    first = next((i for i, source in enumerate(ranked[:MRR_DEPTH], start=1) if source in relevant), None)
    scores["mrr"] = 1.0 / first if first else 0.0
    return scores


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = [*(f"hit@{k}" for k in KS), *(f"recall@{k}" for k in KS), "mrr"]
    overall = {name: round(statistics.mean(r["scores"][name] for r in rows), 4) for name in metric_names}
    by_category: Dict[str, Dict[str, float]] = {}
    for category in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == category]
        by_category[category] = {
            "count": len(subset),
            "hit@1": round(statistics.mean(r["scores"]["hit@1"] for r in subset), 4),
            "hit@3": round(statistics.mean(r["scores"]["hit@3"] for r in subset), 4),
            "mrr": round(statistics.mean(r["scores"]["mrr"] for r in subset), 4),
        }
    return {"overall": overall, "by_category": by_category}


async def evaluate(searcher: Searcher, cases: List[Dict[str, Any]], concurrency: int = 4) -> Dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: List[float] = []

    async def run(case: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            t0 = time.monotonic()
            results = await searcher(case["query"])
            latencies.append((time.monotonic() - t0) * 1000)
        ranked = ranked_sources(results)
        relevant = set(case["relevant"])
        return {
            "query": case["query"],
            "category": case.get("category", "uncategorized"),
            "relevant": sorted(relevant),
            "ranked": ranked[:5],
            "scores": score_query(ranked, relevant),
        }

    rows = await asyncio.gather(*(run(case) for case in cases))
    report = aggregate(list(rows))
    report["latency_ms_p50"] = round(statistics.median(latencies), 1)
    report["misses"] = [
        {k: row[k] for k in ("query", "category", "relevant", "ranked")}
        for row in rows if row["scores"]["hit@3"] == 0
    ]
    return report


class _FallbackCounter(logging.Handler):
    """改写和重排失败时会静默退回原始结果，只打 warning；评测必须把这些次数单独报出来，
    否则"LLM 调用失败"会被误读成"改写/重排没有效果"。"""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "失败" in record.getMessage():
            self.count += 1


def build_knowledge_base(
    embedding: str, documents: List[Dict[str, Any]], workdir: str, cache_dir: Optional[str] = None,
) -> KnowledgeBase:
    """在临时目录里建一个只包含评测语料的知识库，不碰线上数据。"""
    client = chromadb.PersistentClient(path=workdir, settings=chromadb.Settings(anonymized_telemetry=False))
    kb = KnowledgeBase(
        client=client,
        embedding_function=get_embedding_function(embedding, cache_dir=cache_dir),
        collection_name=collection_name_for(embedding, base="retrieval_eval"),
        load_default_docs=False,
    )
    # 先导入旧版本再导入新版本，走真实的版本生命周期（激活新版本会停用同源旧版本）。
    kb.add_documents(sorted(documents, key=lambda d: (d.get("status", "active") == "active", d.get("version", "1.0"))))
    return kb


def build_searchers(
    kb: KnowledgeBase, modes: Sequence[str], top_k: int, rerank_margin: float = 0.03,
) -> Dict[str, Searcher]:
    searchers: Dict[str, Searcher] = {}
    if "vector" in modes:
        async def vector(query: str) -> List[Dict[str, Any]]:
            return await kb.search_async(query, top_k=MRR_DEPTH)
        searchers["vector"] = vector

    llm_modes = [m for m in modes if m in LLM_MODES]
    if not llm_modes:
        return searchers

    from tooling.tool_manager import Tool, ToolManager

    api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(f"模式 {llm_modes} 需要 LLM：请先配置 LLM_API_KEY（与服务使用同一套配置）")
    manager = ToolManager(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL") or None,
        model=os.getenv("LLM_MODEL", "qwen3.7-plus"),
    )
    manager.register(Tool(
        name="knowledge_search", description="评测知识库", handler=kb.search_handler,
        schema={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query"]},
        cache_ttl=0.0, supports_rerank=True,
    ))

    if "rewrite" in modes:
        async def rewrite(query: str) -> List[Dict[str, Any]]:
            sub_queries = await manager.rewrite_query(query, n=3)
            batches = await asyncio.gather(*(kb.search_async(q, top_k=MRR_DEPTH) for q in sub_queries))
            best: Dict[tuple, Dict[str, Any]] = {}
            for item in (i for batch in batches for i in batch):
                key = (item["source_id"], item["version"], item["chunk"])
                if key not in best or item["score"] > best[key]["score"]:
                    best[key] = item
            return sorted(best.values(), key=lambda item: item["score"], reverse=True)
        searchers["rewrite"] = rewrite

    def policy_searcher(policy: str) -> Searcher:
        async def search(query: str) -> List[Dict[str, Any]]:
            result = await manager.search(
                "knowledge_search", query, top_k=MRR_DEPTH, recall_k=MRR_DEPTH,
                rewrite=False, rerank=policy, rerank_margin=rerank_margin,
            )
            search.reranked += int(result.reranked)
            return result.data if result.success else []
        search.reranked = 0
        return search

    if "rerank" in modes:
        searchers["rerank"] = policy_searcher("always")
    if "auto-rerank" in modes:
        searchers["auto-rerank"] = policy_searcher("auto")

    if "rewrite+rerank" in modes:
        async def full(query: str) -> List[Dict[str, Any]]:
            result = await manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
            return result.data if result.success else []
        searchers["rewrite+rerank"] = full

    return searchers


async def run(
    embeddings: Sequence[str], modes: Sequence[str], top_k: int, cache_dir: Optional[str],
    rerank_margin: float = 0.03,
) -> Dict[str, Any]:
    documents, cases = load_dataset()
    report: Dict[str, Any] = {
        "dataset": {"documents": len(documents), "queries": len(cases)},
        "runs": [],
    }
    for embedding in embeddings:
        with tempfile.TemporaryDirectory(prefix="zhiying_retrieval_eval_") as workdir:
            t0 = time.monotonic()
            kb = build_knowledge_base(embedding, documents, workdir, cache_dir=cache_dir)
            index_s = round(time.monotonic() - t0, 1)
            for mode, searcher in build_searchers(kb, modes, top_k, rerank_margin).items():
                counter = _FallbackCounter()
                tool_logger = logging.getLogger("tooling.tool_manager")
                tool_logger.addHandler(counter)
                try:
                    with track_request_usage() as usage:
                        result = await evaluate(searcher, cases, concurrency=1 if mode == "vector" else 4)
                finally:
                    tool_logger.removeHandler(counter)
                spent = usage.summary()
                if hasattr(searcher, "reranked"):
                    result["rerank_rate"] = round(searcher.reranked / max(len(cases), 1), 3)
                result["llm_fallbacks"] = counter.count
                result["llm_calls"] = spent["llm_calls"]
                result["tokens_per_query"] = round(
                    (spent["input_tokens"] + spent["output_tokens"]) / max(len(cases), 1), 1,
                )
                report["runs"].append({"embedding": embedding, "mode": mode, "index_seconds": index_s, **result})
    return report


def format_table(report: Dict[str, Any]) -> str:
    header = (f"{'embedding':<14}{'mode':<16}{'Hit@1':>7}{'Hit@3':>7}{'Hit@5':>7}{'Recall@5':>10}{'MRR':>7}"
              f"{'p50 ms':>9}{'tok/q':>8}{'fallback':>10}")
    lines = [header, "-" * len(header)]
    for r in report["runs"]:
        o = r["overall"]
        lines.append(
            f"{r['embedding']:<14}{r['mode']:<16}{o['hit@1']:>7.3f}{o['hit@3']:>7.3f}{o['hit@5']:>7.3f}"
            f"{o['recall@5']:>10.3f}{o['mrr']:>7.3f}{r['latency_ms_p50']:>9.1f}"
            f"{r.get('tokens_per_query', 0):>8.0f}{r.get('llm_fallbacks', 0):>10}"
            + (f"   rerank {r['rerank_rate']:.0%}" if "rerank_rate" in r else "")
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="知应 RAG 检索评测")
    parser.add_argument("--embedding", nargs="+", default=["default"], choices=SUPPORTED_EMBEDDINGS)
    parser.add_argument("--modes", nargs="+", default=["vector"], choices=MODES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-margin", type=float, default=0.03, help="auto-rerank 的触发阈值")
    parser.add_argument("--cache-dir", default=os.getenv("ZHIYING_EMBEDDING_CACHE_DIR") or None)
    parser.add_argument("--output", help="把完整报告（含每类指标和未命中查询）写入 JSON 文件")
    args = parser.parse_args(argv)

    # 与 API 服务读取同一份 backend/.env，LLM 模式才能拿到 Key 和模型配置。
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")

    logging.basicConfig(level=logging.WARNING)
    # chromadb 0.5 与新版 posthog 不兼容，关闭遥测后仍会打印发送失败的错误日志，与评测无关。
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    report = asyncio.run(run(args.embedding, args.modes, args.top_k, args.cache_dir, args.rerank_margin))
    print(f"数据集：{report['dataset']['documents']} 篇文档版本，{report['dataset']['queries']} 条查询\n")
    print(format_table(report))
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整报告：{args.output}")


if __name__ == "__main__":
    main()
