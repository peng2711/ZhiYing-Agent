"""
离线运行端到端评测：意图识别、对话质量（LLM-as-Judge）和业务闭环。

与 POST /eval/run 使用同一个 EndToEndEvaluator，但不需要启动 Redis 和 ChromaDB 服务：
业务库、任务状态和知识库都建在临时目录或内存里，评测结束即删除，不影响演示数据。
编排器、Skills、业务工具、RAG 检索策略与线上一致。

模型按环节配置（未配置时用 LLM_MODEL）：
  ZHIYING_INTENT_MODEL / ZHIYING_REWRITE_MODEL / ZHIYING_RERANK_MODEL / ZHIYING_COMPOSER_MODEL /
  ZHIYING_MEMORY_MODEL / ZHIYING_JUDGE_MODEL，以及各 Agent 的 ZHIYING_<AGENT>_MODEL。

用法：
  python -m evaluation.run_eval --tag baseline --output report.json
  python -m evaluation.run_eval --limit-intent 5 --limit-dialog 2      # 冒烟
"""
import argparse
import asyncio
import json
import logging
import os
import pathlib
import tempfile
import time
from typing import Any, Dict, List, Optional

BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_ENV = (
    "LLM_MODEL", "ZHIYING_INTENT_MODEL", "ZHIYING_REWRITE_MODEL", "ZHIYING_RERANK_MODEL",
    "ZHIYING_COMPOSER_MODEL", "ZHIYING_MEMORY_MODEL", "ZHIYING_JUDGE_MODEL",
    "ZHIYING_GENERAL_MODEL", "ZHIYING_TECHNICAL_MODEL", "ZHIYING_BILLING_MODEL", "ZHIYING_ESCALATION_MODEL",
)


def _build_evaluator(workdir: pathlib.Path) -> Any:
    import chromadb

    from agents.agent_orchestrator import AgentOrchestrator, build_shared_rag_tools
    from agents.tools import build_business_tools
    from business import BusinessWorkflow, MockBusinessBackend
    from core.intent_recognizer import IntentRecognizer
    from core.skill_loader import SkillManager
    from evaluation.business_evaluator import _TaskStore
    from evaluation.evaluator import EndToEndEvaluator
    from tooling.embeddings import collection_name_for, configured_embedding, configured_embedding_function
    from tooling.knowledge_base import KnowledgeBase
    from tooling.tool_manager import Tool, ToolManager

    api_key = os.environ["LLM_API_KEY"]
    base_url = os.getenv("LLM_BASE_URL") or None
    model = os.getenv("LLM_MODEL", "qwen3.7-plus")

    skills = SkillManager(root_dir=str(BACKEND_DIR / "skills"))
    skills.load()
    orchestrator = AgentOrchestrator(api_key=api_key, base_url=base_url, model=model, skill_manager=skills)

    task_store = _TaskStore()
    backend = MockBusinessBackend(str(workdir / "business.db"))
    orchestrator.set_domain_tools(build_business_tools(backend, task_store))
    orchestrator.set_business_workflow(BusinessWorkflow(backend, task_store))

    tool_manager = ToolManager(api_key=api_key, base_url=base_url, model=model)
    kb = KnowledgeBase(
        client=chromadb.PersistentClient(
            path=str(workdir / "chroma"), settings=chromadb.Settings(anonymized_telemetry=False),
        ),
        embedding_function=configured_embedding_function(),
        collection_name=collection_name_for(configured_embedding()),
    )
    tool_manager.register(Tool(
        name="knowledge_search", description="搜索知识库", handler=kb.search_handler,
        schema={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query"]},
        cache_ttl=0.0, supports_rerank=True,
    ))
    orchestrator.set_shared_tools(build_shared_rag_tools(tool_manager))

    recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
    return EndToEndEvaluator(
        orchestrator=orchestrator, recognizer=recognizer,
        api_key=api_key, base_url=base_url, model=model, baseline_path=None,
    )


def _merge_usage(summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    by_model: Dict[str, Dict[str, int]] = {}
    for summary in summaries:
        for name, bucket in (summary.get("by_model") or {}).items():
            total = by_model.setdefault(name, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            for key in total:
                total[key] += bucket.get(key, 0)
    return by_model


async def run(tag: str, limit_intent: Optional[int], limit_dialog: Optional[int]) -> Dict[str, Any]:
    from core.usage import track_request_usage
    from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES

    intent_cases = DEFAULT_INTENT_CASES[:limit_intent] if limit_intent is not None else DEFAULT_INTENT_CASES
    dialog_cases = DEFAULT_DIALOG_CASES[:limit_dialog] if limit_dialog is not None else DEFAULT_DIALOG_CASES

    with tempfile.TemporaryDirectory(prefix="zhiying_eval_") as tmp:
        evaluator = _build_evaluator(pathlib.Path(tmp))
        t0 = time.monotonic()
        # 外层统计意图评测和 Judge；每轮对话在评测器内部单独统计（线上链路），两者相加即全部调用。
        with track_request_usage() as outer:
            report = await evaluator.run(intent_cases=intent_cases, dialog_cases=dialog_cases)
        elapsed = time.monotonic() - t0

    turn_usages = [r.metadata["token_usage"] for r in report.results if r.metadata.get("token_usage")]
    dialog_overall = [r.scores["overall"] for r in report.results
                      if "overall" in r.scores and not r.metadata.get("judge_failed")]
    if dialog_overall:
        report.avg_scores["overall"] = round(sum(dialog_overall) / len(dialog_overall), 4)
    intent_result = next((r for r in report.results if r.test_id == "intent_recognition"), None)
    intent_errors = [
        {k: case.get(k) for k in ("message", "expected", "predicted", "confidence")}
        for case in (intent_result.metadata.get("cases", []) if intent_result else [])
        if case.get("expected") != case.get("predicted")
    ]
    return {
        "tag": tag,
        "models": {name: os.getenv(name) for name in MODEL_ENV if os.getenv(name)},
        "dataset": {"intent_cases": len(intent_cases), "dialog_cases": len(dialog_cases)},
        "elapsed_s": round(elapsed, 1),
        "pass_rate": report.pass_rate,
        "avg_scores": report.avg_scores,
        "intent_errors": intent_errors,
        "intent_confusions": _confusions(intent_errors),
        "usage_by_model": {
            "serving": _merge_usage(turn_usages),
            "intent_eval_and_judge": _merge_usage([outer.summary()]),
        },
        "results": [
            {"test_id": r.test_id, "passed": r.passed, "scores": r.scores, "detail": r.detail,
             **{k: r.metadata.get(k) for k in ("question", "response", "intent", "expected_intents",
                                                "judge_failed", "judge_error", "tools_used")
                if k in r.metadata}}
            for r in report.results
        ],
    }


def _confusions(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把错例按"预期 → 预测"分组计数，最常见的混淆排在前面。"""
    counts: Dict[tuple, int] = {}
    for error in errors:
        key = (error["expected"], error["predicted"])
        counts[key] = counts.get(key, 0) + 1
    return [{"expected": e, "predicted": p, "count": c}
            for (e, p), c in sorted(counts.items(), key=lambda item: -item[1])]


def _print_summary(result: Dict[str, Any]) -> None:
    s = result["avg_scores"]
    keys = ("intent_accuracy", "overall", "relevance", "accuracy", "completeness", "helpfulness",
            "dialog_intent_match", "primary_task_retention", "citation_coverage_rate",
            "unsafe_execution_rate", "avg_llm_calls_per_turn", "avg_tokens_per_turn")
    print(f"[{result['tag']}] 模型: {result['models']}")
    print(f"耗时 {result['elapsed_s']}s，通过率 {result['pass_rate']:.1%}")
    for key in keys:
        if key in s:
            print(f"  {key:<24}{s[key]}")
    judge_failed = sum(1 for r in result["results"] if r.get("judge_failed"))
    print(f"  judge_failed            {judge_failed}")
    if result["intent_errors"]:
        print(f"  意图错例 {len(result['intent_errors'])} 条，按混淆类型：")
        for item in result["intent_confusions"]:
            print(f"    {item['expected']} → {item['predicted']}: {item['count']}")
        for error in result["intent_errors"]:
            print(f"    [{error['expected']} → {error['predicted']}] {error['message']}")
    for scope, models in result["usage_by_model"].items():
        for name, bucket in models.items():
            print(f"  tokens[{scope}] {name}: {bucket}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="离线运行知应端到端评测")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--limit-intent", type=int)
    parser.add_argument("--limit-dialog", type=int)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

    result = asyncio.run(run(args.tag, args.limit_intent, args.limit_dialog))
    _print_summary(result)
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整报告：{args.output}")


if __name__ == "__main__":
    main()
