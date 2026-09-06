"""
亮点：端到端 Agent 评测框架

核心问题：如何评测端到端 Agent？

评测维度：
  1. 意图识别准确率 —— 预测意图 vs 标注意图，计算 Accuracy / F1
  2. 响应质量评分 —— 用 LLM 作为评判者（LLM-as-Judge），
     从相关性、准确性、完整性、有用性四个维度打分
  3. 端到端对话评测 —— 模拟完整多轮对话，评估整体体验
  4. 回归测试 —— 与历史基线对比，防止性能退化

LLM-as-Judge 是评测 Agent 质量的关键技术：
  人工标注成本高、主观性强；用 LLM 评判可以规模化、可重复。
"""
import asyncio
import json
import logging
import os
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.llm_utils import extract_text_content
from core.llm_client import LLMClient, create_llm_client

from core.intent_recognizer import IntentCategory, IntentRecognizer
from evaluation.business_evaluator import BusinessWorkflowEvaluator

logger = logging.getLogger(__name__)


def _llm_timeout_s() -> float:
    try:
        return max(1.0, float(os.getenv("ZHIYING_LLM_TIMEOUT_S", "45")))
    except (TypeError, ValueError):
        return 45.0


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class IntentTestCase:
    message:          str
    expected_intent:  str
    context:          Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    """LLM-as-Judge 评分结果。"""
    relevance:    float   # 相关性：回答是否针对问题
    accuracy:     float   # 准确性：信息是否正确
    completeness: float   # 完整性：是否完整解决问题
    helpfulness:  float   # 有用性：用户是否能据此行动
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return statistics.mean([self.relevance, self.accuracy, self.completeness, self.helpfulness])


def _score(value: Any, default: float = 0.5) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class EvalResult:
    test_id:    str
    passed:     bool
    scores:     Dict[str, float]
    detail:     str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评测报告。"""
    timestamp:        str
    total:            int
    passed:           int
    pass_rate:        float
    avg_scores:       Dict[str, float]
    regressions:      List[str]          # 相比基线退化的指标
    recommendations:  List[str]
    results:          List[EvalResult]


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    用 LLM 评判 Agent 响应质量。

    为什么用 LLM 而不是人工？
    - 可规模化：数千条测试用例自动评测
    - 可重复：相同输入得到稳定评分
    - 多维度：同时评估相关性、准确性等多个维度

    注意：LLM Judge 本身也有偏差，建议定期用人工标注校准。
    """

    JUDGE_PROMPT = """你是一个客服质量评估专家。请对以下客服响应进行评分。

用户问题: {question}
Agent 响应: {response}
{context_section}

请从以下四个维度评分（0.0-1.0），返回 JSON：
- relevance: 响应是否直接针对用户问题（0=完全无关，1=完全相关）
- accuracy: 信息是否准确无误（0=明显错误，1=完全正确）
- completeness: 是否完整解决了用户需求（0=完全没解决，1=完全解决）
- helpfulness: 用户能否据此采取行动（0=毫无帮助，1=非常有帮助）

只返回 JSON，例如: {{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "helpfulness": 0.85}}"""

    def __init__(self, client: LLMClient, model: str):
        self._client = client
        self._model  = model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> QualityScores:
        ctx_section = f"背景信息: {context}" if context else ""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=ctx_section,
        )
        prompt = self._clean_text(prompt)
        try:
            resp = await asyncio.wait_for(
                self._client.messages.create(
                    model=self._model, max_tokens=256, temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=_llm_timeout_s(),
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            return QualityScores(
                relevance=_score(data.get("relevance")),
                accuracy=_score(data.get("accuracy")),
                completeness=_score(data.get("completeness")),
                helpfulness=_score(data.get("helpfulness")),
            )
        except Exception as ex:
            logger.warning(f"LLM Judge 失败: {ex}")
            return QualityScores(
                0.5, 0.5, 0.5, 0.5,
                judge_failed=True,
                error=str(ex),
            )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """评测意图识别的准确率和 F1。"""

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        predictions, ground_truth = [], []
        case_details: List[Dict[str, Any]] = []

        for case in cases:
            result = await self._recognizer.recognize(case.message)
            predicted = result.intent.value
            predictions.append(predicted)
            ground_truth.append(case.expected_intent)
            case_details.append({
                "message": case.message,
                "expected": case.expected_intent,
                "predicted": predicted,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            })

        # 纯 Python 计算指标
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0.0

        # 每类 F1
        labels = sorted(set(ground_truth + predictions))
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            tp = sum(p == label and g == label for p, g in zip(predictions, ground_truth))
            fp = sum(p == label and g != label for p, g in zip(predictions, ground_truth))
            fn = sum(p != label and g == label for p, g in zip(predictions, ground_truth))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

        macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "macro_f1":   round(macro_f1, 4),
            "per_class":  per_class,
            "total":      len(cases),
            "correct":    correct,
            "cases":      case_details,
        }


# ── 端到端评测器 ──────────────────────────────────────────────────────────────

class EndToEndEvaluator:
    """
    端到端 Agent 评测。

    评测流程：
      1. 运行意图识别评测（准确率/F1）
      2. 运行对话质量评测（LLM-as-Judge）
      3. 与历史基线对比（回归检测）
      4. 生成可操作的优化建议
    """

    # 质量及格线
    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        orchestrator,
        recognizer: IntentRecognizer,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        baseline_path: Optional[str] = None,
    ):
        client = create_llm_client(api_key=api_key, base_url=base_url)

        self._orchestrator     = orchestrator
        self._judge            = LLMJudge(client, model)
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._business_evaluator = BusinessWorkflowEvaluator()
        self._history:         List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline: Optional[EvalReport] = self._load_baseline()

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
    ) -> EvalReport:
        """
        运行完整评测。

        intent_cases: 意图识别测试用例
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
        """
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [], "accuracy": [], "completeness": [], "helpfulness": [],
            "dialog_intent_match": [],
        }

        # 1. 意图识别评测
        intent_metrics: Dict[str, Any] = {}
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = intent_metrics["accuracy"] >= self.PASS_THRESHOLD
            results.append(EvalResult(
                test_id="intent_recognition",
                passed=passed,
                scores={"accuracy": intent_metrics["accuracy"], "macro_f1": intent_metrics["macro_f1"]},
                detail=f"准确率 {intent_metrics['accuracy']:.1%}，Macro-F1 {intent_metrics['macro_f1']:.3f}",
                metadata={
                    "total": intent_metrics.get("total", 0),
                    "correct": intent_metrics.get("correct", 0),
                    "cases": intent_metrics.get("cases", []),
                },
            ))

        # 2. 对话质量评测（调用 orchestrator 产出回复，再用 LLM Judge 评分）
        if dialog_cases:
            for i, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, i)
                results.extend(case_results)
                for r in case_results:
                    for k in all_scores:
                        if k in r.scores:
                            all_scores[k].append(r.scores[k])

        # 3. 确定性业务闭环评测：使用临时数据库，不污染线上或演示订单。
        business_report = await self._business_evaluator.evaluate()
        for item in business_report["results"]:
            results.append(EvalResult(
                test_id=item["test_id"], passed=bool(item["passed"]),
                scores=dict(item.get("scores", {})), detail=str(item.get("detail", "")),
                metadata={"evaluation_type": "deterministic_business_e2e"},
            ))

        # 4. 汇总
        avg_scores = {k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v}
        avg_scores.update(business_report["metrics"])
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]
        rag_answers = [
            item for item in results
            if bool(item.metadata.get("knowledge_used"))
        ]
        if rag_answers:
            avg_scores["citation_coverage_rate"] = round(
                sum(bool(item.metadata.get("citation_count")) for item in rag_answers) / len(rag_answers),
                4,
            )

        passed_count = sum(1 for r in results if r.passed)
        pass_rate    = passed_count / len(results) if results else 0.0

        # 5. 回归检测
        regressions = self._detect_regressions(avg_scores)

        # 6. 优化建议
        recommendations = self._recommendations(avg_scores, intent_metrics)

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(pass_rate, 4),
            avg_scores=avg_scores,
            regressions=regressions,
            recommendations=recommendations,
            results=results,
        )
        self._history.append(report)
        # 评测不应自动覆盖回归基线；只有显式开启时才提升当前报告为 baseline。
        if os.getenv("EVAL_AUTO_PROMOTE_BASELINE", "false").lower() in {"1", "true", "yes"}:
            self._save_baseline(report)
        return report

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        """评测单轮或多轮对话用例。"""
        from agents.agent_orchestrator import Request as OrcReq

        questions = self._dialog_turns(case)
        if not questions:
            return []

        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
        expected_intent = str(case.get("expected_intent") or "").strip()
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []

        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            orch_req = OrcReq(
                message=question,
                user_id=user_id,
                conv_id=conv_id,
                context=context,
                history=history[-6:] if history else None,
            )
            orch_result = await self._orchestrator.run(orch_req)
            actual_answer = orch_result.response

            scores = await self._judge.judge(question, actual_answer, context=context or None)
            actual_intent = orch_result.intent.value if orch_result.intent else ""
            intent_match = (
                1.0 if not expected_intent else float(actual_intent == expected_intent)
            )
            # 对数据集提供了 expected_intent 的用例，把路由正确性纳入端到端
            # 通过条件。只看最终文本会掩盖“答得像但路由错了”的 Agent 回归。
            passed = (
                not scores.judge_failed
                and scores.overall >= self.PASS_THRESHOLD
                and intent_match >= 1.0
            )

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": actual_answer})

            test_id = f"dialog_{case_idx}" if len(questions) == 1 else f"dialog_{case_idx}_turn_{turn_idx}"
            results.append(EvalResult(
                test_id=test_id,
                passed=passed,
                scores={
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "overall": scores.overall,
                    "intent_match": intent_match,
                },
                detail=f"Q: {question[:30]}... → 综合评分 {scores.overall:.3f}",
                metadata={
                    "question": question,
                    "response": actual_answer,
                    "agent_type": orch_result.agent_type.value,
                    "intent": orch_result.intent.value if orch_result.intent else None,
                    "expected_intent": expected_intent or None,
                    "intent_match": bool(intent_match),
                    "turn": turn_idx,
                    "conv_id": conv_id,
                    "judge_failed": scores.judge_failed,
                    "judge_error": scores.error,
                    "knowledge_used": "search_knowledge_base" in orch_result.tools_used,
                    "citation_count": len(orch_result.citations),
                    "tools_used": list(orch_result.tools_used),
                },
            ))

        return results

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(t) for t in turns if str(t).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in history[-8:]]
        return "[评测多轮历史]\n" + "\n".join(lines)

    def _detect_regressions(self, current: Dict[str, float]) -> List[str]:
        """与上一次评测对比，找出退化超过 5% 的指标。"""
        prev_report = self._history[-1] if self._history else self._baseline
        if prev_report is None:
            return []
        prev = prev_report.avg_scores
        regressions = []
        for metric, value in current.items():
            if metric in {"unsafe_execution_rate", "business_p95_latency_ms"} and metric in prev:
                previous = prev[metric]
                worsened = value > previous and (
                    metric == "unsafe_execution_rate"
                    or previous <= 0
                    or (value - previous) / previous > 0.05
                )
                if worsened:
                    regressions.append(f"{metric}: {previous:.3f} → {value:.3f}（越低越好）")
                continue
            if metric in prev and prev[metric] > 0:
                delta = (value - prev[metric]) / prev[metric]
                if delta < -0.05:
                    regressions.append(
                        f"{metric}: {prev[metric]:.3f} → {value:.3f} (退化 {abs(delta):.1%})"
                    )
        return regressions

    def _recommendations(
        self,
        scores: Dict[str, float],
        intent_metrics: Dict[str, Any],
    ) -> List[str]:
        recs = []
        if scores.get("intent_accuracy", 1.0) < 0.90:
            recs.append("意图识别准确率 < 90%：增加 Few-shot 示例，或对低 F1 的意图类别补充训练数据")
        if scores.get("dialog_intent_match", 1.0) < 0.90:
            recs.append("对话路由命中率 < 90%：检查多轮上下文、细粒度意图边界和低置信度澄清策略")
        if scores.get("relevance", 1.0) < 0.80:
            recs.append("相关性偏低：检查 Agent system_prompt，确保 Agent 聚焦于用户问题")
        if scores.get("completeness", 1.0) < 0.80:
            recs.append("完整性偏低：Agent 可能过早结束回答，考虑在 prompt 中要求提供完整解决方案")
        if scores.get("helpfulness", 1.0) < 0.80:
            recs.append("有用性偏低：回答可能过于抽象，考虑要求 Agent 提供具体操作步骤")
        if scores.get("task_completion_rate", 1.0) < 1.0:
            recs.append("业务任务完成率未达 100%：检查任务状态迁移、工具参数和后台状态变化")
        if scores.get("tool_selection_accuracy", 1.0) < 1.0:
            recs.append("业务工具选择不准确：检查角色工具白名单和状态机步骤")
        if scores.get("unsafe_execution_rate", 0.0) > 0:
            recs.append("发现未确认业务写操作：阻断发布并检查服务端确认校验")
        if scores.get("citation_coverage_rate", 1.0) < 1.0:
            recs.append("RAG 引用覆盖率未达 100%：检查检索结果到最终回答的 citation 透传链路")
        if not recs:
            recs.append("所有指标均达标，继续保持")
        return recs

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    @property
    def baseline(self) -> Optional[EvalReport]:
        return self._baseline

    @property
    def latest_report(self) -> Optional[EvalReport]:
        return self._history[-1] if self._history else None

    def promote_latest_baseline(self, expected_timestamp: Optional[str] = None) -> EvalReport:
        """显式将最近一次评测设为基线，时间戳用于防止覆盖错报告。"""
        report = self.latest_report
        if report is None:
            raise ValueError("当前进程还没有可提升的评测报告")
        if expected_timestamp and report.timestamp != expected_timestamp:
            raise ValueError("评测报告已变化，请刷新后重试")
        if not self._baseline_path:
            raise RuntimeError("未配置评测基线存储路径")
        if not self._save_baseline(report):
            raise RuntimeError("评测基线保存失败")
        return report

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_baseline(self, report: EvalReport) -> bool:
        if not self._baseline_path:
            return False
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self._baseline_path.with_suffix(self._baseline_path.suffix + ".tmp")
            temporary_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(self._baseline_path)
            self._baseline = report
            return True
        except Exception as ex:
            logger.warning(f"保存评测基线失败: {ex}")
            return False

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=r.get("test_id", ""),
                    passed=bool(r.get("passed", False)),
                    scores=dict(r.get("scores", {})),
                    detail=r.get("detail", ""),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
        )


# ── 公开评测集（开箱即用）────────────────────────────────────────────────────

_DATASET_DIR = pathlib.Path(__file__).with_name("datasets")
_FALLBACK_INTENT_CASES = [
    {"message": "我的订单什么时候到？", "expected_intent": "logistics"},
    {"message": "帮我开发票", "expected_intent": "invoice"},
    {"message": "退款多久到账？", "expected_intent": "refund"},
    {"message": "登录一直报 401", "expected_intent": "technical_login"},
]
_FALLBACK_DIALOG_CASES = [
    {"question": "我的订单 #12345 还没到，已经超时了"},
    {"turns": ["你好，我想退款", "订单号是 #12345", "退款多久能到账？"]},
]


def _load_dataset_json(filename: str, fallback: Any) -> Any:
    """读取仓库中的公开评测集；文件缺失或格式错误时使用最小兜底集。"""
    path = _DATASET_DIR / filename
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else fallback
    except (OSError, json.JSONDecodeError, TypeError):
        logger.warning("评测集加载失败，使用兜底数据: %s", path)
        return fallback


def load_intent_cases(path: Optional[str] = None) -> List[IntentTestCase]:
    """加载意图评测集，也支持通过路径注入自定义数据集。"""
    source = pathlib.Path(path) if path else _DATASET_DIR / "intent_cases.json"
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        data = _load_dataset_json("intent_cases.json", _FALLBACK_INTENT_CASES)
    cases = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict) or not item.get("message") or not item.get("expected_intent"):
            continue
        cases.append(IntentTestCase(
            message=str(item["message"]),
            expected_intent=str(item["expected_intent"]),
            context=item.get("context") if isinstance(item.get("context"), dict) else None,
        ))
    return cases or [IntentTestCase(**item) for item in _FALLBACK_INTENT_CASES]


def load_dialog_cases(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """加载单轮/多轮对话评测集，也支持通过路径注入自定义数据集。"""
    source = pathlib.Path(path) if path else _DATASET_DIR / "dialog_cases.json"
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        data = _load_dataset_json("dialog_cases.json", _FALLBACK_DIALOG_CASES)
    valid = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and (item.get("question") or item.get("turns")):
            valid.append(item)
    return valid or list(_FALLBACK_DIALOG_CASES)


DEFAULT_INTENT_CASES: List[IntentTestCase] = load_intent_cases()
DEFAULT_DIALOG_CASES: List[Dict[str, Any]] = load_dialog_cases()
