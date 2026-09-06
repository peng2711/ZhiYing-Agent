import json

import pytest

from evaluation.evaluator import (
    DEFAULT_DIALOG_CASES,
    DEFAULT_INTENT_CASES,
    EndToEndEvaluator,
    EvalReport,
)


def test_public_intent_dataset_has_balanced_core_categories():
    assert len(DEFAULT_INTENT_CASES) >= 100
    labels = {case.expected_intent for case in DEFAULT_INTENT_CASES}
    required = {
        "logistics", "order_status", "refund", "invoice", "payment_issue",
        "technical_login", "technical_crash", "account", "complaint",
        "human_handoff", "request", "greeting",
    }
    assert required <= labels
    assert min(sum(case.expected_intent == label for case in DEFAULT_INTENT_CASES) for label in required) >= 10


def test_public_dialog_dataset_contains_single_and_multi_turn_cases():
    assert len(DEFAULT_DIALOG_CASES) >= 20
    assert sum("question" in case for case in DEFAULT_DIALOG_CASES) >= 10
    assert sum(isinstance(case.get("turns"), list) and len(case["turns"]) >= 2 for case in DEFAULT_DIALOG_CASES) >= 10
    assert all(case.get("question") or case.get("turns") for case in DEFAULT_DIALOG_CASES)
    # 对话用例带有期望意图，评测器会将其用于路由轨迹校验。
    assert all(case.get("expected_intent") for case in DEFAULT_DIALOG_CASES)


def test_latest_report_can_be_explicitly_promoted_to_atomic_baseline(tmp_path):
    report = EvalReport(
        timestamp="2026-09-06T12:00:00", total=5, passed=5, pass_rate=1.0,
        avg_scores={"task_completion_rate": 1.0}, regressions=[],
        recommendations=[], results=[],
    )
    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator._history = [report]
    evaluator._baseline = None
    evaluator._baseline_path = tmp_path / "eval" / "baseline.json"

    promoted = evaluator.promote_latest_baseline(report.timestamp)
    assert promoted is report
    assert evaluator.baseline is report
    assert json.loads(evaluator._baseline_path.read_text(encoding="utf-8"))["timestamp"] == report.timestamp
    assert not evaluator._baseline_path.with_suffix(".json.tmp").exists()


def test_baseline_promotion_rejects_stale_report_timestamp(tmp_path):
    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator._history = [EvalReport(
        timestamp="new", total=0, passed=0, pass_rate=0.0, avg_scores={},
        regressions=[], recommendations=[], results=[],
    )]
    evaluator._baseline = None
    evaluator._baseline_path = tmp_path / "baseline.json"
    with pytest.raises(ValueError, match="已变化"):
        evaluator.promote_latest_baseline("old")
