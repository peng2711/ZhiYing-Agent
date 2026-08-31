from evaluation.evaluator import DEFAULT_DIALOG_CASES, DEFAULT_INTENT_CASES


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
