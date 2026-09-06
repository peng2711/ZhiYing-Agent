from core.intent_recognizer import IntentCategory
from core.task_intent import TaskIntentTracker


def test_compound_message_keeps_all_discovered_tasks():
    tracker = TaskIntentTracker()
    state = tracker.update(
        None,
        IntentCategory.ACCOUNT,
        "我想注销账户，但账户里还有一笔未完成订单，可以直接注销吗？",
    )
    assert state["turn_intent"] == "account"
    assert state["active_intent"] == "account"
    assert {"account", "order_status"} <= set(state["primary_intents"])


def test_explicit_topic_switch_changes_active_intent_without_losing_previous_goal():
    tracker = TaskIntentTracker()
    state = tracker.update(None, IntentCategory.TECHNICAL_LOGIN, "登录报 401，而且被重复扣款")
    state = tracker.update(state, IntentCategory.TECHNICAL_LOGIN, "先告诉我登录怎么排查")
    state = tracker.update(state, IntentCategory.PAYMENT_ISSUE, "现在说明扣款需要哪些凭证")
    assert state["active_intent"] == "payment_issue"
    assert {"technical_login", "payment_issue"} <= set(state["primary_intents"])
    assert state["turn"] == 3


def test_generic_followup_does_not_erase_active_task():
    tracker = TaskIntentTracker()
    state = tracker.update(None, IntentCategory.REFUND, "我想申请退款")
    state = tracker.update(state, IntentCategory.QUERY, "那需要多久？")
    assert state["turn_intent"] == "query"
    assert state["active_intent"] == "refund"
    assert state["primary_intents"] == ["refund"]


def test_explicit_keywords_recover_active_task_when_model_returns_other():
    tracker = TaskIntentTracker()
    state = tracker.update(None, IntentCategory.REFUND, "我想退款，也需要开票")
    state = tracker.update(state, IntentCategory.ORDER_STATUS, "订单还没完成")
    state = tracker.update(state, IntentCategory.OTHER, "请分别告诉我退款和开票的条件")
    assert state["turn_intent"] == "other"
    assert state["active_intent"] == "refund"
    assert {"refund", "invoice", "order_status"} <= set(state["primary_intents"])
