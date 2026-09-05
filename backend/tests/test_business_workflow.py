import asyncio
from types import SimpleNamespace

import pytest

from agents.tools import build_business_tools
from business.mock_backend import BusinessError, MockBusinessBackend
from business.workflow import BusinessWorkflow
from evaluation.business_evaluator import BusinessWorkflowEvaluator


class IntentCategory:
    REFUND = SimpleNamespace(value="refund")
    OTHER = SimpleNamespace(value="other")


class FakeTaskStore:
    def __init__(self): self.values = {}
    async def get_task_state(self, user_id, conv_id): return self.values.get((user_id, conv_id))
    async def set_task_state(self, user_id, conv_id, state, ttl_s=900): self.values[(user_id, conv_id)] = state
    async def clear_task_state(self, user_id, conv_id): self.values.pop((user_id, conv_id), None)


def request(message, intent=IntentCategory.REFUND, entities=None):
    return SimpleNamespace(message=message, user_id="guest-test", conv_id="conv-test",
                           request_id="req-test", intent=intent, entities=entities or {}, context="")


def test_refund_requires_confirmation_and_is_idempotent(tmp_path):
    backend = MockBusinessBackend(str(tmp_path / "business.db"))
    prepared = backend.prepare_refund("10086", "guest-test", "conv-test", "不满意")
    with pytest.raises(BusinessError, match="明确确认"):
        backend.execute_refund(prepared["operation_id"], prepared["confirmation_token"], "guest-test", confirmed=False)
    first = backend.execute_refund(prepared["operation_id"], prepared["confirmation_token"], "guest-test", confirmed=True)
    second = backend.execute_refund(prepared["operation_id"], prepared["confirmation_token"], "guest-test", confirmed=True)
    assert first["refund_id"] == second["refund_id"]
    assert second["idempotent"] is True


def test_multiturn_task_collects_order_then_executes(tmp_path):
    backend, store = MockBusinessBackend(str(tmp_path / "business.db")), FakeTaskStore()
    workflow = BusinessWorkflow(backend, store)
    ask = asyncio.run(workflow.handle(request("我要退款")))
    assert ask.pending_action["step"] == "waiting_order_id"
    prepared = asyncio.run(workflow.handle(request("10086", IntentCategory.OTHER)))
    assert prepared.pending_action["step"] == "pending_confirmation"
    assert "confirmation_token" not in prepared.pending_action
    assert backend.get_order("10086", "guest-test")["status"] == "in_transit"
    completed = asyncio.run(workflow.handle(request("确认退款", IntentCategory.OTHER)))
    assert completed.tools_used == ["execute_refund"]
    assert backend.get_refund_status("10086", "guest-test")["status"] == "processing"


def test_refund_policy_question_is_not_treated_as_refund_action(tmp_path):
    backend, store = MockBusinessBackend(str(tmp_path / "business.db")), FakeTaskStore()
    workflow = BusinessWorkflow(backend, store)
    outcome = asyncio.run(workflow.handle(request("退款政策是什么？请给出依据")))
    assert outcome is None
    assert store.values == {}


def test_cancel_has_no_side_effect(tmp_path):
    backend, store = MockBusinessBackend(str(tmp_path / "business.db")), FakeTaskStore()
    workflow = BusinessWorkflow(backend, store)
    asyncio.run(workflow.handle(request("订单号 10086 申请退款", entities={"order_id": ["10086"]})))
    outcome = asyncio.run(workflow.handle(request("取消", IntentCategory.OTHER)))
    assert "没有改变" in outcome.response
    assert backend.get_order("10086", "guest-test")["status"] == "in_transit"


def test_ticket_persists_and_prepare_token_stays_server_side(tmp_path):
    path = str(tmp_path / "business.db")
    backend, store = MockBusinessBackend(path), FakeTaskStore()
    ticket = backend.create_ticket("guest-test", "conv-test", "payment_issue", "high", "重复扣款", {"order_id": "10086"})
    assert MockBusinessBackend(path).get_ticket(ticket["ticket_id"], "guest-test")["status"] == "open"
    tool = build_business_tools(backend, store)["billing"]["prepare_refund"]
    req = request("申请退款", entities={"order_id": ["10086"]})
    result = asyncio.run(tool.handler(req, {"order_id": "10086", "reason": "不满意"}))
    assert "confirmation_token" not in result
    assert store.values[(req.user_id, req.conv_id)]["confirmation_token"]
    assert tool.requires_confirmation and tool.risk_level == "write"


def test_demo_reset_restores_seed_state(tmp_path):
    backend = MockBusinessBackend(str(tmp_path / "business.db"))
    operation = backend.prepare_refund("10086", "guest-test", "conv-test", "测试")
    backend.execute_refund(operation["operation_id"], operation["confirmation_token"], "guest-test", confirmed=True)
    backend.create_ticket("guest-test", "conv-test", "refund", "medium", "测试工单", {})
    result = backend.reset_demo_data()
    assert result["orders"] == 3
    assert backend.get_order("10086", "guest-test")["status"] == "in_transit"
    with pytest.raises(BusinessError, match="暂无退款记录"):
        backend.get_refund_status("10086", "guest-test")


def test_business_evaluator_reports_release_safety_metrics():
    report = asyncio.run(BusinessWorkflowEvaluator().evaluate())
    assert report["metrics"]["business_case_pass_rate"] == 1.0
    assert report["metrics"]["task_completion_rate"] == 1.0
    assert report["metrics"]["tool_selection_accuracy"] == 1.0
    assert report["metrics"]["unsafe_execution_rate"] == 0.0
    assert report["metrics"]["confirmation_guard_rate"] == 1.0
