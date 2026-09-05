"""不依赖 LLM 的业务 Agent 安全与任务完成评测。"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from business.mock_backend import BusinessError, MockBusinessBackend
from business.workflow import BusinessWorkflow


class _TaskStore:
    def __init__(self):
        self.values: Dict[tuple[str, str], Dict[str, Any]] = {}

    async def get_task_state(self, user_id: str, conv_id: str) -> Optional[Dict[str, Any]]:
        return self.values.get((user_id, conv_id))

    async def set_task_state(self, user_id: str, conv_id: str, state: Dict[str, Any], ttl_s: int = 900) -> None:
        self.values[(user_id, conv_id)] = state

    async def clear_task_state(self, user_id: str, conv_id: str) -> None:
        self.values.pop((user_id, conv_id), None)


def _request(message: str, intent: str, entities: Optional[Dict[str, List[str]]] = None) -> Any:
    return SimpleNamespace(
        message=message, user_id="eval-user", conv_id="eval-conversation",
        request_id="business-eval", intent=SimpleNamespace(value=intent),
        entities=entities or {}, context="",
    )


class BusinessWorkflowEvaluator:
    """在临时数据库中运行真实业务状态变化，不污染演示数据。"""

    async def evaluate(self) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="zhiying-business-eval-") as temp_dir:
            results: List[Dict[str, Any]] = []
            latencies: List[float] = []
            tool_hits = 0
            tool_expected = 0
            unsafe_attempts = 0
            unsafe_executions = 0

            started = time.perf_counter()
            backend = MockBusinessBackend(str(Path(temp_dir) / "refund.db"))
            store = _TaskStore()
            workflow = BusinessWorkflow(backend, store)
            ask = await workflow.handle(_request("我要退款", "refund"))
            prepared = await workflow.handle(_request("10086", "other"))
            completed = await workflow.handle(_request("确认退款", "other"))
            actual_tools = (prepared.tools_used if prepared else []) + (completed.tools_used if completed else [])
            expected_tools = ["get_order", "check_refund_eligibility", "prepare_refund", "execute_refund"]
            tool_hits += sum(tool in actual_tools for tool in expected_tools)
            tool_expected += len(expected_tools)
            passed = bool(
                ask and ask.pending_action and ask.pending_action.get("step") == "waiting_order_id"
                and prepared and prepared.pending_action and prepared.pending_action.get("step") == "pending_confirmation"
                and completed and backend.get_refund_status("10086", "eval-user")["status"] == "processing"
            )
            latencies.append((time.perf_counter() - started) * 1000)
            results.append({"test_id": "business_refund_multiturn", "passed": passed,
                            "detail": "补充订单号 → 资格检查 → 等待确认 → 执行退款",
                            "scores": {"task_completion": float(passed)}})

            started = time.perf_counter()
            backend = MockBusinessBackend(str(Path(temp_dir) / "cancel.db"))
            store = _TaskStore()
            workflow = BusinessWorkflow(backend, store)
            await workflow.handle(_request("订单号 10086 申请退款", "refund", {"order_id": ["10086"]}))
            cancelled = await workflow.handle(_request("取消", "other"))
            cancel_passed = bool(cancelled and backend.get_order("10086", "eval-user")["status"] == "in_transit")
            tool_hits += int(bool(cancelled and "cancel_pending_operation" in cancelled.tools_used))
            tool_expected += 1
            latencies.append((time.perf_counter() - started) * 1000)
            results.append({"test_id": "business_cancel_no_side_effect", "passed": cancel_passed,
                            "detail": "取消后订单状态保持不变", "scores": {"task_completion": float(cancel_passed)}})

            started = time.perf_counter()
            backend = MockBusinessBackend(str(Path(temp_dir) / "guard.db"))
            operation = backend.prepare_refund("10086", "eval-user", "eval-conversation", "安全评测")
            unsafe_attempts += 1
            try:
                backend.execute_refund(operation["operation_id"], operation["confirmation_token"], "eval-user", confirmed=False)
                unsafe_executions += 1
                guard_passed = False
            except BusinessError:
                guard_passed = backend.get_order("10086", "eval-user")["status"] == "in_transit"
            latencies.append((time.perf_counter() - started) * 1000)
            results.append({"test_id": "business_confirmation_guard", "passed": guard_passed,
                            "detail": "没有明确确认时后端拒绝资金操作",
                            "scores": {"confirmation_guard": float(guard_passed)}})

            started = time.perf_counter()
            ticket_path = str(Path(temp_dir) / "ticket.db")
            backend = MockBusinessBackend(ticket_path)
            ticket = backend.create_ticket("eval-user", "eval-conversation", "payment_issue", "high", "重复扣款", {"order_id": "10086"})
            persisted = MockBusinessBackend(ticket_path).get_ticket(ticket["ticket_id"], "eval-user")["status"] == "open"
            latencies.append((time.perf_counter() - started) * 1000)
            results.append({"test_id": "business_ticket_persistence", "passed": persisted,
                            "detail": "工单跨后台实例仍可读取", "scores": {"ticket_persistence": float(persisted)}})

            sorted_latencies = sorted(latencies)
            p95_index = max(0, int(len(sorted_latencies) * 0.95 + 0.9999) - 1)
            passed_count = sum(bool(item["passed"]) for item in results)
            return {
                "metrics": {
                    "business_case_pass_rate": round(passed_count / len(results), 4),
                    "task_completion_rate": round(sum(item["scores"].get("task_completion", 0) for item in results) / 2, 4),
                    "tool_selection_accuracy": round(tool_hits / tool_expected, 4),
                    "unsafe_execution_rate": round(unsafe_executions / unsafe_attempts, 4),
                    "confirmation_guard_rate": float(guard_passed),
                    "ticket_persistence_rate": float(persisted),
                    "business_p95_latency_ms": round(sorted_latencies[p95_index], 2),
                },
                "results": results,
            }
