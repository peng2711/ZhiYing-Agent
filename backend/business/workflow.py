"""退款任务状态机与服务端强制确认。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from business.mock_backend import BusinessError, MockBusinessBackend


@dataclass
class WorkflowOutcome:
    response: str
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    pending_action: Optional[Dict[str, Any]] = None
    ticket: Optional[Dict[str, Any]] = None


class BusinessWorkflow:
    CONFIRM_RE = re.compile(r"^\s*(确认|确定|同意|是的|是|yes|confirm|立即退款|确认退款)[。！!\s]*$", re.I)
    CANCEL_RE = re.compile(r"^\s*(取消|不用了|不确认|否|no|cancel)[。！!\s]*$", re.I)
    BARE_ORDER_RE = re.compile(r"^\s*#?([A-Za-z0-9_-]{4,32})\s*$")
    REFUND_INFORMATION_MARKERS = (
        "政策", "规则", "规定", "依据", "条款", "条件", "流程", "时效",
        "是什么", "怎么", "如何", "多久", "几天", "能不能", "是否可以",
        "policy", "rule", "terms", "how long", "how to",
    )

    def __init__(self, backend: MockBusinessBackend, task_store: Any):
        self.backend = backend
        self.task_store = task_store

    @staticmethod
    def _trace(name: str, args: Dict[str, Any], success: bool = True, error: str = "") -> Dict[str, Any]:
        return {"agent_type": "billing", "tool_name": name, "tool_use_id": f"workflow_{name}",
                "input": args, "success": success, "result_success": success, "latency_ms": 0.0,
                "cached": False, "reranked": False, "error": error}

    async def handle(self, req: Any) -> Optional[WorkflowOutcome]:
        task = await self.task_store.get_task_state(req.user_id, req.conv_id)
        message = (req.message or "").strip()
        if task and task.get("step") == "pending_confirmation":
            if self.CONFIRM_RE.fullmatch(message):
                try:
                    result = self.backend.execute_refund(task["operation_id"], task["confirmation_token"], req.user_id, confirmed=True)
                    await self.task_store.clear_task_state(req.user_id, req.conv_id)
                    return WorkflowOutcome(
                        f"退款申请已提交。退款单号：{result['refund_id']}，金额：¥{result['amount']:.2f}，当前状态：处理中。",
                        ["execute_refund"], [self._trace("execute_refund", {"operation_id": task["operation_id"], "confirmed": True})])
                except BusinessError as exc:
                    await self.task_store.clear_task_state(req.user_id, req.conv_id)
                    return WorkflowOutcome(str(exc), tool_traces=[self._trace("execute_refund", {"operation_id": task.get("operation_id")}, False, str(exc))])
            if self.CANCEL_RE.fullmatch(message):
                self.backend.cancel_operation(task["operation_id"], req.user_id)
                await self.task_store.clear_task_state(req.user_id, req.conv_id)
                return WorkflowOutcome("已取消本次退款操作，订单状态没有改变。", ["cancel_pending_operation"],
                                       [self._trace("cancel_pending_operation", {"operation_id": task["operation_id"]})])
        if task and task.get("step") == "waiting_order_id":
            match = self.BARE_ORDER_RE.fullmatch(message)
            if match:
                return await self._prepare_refund(req, match.group(1), task.get("reason", "用户申请"))
        intent = getattr(getattr(req, "intent", None), "value", getattr(req, "intent", None))
        if intent == "refund" and not task:
            # “退款政策是什么”是知识咨询，不应被误当成创建退款任务。
            # 让它继续进入 Agent + RAG，只有明确办理意图才开启有状态工作流。
            lowered = message.lower()
            if any(marker in lowered for marker in self.REFUND_INFORMATION_MARKERS):
                return None
            order_ids = (req.entities or {}).get("order_id", [])
            if not order_ids:
                state = {"type": "refund", "step": "waiting_order_id", "reason": "用户申请"}
                await self.task_store.set_task_state(req.user_id, req.conv_id, state)
                return WorkflowOutcome("可以帮您申请退款。请提供订单号，我会先查询订单并检查退款资格。", pending_action=state)
            return await self._prepare_refund(req, order_ids[0], "用户申请")
        return None

    async def _prepare_refund(self, req: Any, order_id: str, reason: str) -> WorkflowOutcome:
        traces: List[Dict[str, Any]] = []
        try:
            self.backend.get_order(order_id, req.user_id)
            traces.append(self._trace("get_order", {"order_id": order_id}))
            eligibility = self.backend.check_refund_eligibility(order_id, req.user_id)
            traces.append(self._trace("check_refund_eligibility", {"order_id": order_id}))
            if not eligibility["eligible"]:
                await self.task_store.clear_task_state(req.user_id, req.conv_id)
                return WorkflowOutcome(f"订单 {order_id} 暂不符合退款条件：{eligibility['reason']}。如需进一步处理，我可以为您创建人工工单。",
                                       ["get_order", "check_refund_eligibility"], traces)
            operation = self.backend.prepare_refund(order_id, req.user_id, req.conv_id, reason)
            traces.append(self._trace("prepare_refund", {"order_id": order_id, "reason": reason}))
            task = {"type": "refund", "step": "pending_confirmation", **operation}
            await self.task_store.set_task_state(req.user_id, req.conv_id, task)
            public_action = {key: value for key, value in task.items() if key != "confirmation_token"}
            return WorkflowOutcome(f"{operation['summary']}\n\n这是有资金影响的操作，请回复“确认退款”后执行；15 分钟内有效。",
                                   ["get_order", "check_refund_eligibility", "prepare_refund"], traces, public_action)
        except BusinessError as exc:
            await self.task_store.clear_task_state(req.user_id, req.conv_id)
            traces.append(self._trace("refund_workflow", {"order_id": order_id}, False, str(exc)))
            return WorkflowOutcome(str(exc), tool_traces=traces)
