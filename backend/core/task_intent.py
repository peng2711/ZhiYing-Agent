"""多轮对话的轻量任务意图状态，不替代业务工作流状态机。"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from core.intent_recognizer import IntentCategory


TRACKED_INTENTS = {
    IntentCategory.COMPLAINT,
    IntentCategory.ACCOUNT,
    IntentCategory.ACCOUNT_SECURITY,
    IntentCategory.ORDER_STATUS,
    IntentCategory.LOGISTICS,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
    IntentCategory.TECHNICAL_LOGIN,
    IntentCategory.TECHNICAL_CRASH,
    IntentCategory.HUMAN_HANDOFF,
}

_INTENT_HINTS = {
    IntentCategory.REFUND: ("退款", "退货", "退掉", "refund"),
    IntentCategory.INVOICE: ("发票", "抬头", "税号", "开票", "invoice"),
    IntentCategory.PAYMENT_ISSUE: ("重复扣款", "扣了两次", "扣款", "多扣", "支付失败"),
    IntentCategory.TECHNICAL_LOGIN: ("无法登录", "登录失败", "401", "验证码", "登录"),
    IntentCategory.TECHNICAL_CRASH: ("崩溃", "闪退", "500", "报错"),
    IntentCategory.ORDER_STATUS: ("订单状态", "未完成订单", "有没有发货", "是否发货"),
    IntentCategory.LOGISTICS: ("物流", "快递", "配送", "运单"),
    IntentCategory.ACCOUNT: ("注销账户", "修改账户", "绑定手机号", "账户"),
    IntentCategory.HUMAN_HANDOFF: ("转人工", "人工客服", "人工主管", "升级处理"),
    IntentCategory.COMPLAINT: ("投诉", "服务太差", "一直没有处理", "等待了两天"),
}


class TaskIntentTracker:
    """保留多目标任务，同时允许当前轮意图合理切换。"""

    def update(
        self,
        previous: Optional[Dict[str, Any]],
        current_intent: IntentCategory,
        message: str,
    ) -> Dict[str, Any]:
        previous = previous if isinstance(previous, dict) else {}
        primary = self._valid_intents(previous.get("primary_intents", []))
        discovered = self.detect_intents(message)
        if current_intent in TRACKED_INTENTS:
            discovered.insert(0, current_intent)
        for intent in discovered:
            if intent.value not in primary:
                primary.append(intent.value)

        previous_active = str(previous.get("active_intent") or "")
        if current_intent in TRACKED_INTENTS:
            active = current_intent.value
        elif discovered:
            # LLM 偶尔会把显式的复合业务请求归为 other/query；确定性关键词用于
            # 路由兜底，但 turn_intent 仍保留原始模型判断，方便审计与评测。
            active = discovered[0].value
        elif previous_active:
            # “那怎么办/需要多久”之类省略主语的追问继续当前任务。
            active = previous_active
        else:
            active = current_intent.value
        return {
            "primary_intents": primary,
            "active_intent": active,
            "turn_intent": current_intent.value,
            "turn": int(previous.get("turn", 0) or 0) + 1,
        }

    @staticmethod
    def detect_intents(message: str) -> List[IntentCategory]:
        lowered = str(message or "").lower()
        return [
            intent for intent, hints in _INTENT_HINTS.items()
            if any(hint in lowered for hint in hints)
        ]

    @staticmethod
    def _valid_intents(values: Iterable[Any]) -> List[str]:
        valid = {intent.value for intent in TRACKED_INTENTS}
        result: List[str] = []
        for value in values or []:
            text = str(value)
            if text in valid and text not in result:
                result.append(text)
        return result
