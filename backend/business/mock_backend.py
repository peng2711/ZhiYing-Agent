"""具有持久化状态、幂等执行和审计记录的 SQLite 模拟业务后台。"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime] = None) -> str:
    return (value or _now()).isoformat()


class BusinessError(RuntimeError):
    """可安全返回给用户的业务错误。"""


class MockBusinessBackend:
    def __init__(self, db_path: str = "./data/business/business.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._lock = threading.RLock()
        self._init_schema()
        self._seed()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, product_name TEXT NOT NULL,
                    amount REAL NOT NULL, currency TEXT NOT NULL, status TEXT NOT NULL,
                    address TEXT NOT NULL, created_at TEXT NOT NULL, shipped_at TEXT
                );
                CREATE TABLE IF NOT EXISTS logistics (
                    order_id TEXT PRIMARY KEY REFERENCES orders(order_id), carrier TEXT NOT NULL,
                    tracking_no TEXT NOT NULL, status TEXT NOT NULL, latest_event TEXT NOT NULL,
                    estimated_delivery TEXT, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS operations (
                    operation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                    operation_type TEXT NOT NULL, target_id TEXT NOT NULL, payload TEXT NOT NULL,
                    confirmation_token TEXT NOT NULL, status TEXT NOT NULL,
                    expires_at TEXT NOT NULL, created_at TEXT NOT NULL, executed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS refunds (
                    refund_id TEXT PRIMARY KEY, order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
                    user_id TEXT NOT NULL, amount REAL NOT NULL, reason TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
                    issue_type TEXT NOT NULL, priority TEXT NOT NULL, summary TEXT NOT NULL,
                    evidence TEXT NOT NULL, status TEXT NOT NULL, assigned_to TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, action TEXT NOT NULL,
                    target_id TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)

    def _seed(self) -> None:
        with self._lock, self._connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]:
                return
            now = _now()
            conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                ("10086", "*", "Echo Buds Pro", 299.0, "CNY", "in_transit", "上海市浦东新区示例路 88 号", _iso(now - timedelta(days=4)), _iso(now - timedelta(days=3))),
                ("10087", "*", "Echo Mini", 159.0, "CNY", "delivered", "杭州市西湖区示例街 16 号", _iso(now - timedelta(days=12)), _iso(now - timedelta(days=10))),
                ("10088", "*", "Echo Hub", 699.0, "CNY", "paid", "北京市海淀区示例路 6 号", _iso(now - timedelta(hours=8)), None),
            ])
            conn.executemany("INSERT INTO logistics VALUES (?, ?, ?, ?, ?, ?, ?)", [
                ("10086", "顺丰速运", "SF10086001", "运输中", "快件已到达上海浦东集散中心", (now + timedelta(days=1)).date().isoformat(), _iso(now - timedelta(hours=2))),
                ("10087", "中通快递", "ZT10087001", "已签收", "本人签收", None, _iso(now - timedelta(days=8))),
            ])

    @staticmethod
    def _assert_owner(row: sqlite3.Row, user_id: str) -> None:
        if row["user_id"] not in {"*", user_id}:
            raise BusinessError("订单不存在或不属于当前用户")

    def get_order(self, order_id: str, user_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM orders WHERE order_id=?", (str(order_id),)).fetchone()
        if row is None:
            raise BusinessError(f"未找到订单 {order_id}")
        self._assert_owner(row, user_id)
        return {**dict(row), "demo_data": True}

    def get_logistics(self, order_id: str, user_id: str) -> Dict[str, Any]:
        self.get_order(order_id, user_id)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM logistics WHERE order_id=?", (str(order_id),)).fetchone()
        if row is None:
            raise BusinessError("该订单尚未发货，暂无物流信息")
        return {**dict(row), "demo_data": True}

    def check_refund_eligibility(self, order_id: str, user_id: str) -> Dict[str, Any]:
        order = self.get_order(order_id, user_id)
        if order["status"] in {"refund_processing", "refunded", "cancelled"}:
            eligible, reason = False, "订单已进入退款或关闭状态"
        else:
            eligible = _now() - datetime.fromisoformat(order["created_at"]) <= timedelta(days=7)
            reason = "购买时间在 7 天内" if eligible else "订单已超过 7 天无理由退款期限"
        return {"order_id": order_id, "eligible": eligible, "reason": reason,
                "amount": order["amount"], "currency": order["currency"],
                "order_status": order["status"], "requires_confirmation": True}

    def prepare_refund(self, order_id: str, user_id: str, conversation_id: str, reason: str) -> Dict[str, Any]:
        eligibility = self.check_refund_eligibility(order_id, user_id)
        if not eligibility["eligible"]:
            raise BusinessError(eligibility["reason"])
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM operations WHERE user_id=? AND conversation_id=? AND operation_type='refund' AND target_id=? AND status='pending' ORDER BY created_at DESC LIMIT 1",
                (user_id, conversation_id, order_id),
            ).fetchone()
            if row is None or datetime.fromisoformat(row["expires_at"]) <= _now():
                operation_id, token = f"OP{uuid.uuid4().hex[:12].upper()}", uuid.uuid4().hex
                created_at, expires_at = _iso(), _iso(_now() + timedelta(minutes=15))
                conn.execute(
                    "INSERT INTO operations VALUES (?, ?, ?, 'refund', ?, ?, ?, 'pending', ?, ?, NULL)",
                    (operation_id, user_id, conversation_id, order_id, json.dumps({"reason": reason}, ensure_ascii=False), token, expires_at, created_at),
                )
                row = conn.execute("SELECT * FROM operations WHERE operation_id=?", (operation_id,)).fetchone()
        return {"operation_id": row["operation_id"], "confirmation_token": row["confirmation_token"],
                "operation_type": "refund", "order_id": order_id, "amount": eligibility["amount"],
                "currency": eligibility["currency"],
                "summary": f"订单 {order_id} 可申请退款 ¥{eligibility['amount']:.2f}，确认后订单将进入退款处理中。",
                "expires_at": row["expires_at"], "requires_confirmation": True}

    def execute_refund(self, operation_id: str, token: str, user_id: str, *, confirmed: bool) -> Dict[str, Any]:
        if not confirmed:
            raise BusinessError("缺少用户明确确认，拒绝执行退款")
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            op = conn.execute("SELECT * FROM operations WHERE operation_id=?", (operation_id,)).fetchone()
            if op is None or op["user_id"] != user_id or op["confirmation_token"] != token:
                raise BusinessError("退款确认凭证无效")
            if op["status"] == "executed":
                refund = conn.execute("SELECT * FROM refunds WHERE order_id=?", (op["target_id"],)).fetchone()
                return {**dict(refund), "idempotent": True}
            if op["status"] != "pending" or datetime.fromisoformat(op["expires_at"]) <= _now():
                raise BusinessError("退款确认已取消或过期，请重新发起")
            order = conn.execute("SELECT * FROM orders WHERE order_id=?", (op["target_id"],)).fetchone()
            self._assert_owner(order, user_id)
            refund_id, now = f"RF{uuid.uuid4().hex[:12].upper()}", _iso()
            reason = json.loads(op["payload"]).get("reason", "用户申请")
            conn.execute("INSERT INTO refunds VALUES (?, ?, ?, ?, ?, 'processing', ?, ?)",
                         (refund_id, order["order_id"], user_id, order["amount"], reason, now, now))
            conn.execute("UPDATE orders SET status='refund_processing' WHERE order_id=?", (order["order_id"],))
            conn.execute("UPDATE operations SET status='executed', executed_at=? WHERE operation_id=?", (now, operation_id))
            conn.execute("INSERT INTO audit_log(user_id, action, target_id, details, created_at) VALUES (?, 'execute_refund', ?, ?, ?)",
                         (user_id, order["order_id"], json.dumps({"operation_id": operation_id, "refund_id": refund_id}), now))
        return {"refund_id": refund_id, "order_id": order["order_id"], "amount": order["amount"],
                "currency": order["currency"], "status": "processing", "idempotent": False}

    def cancel_operation(self, operation_id: str, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE operations SET status='cancelled' WHERE operation_id=? AND user_id=? AND status='pending'", (operation_id, user_id))

    def get_refund_status(self, order_id: str, user_id: str) -> Dict[str, Any]:
        self.get_order(order_id, user_id)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM refunds WHERE order_id=?", (order_id,)).fetchone()
        if row is None:
            raise BusinessError("该订单暂无退款记录")
        return dict(row)

    def create_ticket(self, user_id: str, conversation_id: str, issue_type: str, priority: str,
                      summary: str, evidence: Dict[str, Any]) -> Dict[str, Any]:
        ticket_id, now = f"CS{_now().strftime('%Y%m%d')}{uuid.uuid4().hex[:6].upper()}", _iso()
        with self._connect() as conn:
            conn.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, 'open', NULL, ?, ?)",
                         (ticket_id, user_id, conversation_id, issue_type, priority, summary[:1000],
                          json.dumps(evidence, ensure_ascii=False), now, now))
        return {"ticket_id": ticket_id, "issue_type": issue_type, "priority": priority,
                "summary": summary[:1000], "evidence": evidence, "status": "open",
                "assigned_to": None, "created_at": now}

    def get_ticket(self, ticket_id: str, user_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE ticket_id=? AND user_id=?", (ticket_id, user_id)).fetchone()
        if row is None:
            raise BusinessError("未找到该工单")
        result = dict(row)
        result["evidence"] = json.loads(result["evidence"])
        return result

    def reset_demo_data(self) -> Dict[str, Any]:
        """清空业务操作并恢复种子订单；仅应由开发环境接口调用。"""
        with self._lock:
            with self._connect() as conn:
                conn.executescript("""
                    DELETE FROM audit_log;
                    DELETE FROM tickets;
                    DELETE FROM refunds;
                    DELETE FROM operations;
                    DELETE FROM logistics;
                    DELETE FROM orders;
                """)
            self._seed()
        return {"orders": 3, "logistics_records": 2, "reset_at": _iso()}
