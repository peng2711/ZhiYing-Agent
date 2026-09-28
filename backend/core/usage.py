"""
LLM Token 用量统计。

所有 LLM 客户端都由 core.llm_client.create_llm_client 创建，并在外面包了一层计量：
每次 messages.create 返回后，读取响应里的 usage 记一笔。

  - 调用方用 `with llm_role("rerank"):` 标注这次调用属于哪个环节，用于按环节拆分成本。
  - API 层用 `track_request_usage()` 开启一次请求的统计。contextvar 会随 asyncio 任务
    和 to_thread 复制，所以并行执行的多个 Agent、检索改写和重排都会累加到同一个请求上，
    而不同请求之间互不影响。
  - 全局累计写入 Prometheus（zhiying_llm_tokens_total / zhiying_llm_calls_total）。

token 口径统一为：input_tokens 包含缓存命中的部分，cached_input_tokens 单独列出。
OpenAI 兼容接口的 prompt_tokens 本来就包含缓存；Anthropic 的 input_tokens 不含缓存读写，
这里会把 cache_read / cache_creation 加回去。

配置 ZHIYING_LLM_PRICING 后按单价估算费用（单位：每百万 token 的价格，币种自定），例如：
  {"qwen3.7-plus": {"input": 0.8, "output": 2.0, "cached_input": 0.16}}
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_current_role: ContextVar[str] = ContextVar("llm_role", default="unknown")
_current_request: ContextVar[Optional["RequestUsage"]] = ContextVar("llm_request_usage", default=None)

try:
    from prometheus_client import Counter

    _TOKENS = Counter("zhiying_llm_tokens_total", "LLM token 用量", ["role", "model", "kind"])
    _CALLS = Counter("zhiying_llm_calls_total", "LLM 调用次数", ["role", "model"])
except Exception:  # pragma: no cover - prometheus_client 是必需依赖，这里只是防御
    _TOKENS = _CALLS = None


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reported: bool = True


@dataclass
class RequestUsage:
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, role: str, model: str, usage: LLMUsage) -> None:
        self.calls.append({
            "role": role,
            "model": model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reported": usage.reported,
        })

    def summary(self) -> Dict[str, Any]:
        by_role: Dict[str, Dict[str, int]] = {}
        by_model: Dict[str, Dict[str, int]] = {}
        for call in self.calls:
            for key, groups in ((call["role"], by_role), (call["model"], by_model)):
                bucket = groups.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
                bucket["calls"] += 1
                bucket["input_tokens"] += call["input_tokens"]
                bucket["output_tokens"] += call["output_tokens"]
        result: Dict[str, Any] = {
            "llm_calls": len(self.calls),
            "input_tokens": sum(c["input_tokens"] for c in self.calls),
            "output_tokens": sum(c["output_tokens"] for c in self.calls),
            "cached_input_tokens": sum(c["cached_input_tokens"] for c in self.calls),
            "unreported_calls": sum(1 for c in self.calls if not c["reported"]),
            "by_role": by_role,
            "by_model": by_model,
        }
        cost = estimate_cost(self.calls)
        if cost is not None:
            result["estimated_cost"] = cost
        return result


@contextmanager
def llm_role(role: str) -> Iterator[None]:
    token = _current_role.set(role)
    try:
        yield
    finally:
        _current_role.reset(token)


@contextmanager
def track_request_usage() -> Iterator[RequestUsage]:
    usage = RequestUsage()
    token = _current_request.set(usage)
    try:
        yield usage
    finally:
        _current_request.reset(token)


def current_request_usage() -> Optional[RequestUsage]:
    return _current_request.get()


def extract_usage(response: Any) -> LLMUsage:
    """从 Anthropic 响应或 llm_client 转换后的响应里取出 token 用量。"""
    usage = getattr(response, "usage", None)
    if usage is None:
        return LLMUsage(reported=False)

    def value(key: str) -> int:
        raw = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        return int(raw or 0)

    cache_read = value("cache_read_input_tokens")
    cache_creation = value("cache_creation_input_tokens")
    return LLMUsage(
        input_tokens=value("input_tokens") + cache_read + cache_creation,
        output_tokens=value("output_tokens"),
        cached_input_tokens=cache_read,
    )


def record_llm_call(model: str, response: Any) -> None:
    """记录一次 LLM 调用；统计失败只记日志，不能影响业务调用。"""
    try:
        usage = extract_usage(response)
        role = _current_role.get()
        model = str(model or "unknown")
        request = _current_request.get()
        if request is not None:
            request.add(role, model, usage)
        if _CALLS is not None:
            _CALLS.labels(role=role, model=model).inc()
            _TOKENS.labels(role=role, model=model, kind="input").inc(usage.input_tokens)
            _TOKENS.labels(role=role, model=model, kind="output").inc(usage.output_tokens)
            _TOKENS.labels(role=role, model=model, kind="cached_input").inc(usage.cached_input_tokens)
    except Exception as ex:
        logger.warning("记录 LLM 用量失败: %s", ex)


def _pricing() -> Dict[str, Dict[str, float]]:
    raw = os.getenv("ZHIYING_LLM_PRICING", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except ValueError:
        logger.warning("ZHIYING_LLM_PRICING 不是合法 JSON，已忽略")
        return {}


def estimate_cost(calls: List[Dict[str, Any]]) -> Optional[float]:
    pricing = _pricing()
    if not pricing:
        return None
    total = 0.0
    for call in calls:
        price = pricing.get(call["model"])
        if not price:
            continue
        cached = call["cached_input_tokens"]
        uncached = call["input_tokens"] - cached
        total += uncached * float(price.get("input", 0))
        total += cached * float(price.get("cached_input", price.get("input", 0)))
        total += call["output_tokens"] * float(price.get("output", 0))
    return round(total / 1_000_000, 6)
