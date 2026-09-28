import asyncio
import types

from prometheus_client import REGISTRY

from core.llm_client import MeteredClient, openai_response_to_anthropic
from core.usage import estimate_cost, extract_usage, llm_role, track_request_usage
from evaluation.evaluator import EndToEndEvaluator


class FakeMessages:
    def __init__(self, usage=None, delay=0.0):
        self.usage = usage if usage is not None else {"input_tokens": 100, "output_tokens": 20}
        self.delay = delay

    async def create(self, **kwargs):
        await asyncio.sleep(self.delay)
        return types.SimpleNamespace(content=[], usage=self.usage)


def metered(**kwargs):
    return MeteredClient(types.SimpleNamespace(messages=FakeMessages(**kwargs)))


def test_anthropic_usage_counts_cache_reads_and_writes_as_input():
    usage = extract_usage(types.SimpleNamespace(usage={
        "input_tokens": 50, "output_tokens": 10,
        "cache_read_input_tokens": 400, "cache_creation_input_tokens": 30,
    }))
    assert usage.input_tokens == 480
    assert usage.cached_input_tokens == 400
    assert usage.output_tokens == 10


def test_openai_usage_is_normalized_to_the_same_convention():
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="好的", tool_calls=None))],
        usage=types.SimpleNamespace(
            prompt_tokens=1200, completion_tokens=80,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=1000),
        ),
    )
    usage = extract_usage(openai_response_to_anthropic(response))
    # OpenAI 的 prompt_tokens 已含缓存，归一化后总输入不能被重复计算。
    assert usage.input_tokens == 1200
    assert usage.cached_input_tokens == 1000
    assert usage.output_tokens == 80


def test_request_usage_is_split_by_role_and_includes_parallel_subtasks():
    client = metered()

    async def call(role):
        with llm_role(role):
            await client.messages.create(model="m1", messages=[])

    async def handle_request():
        with track_request_usage() as usage:
            await call("intent")
            await asyncio.gather(call("agent:billing"), call("agent:technical"), call("rerank"))
            return usage.summary()

    summary = asyncio.run(handle_request())

    assert summary["llm_calls"] == 4
    assert summary["input_tokens"] == 400
    assert summary["output_tokens"] == 80
    assert set(summary["by_role"]) == {"intent", "agent:billing", "agent:technical", "rerank"}


def test_concurrent_requests_do_not_mix_usage():
    small, large = metered(delay=0.02), metered(usage={"input_tokens": 1000, "output_tokens": 500}, delay=0.01)

    async def handle(client, calls):
        with track_request_usage() as usage:
            for _ in range(calls):
                await client.messages.create(model="m1", messages=[])
            return usage.summary()

    async def main():
        return await asyncio.gather(handle(small, 3), handle(large, 1))

    a, b = asyncio.run(main())

    assert (a["llm_calls"], a["input_tokens"]) == (3, 300)
    assert (b["llm_calls"], b["input_tokens"]) == (1, 1000)


def test_missing_usage_is_counted_as_unreported_and_never_raises():
    class NoUsage:
        async def create(self, **kwargs):
            return types.SimpleNamespace(content=[])

    client = MeteredClient(types.SimpleNamespace(messages=NoUsage()))

    async def main():
        with track_request_usage() as usage:
            await client.messages.create(model="m1", messages=[])
            return usage.summary()

    summary = asyncio.run(main())
    assert summary["llm_calls"] == 1
    assert summary["unreported_calls"] == 1


def test_calls_outside_a_request_still_reach_prometheus():
    labels = {"role": "memory_profile", "model": "prom-model", "kind": "output"}
    before = REGISTRY.get_sample_value("zhiying_llm_tokens_total", labels) or 0.0

    async def main():
        with llm_role("memory_profile"):
            await metered().messages.create(model="prom-model", messages=[])

    asyncio.run(main())
    assert REGISTRY.get_sample_value("zhiying_llm_tokens_total", labels) == before + 20


def test_cost_estimate_uses_cached_input_price(monkeypatch):
    monkeypatch.setenv("ZHIYING_LLM_PRICING", '{"m1": {"input": 2, "cached_input": 0.5, "output": 8}}')
    calls = [{"model": "m1", "input_tokens": 1_000_000, "cached_input_tokens": 600_000, "output_tokens": 100_000},
             {"model": "unpriced", "input_tokens": 999, "cached_input_tokens": 0, "output_tokens": 999}]
    # 40 万未缓存 × 2 + 60 万缓存 × 0.5 + 10 万输出 × 8，单位是每百万 token
    assert estimate_cost(calls) == 1.9
    monkeypatch.delenv("ZHIYING_LLM_PRICING")
    assert estimate_cost(calls) is None


def test_token_growth_is_reported_as_regression():
    evaluator = EndToEndEvaluator.__new__(EndToEndEvaluator)
    evaluator._history = [types.SimpleNamespace(avg_scores={"avg_tokens_per_turn": 1000.0, "overall": 0.8})]
    evaluator._baseline = None

    regressions = evaluator._detect_regressions({"avg_tokens_per_turn": 1200.0, "overall": 0.8})
    assert len(regressions) == 1 and "avg_tokens_per_turn" in regressions[0]
    assert evaluator._detect_regressions({"avg_tokens_per_turn": 800.0, "overall": 0.8}) == []
