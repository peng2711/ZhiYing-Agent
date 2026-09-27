import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import api.main as main


def test_eval_run_rejects_concurrent_runs(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    class SlowEvaluator:
        async def run(self, intent_cases, dialog_cases):
            calls.append(1)
            started.set()
            await release.wait()
            return SimpleNamespace(timestamp="t", pass_rate=1.0, total=0, passed=0,
                                   avg_scores={}, regressions=[], recommendations=[], results=[])

    monkeypatch.setattr(main, "_evaluator", SlowEvaluator())
    monkeypatch.setattr(main, "_eval_lock", asyncio.Lock())

    async def scenario():
        first = asyncio.create_task(main.run_eval(None))
        await started.wait()
        with pytest.raises(HTTPException) as exc_info:
            await main.run_eval(None)
        release.set()
        report = await first
        # 前一次结束后可以再次运行。
        await main.run_eval(None)
        return exc_info.value, report

    error, report = asyncio.run(scenario())

    assert error.status_code == 409
    assert report["pass_rate"] == 1.0
    assert len(calls) == 2
