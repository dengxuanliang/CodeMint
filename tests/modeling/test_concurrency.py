from __future__ import annotations

import asyncio

import pytest

from codemint.modeling.concurrency import gather_limited


@pytest.mark.asyncio
async def test_gather_limited_bounds_concurrency() -> None:
    running = 0
    peak = 0

    async def worker(value: int) -> int:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0)
        running -= 1
        return value * 2

    results = await gather_limited(2, [worker(i) for i in range(5)])

    assert results == [0, 2, 4, 6, 8]
    assert peak == 2


@pytest.mark.asyncio
async def test_gather_limited_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        await gather_limited(0, [])
