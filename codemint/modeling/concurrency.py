from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Iterable
from typing import TypeVar


T = TypeVar("T")


async def gather_limited(limit: int, awaitables: Iterable[Awaitable[T]]) -> list[T]:
    if limit <= 0:
        _close_awaitables(awaitables)
        raise ValueError("limit must be greater than 0")

    semaphore = asyncio.Semaphore(limit)

    async def run(awaitable: Awaitable[T]) -> T:
        async with semaphore:
            return await awaitable

    tasks = [run(awaitable) for awaitable in awaitables]
    return list(await asyncio.gather(*tasks))


def _close_awaitables(awaitables: Iterable[Awaitable[T]]) -> None:
    for awaitable in awaitables:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
