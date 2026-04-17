from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from typing import TypeVar


T = TypeVar("T")


async def gather_limited(limit: int, awaitables: Iterable[Awaitable[T]]) -> list[T]:
    semaphore = asyncio.Semaphore(limit)

    async def run(awaitable: Awaitable[T]) -> T:
        async with semaphore:
            return await awaitable

    tasks = [run(awaitable) for awaitable in awaitables]
    return list(await asyncio.gather(*tasks))

