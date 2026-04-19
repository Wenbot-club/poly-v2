"""
CompositeSignalProvider: merges multiple SignalProviders into one stream.

All sub-providers run concurrently as asyncio tasks. Ticks from any provider
are forwarded to a shared queue and yielded in arrival order (no reordering).

feed_state aggregation (checked dynamically against each sub-provider):
  "live"         if at least one child is live
  "connecting"   elif at least one child is connecting
  "stale"        elif at least one child is stale
  "disconnected" otherwise (all disconnected)

source_name is built as "<p1.source_name>+<p2.source_name>+..." so that
LiveRTDSSummary.source reflects the full composite, e.g. "binance+coinbase".
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Literal

from .base import SignalProvider


class CompositeSignalProvider:
    """
    Merges N SignalProviders into a single async signal stream.

    connect(symbol) is forwarded to all sub-providers concurrently.
    close() is forwarded to all sub-providers concurrently.
    iter_signals() merges all sub-streams until all providers are exhausted.

    Satisfies the SignalProvider protocol.
    """

    def __init__(self, providers: List[SignalProvider]) -> None:
        if not providers:
            raise ValueError("CompositeSignalProvider requires at least one provider")
        self._providers = providers
        self.source_name: str = "+".join(
            getattr(p, "source_name", "unknown") for p in providers
        )

    @property
    def feed_state(self) -> Literal["connecting", "live", "stale", "disconnected"]:
        states = {p.feed_state for p in self._providers}
        if "live" in states:
            return "live"
        if "connecting" in states:
            return "connecting"
        if "stale" in states:
            return "stale"
        return "disconnected"

    async def connect(self, symbol: str) -> None:
        """Forward connect() to all sub-providers concurrently."""
        await asyncio.gather(*(p.connect(symbol) for p in self._providers))

    async def close(self) -> None:
        """Forward close() to all sub-providers concurrently."""
        await asyncio.gather(*(p.close() for p in self._providers))

    async def iter_signals(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield ticks from all sub-providers until all are exhausted.

        Uses a shared queue: each provider runs as a background task that
        puts its ticks into the queue. A sentinel is placed in the queue
        only after the last provider's task completes.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue()
        _SENTINEL = object()
        remaining = [len(self._providers)]

        async def _feed(provider: SignalProvider) -> None:
            try:
                async for msg in provider.iter_signals():
                    await queue.put(msg)
            finally:
                remaining[0] -= 1
                if remaining[0] == 0:
                    await queue.put(_SENTINEL)

        tasks = [asyncio.create_task(_feed(p)) for p in self._providers]
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                yield item
        finally:
            for t in tasks:
                t.cancel()
            # Drain queue to unblock any waiting producers before cleanup.
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
