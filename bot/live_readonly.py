"""
Autonomous read-only live session orchestrator.

Wires together:
  AsyncDiscoveryProvider → MarketContext
  StateFactory           → LocalState
  MarketDataProvider     → normalized market messages
  MarketMessageRouter    → LocalState updates

No strategy, no fair-value, no PTB, no execution, no RTDS, no user channel.

Feed-state transition tracking:
  Transitions are detected at two points:
    1. Immediately after connect() — captures "disconnected → connecting" or
       "connecting → live" if the provider advances state during connect().
    2. On each received message — captures "connecting → live" and "live → stale".
    3. In the finally block after close() — captures the terminal "→ disconnected".
  One gap: a transition to "stale" that happens between messages (no yield) is
  not immediately visible; it appears only when the next message arrives.
  Documented here, not hidden.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .domain import LocalState, utc_now_ms
from .providers.base import AsyncDiscoveryProvider, MarketDataProvider
from .routers.ws_market import MarketMessageRouter
from .settings import DEFAULT_CONFIG, RuntimeConfig
from .state import StateFactory


@dataclass(slots=True)
class LiveReadonlySummary:
    market_id: str
    yes_token_id: str
    no_token_id: str
    total_messages: int
    book_count: int
    price_change_count: int
    other_count: int
    yes_snapshotted: bool
    no_snapshotted: bool
    feed_state_transitions: List[Tuple[str, str]]
    final_feed_state: str
    started_at_ms: int
    ended_at_ms: int


class LiveReadonlySession:
    """
    Autonomous read-only live session.

    Lifecycle (standalone):
        session = LiveReadonlySession(discovery=..., provider=...)
        summary = await session.run_for(duration=30)
        state   = session.state   # LocalState after run

    Lifecycle (shared state, e.g. from LiveCombinedSession):
        session = LiveReadonlySession(discovery=..., provider=..., state=shared_state)
        # run_for() skips discovery and uses state.market directly

    The provider, discovery, and state are injected — pass fakes for testing.
    """

    def __init__(
        self,
        *,
        discovery: AsyncDiscoveryProvider,
        provider: MarketDataProvider,
        config: RuntimeConfig = DEFAULT_CONFIG,
        router: Optional[MarketMessageRouter] = None,
        state: Optional[LocalState] = None,
    ) -> None:
        self._discovery = discovery
        self._provider = provider
        self._config = config
        self._router = router or MarketMessageRouter()
        # Pre-populate with an external state to skip discovery in run_for().
        # Accessible after run_for() / run_forever() for inspection and testing.
        self.state: Optional[LocalState] = state

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    async def run_for(self, duration: int) -> LiveReadonlySummary:
        """
        Run the session for `duration` seconds, then return a summary.

        Exits cleanly whether the provider exhausts its messages before the
        timeout (normal in tests / finite fake providers) or the timeout fires.
        Neither path raises an exception.
        """
        started_at_ms = utc_now_ms()

        if self.state is not None:
            state = self.state
            market = state.market
        else:
            market = await self._discovery.find_active_btc_15m_market()
            state = StateFactory(self._config).create(market)
            self.state = state

        counters: Dict[str, int] = {"total": 0, "book": 0, "price_change": 0, "other": 0}
        snapshotted: set[str] = set()
        transitions: List[Tuple[str, str]] = []

        # Capture state before connect so "disconnected → connecting" is recorded.
        last_feed_state = self._provider.feed_state
        await self._provider.connect([market.yes_token_id, market.no_token_id])
        self._record_transition(transitions, last_feed_state, self._provider.feed_state)
        last_feed_state = self._provider.feed_state

        async def _run_loop() -> None:
            nonlocal last_feed_state
            async for msg in self._provider.iter_messages():
                # Detect feed_state change on each received message.
                current = self._provider.feed_state
                self._record_transition(transitions, last_feed_state, current)
                last_feed_state = current

                event_type = self._process_message(state, msg)
                counters["total"] += 1
                if event_type == "book":
                    counters["book"] += 1
                    asset_id = str(msg.get("asset_id", ""))
                    if asset_id:
                        snapshotted.add(asset_id)
                elif event_type == "price_change":
                    counters["price_change"] += 1
                else:
                    counters["other"] += 1

        try:
            await asyncio.wait_for(_run_loop(), timeout=float(duration))
        except asyncio.TimeoutError:
            pass  # normal exit after duration
        finally:
            await self._provider.close()
            # Record terminal transition to disconnected (point 3 in module docstring).
            final_state = self._provider.feed_state
            self._record_transition(transitions, last_feed_state, final_state)
            last_feed_state = final_state

        return LiveReadonlySummary(
            market_id=market.market_id,
            yes_token_id=market.yes_token_id,
            no_token_id=market.no_token_id,
            total_messages=counters["total"],
            book_count=counters["book"],
            price_change_count=counters["price_change"],
            other_count=counters["other"],
            yes_snapshotted=market.yes_token_id in snapshotted,
            no_snapshotted=market.no_token_id in snapshotted,
            feed_state_transitions=transitions,
            final_feed_state=last_feed_state,
            started_at_ms=started_at_ms,
            ended_at_ms=utc_now_ms(),
        )

    async def run_forever(self) -> None:
        """
        Run indefinitely until close() is called (from another task) or the
        provider disconnects.  No summary is returned; inspect self.state directly.
        """
        if self.state is not None:
            state = self.state
            market = state.market
        else:
            market = await self._discovery.find_active_btc_15m_market()
            state = StateFactory(self._config).create(market)
            self.state = state
        await self._provider.connect([market.yes_token_id, market.no_token_id])
        try:
            async for msg in self._provider.iter_messages():
                self._process_message(state, msg)
        finally:
            await self._provider.close()

    async def close(self) -> None:
        """Signal the provider to disconnect; iter_messages() will exit cleanly."""
        await self._provider.close()

    # ---------------------------------------------------------------------- #
    # Internals                                                                #
    # ---------------------------------------------------------------------- #

    def _process_message(self, state: LocalState, msg: Dict[str, Any]) -> str:
        """
        Apply one normalized message to state via the router.
        Returns event_type string for counter tracking.
        Testable in isolation without a live provider.
        """
        self._router.apply(state, msg)
        return str(msg.get("event_type", "unknown"))

    @staticmethod
    def _record_transition(
        transitions: List[Tuple[str, str]], prev: str, current: str
    ) -> None:
        if current != prev:
            transitions.append((prev, current))
