"""
Autonomous read-only live RTDS signal session.

Wires together:
  SignalProvider (e.g. BinanceSignalProvider)
  RTDSMessageRouter → LocalState updates (binance_ticks, tape_ewma, etc.)

No strategy, no fair-value, no PTB, no execution, no market book feed.

State injection:
  state is injectable so PR #6 (combined market + RTDS session) can share a
  single LocalState between LiveReadonlySession and LiveRTDSSession without
  refactoring. When state=None, a standalone placeholder state is created
  internally (market field is a stub; only RTDS deques matter for this use).

Feed-state transition tracking follows the same convention as LiveReadonlySession:
  1. After connect() — captures "disconnected → connecting"
  2. On each received tick — captures "connecting → live", "live → stale"
  3. In the finally block after close() — captures terminal "→ disconnected"
  Gap: a transition to "stale" between ticks (no yield) is not visible until
  the next tick arrives. Documented, not hidden.

Chainlink:
  RTDSMessageRouter routes source="chainlink" → register_chainlink_tick().
  PolymarketChainlinkSignalProvider (bot.providers.polymarket_chainlink_signal)
  supplies the Polymarket RTDS Chainlink feed without auth.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from .domain import (
    ClobMarketInfo,
    LocalState,
    MarketContext,
    TokenBook,
)
from .providers.base import SignalProvider
from .routers.ws_rtds import RTDSMessageRouter
from .settings import DEFAULT_CONFIG, RuntimeConfig


@dataclass(slots=True)
class LiveRTDSSummary:
    symbol: str
    source: str
    total_ticks: int
    first_value: Optional[float]
    last_value: Optional[float]
    min_value: Optional[float]     # None if no ticks received
    max_value: Optional[float]     # None if no ticks received
    feed_state_transitions: List[Tuple[str, str]]
    final_feed_state: str
    started_at_ms: int
    ended_at_ms: int


def _make_rtds_standalone_state() -> LocalState:
    """
    Minimal LocalState for standalone RTDS use.
    The market field is a non-functional stub — only RTDS deques are used.
    In combined sessions (PR #6), pass the real market state instead.
    """
    stub_market = MarketContext(
        market_id="rtds-standalone",
        condition_id="0x0",
        title="RTDS standalone",
        slug="rtds-standalone",
        start_ts_ms=0,
        end_ts_ms=0,
        yes_token_id="",
        no_token_id="",
        clob=ClobMarketInfo(
            tokens=[],
            min_order_size=0.0,
            min_tick_size=0.0,
            maker_base_fee_bps=0,
            taker_base_fee_bps=0,
            taker_delay_enabled=False,
            min_order_age_s=0.0,
            fee_rate=0.0,
            fee_exponent=1.0,
        ),
    )
    return LocalState(
        market=stub_market,
        yes_book=TokenBook(asset_id=""),
        no_book=TokenBook(asset_id=""),
    )


class LiveRTDSSession:
    """
    Autonomous read-only live RTDS session.

    Typical usage:
        async with aiohttp.ClientSession() as http_session:
            provider = BinanceSignalProvider(http_session)
            session = LiveRTDSSession(signal_provider=provider)
            summary = await session.run_for(duration=30)
            state = session.state   # LocalState with binance_ticks populated

    For combined market + RTDS (PR #6), pass a shared LocalState:
        session = LiveRTDSSession(signal_provider=provider, state=shared_state)
    """

    def __init__(
        self,
        *,
        signal_provider: SignalProvider,
        state: Optional[LocalState] = None,
        config: RuntimeConfig = DEFAULT_CONFIG,
        router: Optional[RTDSMessageRouter] = None,
        symbol: str = "btc/usd",
    ) -> None:
        self._signal_provider = signal_provider
        self._config = config
        self._router = router or RTDSMessageRouter(config)
        self._symbol = symbol
        # state is exposed publicly for post-run inspection and test assertions.
        self.state: Optional[LocalState] = state

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    async def run_for(self, duration: int) -> LiveRTDSSummary:
        """
        Run the RTDS session for `duration` seconds, then return a summary.

        Exits cleanly whether the provider exhausts its ticks before the
        timeout (tests / fake providers) or the timeout fires (live).
        Neither path raises.
        """
        from .domain import utc_now_ms
        started_at_ms = utc_now_ms()

        state = self.state if self.state is not None else _make_rtds_standalone_state()
        self.state = state

        total_ticks = 0
        first_value: Optional[float] = None
        last_value: Optional[float] = None
        min_value: Optional[float] = None
        max_value: Optional[float] = None

        transitions: List[Tuple[str, str]] = []
        last_feed_state = self._signal_provider.feed_state

        await self._signal_provider.connect(self._symbol)
        _record_transition(transitions, last_feed_state, self._signal_provider.feed_state)
        last_feed_state = self._signal_provider.feed_state

        async def _run_loop() -> None:
            nonlocal last_feed_state, total_ticks, first_value, last_value
            nonlocal min_value, max_value

            async for tick in self._signal_provider.iter_signals():
                current = self._signal_provider.feed_state
                _record_transition(transitions, last_feed_state, current)
                last_feed_state = current

                self._router.apply(state, tick)
                total_ticks += 1

                value = float(tick["value"])
                if first_value is None:
                    first_value = value
                last_value = value
                min_value = value if min_value is None else min(min_value, value)
                max_value = value if max_value is None else max(max_value, value)

        try:
            await asyncio.wait_for(_run_loop(), timeout=float(duration))
        except asyncio.TimeoutError:
            pass
        finally:
            await self._signal_provider.close()
            final_state = self._signal_provider.feed_state
            _record_transition(transitions, last_feed_state, final_state)
            last_feed_state = final_state

        return LiveRTDSSummary(
            symbol=self._symbol,
            source=getattr(self._signal_provider, "source_name", "binance"),
            total_ticks=total_ticks,
            first_value=first_value,
            last_value=last_value,
            min_value=min_value,
            max_value=max_value,
            feed_state_transitions=transitions,
            final_feed_state=last_feed_state,
            started_at_ms=started_at_ms,
            ended_at_ms=utc_now_ms(),
        )

    async def run_forever(self) -> None:
        """
        Run indefinitely until close() is called from another task.
        Inspect self.state for accumulated ticks.
        """
        state = self.state if self.state is not None else _make_rtds_standalone_state()
        self.state = state
        await self._signal_provider.connect(self._symbol)
        try:
            async for tick in self._signal_provider.iter_signals():
                self._router.apply(state, tick)
        finally:
            await self._signal_provider.close()

    async def close(self) -> None:
        """Signal the provider to disconnect; iter_signals() exits cleanly."""
        await self._signal_provider.close()


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _record_transition(
    transitions: List[Tuple[str, str]], prev: str, current: str
) -> None:
    if current != prev:
        transitions.append((prev, current))
