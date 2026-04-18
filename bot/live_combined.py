"""
Combined read-only orchestrator: market book feed + RTDS signal feed on a
single shared LocalState.

LiveCombinedSession owns the full lifecycle:
  1. Discovery ONCE (not twice — invariant)
  2. Single LocalState created from that market
  3. LiveReadonlySession and LiveRTDSSession instantiated with the shared state
  4. asyncio.gather() runs both sub-sessions concurrently

Since asyncio is single-threaded, concurrent writes to LocalState from the
two loops are safe — only one executes at a time (cooperative multitasking).
The two loops modify disjoint parts of the state:
  market loop  → yes_book, no_book, simulated_now_ms
  RTDS loop    → binance_ticks, last_binance, tape_ewma

No strategy, no fair-value, no PTB, no execution, no paper fills.
Goal: prove clean cohabitation of two live feeds on one state.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from .domain import LocalState
from .live_readonly import LiveReadonlySession, LiveReadonlySummary
from .live_rtds import LiveRTDSSession, LiveRTDSSummary
from .providers.base import AsyncDiscoveryProvider, MarketDataProvider, SignalProvider
from .routers.ws_market import MarketMessageRouter
from .routers.ws_rtds import RTDSMessageRouter
from .settings import DEFAULT_CONFIG, RuntimeConfig
from .state import StateFactory


@dataclass(slots=True)
class LiveCombinedSummary:
    market: LiveReadonlySummary
    rtds: LiveRTDSSummary


class LiveCombinedSession:
    """
    Autonomous combined read-only orchestrator.

    Typical usage:
        async with aiohttp.ClientSession() as http:
            session = LiveCombinedSession(
                discovery=PolymarketDiscoveryProvider(http),
                market_provider=PolymarketMarketDataProvider(http),
                signal_provider=BinanceSignalProvider(http),
            )
            summary = await session.run_for(duration=30)
            state = session.state  # single LocalState with both feeds applied

    Discovery is called exactly once. The resulting LocalState is shared between
    the market and RTDS sub-sessions — never duplicated.
    """

    def __init__(
        self,
        *,
        discovery: AsyncDiscoveryProvider,
        market_provider: MarketDataProvider,
        signal_provider: SignalProvider,
        config: RuntimeConfig = DEFAULT_CONFIG,
        market_router: Optional[MarketMessageRouter] = None,
        rtds_router: Optional[RTDSMessageRouter] = None,
    ) -> None:
        self._discovery = discovery
        self._market_provider = market_provider
        self._signal_provider = signal_provider
        self._config = config
        self._market_router = market_router
        self._rtds_router = rtds_router
        self.state: Optional[LocalState] = None

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    async def run_for(self, duration: int) -> LiveCombinedSummary:
        """
        Run both market and RTDS feeds for `duration` seconds on a shared state.

        Discovery is called once. Each sub-session manages its own provider
        connection and timeout. Both finish around the same time (same duration).
        Returns a combined summary regardless of whether providers exhaust early
        or the timeout fires.
        """
        market = await self._discovery.find_active_btc_15m_market()
        state = StateFactory(self._config).create(market)
        self.state = state

        market_session = LiveReadonlySession(
            discovery=self._discovery,
            provider=self._market_provider,
            config=self._config,
            router=self._market_router,
            state=state,            # injected — skips discovery inside run_for
        )
        rtds_session = LiveRTDSSession(
            signal_provider=self._signal_provider,
            state=state,            # same shared state
            config=self._config,
            router=self._rtds_router,
        )

        market_summary, rtds_summary = await asyncio.gather(
            market_session.run_for(duration),
            rtds_session.run_for(duration),
        )

        return LiveCombinedSummary(market=market_summary, rtds=rtds_summary)

    async def run_forever(self) -> None:
        """
        Run both feeds indefinitely until close() is called from another task.
        Inspect self.state for accumulated data.
        """
        market = await self._discovery.find_active_btc_15m_market()
        state = StateFactory(self._config).create(market)
        self.state = state

        market_session = LiveReadonlySession(
            discovery=self._discovery,
            provider=self._market_provider,
            config=self._config,
            router=self._market_router,
            state=state,
        )
        rtds_session = LiveRTDSSession(
            signal_provider=self._signal_provider,
            state=state,
            config=self._config,
            router=self._rtds_router,
        )

        await asyncio.gather(
            market_session.run_forever(),
            rtds_session.run_forever(),
        )

    async def close(self) -> None:
        """Close both providers; run_forever() / iter loops exit cleanly."""
        await asyncio.gather(
            self._market_provider.close(),
            self._signal_provider.close(),
        )
