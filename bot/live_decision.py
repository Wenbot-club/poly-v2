"""
Combined read-only orchestrator with non-executing strategy layer.

LiveDecisionSession wires together:
  LiveReadonlySession              — market book feed
  LiveRTDSSession                  — signal feed (Binance + Chainlink composite)
  FairValueEngine                  — fair value computation (requires last_chainlink)
  PTBLocker                        — price-to-beat lock
  Strategy                         — produces DesiredQuotes

Discovery is called exactly once. Both feeds share a single LocalState.
The decision loop polls state every `decision_poll_ms` ms, fires when the
market book or a signal tick has advanced, and deduplicates identical decisions.

A final synchronous pass runs after both feeds complete, catching any state
changes that occurred after the last polling interval.

Price anchor: FairValueEngine.compute() raises RuntimeError when last_chainlink
is None. The session catches this, increments skipped_fair_value_count, records
last_fair_value_error, and continues.

To produce real decisions in live operation, pass a CompositeSignalProvider
combining BinanceSignalProvider + PolymarketChainlinkSignalProvider. Chainlink
ticks from the Polymarket RTDS feed (wss://ws-live-data.polymarket.com, topic
crypto_prices_chainlink, no auth required) populate last_chainlink natively via
RTDSMessageRouter → register_chainlink_tick(). Not on-chain Chainlink; no auth.

With composite provider: decision_count > 0, skipped_fair_value_count == 0.
Without anchor source: decision_count == 0, skipped counts explain why.

No execution, no orders, no paper fills, no AsyncLocalRunner changes.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal, Optional

from .domain import DesiredQuotes, FairValueSnapshot, LocalState, utc_now_ms
from .fair_value import FairValueEngine
from .live_readonly import LiveReadonlySession, LiveReadonlySummary
from .live_rtds import LiveRTDSSession, LiveRTDSSummary
from .providers.base import AsyncDiscoveryProvider, MarketDataProvider, SignalProvider
from .ptb import PTBLocker
from .routers.ws_market import MarketMessageRouter
from .routers.ws_rtds import RTDSMessageRouter
from .settings import DEFAULT_CONFIG, RuntimeConfig
from .state import StateFactory
from .strategy.base import Strategy


@dataclass(slots=True)
class DecisionSnapshot:
    now_ms: int
    trigger: Literal["market", "rtds"]
    fair: FairValueSnapshot
    desired_quotes: DesiredQuotes
    ptb_locked: bool
    ptb_value: Optional[float]


@dataclass(slots=True)
class LiveDecisionSummary:
    market: LiveReadonlySummary
    rtds: LiveRTDSSummary
    decision_count: int
    first_decision_ts_ms: Optional[int]
    last_decision_ts_ms: Optional[int]
    last_desired_quotes: Optional[DesiredQuotes]
    skipped_fair_value_count: int
    last_fair_value_error: Optional[str]


class LiveDecisionSession:
    """
    Non-executing combined session: live feeds + strategy layer, no orders posted.

    Typical usage:
        session = LiveDecisionSession(
            discovery=..., market_provider=..., signal_provider=..., strategy=MyStrategy(),
        )
        summary = await session.run_for(duration=30)
        decisions = session.decisions   # list[DecisionSnapshot], in-memory only

    fair_engine and ptb_locker are injectable for tests. By default, created
    internally from config. FairValueEngine.compute() requires last_chainlink —
    when it raises, the cycle is skipped and counted in the summary.
    """

    def __init__(
        self,
        *,
        discovery: AsyncDiscoveryProvider,
        market_provider: MarketDataProvider,
        signal_provider: SignalProvider,
        strategy: Strategy,
        config: RuntimeConfig = DEFAULT_CONFIG,
        market_router: Optional[MarketMessageRouter] = None,
        rtds_router: Optional[RTDSMessageRouter] = None,
        fair_engine: Optional[FairValueEngine] = None,
        ptb_locker: Optional[PTBLocker] = None,
        decision_poll_ms: int = 100,
    ) -> None:
        self._discovery = discovery
        self._market_provider = market_provider
        self._signal_provider = signal_provider
        self._strategy = strategy
        self._config = config
        self._market_router = market_router
        self._rtds_router = rtds_router
        self._fair_engine = fair_engine or FairValueEngine(config=config)
        self._ptb_locker = ptb_locker or PTBLocker(config=config)
        self._decision_poll_ms = decision_poll_ms
        self.state: Optional[LocalState] = None
        self.decisions: list[DecisionSnapshot] = []
        # Per-run tracking (reset in run_for / run_forever)
        self._last_binance_seq: Optional[int] = None
        self._last_yes_book_ts: int = 0
        self._last_dedup_key: Optional[tuple] = None
        self._skipped_fair_count: int = 0
        self._last_fair_error: Optional[str] = None

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    async def run_for(self, duration: int) -> LiveDecisionSummary:
        """
        Run both feeds + strategy layer for `duration` seconds.

        The decision loop runs as a background task, cancelled when both feeds
        finish. A final synchronous pass runs after cancellation to catch any
        state changes that landed after the last poll interval.
        """
        market = await self._discovery.find_active_btc_15m_market()
        state = StateFactory(self._config).create(market)
        self.state = state
        self._reset_run_state()

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

        decision_task = asyncio.create_task(self._decision_loop(state))

        try:
            market_summary, rtds_summary = await asyncio.gather(
                market_session.run_for(duration),
                rtds_session.run_for(duration),
            )
        finally:
            decision_task.cancel()
            try:
                await decision_task
            except asyncio.CancelledError:
                pass
            # Final pass: pick up state changes after last polling interval.
            self._poll_and_decide(state, utc_now_ms())

        first_ts = self.decisions[0].now_ms if self.decisions else None
        last_ts = self.decisions[-1].now_ms if self.decisions else None
        last_dq = self.decisions[-1].desired_quotes if self.decisions else None

        return LiveDecisionSummary(
            market=market_summary,
            rtds=rtds_summary,
            decision_count=len(self.decisions),
            first_decision_ts_ms=first_ts,
            last_decision_ts_ms=last_ts,
            last_desired_quotes=last_dq,
            skipped_fair_value_count=self._skipped_fair_count,
            last_fair_value_error=self._last_fair_error,
        )

    async def run_forever(self) -> None:
        """
        Run indefinitely until close() is called from another task.
        Inspect self.state and self.decisions for accumulated data.
        """
        market = await self._discovery.find_active_btc_15m_market()
        state = StateFactory(self._config).create(market)
        self.state = state
        self._reset_run_state()

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

        decision_task = asyncio.create_task(self._decision_loop(state))

        try:
            await asyncio.gather(
                market_session.run_forever(),
                rtds_session.run_forever(),
            )
        finally:
            decision_task.cancel()
            try:
                await decision_task
            except asyncio.CancelledError:
                pass

    async def close(self) -> None:
        """Close both providers; run_forever() exits cleanly."""
        await asyncio.gather(
            self._market_provider.close(),
            self._signal_provider.close(),
        )

    # ---------------------------------------------------------------------- #
    # Decision loop                                                            #
    # ---------------------------------------------------------------------- #

    async def _decision_loop(self, state: LocalState) -> None:
        while True:
            self._poll_and_decide(state, utc_now_ms())
            await asyncio.sleep(self._decision_poll_ms / 1000.0)

    def _poll_and_decide(self, state: LocalState, now_ms: int) -> None:
        """
        Check whether state has advanced since last poll; if so, attempt a
        decision cycle. Shared between the background loop and the final pass.

        Tracking variables are always updated when a change is detected,
        even if the decision is suppressed by the dedup guard.
        """
        has_new_binance = (
            state.last_binance is not None
            and state.last_binance.sequence_no != self._last_binance_seq
        )
        has_new_book = state.yes_book.timestamp_ms > self._last_yes_book_ts

        if not (has_new_binance or has_new_book):
            return

        trigger: Literal["market", "rtds"] = "rtds" if has_new_binance else "market"

        if has_new_binance and state.last_binance is not None:
            self._last_binance_seq = state.last_binance.sequence_no
        if has_new_book:
            self._last_yes_book_ts = state.yes_book.timestamp_ms

        try:
            fair = self._fair_engine.compute(state, now_ms)
        except RuntimeError as exc:
            self._skipped_fair_count += 1
            self._last_fair_error = str(exc)
            return

        ptb = self._ptb_locker.try_lock(state, now_ms)
        desired_quotes = self._strategy.build(state, fair, now_ms)

        key = self._dedup_key_from(fair, desired_quotes)
        if key == self._last_dedup_key:
            return

        self._last_dedup_key = key
        self.decisions.append(
            DecisionSnapshot(
                now_ms=now_ms,
                trigger=trigger,
                fair=fair,
                desired_quotes=desired_quotes,
                ptb_locked=ptb.locked,
                ptb_value=ptb.ptb_value,
            )
        )

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    def _reset_run_state(self) -> None:
        self.decisions = []
        self._last_binance_seq = None
        self._last_yes_book_ts = 0
        self._last_dedup_key = None
        self._skipped_fair_count = 0
        self._last_fair_error = None

    @staticmethod
    def _dedup_key_from(fair: FairValueSnapshot, dq: DesiredQuotes) -> tuple:
        return (
            round(fair.p_up, 4),
            dq.bid.price,
            dq.ask.price,
            dq.bid.size,
            dq.ask.size,
            dq.bid.enabled,
            dq.ask.enabled,
        )
