"""
Combined read-only orchestrator with conservative paper execution layer.

LivePaperSession wires together:
  LiveReadonlySession  — market book feed
  LiveRTDSSession      — signal feed (Binance, or composite Binance+Coinbase)
  FairValueEngine      — fair value computation (requires last_chainlink)
  PTBLocker            — price-to-beat lock
  Strategy             — produces DesiredQuotes
  MockExecutionEngine  — simulated order posting, cancellation, fill

Discovery called exactly once. Both feeds share a single LocalState.
State mutations flow exclusively through MockExecutionEngine → QueueingUserRouter
→ UserMessageRouter.apply() → state. No direct mutation of open_orders or inventory.

Execution pattern:
  Each decision cycle (new dedup-passing decision):
    → _sync_quotes_impl() posts/cancels simulated bid (bid-only, mirrors AsyncLocalRunner)
  Each poll with state change (regardless of decision dedup):
    → _check_fills() simulates fill when top_ask.price <= live_bid.price (conservative)

Price anchor: FairValueEngine.compute() raises RuntimeError when last_chainlink is None.
Caught as RuntimeError only — other exceptions propagate. Counted in skipped_fair_value_count.

To post simulated orders in live operation, pass a CompositeSignalProvider combining
BinanceSignalProvider + CoinbaseAnchorProvider. Coinbase ticks fill the internal anchor
slot (last_chainlink). This is a Coinbase price anchor, NOT a Chainlink oracle.

No real orders, no real fills, no PnL (no outcome available). Honest counters only.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from .async_runner import QueueingUserRouter
from .domain import DesiredQuotes, FairValueSnapshot, LocalState, utc_now_ms
from .execution.base import ExecutionGateway
from .execution.paper import MockExecutionEngine
from .fair_value import FairValueEngine
from .live_decision import DecisionSnapshot
from .live_readonly import LiveReadonlySession, LiveReadonlySummary
from .live_rtds import LiveRTDSSession, LiveRTDSSummary
from .providers.base import AsyncDiscoveryProvider, MarketDataProvider, SignalProvider
from .ptb import PTBLocker
from .routers.ws_market import MarketMessageRouter
from .routers.ws_rtds import RTDSMessageRouter
from .routers.ws_user import UserMessageRouter
from .settings import DEFAULT_CONFIG, RuntimeConfig
from .state import StateFactory
from .strategy.base import Strategy


@dataclass(slots=True)
class LivePaperSummary:
    market: LiveReadonlySummary
    rtds: LiveRTDSSummary
    decision_count: int
    first_decision_ts_ms: Optional[int]
    last_decision_ts_ms: Optional[int]
    skipped_fair_value_count: int
    last_fair_value_error: Optional[str]
    orders_posted: int
    orders_cancelled: int
    orders_rejected: int
    fills_simulated: int
    final_pusd_free: float
    final_up_free: float
    last_rejection_reason: Optional[str]


class LivePaperSession:
    """
    Non-executing paper session: live feeds + strategy layer + simulated execution.

    Orders are simulated via MockExecutionEngine — nothing is posted to any exchange.
    All state mutations (inventory, reservations, order lifecycle) flow through
    MockExecutionEngine → QueueingUserRouter → UserMessageRouter.apply().

    initial_pusd: if None, seeds from config.default_working_capital_usd.

    execution_engine_factory: receives the runtime QueueingUserRouter; if None,
    defaults to MockExecutionEngine(config, user_router=queueing_router).

    FairValueEngine and PTBLocker are injectable for tests.
    """

    def __init__(
        self,
        *,
        discovery: AsyncDiscoveryProvider,
        market_provider: MarketDataProvider,
        signal_provider: SignalProvider,
        strategy: Strategy,
        config: RuntimeConfig = DEFAULT_CONFIG,
        initial_pusd: Optional[float] = None,
        market_router: Optional[MarketMessageRouter] = None,
        rtds_router: Optional[RTDSMessageRouter] = None,
        user_router: Optional[UserMessageRouter] = None,
        fair_engine: Optional[FairValueEngine] = None,
        ptb_locker: Optional[PTBLocker] = None,
        execution_engine_factory: Optional[Callable[[QueueingUserRouter], ExecutionGateway]] = None,
        decision_poll_ms: int = 100,
    ) -> None:
        self._discovery = discovery
        self._market_provider = market_provider
        self._signal_provider = signal_provider
        self._strategy = strategy
        self._config = config
        self._initial_pusd = initial_pusd
        self._market_router = market_router
        self._rtds_router = rtds_router
        self._user_router = user_router or UserMessageRouter()
        self._fair_engine = fair_engine or FairValueEngine(config=config)
        self._ptb_locker = ptb_locker or PTBLocker(config=config)
        self._execution_engine_factory = execution_engine_factory
        self._decision_poll_ms = decision_poll_ms

        self.state: Optional[LocalState] = None
        self.decisions: list[DecisionSnapshot] = []

        # Per-run state; reset in _reset_run_state()
        self._user_queue: Optional[asyncio.Queue] = None
        self._engine: Optional[ExecutionGateway] = None
        self._last_binance_seq: Optional[int] = None
        self._last_yes_book_ts: int = 0
        self._last_dedup_key: Optional[tuple] = None
        self._skipped_fair_count: int = 0
        self._last_fair_error: Optional[str] = None
        self._orders_posted: int = 0
        self._orders_cancelled: int = 0
        self._orders_rejected: int = 0
        self._fills_simulated: int = 0
        self._last_rejection_reason: Optional[str] = None

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    async def run_for(self, duration: int) -> LivePaperSummary:
        """
        Run both feeds + strategy + paper execution for `duration` seconds.

        Decision loop runs as a background task, cancelled when feeds finish.
        Final synchronous pass after cancellation catches state changes that
        arrived after the last poll interval.
        """
        market = await self._discovery.find_active_btc_15m_market()
        state = StateFactory(self._config).create(market)
        pusd = (
            self._initial_pusd
            if self._initial_pusd is not None
            else self._config.default_working_capital_usd
        )
        state.inventory.pusd_free = pusd
        self.state = state
        self._reset_run_state()

        user_queue: asyncio.Queue = asyncio.Queue()
        self._user_queue = user_queue
        queueing_router = QueueingUserRouter(user_queue)
        self._engine = (
            self._execution_engine_factory(queueing_router)
            if self._execution_engine_factory is not None
            else MockExecutionEngine(config=self._config, user_router=queueing_router)
        )

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
            self._poll_and_execute(state, utc_now_ms())

        first_ts = self.decisions[0].now_ms if self.decisions else None
        last_ts = self.decisions[-1].now_ms if self.decisions else None

        return LivePaperSummary(
            market=market_summary,
            rtds=rtds_summary,
            decision_count=len(self.decisions),
            first_decision_ts_ms=first_ts,
            last_decision_ts_ms=last_ts,
            skipped_fair_value_count=self._skipped_fair_count,
            last_fair_value_error=self._last_fair_error,
            orders_posted=self._orders_posted,
            orders_cancelled=self._orders_cancelled,
            orders_rejected=self._orders_rejected,
            fills_simulated=self._fills_simulated,
            final_pusd_free=state.inventory.pusd_free,
            final_up_free=state.inventory.up_free,
            last_rejection_reason=self._last_rejection_reason,
        )

    async def run_forever(self) -> None:
        """
        Run indefinitely until close() is called from another task.
        Inspect self.state and self.decisions for accumulated data.
        """
        market = await self._discovery.find_active_btc_15m_market()
        state = StateFactory(self._config).create(market)
        pusd = (
            self._initial_pusd
            if self._initial_pusd is not None
            else self._config.default_working_capital_usd
        )
        state.inventory.pusd_free = pusd
        self.state = state
        self._reset_run_state()

        user_queue: asyncio.Queue = asyncio.Queue()
        self._user_queue = user_queue
        queueing_router = QueueingUserRouter(user_queue)
        self._engine = (
            self._execution_engine_factory(queueing_router)
            if self._execution_engine_factory is not None
            else MockExecutionEngine(config=self._config, user_router=queueing_router)
        )

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
    # Decision + execution loop                                                #
    # ---------------------------------------------------------------------- #

    async def _decision_loop(self, state: LocalState) -> None:
        while True:
            self._poll_and_execute(state, utc_now_ms())
            await asyncio.sleep(self._decision_poll_ms / 1000.0)

    def _poll_and_execute(self, state: LocalState, now_ms: int) -> None:
        """
        Check for state changes, run decision cycle, sync quotes, check fills.

        Decision (fair → strategy → dedup) triggers _sync_quotes_impl.
        Fill check runs on every state change regardless of decision dedup.
        RuntimeError from fair_engine is counted; all other exceptions propagate.
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
            self._check_fills(state, now_ms)
            return

        ptb = self._ptb_locker.try_lock(state, now_ms)
        desired = self._strategy.build(state, fair, now_ms)

        key = self._dedup_key_from(fair, desired)
        if key != self._last_dedup_key:
            self._last_dedup_key = key
            self.decisions.append(
                DecisionSnapshot(
                    now_ms=now_ms,
                    trigger=trigger,
                    fair=fair,
                    desired_quotes=desired,
                    ptb_locked=ptb.locked,
                    ptb_value=ptb.ptb_value,
                )
            )
            self._sync_quotes_impl(state, desired, now_ms)

        self._check_fills(state, now_ms)

    # ---------------------------------------------------------------------- #
    # Paper execution helpers                                                  #
    # ---------------------------------------------------------------------- #

    def _sync_quotes_impl(
        self, state: LocalState, desired: DesiredQuotes, now_ms: int
    ) -> None:
        """
        Bid-only quote synchronisation, mirroring AsyncLocalRunner._sync_quotes().
        All order events flow through engine → QueueingUserRouter → drained here.
        """
        assert self._engine is not None
        threshold = self._config.thresholds.quote.size_change_reprice_ratio.value
        current_bid_id = state.live_bid_order_id
        current_bid = state.open_orders.get(current_bid_id) if current_bid_id else None

        if desired.bid.enabled and desired.bid.price is not None and desired.bid.size > 0:
            if current_bid is None:
                order_id = self._engine.post_order(
                    state,
                    asset_id=state.market.yes_token_id,
                    side=desired.bid.side,
                    price=desired.bid.price,
                    size=desired.bid.size,
                    now_ms=now_ms,
                    slot="bid",
                )
                if order_id is None:
                    self._orders_rejected += 1
                    self._capture_rejection_reason(state)
                else:
                    self._orders_posted += 1
                self._drain_user_queue(state)
            else:
                remaining = current_bid.remaining
                size_changed = (
                    abs(remaining - desired.bid.size) / remaining >= threshold
                    if remaining > 0
                    else False
                )
                price_changed = abs(current_bid.price - desired.bid.price) > 1e-12
                if price_changed or size_changed:
                    cancel_action = self._engine.cancel_order(
                        state, current_bid_id, now_ms, reason="reprice"
                    )
                    self._drain_user_queue(state)
                    if cancel_action is not None:
                        self._orders_cancelled += 1
                    order_id = self._engine.post_order(
                        state,
                        asset_id=state.market.yes_token_id,
                        side=desired.bid.side,
                        price=desired.bid.price,
                        size=desired.bid.size,
                        now_ms=now_ms + 1,
                        slot="bid",
                    )
                    if order_id is None:
                        self._orders_rejected += 1
                        self._capture_rejection_reason(state)
                    else:
                        self._orders_posted += 1
                    self._drain_user_queue(state)
        else:
            if current_bid is not None:
                cancel_action = self._engine.cancel_order(
                    state, current_bid_id, now_ms, reason="bid_disabled"
                )
                self._drain_user_queue(state)
                if cancel_action is not None:
                    self._orders_cancelled += 1

    def _check_fills(self, state: LocalState, now_ms: int) -> None:
        """
        Conservative fill simulation: trigger fill when top_ask.price <= live_bid.price.
        fill_size = min(order.remaining, top_ask.size).
        """
        assert self._engine is not None
        bid_id = state.live_bid_order_id
        if not bid_id or bid_id not in state.open_orders:
            return
        order = state.open_orders[bid_id]
        top_ask = state.yes_book.top_ask()
        if top_ask is not None and top_ask.price <= order.price:
            fill_size = min(order.remaining, top_ask.size)
            actions = self._engine.simulate_fill(
                state, order_id=bid_id, fill_size=fill_size, now_ms=now_ms
            )
            self._drain_user_queue(state)
            if actions:
                self._fills_simulated += 1

    def _drain_user_queue(self, state: LocalState) -> None:
        """Apply all pending user queue messages to state synchronously."""
        assert self._user_queue is not None
        while not self._user_queue.empty():
            msg = self._user_queue.get_nowait()
            self._user_router.apply(state, msg)

    def _capture_rejection_reason(self, state: LocalState) -> None:
        """Read rejection reason from the most recent warn log after post_order() → None."""
        if state.logs and state.logs[-1].message == "mock_order_rejected":
            self._last_rejection_reason = str(
                state.logs[-1].payload.get("reason", "unknown")
            )

    def _reset_run_state(self) -> None:
        self.decisions = []
        self._user_queue = None
        self._engine = None
        self._last_binance_seq = None
        self._last_yes_book_ts = 0
        self._last_dedup_key = None
        self._skipped_fair_count = 0
        self._last_fair_error = None
        self._orders_posted = 0
        self._orders_cancelled = 0
        self._orders_rejected = 0
        self._fills_simulated = 0
        self._last_rejection_reason = None

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
