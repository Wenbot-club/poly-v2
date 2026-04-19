"""
Combined read-only orchestrator with conservative paper execution layer.

LivePaperSession wires together:
  LiveReadonlySession              — market book feed
  LiveRTDSSession                  — signal feed (Binance + Chainlink composite)
  FairValueEngine                  — fair value computation (requires last_chainlink)
  PTBLocker                        — price-to-beat lock
  Strategy                         — produces DesiredQuotes
  MockExecutionEngine              — simulated order posting, cancellation, fill

Discovery called exactly once. Both feeds share a single LocalState.
State mutations flow exclusively through MockExecutionEngine → QueueingUserRouter
→ UserMessageRouter.apply() → state. No direct mutation of open_orders or inventory.

Execution pattern:
  Each decision cycle (new dedup-passing decision):
    → _sync_quotes_impl() posts/cancels simulated bid (bid-only, mirrors AsyncLocalRunner)
  Each poll with state change (regardless of decision dedup):
    → _check_fills() simulates fill when top_ask.price <= live_bid.price (conservative)

Peak tracking:
  _update_peaks() is called inside _drain_user_queue() after every state mutation.
  This ensures max_pusd_reserved captures the reservation from post_order before
  any fill releases it, and max_up_inventory captures the peak after fills.

Price anchor: FairValueEngine.compute() raises RuntimeError when last_chainlink is None.
Caught as RuntimeError only — other exceptions propagate. Counted in skipped_fair_value_count.

To post simulated orders in live operation, pass a CompositeSignalProvider combining
BinanceSignalProvider + PolymarketChainlinkSignalProvider. Chainlink ticks from the
Polymarket RTDS feed (wss://ws-live-data.polymarket.com, topic crypto_prices_chainlink,
no auth) populate last_chainlink natively via RTDSMessageRouter. Not on-chain Chainlink.

No real orders, no real fills, no PnL (no outcome available). Honest counters only.

Event log:
  session.events is a list[dict] populated during run_for(). Call
  paper_journal.write_jsonl(session.events, path) after run_for() to persist.
  run_for() never writes to disk itself.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    # Decision layer
    decision_count: int
    first_decision_ts_ms: Optional[int]
    last_decision_ts_ms: Optional[int]
    skipped_fair_value_count: int
    last_fair_value_error: Optional[str]
    # Execution counters
    orders_posted: int
    orders_cancelled: int
    orders_rejected: int
    fills_simulated: int
    filled_orders: int          # unique order_ids with ≥1 fill
    last_rejection_reason: Optional[str]
    # Derived ratios (None when denominator == 0)
    fill_rate: Optional[float]              # filled_orders / orders_posted
    rejection_rate: Optional[float]         # orders_rejected / (orders_posted + orders_rejected)
    cancel_to_post_ratio: Optional[float]   # orders_cancelled / orders_posted
    # Peak inventory (tracked live via _update_peaks after each _drain_user_queue)
    max_up_inventory: float
    max_pusd_reserved: float
    # Fill timestamps
    first_fill_ts_ms: Optional[int]
    last_fill_ts_ms: Optional[int]
    # Final inventory
    final_pusd_free: float
    final_up_free: float
    final_pusd_reserved: float
    # Mark-to-market PnL (conservative: YES inventory valued at yes_book top bid)
    # pnl_total_mark == pnl_unrealized_mark in bid-only mode (no sells → no realization).
    # All mark fields are None when up_free > 0 and yes_book has no bids.
    # When up_free == 0 the portfolio value is exact (no mark uncertainty).
    portfolio_value_start: float            # initial_pusd — all cash at t0
    portfolio_value_end_mark: Optional[float]  # pusd_free + up_free * mark_price
    pnl_total_mark: Optional[float]         # portfolio_value_end_mark - portfolio_value_start
    pnl_unrealized_mark: Optional[float]    # up_free * mark_price - cost_basis
    cost_basis: float                       # PUSD spent acquiring YES = initial_pusd - pusd_free
    mark_price: Optional[float]             # yes_book.top_bid().price at session end
    mark_source: str                        # always "yes_book_top_bid"


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

    After run_for(), session.events holds the ordered event log. Pass it to
    paper_journal.write_jsonl() to persist — run_for() never writes to disk.
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
        self.events: list[dict] = []

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
        self._filled_order_ids: set[str] = set()
        self._first_fill_ts_ms: Optional[int] = None
        self._last_fill_ts_ms: Optional[int] = None
        self._max_up_inventory: float = 0.0
        self._max_pusd_reserved: float = 0.0
        self._last_rejection_reason: Optional[str] = None
        self._portfolio_value_start: float = 0.0

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    async def run_for(self, duration: int) -> LivePaperSummary:
        """
        Run both feeds + strategy + paper execution for `duration` seconds.

        Decision loop runs as a background task, cancelled when feeds finish.
        Final synchronous pass after cancellation catches state changes that
        arrived after the last poll interval.

        After this returns, session.events holds the ordered JSONL-ready event log.
        Call paper_journal.write_jsonl(session.events, path) to persist it.
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
        self._portfolio_value_start = pusd

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

        posted = self._orders_posted
        filled = len(self._filled_order_ids)
        rejected = self._orders_rejected
        cancelled = self._orders_cancelled

        fill_rate = filled / posted if posted > 0 else None
        rejection_rate = rejected / (posted + rejected) if (posted + rejected) > 0 else None
        cancel_to_post_ratio = cancelled / posted if posted > 0 else None

        first_ts = self.decisions[0].now_ms if self.decisions else None
        last_ts = self.decisions[-1].now_ms if self.decisions else None

        final_pusd_free = state.inventory.pusd_free
        final_pusd_reserved = state.inventory.pusd_reserved_for_bids
        final_up_free = state.inventory.up_free

        portfolio_value_start = float(pusd)
        cost_basis = portfolio_value_start - final_pusd_free

        top_bid = state.yes_book.top_bid()
        mark_price = top_bid.price if top_bid is not None else None
        mark_source = "yes_book_top_bid"

        if final_up_free > 0.0:
            if mark_price is None:
                portfolio_value_end_mark: Optional[float] = None
                pnl_total_mark: Optional[float] = None
                pnl_unrealized_mark: Optional[float] = None
            else:
                inventory_mark_value = final_up_free * mark_price
                portfolio_value_end_mark = final_pusd_free + inventory_mark_value
                pnl_total_mark = portfolio_value_end_mark - portfolio_value_start
                pnl_unrealized_mark = inventory_mark_value - cost_basis
        else:
            # No YES inventory to mark: portfolio value is still known even without a book.
            portfolio_value_end_mark = final_pusd_free
            pnl_total_mark = portfolio_value_end_mark - portfolio_value_start
            pnl_unrealized_mark = 0.0

        self.events.append({
            "ts_ms": utc_now_ms(),
            "event": "session_end",
            "portfolio_value_start": portfolio_value_start,
            "portfolio_value_end_mark": portfolio_value_end_mark,
            "pnl_total_mark": pnl_total_mark,
            "pnl_unrealized_mark": pnl_unrealized_mark,
            "cost_basis": cost_basis,
            "mark_price": mark_price,
            "mark_source": mark_source,
            "up_free": final_up_free,
            "pusd_free": final_pusd_free,
            "pusd_reserved": final_pusd_reserved,
        })

        return LivePaperSummary(
            market=market_summary,
            rtds=rtds_summary,
            decision_count=len(self.decisions),
            first_decision_ts_ms=first_ts,
            last_decision_ts_ms=last_ts,
            skipped_fair_value_count=self._skipped_fair_count,
            last_fair_value_error=self._last_fair_error,
            orders_posted=posted,
            orders_cancelled=cancelled,
            orders_rejected=rejected,
            fills_simulated=self._fills_simulated,
            filled_orders=filled,
            fill_rate=fill_rate,
            rejection_rate=rejection_rate,
            cancel_to_post_ratio=cancel_to_post_ratio,
            max_up_inventory=self._max_up_inventory,
            max_pusd_reserved=self._max_pusd_reserved,
            first_fill_ts_ms=self._first_fill_ts_ms,
            last_fill_ts_ms=self._last_fill_ts_ms,
            final_pusd_free=final_pusd_free,
            final_up_free=final_up_free,
            final_pusd_reserved=final_pusd_reserved,
            last_rejection_reason=self._last_rejection_reason,
            portfolio_value_start=portfolio_value_start,
            portfolio_value_end_mark=portfolio_value_end_mark,
            pnl_total_mark=pnl_total_mark,
            pnl_unrealized_mark=pnl_unrealized_mark,
            cost_basis=cost_basis,
            mark_price=mark_price,
            mark_source=mark_source,
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
        self._portfolio_value_start = pusd

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
            self.events.append({
                "ts_ms": now_ms,
                "event": "decision",
                "trigger": trigger,
                "p_up": round(fair.p_up, 6),
                "bid_price": desired.bid.price,
                "bid_enabled": desired.bid.enabled,
                "ask_price": desired.ask.price,
                "ask_enabled": desired.ask.enabled,
                "ptb_locked": ptb.locked,
                "ptb_value": ptb.ptb_value,
            })
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
                    reason = self._extract_rejection_reason(state)
                    self._last_rejection_reason = reason
                    self.events.append({
                        "ts_ms": now_ms,
                        "event": "order_rejected",
                        "side": desired.bid.side,
                        "price": desired.bid.price,
                        "size": desired.bid.size,
                        "reason": reason,
                    })
                else:
                    self._orders_posted += 1
                    self.events.append({
                        "ts_ms": now_ms,
                        "event": "order_posted",
                        "order_id": order_id,
                        "side": desired.bid.side,
                        "price": desired.bid.price,
                        "size": desired.bid.size,
                    })
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
                        self.events.append({
                            "ts_ms": now_ms,
                            "event": "order_cancelled",
                            "order_id": current_bid_id,
                            "reason": "reprice",
                        })
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
                        reason = self._extract_rejection_reason(state)
                        self._last_rejection_reason = reason
                        self.events.append({
                            "ts_ms": now_ms,
                            "event": "order_rejected",
                            "side": desired.bid.side,
                            "price": desired.bid.price,
                            "size": desired.bid.size,
                            "reason": reason,
                        })
                    else:
                        self._orders_posted += 1
                        self.events.append({
                            "ts_ms": now_ms,
                            "event": "order_posted",
                            "order_id": order_id,
                            "side": desired.bid.side,
                            "price": desired.bid.price,
                            "size": desired.bid.size,
                        })
                    self._drain_user_queue(state)
        else:
            if current_bid is not None:
                cancel_action = self._engine.cancel_order(
                    state, current_bid_id, now_ms, reason="bid_disabled"
                )
                self._drain_user_queue(state)
                if cancel_action is not None:
                    self._orders_cancelled += 1
                    self.events.append({
                        "ts_ms": now_ms,
                        "event": "order_cancelled",
                        "order_id": current_bid_id,
                        "reason": "bid_disabled",
                    })

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
            fill_price = order.price
            actions = self._engine.simulate_fill(
                state, order_id=bid_id, fill_size=fill_size, now_ms=now_ms
            )
            self._drain_user_queue(state)
            if actions:
                self._fills_simulated += 1
                self._filled_order_ids.add(bid_id)
                if self._first_fill_ts_ms is None:
                    self._first_fill_ts_ms = now_ms
                self._last_fill_ts_ms = now_ms
                self.events.append({
                    "ts_ms": now_ms,
                    "event": "fill_simulated",
                    "order_id": bid_id,
                    "fill_price": fill_price,
                    "fill_size": fill_size,
                })

    def _drain_user_queue(self, state: LocalState) -> None:
        """Apply all pending user queue messages to state, then update peak inventory."""
        assert self._user_queue is not None
        while not self._user_queue.empty():
            msg = self._user_queue.get_nowait()
            self._user_router.apply(state, msg)
        self._update_peaks(state)

    def _update_peaks(self, state: LocalState) -> None:
        up = state.inventory.up_free
        reserved = state.inventory.pusd_reserved_for_bids
        if up > self._max_up_inventory:
            self._max_up_inventory = up
        if reserved > self._max_pusd_reserved:
            self._max_pusd_reserved = reserved

    def _extract_rejection_reason(self, state: LocalState) -> str:
        """Read canonical rejection reason from the last warn log after post_order() → None."""
        if state.logs and state.logs[-1].message == "mock_order_rejected":
            return str(state.logs[-1].payload.get("reason", "unknown"))
        return "unknown"

    def _reset_run_state(self) -> None:
        self.decisions = []
        self.events = []
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
        self._filled_order_ids = set()
        self._first_fill_ts_ms = None
        self._last_fill_ts_ms = None
        self._max_up_inventory = 0.0
        self._max_pusd_reserved = 0.0
        self._last_rejection_reason = None
        self._portfolio_value_start = 0.0

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
