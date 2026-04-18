from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .execution import MockExecutionEngine
from .fair_value import FairValueEngine
from .ptb import PTBLocker
from .quote_policy import QuotePolicy
from .recorder import JSONLRecorder
from .settings import RuntimeConfig
from .types import DesiredQuotes, LocalState
from .ws_market import MarketMessageRouter
from .ws_rtds import RTDSMessageRouter
from .ws_user import UserMessageRouter


class QueueingUserRouter:
    def __init__(self, queue: asyncio.Queue[Dict[str, Any]], recorder: Optional[JSONLRecorder] = None) -> None:
        self.queue = queue
        self.recorder = recorder

    def apply(self, state: LocalState, message: Dict[str, Any]) -> None:
        if self.recorder is not None:
            self.recorder.record("user_message", dict(message))
        self.queue.put_nowait(message)


@dataclass(slots=True)
class AsyncRunSummary:
    actions: List[str]
    initial_quote: dict
    reeval_quote: dict
    final_state: dict
    temporal_checks: List[dict]


@dataclass
class AsyncLocalRunner:
    config: RuntimeConfig
    state: LocalState
    event_script: List[Dict[str, Any]]
    recorder: Optional[JSONLRecorder] = None

    async def run(self) -> AsyncRunSummary:
        market_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        rtds_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        user_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        control_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        sequencing_done = asyncio.Event()
        control_done = asyncio.Event()

        market_router = MarketMessageRouter()
        rtds_router = RTDSMessageRouter(self.config)
        user_router = UserMessageRouter()
        queueing_user_router = QueueingUserRouter(user_queue, recorder=self.recorder)
        engine = MockExecutionEngine(config=self.config, user_router=queueing_user_router)
        ptb_locker = PTBLocker(self.config)
        fair_engine = FairValueEngine(self.config)
        quote_policy = QuotePolicy(self.config)

        actions: List[str] = []
        temporal_checks: List[dict] = []
        initial_quote: Optional[dict] = None
        reeval_quote: Optional[dict] = None

        if self.recorder is not None:
            self.recorder.record(
                "run_header",
                {
                    "initial_clock_ms": self.state.simulated_now_ms,
                    "initial_inventory": {
                        "up_free": self.state.inventory.up_free,
                        "down_free": self.state.inventory.down_free,
                        "pusd_free": self.state.inventory.pusd_free,
                        "up_target": self.state.inventory.up_target,
                    },
                    "script_length": len(self.event_script),
                },
            )

        async def sequencer() -> None:
            for step in self.event_script:
                channel = str(step["channel"])
                payload = dict(step["payload"])
                if channel == "market":
                    market_queue.put_nowait(payload)
                    await market_queue.join()
                elif channel == "rtds":
                    rtds_queue.put_nowait(payload)
                    await rtds_queue.join()
                elif channel == "control":
                    control_queue.put_nowait(payload)
                    await control_queue.join()
                    await user_queue.join()
                else:
                    raise ValueError(f"unsupported scripted channel: {channel!r}")
            sequencing_done.set()

        async def market_consumer() -> None:
            while not (sequencing_done.is_set() and market_queue.empty()):
                try:
                    msg = await asyncio.wait_for(market_queue.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    continue
                try:
                    market_router.apply(self.state, msg)
                finally:
                    market_queue.task_done()

        async def rtds_consumer() -> None:
            while not (sequencing_done.is_set() and rtds_queue.empty()):
                try:
                    msg = await asyncio.wait_for(rtds_queue.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    continue
                try:
                    rtds_router.apply(self.state, msg)
                finally:
                    rtds_queue.task_done()

        async def user_consumer() -> None:
            while True:
                try:
                    msg = await asyncio.wait_for(user_queue.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    if sequencing_done.is_set() and control_done.is_set() and user_queue.empty():
                        break
                    continue
                try:
                    user_router.apply(self.state, msg)
                finally:
                    user_queue.task_done()

        async def control_consumer() -> None:
            nonlocal initial_quote, reeval_quote
            try:
                while not (sequencing_done.is_set() and control_queue.empty()):
                    try:
                        tick = await asyncio.wait_for(control_queue.get(), timeout=0.01)
                    except asyncio.TimeoutError:
                        continue
                    try:
                        now_ms = int(tick["now_ms"])
                        op = str(tick["op"])
                        self.state.set_clock(now_ms)

                        if op == "quote":
                            ptb_locker.try_lock(self.state, now_ms=now_ms)
                            self._assert_temporal_consistency(now_ms, temporal_checks, label=str(tick.get("label", "quote")))
                            fair = fair_engine.compute(self.state, now_ms=now_ms)
                            desired = quote_policy.build(self.state, fair, now_ms=now_ms)
                            sync_actions = self._sync_quotes(engine, desired, now_ms)
                            actions.extend(sync_actions)
                            self._record_actions(sync_actions, now_ms)
                            await user_queue.join()
                            quote_snapshot = {
                                "now_ms": now_ms,
                                "bid_enabled": desired.bid.enabled,
                                "bid_price": desired.bid.price,
                                "bid_size": desired.bid.size,
                                "ask_enabled": desired.ask.enabled,
                                "ask_price": desired.ask.price,
                                "ask_size": desired.ask.size,
                                "actions": sync_actions,
                            }
                            if initial_quote is None:
                                initial_quote = quote_snapshot
                            else:
                                reeval_quote = quote_snapshot
                        elif op == "fill_partial":
                            order_id = self.state.live_bid_order_id
                            if order_id is None:
                                raise RuntimeError("fill_partial requested but no live bid order exists")
                            fill_actions = engine.simulate_fill(self.state, order_id=order_id, fill_size=float(tick["fill_size"]), now_ms=now_ms)
                            actions.extend(fill_actions)
                            self._record_actions(fill_actions, now_ms)
                            await user_queue.join()
                        elif op == "cancel_final":
                            cancel_actions = engine.cancel_all(self.state, now_ms=now_ms, reason=str(tick["reason"]))
                            actions.extend(cancel_actions)
                            self._record_actions(cancel_actions, now_ms)
                            await user_queue.join()
                        else:
                            raise ValueError(f"unsupported control op: {op!r}")
                    finally:
                        control_queue.task_done()
            finally:
                control_done.set()

        await asyncio.gather(sequencer(), market_consumer(), rtds_consumer(), user_consumer(), control_consumer())

        summary = AsyncRunSummary(
            actions=actions,
            initial_quote=initial_quote or {},
            reeval_quote=reeval_quote or {},
            final_state=self._snapshot_state(),
            temporal_checks=temporal_checks,
        )
        if self.recorder is not None:
            self.recorder.record("final_state", {"state": summary.final_state, "actions": list(actions)})
        return summary

    def _record_actions(self, sync_actions: List[str], now_ms: int) -> None:
        if self.recorder is None:
            return
        for action in sync_actions:
            self.recorder.record("execution_action", {"timestamp_ms": now_ms, "action": action})

    def _assert_temporal_consistency(self, now_ms: int, sink: List[dict], *, label: str) -> None:
        payload_candidates = [self.state.yes_book.timestamp_ms, self.state.no_book.timestamp_ms]
        recv_candidates = [self.state.yes_book.timestamp_ms, self.state.no_book.timestamp_ms]
        if self.state.last_chainlink is not None:
            payload_candidates.append(self.state.last_chainlink.timestamp_ms)
            recv_candidates.append(self.state.last_chainlink.recv_timestamp_ms)
        if self.state.last_binance is not None:
            payload_candidates.append(self.state.last_binance.timestamp_ms)
            recv_candidates.append(self.state.last_binance.recv_timestamp_ms)
        max_payload_ts = max(payload_candidates)
        max_recv_ts = max(recv_candidates)
        assert max_payload_ts <= now_ms, (label, max_payload_ts, now_ms)
        assert max_recv_ts <= now_ms, (label, max_recv_ts, now_ms)
        sink.append({"label": label, "now_ms": now_ms, "max_payload_ts": max_payload_ts, "max_recv_ts": max_recv_ts, "temporal_assertions_passed": True})

    def _sync_quotes(self, engine: MockExecutionEngine, desired: DesiredQuotes, now_ms: int) -> List[str]:
        actions: List[str] = []
        threshold = self.config.thresholds.quote.size_change_reprice_ratio.value
        current_bid = self.state.open_orders.get(self.state.live_bid_order_id) if self.state.live_bid_order_id else None
        if desired.bid.enabled and desired.bid.price is not None and desired.bid.size > 0:
            if current_bid is None:
                order_id = engine.post_order(self.state, asset_id=self.state.market.yes_token_id, side=desired.bid.side, price=desired.bid.price, size=desired.bid.size, now_ms=now_ms, slot="bid")
                if order_id is not None:
                    actions.append(f"post:{order_id}")
            else:
                remaining = current_bid.remaining
                size_changed = False
                if remaining > 0:
                    size_changed = abs(remaining - desired.bid.size) / remaining >= threshold
                price_changed = abs(current_bid.price - desired.bid.price) > 1e-12
                if price_changed or size_changed:
                    cancel_action = engine.cancel_order(self.state, current_bid.order_id, now_ms, reason="reprice")
                    if cancel_action is not None:
                        actions.append(cancel_action)
                    order_id = engine.post_order(self.state, asset_id=self.state.market.yes_token_id, side=desired.bid.side, price=desired.bid.price, size=desired.bid.size, now_ms=now_ms + 1, slot="bid")
                    if order_id is not None:
                        actions.append(f"post:{order_id}")
        else:
            if current_bid is not None:
                cancel_action = engine.cancel_order(self.state, current_bid.order_id, now_ms, reason="bid_disabled")
                if cancel_action is not None:
                    actions.append(cancel_action)
        return actions

    def _snapshot_state(self) -> dict:
        return {
            "open_orders": {oid: {"side": order.side, "price": order.price, "remaining": order.remaining, "status": order.status} for oid, order in self.state.open_orders.items()},
            "live_bid_order_id": self.state.live_bid_order_id,
            "live_ask_order_id": self.state.live_ask_order_id,
            "pusd_free": round(self.state.inventory.pusd_free, 6),
            "up_free": round(self.state.inventory.up_free, 6),
            "pusd_reserved_for_bids": round(self.state.inventory.pusd_reserved_for_bids, 6),
            "up_reserved_for_asks": round(self.state.inventory.up_reserved_for_asks, 6),
            "available_pusd": round(self.state.inventory.available_pusd(), 6),
            "available_up": round(self.state.inventory.available_up(), 6),
        }
