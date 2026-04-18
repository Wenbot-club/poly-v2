from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import List, Optional

from .settings import RuntimeConfig
from .types import LocalState, normalize_side
from .ws_user import UserMessageRouter


@dataclass
class MockExecutionEngine:
    config: RuntimeConfig
    user_router: UserMessageRouter = field(default_factory=UserMessageRouter)
    order_seq: count = field(default_factory=lambda: count(1))
    trade_seq: count = field(default_factory=lambda: count(1))

    def post_order(
        self,
        state: LocalState,
        *,
        asset_id: str,
        side: str,
        price: float,
        size: float,
        now_ms: int,
        slot: str,
    ) -> Optional[str]:
        side_norm = normalize_side(side, field_name="post_order.side")
        validation_error = self._validate_post_only(
            state,
            asset_id=asset_id,
            side=side_norm,
            price=price,
            size=size,
        )
        if validation_error is not None:
            state.log(
                "WARN",
                "mock_order_rejected",
                ts_ms=now_ms,
                asset_id=asset_id,
                side=side_norm,
                price=price,
                size=size,
                reason=validation_error,
                slot=slot,
            )
            return None

        order_id = f"mock-{next(self.order_seq)}"
        self.user_router.apply(
            state,
            {
                "event_type": "order",
                "timestamp": now_ms,
                "order_id": order_id,
                "asset_id": asset_id,
                "side": side_norm,
                "price": price,
                "original_size": size,
                "size_matched": 0.0,
                "status": "LIVE",
            },
        )
        state.log(
            "INFO",
            "mock_order_post_requested",
            ts_ms=now_ms,
            order_id=order_id,
            asset_id=asset_id,
            side=side_norm,
            price=price,
            size=size,
            slot=slot,
        )
        return order_id

    def cancel_order(self, state: LocalState, order_id: str, now_ms: int, reason: str) -> Optional[str]:
        order = state.open_orders.get(order_id)
        if order is None:
            state.log("WARN", "mock_cancel_ignored_unknown_order", ts_ms=now_ms, order_id=order_id, reason=reason)
            return None
        self.user_router.apply(
            state,
            {
                "event_type": "order",
                "timestamp": now_ms,
                "order_id": order_id,
                "asset_id": order.asset_id,
                "side": order.side,
                "price": order.price,
                "original_size": order.size,
                "size_matched": order.size - order.remaining,
                "status": "CANCELED",
            },
        )
        state.log(
            "INFO",
            "mock_order_cancel_requested",
            ts_ms=now_ms,
            order_id=order_id,
            reason=reason,
        )
        return f"cancel:{order_id}"

    def cancel_all(self, state: LocalState, now_ms: int, reason: str) -> List[str]:
        actions: List[str] = []
        for order_id in list(state.open_orders.keys()):
            action = self.cancel_order(state, order_id, now_ms, reason=reason)
            if action is not None:
                actions.append(action)
        return actions

    def simulate_fill(
        self,
        state: LocalState,
        *,
        order_id: str,
        fill_size: float,
        now_ms: int,
    ) -> List[str]:
        order = state.open_orders.get(order_id)
        if order is None:
            state.log("WARN", "mock_fill_ignored_unknown_order", ts_ms=now_ms, order_id=order_id)
            return []

        fill_size = min(fill_size, order.remaining)
        if fill_size <= 0:
            state.log("WARN", "mock_fill_ignored_nonpositive", ts_ms=now_ms, order_id=order_id, fill_size=fill_size)
            return []

        new_matched = (order.size - order.remaining) + fill_size
        status = "FILLED" if abs(new_matched - order.size) <= 1e-12 else "PARTIALLY_FILLED"

        self.user_router.apply(
            state,
            {
                "event_type": "trade",
                "timestamp": now_ms,
                "trade_id": f"mock-trade-{next(self.trade_seq)}",
                "order_id": order_id,
                "asset_id": order.asset_id,
                "side": order.side,
                "price": order.price,
                "size": fill_size,
            },
        )
        self.user_router.apply(
            state,
            {
                "event_type": "order",
                "timestamp": now_ms,
                "order_id": order_id,
                "asset_id": order.asset_id,
                "side": order.side,
                "price": order.price,
                "original_size": order.size,
                "size_matched": new_matched,
                "status": status,
            },
        )
        state.log(
            "INFO",
            "mock_fill_simulated",
            ts_ms=now_ms,
            order_id=order_id,
            fill_size=fill_size,
            status=status,
        )
        return [f"fill:{order_id}:{fill_size}", f"status:{order_id}:{status}"]

    @staticmethod
    def _validate_post_only(
        state: LocalState,
        *,
        asset_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Optional[str]:
        if asset_id != state.market.yes_token_id:
            return "mock_engine_only_supports_yes_token"
        side_norm = normalize_side(side, field_name="_validate_post_only.side")
        if size < state.market.clob.min_order_size:
            return "size_below_min_order_size"
        tick = state.market.clob.min_tick_size
        if abs(round(price / tick) * tick - price) > 1e-9:
            return "invalid_tick_size"
        top_bid = state.yes_book.top_bid()
        top_ask = state.yes_book.top_ask()
        if top_bid is None or top_ask is None:
            return "book_incomplete"
        if side_norm == "BUY" and price >= top_ask.price:
            return "would_cross_ask"
        if side_norm == "SELL" and price <= top_bid.price:
            return "would_cross_bid"
        if side_norm == "BUY":
            if state.inventory.available_pusd() + 1e-12 < price * size:
                return "not_enough_pusd"
        else:
            if state.inventory.available_up() + 1e-12 < size:
                return "not_enough_up_inventory"
        return None
