from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from .types import LiveOrder, LocalState, UserOrderEvent, UserTradeEvent, normalize_side


@dataclass
class UserMessageRouter:
    def apply(self, state: LocalState, message: Dict[str, Any]) -> None:
        event_type = str(message.get("event_type", "")).lower()
        if event_type == "order":
            self._apply_order(state, message)
        elif event_type == "trade":
            self._apply_trade(state, message)
        else:
            state.log("WARN", "user_message_ignored", event_type=event_type)

    def _apply_order(self, state: LocalState, message: Dict[str, Any]) -> None:
        ts_ms = int(message["timestamp"])
        state.set_clock(ts_ms)

        order_id = str(message["order_id"])
        asset_id = str(message["asset_id"])
        side = normalize_side(message["side"], field_name="order.side")
        price = float(message["price"])
        original_size = float(message["original_size"])
        size_matched = float(message.get("size_matched", 0.0))
        remaining = max(0.0, original_size - size_matched)
        status = str(message["status"]).upper()

        event = UserOrderEvent(
            order_id=order_id,
            asset_id=asset_id,
            side=side,
            price=price,
            original_size=original_size,
            size_matched=size_matched,
            remaining=remaining,
            status=status,
            timestamp_ms=ts_ms,
        )
        state.user_order_events.append(event)

        if status in {"LIVE", "OPEN", "PARTIALLY_FILLED"}:
            live = state.open_orders.get(order_id)
            if live is None:
                state.open_orders[order_id] = LiveOrder(
                    order_id=order_id,
                    asset_id=asset_id,
                    side=side,
                    price=price,
                    size=original_size,
                    remaining=remaining,
                    status=status,
                    created_ts_ms=ts_ms,
                    updated_ts_ms=ts_ms,
                )
            else:
                live.price = price
                live.size = original_size
                live.remaining = remaining
                live.status = status
                live.updated_ts_ms = ts_ms
        else:
            state.open_orders.pop(order_id, None)

        if asset_id == state.market.yes_token_id:
            if status in {"LIVE", "OPEN", "PARTIALLY_FILLED"}:
                if side == "BUY":
                    state.live_bid_order_id = order_id
                else:
                    state.live_ask_order_id = order_id
            else:
                if state.live_bid_order_id == order_id:
                    state.live_bid_order_id = None
                if state.live_ask_order_id == order_id:
                    state.live_ask_order_id = None

        _recompute_live_exposure_and_reservations(state)
        state.log(
            "INFO",
            "user_order_applied",
            ts_ms=ts_ms,
            order_id=order_id,
            asset_id=asset_id,
            side=side,
            price=price,
            original_size=original_size,
            size_matched=size_matched,
            remaining=remaining,
            status=status,
            live_bid_order_id=state.live_bid_order_id,
            live_ask_order_id=state.live_ask_order_id,
        )

    def _apply_trade(self, state: LocalState, message: Dict[str, Any]) -> None:
        ts_ms = int(message["timestamp"])
        state.set_clock(ts_ms)

        trade_id = str(message["trade_id"])
        asset_id = str(message["asset_id"])
        side = normalize_side(message["side"], field_name="trade.side")
        price = float(message["price"])
        size = float(message["size"])
        order_id = str(message["order_id"]) if message.get("order_id") is not None else None

        event = UserTradeEvent(
            trade_id=trade_id,
            asset_id=asset_id,
            side=side,
            price=price,
            size=size,
            timestamp_ms=ts_ms,
            order_id=order_id,
        )
        state.user_trade_events.append(event)

        if asset_id == state.market.yes_token_id:
            if side == "BUY":
                state.inventory.up_free += size
                state.inventory.pusd_free -= price * size
            else:
                state.inventory.up_free -= size
                state.inventory.pusd_free += price * size

        state.log(
            "INFO",
            "user_trade_applied",
            ts_ms=ts_ms,
            trade_id=trade_id,
            order_id=order_id,
            asset_id=asset_id,
            side=side,
            price=price,
            size=size,
            up_free=round(state.inventory.up_free, 6),
            pusd_free=round(state.inventory.pusd_free, 6),
        )


def _recompute_live_exposure_and_reservations(state: LocalState) -> None:
    up_live_bids = 0.0
    up_live_asks = 0.0
    pusd_reserved = 0.0
    up_reserved = 0.0
    for order in state.open_orders.values():
        if order.asset_id != state.market.yes_token_id:
            continue
        if order.status not in {"LIVE", "OPEN", "PARTIALLY_FILLED"}:
            continue
        if order.side == "BUY":
            up_live_bids += order.remaining
            pusd_reserved += order.remaining * order.price
        else:
            up_live_asks += order.remaining
            up_reserved += order.remaining
    state.inventory.up_live_bids = up_live_bids
    state.inventory.up_live_asks = up_live_asks
    state.inventory.pusd_reserved_for_bids = round(pusd_reserved, 12)
    state.inventory.up_reserved_for_asks = round(up_reserved, 12)


@dataclass
class MockUserStream:
    messages: Iterable[Dict[str, Any]]

    def run(self, state: LocalState) -> None:
        router = UserMessageRouter()
        for msg in self.messages:
            router.apply(state, msg)
