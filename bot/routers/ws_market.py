from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from ..domain import LocalState, normalize_side


@dataclass
class MarketMessageRouter:
    def apply(self, state: LocalState, message: Dict[str, Any]) -> None:
        event_type = message.get("event_type")
        if event_type == "book":
            self._apply_book(state, message)
        elif event_type == "price_change":
            self._apply_price_change(state, message)
        elif event_type == "best_bid_ask":
            self._apply_best_bid_ask(state, message)
        elif event_type == "tick_size_change":
            self._apply_tick_size_change(state, message)
        elif event_type == "last_trade_price":
            self._apply_last_trade_price(state, message)
        else:
            state.log("WARN", "market_message_ignored", event_type=str(event_type))

    def _apply_book(self, state: LocalState, message: Dict[str, Any]) -> None:
        book = _select_book(state, str(message["asset_id"]))
        book.bids = {
            float(level["price"]): float(level["size"])
            for level in message.get("bids", [])
            if float(level["size"]) > 0
        }
        book.asks = {
            float(level["price"]): float(level["size"])
            for level in message.get("asks", [])
            if float(level["size"]) > 0
        }
        book.timestamp_ms = int(message["timestamp"])
        _refresh_best(book)
        state.set_clock(book.timestamp_ms)
        state.log("INFO", "book_snapshot_applied", ts_ms=book.timestamp_ms, asset_id=book.asset_id)

    def _apply_price_change(self, state: LocalState, message: Dict[str, Any]) -> None:
        ts_ms = int(message["timestamp"])
        state.set_clock(ts_ms)
        for change in message.get("price_changes", []):
            book = _select_book(state, str(change["asset_id"]))
            price = float(change["price"])
            size = float(change["size"])
            side = normalize_side(change["side"], field_name="price_changes[].side")
            target = book.bids if side == "BUY" else book.asks
            if size <= 0:
                target.pop(price, None)
            else:
                target[price] = size
            book.timestamp_ms = ts_ms
            if "best_bid" in change and "best_ask" in change:
                book.best.bid = float(change["best_bid"])
                book.best.ask = float(change["best_ask"])
                book.best.spread = max(0.0, book.best.ask - book.best.bid)
                book.best.timestamp_ms = ts_ms
            else:
                _refresh_best(book)
        state.log("INFO", "price_change_applied", ts_ms=ts_ms)

    def _apply_best_bid_ask(self, state: LocalState, message: Dict[str, Any]) -> None:
        book = _select_book(state, str(message["asset_id"]))
        book.best.bid = float(message["best_bid"])
        book.best.ask = float(message["best_ask"])
        book.best.spread = float(message.get("spread", book.best.ask - book.best.bid))
        book.best.timestamp_ms = int(message["timestamp"])
        book.timestamp_ms = book.best.timestamp_ms
        state.set_clock(book.timestamp_ms)
        state.log(
            "INFO",
            "best_bid_ask_applied",
            ts_ms=book.timestamp_ms,
            asset_id=book.asset_id,
            bid=book.best.bid,
            ask=book.best.ask,
        )

    def _apply_tick_size_change(self, state: LocalState, message: Dict[str, Any]) -> None:
        book = _select_book(state, str(message["asset_id"]))
        old_tick = book.tick_size
        book.tick_size = float(message["new_tick_size"])
        book.timestamp_ms = int(message["timestamp"])
        state.set_clock(book.timestamp_ms)
        state.log(
            "WARN",
            "tick_size_changed",
            ts_ms=book.timestamp_ms,
            asset_id=book.asset_id,
            old_tick=old_tick,
            new_tick=book.tick_size,
        )

    def _apply_last_trade_price(self, state: LocalState, message: Dict[str, Any]) -> None:
        book = _select_book(state, str(message["asset_id"]))
        book.last_trade_price = float(message["price"])
        book.last_trade_side = normalize_side(message["side"], field_name="last_trade_price.side")
        book.timestamp_ms = int(message["timestamp"])
        state.set_clock(book.timestamp_ms)
        state.log(
            "INFO",
            "last_trade_price_applied",
            ts_ms=book.timestamp_ms,
            asset_id=book.asset_id,
            price=book.last_trade_price,
            side=book.last_trade_side,
        )


def _select_book(state: LocalState, asset_id: str):
    if asset_id == state.market.yes_token_id:
        return state.yes_book
    if asset_id == state.market.no_token_id:
        return state.no_book
    raise KeyError(f"Unknown asset_id: {asset_id}")


def _refresh_best(book) -> None:
    bid = max(book.bids.keys()) if book.bids else 0.0
    ask = min(book.asks.keys()) if book.asks else 1.0
    book.best.bid = bid
    book.best.ask = ask
    book.best.spread = max(0.0, ask - bid)
    book.best.timestamp_ms = book.timestamp_ms


@dataclass
class MockMarketStream:
    messages: Iterable[Dict[str, Any]]

    def run(self, state: LocalState) -> None:
        router = MarketMessageRouter()
        for msg in self.messages:
            router.apply(state, msg)
