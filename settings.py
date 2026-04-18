from __future__ import annotations

from dataclasses import dataclass
import math

from .types import DesiredOrder, DesiredQuotes, FairValueSnapshot, LocalState


@dataclass
class QuotePolicy:
    config: object

    def build(self, state: LocalState, fair: FairValueSnapshot, now_ms: int) -> DesiredQuotes:
        state.set_clock(now_ms)
        tick = state.market.clob.min_tick_size
        quote_notional_usd = 5.0

        raw_bid = fair.p_up - 0.015
        best_ask_cap = state.yes_book.best.ask - tick
        bid_price = math.floor(min(raw_bid, best_ask_cap) / tick) * tick
        bid_price = round(max(bid_price, tick), 2)
        bid_size = math.floor((quote_notional_usd / bid_price) * 1000.0) / 1000.0
        bid_enabled = state.inventory.available_pusd() + 1e-12 >= bid_price * bid_size

        ask_enabled = state.inventory.available_up() >= state.market.clob.min_order_size and fair.p_up < 0.5
        ask_price = None
        ask_size = 0.0
        if ask_enabled:
            raw_ask = fair.p_up + 0.015
            best_bid_floor = state.yes_book.best.bid + tick
            ask_price = math.ceil(max(raw_ask, best_bid_floor) / tick) * tick
            ask_price = round(ask_price, 2)
            ask_size = math.floor((quote_notional_usd / ask_price) * 1000.0) / 1000.0

        inventory_skew = round((state.inventory.up_free * bid_price - quote_notional_usd) / max(quote_notional_usd, 1.0), 6)
        quotes = DesiredQuotes(
            bid=DesiredOrder(
                enabled=bid_enabled,
                side="BUY",
                price=bid_price if bid_enabled else None,
                size=bid_size if bid_enabled else 0.0,
                reason="quote_bid" if bid_enabled else "bid_disabled",
            ),
            ask=DesiredOrder(
                enabled=ask_enabled,
                side="SELL",
                price=ask_price if ask_enabled else None,
                size=ask_size if ask_enabled else 0.0,
                reason="quote_ask" if ask_enabled else "ask_disabled",
            ),
            mode="two_sided" if ask_enabled else "bid_only",
            inventory_skew=inventory_skew,
            timestamp_ms=now_ms,
        )
        state.desired_quotes = quotes
        state.log(
            "INFO",
            "desired_quotes_built",
            ts_ms=now_ms,
            bid=quotes.bid.price,
            ask=quotes.ask.price,
            bid_size=quotes.bid.size,
            ask_size=quotes.ask.size,
            inventory_skew=quotes.inventory_skew,
        )
        return quotes
