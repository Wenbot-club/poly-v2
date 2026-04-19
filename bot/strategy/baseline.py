from __future__ import annotations

import math
from dataclasses import dataclass

from ..domain import DesiredOrder, DesiredQuotes, FairValueSnapshot, LocalState


# ---------------------------------------------------------------------------
# Edge constants (all in 0..1 probability space)
# ---------------------------------------------------------------------------

BASE_EDGE: float     = 0.01  # minimum half-spread
K_SPREAD: float      = 0.10  # edge scaling on book spread
K_UNCERTAINTY: float = 0.50  # edge scaling on |gap_z| (prob-point per sigma unit)
Z_CAP: float         = 3.0   # clip |gap_z| before applying uncertainty term
MIN_EDGE: float      = 0.01  # floor on computed edge
MAX_EDGE: float      = 0.10  # cap on computed edge

# ---------------------------------------------------------------------------
# Gate constants
# ---------------------------------------------------------------------------

MAX_BOOK_SPREAD: float    = 0.15   # reject if top_ask - top_bid exceeds this
TAU_GATE_S: float         = 30.0   # belt+suspenders — _poll_and_execute also gates
CHAINLINK_MAX_AGE_MS: int = 5_000  # 5 seconds
BINANCE_MAX_AGE_MS: int   = 2_000  # 2 seconds
FAIR_SANITY_MIN: float    = 0.02   # reject if p_up ≤ this or ≥ 1 − this

# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

QUOTE_NOTIONAL_USD: float = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _disabled(reason: str, now_ms: int) -> DesiredQuotes:
    off = DesiredOrder(enabled=False, side="BUY", price=None, size=0.0, reason=reason)
    return DesiredQuotes(
        bid=off,
        ask=DesiredOrder(enabled=False, side="SELL", price=None, size=0.0, reason=reason),
        mode="gated",
        inventory_skew=0.0,
        timestamp_ms=now_ms,
    )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass
class QuotePolicy:
    config: object

    def build(self, state: LocalState, fair: FairValueSnapshot, now_ms: int) -> DesiredQuotes:
        state.set_clock(now_ms)
        tick = state.market.clob.min_tick_size

        # ------------------------------------------------------------------ #
        # Gates — return all-disabled if any condition fails                  #
        # ------------------------------------------------------------------ #

        top_bid = state.yes_book.top_bid()
        top_ask = state.yes_book.top_ask()
        if top_bid is None or top_ask is None:
            return _disabled("book_gate", now_ms)

        book_spread = top_ask.price - top_bid.price
        if top_bid.price <= 0.0 or top_ask.price >= 1.0 or book_spread > MAX_BOOK_SPREAD:
            return _disabled("book_gate", now_ms)

        if state.last_chainlink is not None:
            if now_ms - state.last_chainlink.recv_timestamp_ms > CHAINLINK_MAX_AGE_MS:
                return _disabled("chainlink_stale", now_ms)

        if state.last_binance is not None:
            if now_ms - state.last_binance.recv_timestamp_ms > BINANCE_MAX_AGE_MS:
                return _disabled("binance_stale", now_ms)

        if fair.tau_s < TAU_GATE_S:
            return _disabled("tau_gate", now_ms)

        if fair.p_up <= FAIR_SANITY_MIN or fair.p_up >= 1.0 - FAIR_SANITY_MIN:
            return _disabled("fair_sanity", now_ms)

        # ------------------------------------------------------------------ #
        # Edge computation (all in 0..1 space)                                #
        # ------------------------------------------------------------------ #

        gap_z_clipped    = min(abs(fair.gap_z), Z_CAP)
        spread_term      = K_SPREAD * book_spread
        uncertainty_term = K_UNCERTAINTY * gap_z_clipped * 0.01
        edge = max(MIN_EDGE, min(MAX_EDGE, BASE_EDGE + spread_term + uncertainty_term))

        # ------------------------------------------------------------------ #
        # Inventory skew — long position widens bid edge, tightens ask edge  #
        # ------------------------------------------------------------------ #

        up_usd = state.inventory.up_free * fair.p_up
        skew   = min(up_usd / QUOTE_NOTIONAL_USD, 1.0) if QUOTE_NOTIONAL_USD > 0 else 0.0
        bid_edge = edge * (1.0 + 0.5 * skew)
        ask_edge = edge * (1.0 - 0.25 * skew)

        # ------------------------------------------------------------------ #
        # Bid                                                                  #
        # ------------------------------------------------------------------ #

        raw_bid   = fair.p_up - bid_edge
        bid_price = math.floor(min(raw_bid, top_ask.price - tick) / tick) * tick
        bid_price = round(max(bid_price, tick), 2)
        bid_size  = math.floor((QUOTE_NOTIONAL_USD / bid_price) * 1000.0) / 1000.0
        bid_enabled = (
            bid_size > 0
            and state.inventory.available_pusd() + 1e-12 >= bid_price * bid_size
        )

        # ------------------------------------------------------------------ #
        # Ask — sell existing YES inventory only, never short                 #
        # ------------------------------------------------------------------ #

        ask_price   = None
        ask_size    = 0.0
        ask_enabled = False

        if state.inventory.up_free >= state.market.clob.min_order_size:
            raw_ask   = fair.p_up + ask_edge
            ask_price = math.ceil(max(raw_ask, top_bid.price + tick) / tick) * tick
            ask_price = round(min(ask_price, 1.0 - tick), 2)
            ask_size  = math.floor((QUOTE_NOTIONAL_USD / ask_price) * 1000.0) / 1000.0
            # Guard: enough available inventory to cover the computed size.
            ask_enabled = (
                ask_size > 0
                and ask_price <= 1.0 - tick
                and state.inventory.available_up() >= ask_size
            )
            if not ask_enabled:
                ask_price = None
                ask_size  = 0.0

        inventory_skew = round(
            (state.inventory.up_free * fair.p_up - QUOTE_NOTIONAL_USD)
            / max(QUOTE_NOTIONAL_USD, 1.0),
            6,
        )

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
