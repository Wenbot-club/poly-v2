"""Directional (FLAT → LONG) strategy for BTC M15."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..domain import DesiredOrder, DesiredQuotes, FairValueSnapshot, LocalState, PositionState
from ..settings import RuntimeConfig


def _disabled(reason: str, now_ms: int, strategy_state: str = "flat") -> DesiredQuotes:
    off = DesiredOrder(enabled=False, side="BUY", price=None, size=0.0, reason=reason)
    return DesiredQuotes(
        bid=off,
        ask=DesiredOrder(enabled=False, side="SELL", price=None, size=0.0, reason=reason),
        mode="gated",
        inventory_skew=0.0,
        timestamp_ms=now_ms,
        strategy_state=strategy_state,
    )


def _no_bid() -> DesiredOrder:
    return DesiredOrder(enabled=False, side="BUY", price=None, size=0.0, reason="no_pyramid")


@dataclass
class DirectionalPolicyV2:
    """
    FLAT/LONG state machine for BTC M15.

    Entry: bid at or just below best_ask when fair.p_up - best_ask >= min_entry_edge_prob.
    Exit priority: exit_force → exit_stop_loss → exit_edge_lost → exit_take_profit → hold.
    No shorts, no pyramiding, one position at a time.

    exit_force bypasses freshness gates so we can always exit near window end.
    All other decisions require fresh Binance and Chainlink signals.
    """

    config: RuntimeConfig

    def build(self, state: LocalState, fair: FairValueSnapshot, now_ms: int) -> DesiredQuotes:
        result = self._build_impl(state, fair, now_ms)
        state.desired_quotes = result
        return result

    def _build_impl(self, state: LocalState, fair: FairValueSnapshot, now_ms: int) -> DesiredQuotes:
        state.set_clock(now_ms)
        pos = state.position
        d = self.config.thresholds.directional
        tick = state.market.clob.min_tick_size
        s = "long" if pos.qty > 0 else "flat"

        # Book must be present for any order
        top_bid = state.yes_book.top_bid()
        top_ask = state.yes_book.top_ask()
        if top_bid is None or top_ask is None:
            return _disabled("book_gate", now_ms, s)
        if top_bid.price <= 0.0 or top_ask.price >= 1.0:
            return _disabled("book_gate", now_ms, s)

        # exit_force bypasses freshness gates — we just want out before window closes
        if pos.qty > 0 and fair.tau_s < d.force_exit_tau_s:
            pnl = top_bid.price - pos.avg_cost
            return self._make_exit(
                pos, top_bid, tick, state, now_ms,
                pnl_per_share=pnl,
                reason="exit_force",
                aggressive=True,
            )

        # Freshness gates (config-driven thresholds)
        f = self.config.thresholds.freshness
        if state.last_binance is not None:
            if now_ms - state.last_binance.recv_timestamp_ms > f.binance_max_age_ms.value:
                return _disabled("binance_stale", now_ms, s)
        if state.last_chainlink is not None:
            if now_ms - state.last_chainlink.recv_timestamp_ms > d.chainlink_max_age_ms:
                return _disabled("chainlink_stale", now_ms, s)

        if pos.qty > 0:
            return self._build_long(state, fair, now_ms, tick, d, pos, top_bid, top_ask)

        if fair.tau_s < d.min_tau_to_enter_s:
            return _disabled("tau_gate", now_ms, "flat")

        return self._build_flat(state, fair, now_ms, tick, d, top_bid, top_ask)

    # ---------------------------------------------------------------------- #
    # LONG state: manage exit                                                  #
    # ---------------------------------------------------------------------- #

    def _build_long(
        self, state, fair, now_ms, tick, d, pos, top_bid, top_ask
    ) -> DesiredQuotes:
        pnl = top_bid.price - pos.avg_cost           # realizable PnL at best_bid
        unrealized_edge = fair.p_up - top_bid.price   # remaining expected edge
        is_aggressive = fair.tau_s <= d.aggressive_exit_tau_s

        if pnl < d.stop_loss_prob:
            return self._make_exit(pos, top_bid, tick, state, now_ms, pnl,
                                   reason="exit_stop_loss", aggressive=True)

        if unrealized_edge < d.edge_lost_exit_prob:
            suffix = "_aggressive" if is_aggressive else "_passive"
            return self._make_exit(pos, top_bid, tick, state, now_ms, pnl,
                                   reason=f"exit_edge_lost{suffix}", aggressive=is_aggressive)

        if pnl >= d.take_profit_prob:
            suffix = "_aggressive" if is_aggressive else "_passive"
            return self._make_exit(pos, top_bid, tick, state, now_ms, pnl,
                                   reason=f"exit_take_profit{suffix}", aggressive=is_aggressive)

        return DesiredQuotes(
            bid=_no_bid(),
            ask=DesiredOrder(enabled=False, side="SELL", price=None, size=0.0, reason="hold"),
            mode="long_hold",
            inventory_skew=0.0,
            timestamp_ms=now_ms,
            strategy_state="long",
            pnl_per_share=round(pnl, 6),
        )

    def _make_exit(
        self, pos: PositionState, top_bid, tick, state, now_ms,
        pnl_per_share: float, reason: str, aggressive: bool,
    ) -> DesiredQuotes:
        if aggressive:
            ask_price = top_bid.price
        else:
            ask_price = round(min(top_bid.price + tick, 1.0 - tick), 2)

        exit_size = pos.qty
        min_order = state.market.clob.min_order_size
        ask_enabled = (
            exit_size >= min_order
            and ask_price > 0.0
            and state.inventory.available_up() + 1e-12 >= exit_size
        )

        return DesiredQuotes(
            bid=_no_bid(),
            ask=DesiredOrder(
                enabled=ask_enabled,
                side="SELL",
                price=ask_price if ask_enabled else None,
                size=exit_size if ask_enabled else 0.0,
                reason=reason,
            ),
            mode="long_exit",
            inventory_skew=0.0,
            timestamp_ms=now_ms,
            strategy_state="long",
            pnl_per_share=round(pnl_per_share, 6),
            exit_candidate_reason=reason,
        )

    # ---------------------------------------------------------------------- #
    # FLAT state: consider entry                                               #
    # ---------------------------------------------------------------------- #

    def _build_flat(self, state, fair, now_ms, tick, d, top_bid, top_ask) -> DesiredQuotes:
        entry_edge = fair.p_up - top_ask.price

        if entry_edge < d.min_entry_edge_prob:
            return _disabled("no_edge", now_ms, "flat")

        if entry_edge >= d.aggressive_entry_edge_prob:
            bid_price = top_ask.price
            entry_mode = "enter_long_aggressive"
        else:
            bid_price = round(max(top_ask.price - tick, tick), 2)
            entry_mode = "enter_long_passive"

        bid_size = math.floor(
            (d.entry_notional_usd / max(bid_price, tick)) * 1000.0
        ) / 1000.0
        max_size = math.floor(
            (d.max_position_notional_usd / max(bid_price, tick)) * 1000.0
        ) / 1000.0
        bid_size = min(bid_size, max_size)

        min_order = state.market.clob.min_order_size
        if bid_size < min_order:
            return _disabled("size_too_small", now_ms, "flat")

        bid_enabled = (
            bid_size > 0
            and state.inventory.available_pusd() + 1e-12 >= bid_price * bid_size
        )

        return DesiredQuotes(
            bid=DesiredOrder(
                enabled=bid_enabled,
                side="BUY",
                price=bid_price if bid_enabled else None,
                size=bid_size if bid_enabled else 0.0,
                reason=entry_mode if bid_enabled else "no_capital",
            ),
            ask=DesiredOrder(enabled=False, side="SELL", price=None, size=0.0, reason="flat"),
            mode="flat_entry",
            inventory_skew=0.0,
            timestamp_ms=now_ms,
            strategy_state="flat",
            entry_edge=round(entry_edge, 6),
        )
