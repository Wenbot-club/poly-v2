"""
Per-trade and campaign-level summary structures for BTC M5.

TradeRecord  — one entry per window, populated by M5Session.
M5CampaignSummary — aggregated across all windows, produced by aggregate_trades().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TradeRecord:
    window_ts: int

    # PTB
    ptb: Optional[float] = None
    ptb_source: Optional[str] = None           # "ssr" | "api" | "chainlink"

    # LEG1
    entry_mode: Optional[str] = None           # "early" | "baseline"
    entry_side: Optional[str] = None           # "up" | "down"
    entry_elapsed_s: Optional[float] = None
    entry_price: Optional[float] = None
    entry_shares: Optional[float] = None

    # Probabilistic model observation at entry (EARLY only)
    p_model_up_at_entry: Optional[float] = None
    edge_up_at_entry: Optional[float] = None
    edge_down_at_entry: Optional[float] = None
    z_gap_at_entry: Optional[float] = None
    sigma_to_close_at_entry: Optional[float] = None

    # HEDGE
    hedged: bool = False
    hedge_elapsed_s: Optional[float] = None
    hedge_side: Optional[str] = None           # "up" | "down"
    hedge_price: Optional[float] = None
    hedge_shares: Optional[float] = None
    hedge_trigger_btc: Optional[float] = None

    # Execution guards
    hedge_blocked_by_cutoff: bool = False
    price_insane_block_count: int = 0

    # Paper execution trace — LEG1 (separate from HEDGE)
    leg1_observed_ask: Optional[float] = None
    leg1_attempted_price: Optional[float] = None
    leg1_slippage: Optional[float] = None      # attempted - observed
    leg1_fill_retries: int = 0

    # Paper execution trace — HEDGE
    hedge_observed_ask: Optional[float] = None
    hedge_attempted_price: Optional[float] = None
    hedge_slippage: Optional[float] = None
    hedge_fill_retries: int = 0

    # Settlement
    result: Optional[str] = None               # "up" | "down"
    pnl_leg1: Optional[float] = None
    pnl_hedge: Optional[float] = None
    net_pnl: Optional[float] = None

    # Abort / block
    abort_reason: Optional[str] = None         # "ptb_unavailable" | "tokens_unavailable" | ...
    entry_block_reason: Optional[str] = None   # "noise_zone" | "edge_not_enough" | "probability_not_strong_enough" | ...


@dataclass
class M5CampaignSummary:
    windows_seen: int
    ptb_fail_count: int
    token_setup_fail_count: int
    leg1_entered_count: int
    early_entry_count: int
    baseline_entry_count: int
    hedge_triggered_count: int
    hedge_blocked_by_cutoff_count: int
    price_insane_block_count: int
    pnl_leg1_total: float
    pnl_hedge_total: float
    net_pnl_total: float
    avg_leg1_entry_price: Optional[float]
    avg_hedge_entry_price: Optional[float]
    avg_leg1_slippage: Optional[float]
    avg_hedge_slippage: Optional[float]
    # Probabilistic model diagnostics
    avg_p_model_up_at_entry: Optional[float] = None
    avg_sigma_to_close_at_entry: Optional[float] = None
    blocked_by_noise_zone_count: int = 0
    blocked_by_edge_count: int = 0
    blocked_by_probability_count: int = 0
    trades: list = field(default_factory=list)  # list[TradeRecord]


def aggregate_trades(trades: list) -> M5CampaignSummary:
    """Aggregate a list of TradeRecord into an M5CampaignSummary."""
    entered = [t for t in trades if t.entry_side is not None]
    hedged = [t for t in trades if t.hedged]

    pnl_leg1 = sum(t.pnl_leg1 for t in trades if t.pnl_leg1 is not None)
    pnl_hedge = sum(t.pnl_hedge for t in trades if t.pnl_hedge is not None)

    leg1_prices = [t.entry_price for t in entered if t.entry_price is not None]
    hedge_prices = [t.hedge_price for t in hedged if t.hedge_price is not None]
    leg1_slippages = [t.leg1_slippage for t in entered if t.leg1_slippage is not None]
    hedge_slippages = [t.hedge_slippage for t in hedged if t.hedge_slippage is not None]

    early_entered = [t for t in entered if t.entry_mode == "early"]
    p_vals = [t.p_model_up_at_entry for t in early_entered if t.p_model_up_at_entry is not None]
    sigma_vals = [t.sigma_to_close_at_entry for t in early_entered if t.sigma_to_close_at_entry is not None]

    return M5CampaignSummary(
        windows_seen=len(trades),
        ptb_fail_count=sum(1 for t in trades if t.abort_reason == "ptb_unavailable"),
        token_setup_fail_count=sum(1 for t in trades if t.abort_reason == "tokens_unavailable"),
        leg1_entered_count=len(entered),
        early_entry_count=len(early_entered),
        baseline_entry_count=sum(1 for t in entered if t.entry_mode == "baseline"),
        hedge_triggered_count=len(hedged),
        hedge_blocked_by_cutoff_count=sum(1 for t in trades if t.hedge_blocked_by_cutoff),
        price_insane_block_count=sum(t.price_insane_block_count for t in trades),
        pnl_leg1_total=round(pnl_leg1, 6),
        pnl_hedge_total=round(pnl_hedge, 6),
        net_pnl_total=round(pnl_leg1 + pnl_hedge, 6),
        avg_leg1_entry_price=(
            sum(leg1_prices) / len(leg1_prices) if leg1_prices else None
        ),
        avg_hedge_entry_price=(
            sum(hedge_prices) / len(hedge_prices) if hedge_prices else None
        ),
        avg_leg1_slippage=(
            sum(leg1_slippages) / len(leg1_slippages) if leg1_slippages else None
        ),
        avg_hedge_slippage=(
            sum(hedge_slippages) / len(hedge_slippages) if hedge_slippages else None
        ),
        avg_p_model_up_at_entry=(
            sum(p_vals) / len(p_vals) if p_vals else None
        ),
        avg_sigma_to_close_at_entry=(
            sum(sigma_vals) / len(sigma_vals) if sigma_vals else None
        ),
        blocked_by_noise_zone_count=sum(
            1 for t in trades if t.entry_block_reason == "noise_zone"
        ),
        blocked_by_edge_count=sum(
            1 for t in trades if t.entry_block_reason == "edge_not_enough"
        ),
        blocked_by_probability_count=sum(
            1 for t in trades if t.entry_block_reason == "probability_not_strong_enough"
        ),
        trades=trades,
    )
