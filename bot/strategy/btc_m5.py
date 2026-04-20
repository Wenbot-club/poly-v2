"""
Pure BTC M5 strategy logic — no I/O.

Implements:
  - Probabilistic PTB-close model for EARLY entry (entry_scan_start_s – entry_scan_end_s)
  - estimate_sigma_to_close: rolling vol from BtcHistory samples, scaled by sqrt(tau_s)
  - compute_entry_signal: sigmoid(z_gap + momentum), edge vs Polymarket price
  - baseline direction at entry_scan_end_s
  - single hedge trigger (Binance vs PTB)
  - settlement P&L calculation (win = (1-entry)*shares, loss = -usd_staked)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Volatility estimation
# ---------------------------------------------------------------------------

def estimate_sigma_to_close(
    samples: list[tuple[int, float]],
    tau_s: float,
    sigma_floor_usd: float = 5.0,
) -> float:
    """
    Estimate remaining price vol (USD) from now to window close.

    Computes mean(dp² / dt_s) over the provided (ts_ms, price) samples —
    an estimate of variance per second — then scales by sqrt(tau_s).

    Returns max(estimate, sigma_floor_usd).
    """
    if len(samples) < 3:
        return sigma_floor_usd

    contributions: list[float] = []
    for i in range(1, len(samples)):
        dt_s = (samples[i][0] - samples[i - 1][0]) / 1000.0
        if dt_s <= 0:
            continue
        dp = samples[i][1] - samples[i - 1][1]
        contributions.append(dp * dp / dt_s)

    if not contributions:
        return sigma_floor_usd

    var_per_s = sum(contributions) / len(contributions)
    sigma_to_close = math.sqrt(var_per_s * max(tau_s, 0.0))
    return max(sigma_to_close, sigma_floor_usd)


# ---------------------------------------------------------------------------
# Probabilistic entry signal
# ---------------------------------------------------------------------------

@dataclass
class EntrySignal:
    direction: Optional[str]        # "up" | "down" | None
    p_model_up: float               # 0–1
    z_gap: float
    sigma_to_close: float
    edge_up: float                  # p_model_up - price_up
    edge_down: float                # (1 - p_model_up) - price_down
    block_reason: Optional[str]     # None when direction is set


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_entry_signal(
    btc: float,
    ptb: float,
    tau_s: float,
    btc_samples: list[tuple[int, float]],
    btc_10s: Optional[float],
    btc_30s: Optional[float],
    price_up: Optional[float],
    price_down: Optional[float],
    *,
    sigma_floor_usd: float = 5.0,
    z_gap_min: float = 0.35,
    p_enter_up_min: float = 0.60,
    p_enter_down_max: float = 0.40,
    min_entry_edge: float = 0.06,
) -> EntrySignal:
    """
    Probabilistic PTB-close model.

    score_raw = 1.35 * z_gap + 0.35 * mom10_norm + 0.20 * mom30_norm
    p_model_up = sigmoid(score_raw)

    Chainlink is not used here — it remains a PTB fallback only.

    Entry gates (in order):
      1. abs(z_gap) < z_gap_min              → block_reason="noise_zone"
      2. p_model_up >= p_enter_up_min
           edge_up >= min_entry_edge         → direction="up"
           else                             → block_reason="edge_not_enough"
      3. p_model_up <= p_enter_down_max
           edge_down >= min_entry_edge       → direction="down"
           else                             → block_reason="edge_not_enough"
      4. otherwise                          → block_reason="probability_not_strong_enough"
    """
    sigma = estimate_sigma_to_close(btc_samples, tau_s, sigma_floor_usd)
    gap = btc - ptb
    z_gap = gap / sigma

    mom10_norm = (btc - btc_10s) / sigma if btc_10s is not None else 0.0
    mom30_norm = (btc - btc_30s) / sigma if btc_30s is not None else 0.0

    score_raw = 1.35 * z_gap + 0.35 * mom10_norm + 0.20 * mom30_norm
    p_model_up = _sigmoid(score_raw)

    _price_up = price_up if price_up is not None else 0.50
    _price_down = price_down if price_down is not None else 0.50
    edge_up = p_model_up - _price_up
    edge_down = (1.0 - p_model_up) - _price_down

    if abs(z_gap) < z_gap_min:
        return EntrySignal(
            direction=None, p_model_up=p_model_up,
            z_gap=z_gap, sigma_to_close=sigma,
            edge_up=edge_up, edge_down=edge_down,
            block_reason="noise_zone",
        )

    if p_model_up >= p_enter_up_min:
        if edge_up >= min_entry_edge:
            return EntrySignal(
                direction="up", p_model_up=p_model_up,
                z_gap=z_gap, sigma_to_close=sigma,
                edge_up=edge_up, edge_down=edge_down,
                block_reason=None,
            )
        return EntrySignal(
            direction=None, p_model_up=p_model_up,
            z_gap=z_gap, sigma_to_close=sigma,
            edge_up=edge_up, edge_down=edge_down,
            block_reason="edge_not_enough",
        )

    if p_model_up <= p_enter_down_max:
        if edge_down >= min_entry_edge:
            return EntrySignal(
                direction="down", p_model_up=p_model_up,
                z_gap=z_gap, sigma_to_close=sigma,
                edge_up=edge_up, edge_down=edge_down,
                block_reason=None,
            )
        return EntrySignal(
            direction=None, p_model_up=p_model_up,
            z_gap=z_gap, sigma_to_close=sigma,
            edge_up=edge_up, edge_down=edge_down,
            block_reason="edge_not_enough",
        )

    return EntrySignal(
        direction=None, p_model_up=p_model_up,
        z_gap=z_gap, sigma_to_close=sigma,
        edge_up=edge_up, edge_down=edge_down,
        block_reason="probability_not_strong_enough",
    )


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def baseline_direction(btc: float, ptb: float) -> Optional[str]:
    """'up' if btc > ptb, 'down' if btc < ptb, None if equal."""
    if btc > ptb:
        return "up"
    if btc < ptb:
        return "down"
    return None


# ---------------------------------------------------------------------------
# Hedge trigger
# ---------------------------------------------------------------------------

def should_hedge(leg1_side: str, btc: float, ptb: float, threshold: float = 1.0) -> bool:
    """True when Binance price has crossed PTB by threshold against LEG1 direction."""
    if leg1_side == "up":
        return btc < ptb - threshold
    return btc > ptb + threshold


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

@dataclass
class SettlementResult:
    result: str         # "up" | "down"
    pnl_leg1: float
    pnl_hedge: float
    net_pnl: float


def compute_settlement(
    close_price: float,
    open_price: float,
    leg1_side: str,
    leg1_entry_price: float,
    leg1_shares: float,
    leg1_usd_staked: float,
    hedge_side: Optional[str] = None,
    hedge_entry_price: Optional[float] = None,
    hedge_shares: Optional[float] = None,
    hedge_usd_staked: Optional[float] = None,
) -> SettlementResult:
    """
    result = 'up' if close >= open, else 'down'.
    win pnl  = (1 - entry_price) * shares
    loss pnl = -usd_staked
    """
    result = "up" if close_price >= open_price else "down"

    pnl_leg1 = (
        (1.0 - leg1_entry_price) * leg1_shares
        if result == leg1_side
        else -leg1_usd_staked
    )

    pnl_hedge = 0.0
    if (hedge_side is not None and hedge_entry_price is not None
            and hedge_shares is not None and hedge_usd_staked is not None):
        pnl_hedge = (
            (1.0 - hedge_entry_price) * hedge_shares
            if result == hedge_side
            else -hedge_usd_staked
        )

    return SettlementResult(
        result=result,
        pnl_leg1=round(pnl_leg1, 8),
        pnl_hedge=round(pnl_hedge, 8),
        net_pnl=round(pnl_leg1 + pnl_hedge, 8),
    )
