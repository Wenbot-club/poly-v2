"""
Pure BTC M5 strategy logic — no I/O.

Implements:
  - 8-signal consensus for EARLY entry (140-170s)
  - baseline direction at 170s
  - single hedge trigger (Binance vs PTB)
  - settlement P&L calculation (win = (1-entry)*shares, loss = -usd_staked)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------

@dataclass
class ConsensusResult:
    score: float            # 0-100; >50 = up-leaning
    up_votes: int
    down_votes: int
    non_neutral: int
    direction: Optional[str]    # "up" | "down" | None
    reason: Optional[str]       # None when direction is set


def compute_consensus(
    btc: float,
    ptb: float,
    chainlink: Optional[float],
    btc_10s: Optional[float],
    btc_30s: Optional[float],
    btc_60s: Optional[float],
    price_up: Optional[float],
    *,
    threshold: float = 88.0,
    min_non_neutral: int = 3,
) -> ConsensusResult:
    """
    8-signal binary consensus.

    Signals (each returns +1=up, -1=down, 0=neutral):
      1. btc vs ptb
      2. chainlink vs ptb
      3. btc vs btc_10s_ago
      4. btc vs btc_30s_ago
      5. btc vs btc_60s_ago
      6. price_up token: >0.55=+1, <0.45=-1, else 0
      7. btc - ptb > 5 → +1, < -5 → -1
      8. btc - ptb > 20 → +1, < -20 → -1

    score = 100 * up_votes / non_neutral_votes
    Requires min_non_neutral votes; otherwise direction=None.
    threshold=88 → ≥88 = UP, ≤12 = DOWN, else None.
    """
    gap = btc - ptb

    def _cmp(a: Optional[float], b: Optional[float]) -> int:
        if a is None or b is None:
            return 0
        if a > b:
            return 1
        if a < b:
            return -1
        return 0

    votes = [
        _cmp(btc, ptb),
        _cmp(chainlink, ptb),
        _cmp(btc, btc_10s),
        _cmp(btc, btc_30s),
        _cmp(btc, btc_60s),
        (1 if price_up is not None and price_up > 0.55
         else -1 if price_up is not None and price_up < 0.45
         else 0),
        (1 if gap > 5 else -1 if gap < -5 else 0),
        (1 if gap > 20 else -1 if gap < -20 else 0),
    ]

    up_votes = sum(1 for v in votes if v == 1)
    down_votes = sum(1 for v in votes if v == -1)
    non_neutral = up_votes + down_votes

    if non_neutral < min_non_neutral:
        return ConsensusResult(
            score=50.0, up_votes=up_votes, down_votes=down_votes,
            non_neutral=non_neutral, direction=None,
            reason="insufficient_non_neutral",
        )

    score = 100.0 * up_votes / non_neutral

    if score >= threshold:
        direction, reason = "up", None
    elif score <= (100.0 - threshold):
        direction, reason = "down", None
    else:
        direction, reason = None, "score_inconclusive"

    return ConsensusResult(
        score=score, up_votes=up_votes, down_votes=down_votes,
        non_neutral=non_neutral, direction=direction, reason=reason,
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
