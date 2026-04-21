"""
Attribution report tests — 10 tests.

Tests 1-2 : leg1_standalone vs hedge_incremental attribution
Tests 3-4 : early / baseline bucket splits
Tests 5-6 : hedged / non_hedged bucket splits
Tests 7-8 : up / down entry side splits
Tests 9-10: flat vs compound bankroll simulation + max drawdown
"""
from __future__ import annotations

import math
import pytest

from bot.m5_attribution import (
    BucketStats,
    compute_attribution,
    export_attribution,
)
from bot.m5_summary import TradeRecord


# ---------------------------------------------------------------------------
# Minimal TradeRecord builder
# ---------------------------------------------------------------------------

def _trade(
    *,
    window_ts: int = 1_746_000_000,
    entry_mode: str | None = "baseline",
    entry_side: str | None = "up",
    entry_price: float | None = 0.60,
    hedged: bool = False,
    hedge_price: float | None = None,
    hedge_side: str | None = None,
    result: str | None = "up",
    pnl_leg1: float | None = None,
    pnl_hedge: float | None = None,
    net_pnl: float | None = None,
) -> TradeRecord:
    t = TradeRecord(window_ts=window_ts)
    t.entry_mode = entry_mode
    t.entry_side = entry_side
    t.entry_price = entry_price
    t.hedged = hedged
    t.hedge_price = hedge_price
    t.hedge_side = hedge_side
    t.result = result
    # Default PnL for a simple UP win at price 0.60 if not overridden
    if pnl_leg1 is None and entry_side is not None and result is not None:
        if result == entry_side:
            p = entry_price or 0.60
            pnl_leg1 = round((1 - p) / p, 6)
        else:
            pnl_leg1 = -1.0
    if pnl_hedge is None:
        pnl_hedge = 0.0
    if net_pnl is None and pnl_leg1 is not None:
        net_pnl = round((pnl_leg1 or 0.0) + (pnl_hedge or 0.0), 6)
    t.pnl_leg1 = pnl_leg1
    t.pnl_hedge = pnl_hedge
    t.net_pnl = net_pnl
    return t


# ---------------------------------------------------------------------------
# Tests 1-2: attribution split
# ---------------------------------------------------------------------------

def test_attribution_leg1_standalone_no_hedge():
    """All trades are unhedged → hedge_incremental = 0."""
    trades = [
        _trade(entry_side="up", result="up", pnl_leg1=0.5, pnl_hedge=0.0, net_pnl=0.5),
        _trade(entry_side="down", result="down", pnl_leg1=0.3, pnl_hedge=0.0, net_pnl=0.3),
    ]
    r = compute_attribution(trades)
    assert r.leg1_standalone_net_total == pytest.approx(0.8, rel=1e-6)
    assert r.hedge_incremental_total == pytest.approx(0.0)
    assert r.net_pnl_total == pytest.approx(0.8, rel=1e-6)


def test_attribution_hedge_incremental_separate_from_leg1():
    """LEG1 loses, HEDGE wins — attribution splits correctly."""
    # leg1 UP loses → pnl_leg1 = -1.0
    # hedge DOWN wins → pnl_hedge = (1 - 0.41) * (2 / 0.41) ≈ 2.878
    pnl_h = (1.0 - 0.41) * (2.0 / 0.41)
    trades = [
        _trade(
            entry_side="up", result="down",
            pnl_leg1=-1.0, pnl_hedge=pnl_h, net_pnl=-1.0 + pnl_h,
            hedged=True, hedge_side="down", hedge_price=0.41,
        )
    ]
    r = compute_attribution(trades)
    assert r.leg1_standalone_net_total == pytest.approx(-1.0)
    assert r.hedge_incremental_total == pytest.approx(pnl_h, rel=1e-6)
    assert r.net_pnl_total == pytest.approx(-1.0 + pnl_h, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests 3-4: early / baseline bucket splits
# ---------------------------------------------------------------------------

def test_early_baseline_counts_and_pnl():
    """early and baseline buckets sum to total entered."""
    trades = [
        _trade(entry_mode="early",    entry_side="up",   result="up",   pnl_leg1=0.5, net_pnl=0.5),
        _trade(entry_mode="early",    entry_side="down", result="down", pnl_leg1=0.4, net_pnl=0.4),
        _trade(entry_mode="baseline", entry_side="up",   result="down", pnl_leg1=-1.0, net_pnl=-1.0),
    ]
    r = compute_attribution(trades)
    assert r.early.count == 2
    assert r.baseline.count == 1
    assert r.early.pnl_leg1_total == pytest.approx(0.9, rel=1e-6)
    assert r.baseline.pnl_leg1_total == pytest.approx(-1.0)
    assert r.early.win_rate == pytest.approx(1.0)
    assert r.baseline.win_rate == pytest.approx(0.0)


def test_bucket_avg_entry_price():
    """avg_entry_price is mean of entry_price within bucket."""
    trades = [
        _trade(entry_mode="early", entry_side="up", entry_price=0.60, result="up", pnl_leg1=0.5, net_pnl=0.5),
        _trade(entry_mode="early", entry_side="up", entry_price=0.50, result="up", pnl_leg1=1.0, net_pnl=1.0),
    ]
    r = compute_attribution(trades)
    assert r.early.avg_entry_price == pytest.approx(0.55, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests 5-6: hedged / non_hedged splits
# ---------------------------------------------------------------------------

def test_hedged_non_hedged_counts():
    """hedged and non_hedged bucket counts are mutually exclusive and exhaustive."""
    trades = [
        _trade(hedged=True,  pnl_leg1=-1.0, pnl_hedge=2.0, net_pnl=1.0),
        _trade(hedged=False, pnl_leg1=0.5,  pnl_hedge=0.0, net_pnl=0.5),
        _trade(hedged=False, pnl_leg1=0.3,  pnl_hedge=0.0, net_pnl=0.3),
    ]
    r = compute_attribution(trades)
    assert r.hedged.count == 1
    assert r.non_hedged.count == 2
    assert r.hedged.hedge_triggered_count == 1
    assert r.non_hedged.hedge_triggered_count == 0
    assert r.hedged.net_pnl_total == pytest.approx(1.0)
    assert r.non_hedged.net_pnl_total == pytest.approx(0.8, rel=1e-6)


def test_hedged_win_rate():
    """win_rate in hedged bucket counts result == entry_side."""
    trades = [
        _trade(hedged=True, entry_side="up", result="up",   pnl_leg1=0.5, net_pnl=0.5),
        _trade(hedged=True, entry_side="up", result="down", pnl_leg1=-1.0, pnl_hedge=2.0, net_pnl=1.0),
    ]
    r = compute_attribution(trades)
    # 1 win out of 2 hedged trades (result=="up" for first only)
    assert r.hedged.win_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests 7-8: up / down entry side splits
# ---------------------------------------------------------------------------

def test_up_down_splits_net_pnl():
    """up and down buckets carry independent PnL totals."""
    trades = [
        _trade(entry_side="up",   result="up",   pnl_leg1=0.6, net_pnl=0.6),
        _trade(entry_side="up",   result="down", pnl_leg1=-1.0, net_pnl=-1.0),
        _trade(entry_side="down", result="down", pnl_leg1=0.4, net_pnl=0.4),
    ]
    r = compute_attribution(trades)
    assert r.up.count == 2
    assert r.down.count == 1
    assert r.up.net_pnl_total == pytest.approx(-0.4, rel=1e-6)
    assert r.down.net_pnl_total == pytest.approx(0.4, rel=1e-6)


def test_up_down_win_rates_independent():
    """win_rate is computed separately for up and down buckets."""
    trades = [
        _trade(entry_side="up",   result="up",   pnl_leg1=0.5, net_pnl=0.5),   # up win
        _trade(entry_side="up",   result="down", pnl_leg1=-1.0, net_pnl=-1.0), # up loss
        _trade(entry_side="down", result="down", pnl_leg1=0.5, net_pnl=0.5),   # down win
        _trade(entry_side="down", result="down", pnl_leg1=0.5, net_pnl=0.5),   # down win
    ]
    r = compute_attribution(trades)
    assert r.up.win_rate == pytest.approx(0.5)
    assert r.down.win_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests 9-10: bankroll simulation
# ---------------------------------------------------------------------------

def test_flat_bankroll_accumulates_linearly():
    """Flat bankroll = initial + sum(net_pnl), independent of order."""
    trades = [
        _trade(pnl_leg1=0.5, net_pnl=0.5),
        _trade(pnl_leg1=-1.0, net_pnl=-1.0),
        _trade(pnl_leg1=0.3, net_pnl=0.3),
    ]
    r = compute_attribution(trades, initial_bankroll=100.0)
    expected_flat = 100.0 + 0.5 - 1.0 + 0.3
    assert r.bankroll.flat_final == pytest.approx(expected_flat, rel=1e-6)
    assert r.bankroll.flat_return_pct == pytest.approx((expected_flat - 100.0) / 100.0 * 100.0, rel=1e-6)


def test_compound_bankroll_grows_faster_on_positive_run():
    """Compound grows faster than flat when all trades are winners."""
    trades = [_trade(pnl_leg1=0.5, net_pnl=0.5) for _ in range(4)]
    r = compute_attribution(trades, initial_bankroll=100.0)
    assert r.bankroll.compound_final > r.bankroll.flat_final


def test_max_drawdown_flat_detected():
    """Max drawdown is peak-to-trough over the curve."""
    # +2 then -3 → peak=102, trough=99 → dd=3
    trades = [
        _trade(pnl_leg1=2.0, net_pnl=2.0),
        _trade(pnl_leg1=-3.0, net_pnl=-3.0),
        _trade(pnl_leg1=1.0, net_pnl=1.0),
    ]
    r = compute_attribution(trades, initial_bankroll=100.0)
    assert r.bankroll.max_drawdown_flat == pytest.approx(3.0, rel=1e-6)


def test_per_window_rows_count_matches_trades():
    """rows list has one entry per trade (including non-entered)."""
    aborted = TradeRecord(window_ts=1_746_000_000)  # no entry, no pnl
    aborted.abort_reason = "ptb_unavailable"
    trades = [
        aborted,
        _trade(pnl_leg1=0.5, net_pnl=0.5),
        _trade(pnl_leg1=-1.0, net_pnl=-1.0),
    ]
    r = compute_attribution(trades)
    assert len(r.rows) == 3
    # Aborted window: bankroll unchanged
    assert r.rows[0].flat_bankroll_after == pytest.approx(100.0)
    assert r.rows[0].net_pnl is None
    # After win: bankroll = 100.5
    assert r.rows[1].flat_bankroll_after == pytest.approx(100.5, rel=1e-6)


def test_csv_export_row_count(tmp_path):
    """CSV export writes header + one row per trade."""
    trades = [
        _trade(pnl_leg1=0.5, net_pnl=0.5),
        _trade(entry_side=None, net_pnl=None),  # non-entered
    ]
    # Fix: non-entered trade has no entry_side and no net_pnl
    trades[1].pnl_leg1 = None
    r = compute_attribution(trades)
    out = tmp_path / "attr.csv"
    export_attribution(r, out)
    lines = out.read_text().splitlines()
    assert len(lines) == 3  # header + 2 rows
