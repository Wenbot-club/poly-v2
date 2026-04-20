"""
Backtest engine tests — 9 tests.

Tests 1-2 : Simple settlement (UP win, DOWN win)
Tests 3   : LEG1 loses / HEDGE wins — PnL attribution
Tests 4-5 : Position limits (max 1 LEG1, max 1 HEDGE)
Test  6   : baseline_ptb takes correct direction at 170s
Tests 7-8 : current_m5 plugs into engine; replays EARLY and BASELINE
Test  9   : aggregate_results cumulative totals
"""
from __future__ import annotations

import pytest

from offline.data_types import TickData, WindowData
from offline.engine import BacktestEngine
from offline.reporting import aggregate_results
from offline.strategies.baseline_ptb import BaselinePtbStrategy
from offline.strategies.current_m5 import CurrentM5Strategy
from offline.strategy_interface import Action, M5Strategy, TickState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _window(
    result: str,
    ptb_api: float = 84_000.0,
    close_price: float | None = None,
    ticks: list[TickData] | None = None,
) -> WindowData:
    """Build a minimal WindowData for testing."""
    if close_price is None:
        close_price = ptb_api + (100.0 if result == "up" else -100.0)
    return WindowData(
        window_ts=1_746_000_000,
        ptb_api=ptb_api,
        close_price=close_price,
        result=result,
        ticks=ticks or [],
    )


def _tick(sec: int, binance: float = 84_010.0,
          up_ask: float = 0.60, dn_ask: float = 0.42) -> TickData:
    return TickData(sec=sec, binance=binance,
                    price_up_ask=up_ask, price_down_ask=dn_ask)


class _FixedAction(M5Strategy):
    """Strategy that fires a single action at a given second, then NOOPs."""
    def __init__(self, action: Action, at_sec: int) -> None:
        self._action = action
        self._at_sec = at_sec
        self._fired = False

    def reset(self) -> None:
        self._fired = False

    def on_tick(self, state: TickState) -> Action:
        if not self._fired and state.sec == self._at_sec:
            self._fired = True
            return self._action
        return Action.NOOP


class _SequenceStrategy(M5Strategy):
    """Fires (sec, action) pairs in order."""
    def __init__(self, seq: list[tuple[int, Action]]) -> None:
        self._seq = list(seq)
        self._idx = 0

    def reset(self) -> None:
        self._idx = 0

    def on_tick(self, state: TickState) -> Action:
        if self._idx < len(self._seq) and state.sec == self._seq[self._idx][0]:
            _, action = self._seq[self._idx]
            self._idx += 1
            return action
        return Action.NOOP


ENGINE = BacktestEngine(leg1_usd_stake=1.0, hedge_usd_stake=2.0)


# ---------------------------------------------------------------------------
# Tests 1-2 : Simple settlement
# ---------------------------------------------------------------------------

def test_up_win_no_hedge_pnl():
    """Entry UP at 0.60, result UP → pnl_leg1 = (1 - 0.60) * (1/0.60)."""
    ticks = [_tick(170, binance=84_100.0, up_ask=0.60)]
    w = _window("up", ptb_api=84_000.0, close_price=84_200.0, ticks=ticks)
    r = ENGINE.run_window(w, _FixedAction(Action.BUY_UP_LEG1, at_sec=170))
    assert r.entry_taken
    assert r.entry_side == "up"
    assert r.result == "up"
    assert r.pnl_leg1 == pytest.approx((1.0 - 0.60) * (1.0 / 0.60), rel=1e-6)
    assert not r.hedged
    assert r.pnl_hedge == 0.0
    assert r.net_pnl == pytest.approx(r.pnl_leg1)


def test_down_win_no_hedge_pnl():
    """Entry DOWN at 0.42, result DOWN → pnl_leg1 = (1 - 0.42) * (1/0.42)."""
    ticks = [_tick(170, binance=83_900.0, dn_ask=0.42)]
    w = _window("down", ptb_api=84_000.0, close_price=83_800.0, ticks=ticks)
    r = ENGINE.run_window(w, _FixedAction(Action.BUY_DOWN_LEG1, at_sec=170))
    assert r.entry_side == "down"
    assert r.result == "down"
    assert r.pnl_leg1 == pytest.approx((1.0 - 0.42) * (1.0 / 0.42), rel=1e-6)
    assert r.net_pnl == pytest.approx(r.pnl_leg1)


# ---------------------------------------------------------------------------
# Test 3 : LEG1 loses / HEDGE wins — attribution
# ---------------------------------------------------------------------------

def test_leg1_loss_hedge_win_attribution():
    """LEG1=UP loses, HEDGE=DOWN wins. Check independent attribution."""
    ticks = [
        _tick(170, binance=84_100.0, up_ask=0.60, dn_ask=0.42),  # leg1 UP
        _tick(200, binance=82_900.0, up_ask=0.61, dn_ask=0.41),  # hedge DOWN
    ]
    w = _window("down", ptb_api=84_000.0, close_price=83_800.0, ticks=ticks)
    seq = [
        (170, Action.BUY_UP_LEG1),
        (200, Action.BUY_DOWN_HEDGE),
    ]
    r = ENGINE.run_window(w, _SequenceStrategy(seq))

    assert r.entry_side == "up"
    assert r.hedge_side == "down"
    assert r.result == "down"
    assert r.pnl_leg1 == pytest.approx(-1.0)                       # loss
    assert r.pnl_hedge == pytest.approx((1.0 - 0.41) * (2.0 / 0.41), rel=1e-6)  # win
    assert r.net_pnl == pytest.approx(r.pnl_leg1 + r.pnl_hedge, rel=1e-6)


# ---------------------------------------------------------------------------
# Tests 4-5 : Position limits
# ---------------------------------------------------------------------------

def test_max_one_leg1_enforced():
    """Engine ignores second BUY_UP_LEG1 if entry already taken."""
    ticks = [
        _tick(170, up_ask=0.60),
        _tick(171, up_ask=0.61),  # second attempt at different price
    ]
    w = _window("up", ticks=ticks)
    seq = [(170, Action.BUY_UP_LEG1), (171, Action.BUY_UP_LEG1)]
    r = ENGINE.run_window(w, _SequenceStrategy(seq))
    assert r.entry_price == pytest.approx(0.60)   # only first fill counts
    assert r.entry_sec == 170


def test_max_one_hedge_enforced():
    """Engine ignores second BUY_DOWN_HEDGE if hedge already taken."""
    ticks = [
        _tick(170, up_ask=0.60),
        _tick(180, dn_ask=0.42),  # first hedge
        _tick(181, dn_ask=0.38),  # second attempt at better price
    ]
    w = _window("up", ticks=ticks)
    seq = [
        (170, Action.BUY_UP_LEG1),
        (180, Action.BUY_DOWN_HEDGE),
        (181, Action.BUY_DOWN_HEDGE),
    ]
    r = ENGINE.run_window(w, _SequenceStrategy(seq))
    assert r.hedge_price == pytest.approx(0.42)   # only first fill counts
    assert r.hedge_sec == 180


# ---------------------------------------------------------------------------
# Test 6 : baseline_ptb direction
# ---------------------------------------------------------------------------

def test_baseline_ptb_buys_up_when_btc_above_ptb():
    """BaselinePtbStrategy fires BUY_UP_LEG1 at 170s when btc > ptb."""
    ptb = 84_000.0
    ticks = [
        _tick(sec=169, binance=84_100.0),   # before entry — should NOOP
        _tick(sec=170, binance=84_100.0, up_ask=0.58),
    ]
    w = _window("up", ptb_api=ptb, ticks=ticks)
    r = ENGINE.run_window(w, BaselinePtbStrategy())
    assert r.entry_taken
    assert r.entry_side == "up"
    assert r.entry_sec == 170
    assert r.entry_price == pytest.approx(0.58)


def test_baseline_ptb_buys_down_when_btc_below_ptb():
    """BaselinePtbStrategy fires BUY_DOWN_LEG1 at 170s when btc < ptb."""
    ptb = 84_000.0
    ticks = [_tick(sec=170, binance=83_900.0, dn_ask=0.44)]
    w = _window("down", ptb_api=ptb, ticks=ticks)
    r = ENGINE.run_window(w, BaselinePtbStrategy())
    assert r.entry_side == "down"
    assert r.entry_price == pytest.approx(0.44)


# ---------------------------------------------------------------------------
# Tests 7-8 : current_m5 plugs into engine
# ---------------------------------------------------------------------------

def test_current_m5_enters_at_baseline_when_no_early_signal():
    """CurrentM5Strategy falls back to baseline at 170s when z_gap is in noise zone."""
    ptb = 84_000.0
    # BTC == PTB → z_gap = 0 < z_gap_min (0.35) → noise_zone → no early entry
    # At 170s baseline_direction: btc == ptb, so direction is None → no entry;
    # offset btc by 1 USD just above ptb so baseline fires UP.
    ticks = [_tick(sec=s, binance=84_001.0) for s in range(300)]
    w = _window("up", ptb_api=ptb, ticks=ticks)
    r = ENGINE.run_window(w, CurrentM5Strategy())
    assert r.entry_taken
    assert r.entry_sec == 170
    assert r.entry_mode == "baseline"
    assert r.entry_side == "up"


def test_current_m5_enters_early_with_strong_signal():
    """CurrentM5Strategy takes EARLY entry when z_gap >> z_gap_min."""
    ptb = 84_000.0
    # Build 60s of flat history at ptb, then spike BTC by $100 at sec=145
    # sigma_floor = 5 → z_gap = 100/5 = 20 >> 0.35 → early entry
    ticks = []
    for s in range(85, 146):   # 60 ticks of history before scan window
        ticks.append(TickData(sec=s, binance=ptb, price_up_ask=0.60, price_down_ask=0.42))
    # At sec=145 (inside early scan [140,170)), btc spikes up
    ticks.append(TickData(sec=145, binance=ptb + 100.0, price_up_ask=0.60, price_down_ask=0.42))
    ticks += [TickData(sec=s, binance=ptb + 100.0, price_up_ask=0.60, price_down_ask=0.42)
              for s in range(146, 300)]

    w = _window("up", ptb_api=ptb, ticks=ticks)
    r = ENGINE.run_window(w, CurrentM5Strategy())
    assert r.entry_taken
    assert r.entry_mode == "early"
    assert r.entry_sec is not None and r.entry_sec < 170


# ---------------------------------------------------------------------------
# Test 9 : aggregate_results cumulative totals
# ---------------------------------------------------------------------------

def test_aggregate_results_cumulative_totals():
    """aggregate_results computes correct totals and splits across N windows."""
    ptb = 84_000.0
    ticks_up = [_tick(170, binance=84_100.0, up_ask=0.60)]
    ticks_dn = [_tick(170, binance=83_900.0, dn_ask=0.42)]

    windows = [
        _window("up",   ptb_api=ptb, close_price=84_200.0, ticks=ticks_up),
        _window("down", ptb_api=ptb, close_price=83_800.0, ticks=ticks_dn),
        _window("up",   ptb_api=ptb, close_price=84_200.0, ticks=ticks_up),
    ]
    strategy = BaselinePtbStrategy()
    results = ENGINE.run_campaign(windows, strategy)
    report = aggregate_results(results, strategy_name="test")

    assert report.windows_total == 3
    assert report.trades_taken == 3
    assert report.baseline_count == 3
    assert report.early_count == 0

    # Window 1 UP: entry UP → win; Window 2 DOWN: entry DOWN → win; Window 3 UP: entry UP → win
    assert report.win_rate == pytest.approx(1.0)

    # Each trade has its own pnl; net_pnl_total = pnl_leg1_total (no hedges)
    assert report.hedge_triggered_count == 0
    assert report.net_pnl_total == pytest.approx(report.pnl_leg1_total, rel=1e-6)
    assert report.pnl_leg1_total > 0   # all wins
