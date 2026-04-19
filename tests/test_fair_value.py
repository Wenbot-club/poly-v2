"""Tests for bot/fair_value.py — deterministic, no network."""
from __future__ import annotations

import pytest

from bot.domain import (
    ClobMarketInfo,
    ClobToken,
    MarketContext,
    PriceTick,
)
from bot.fair_value import (
    SIGMA_FLOOR_USD,
    K_SIGNAL,
    SIGNAL_CAP,
    FairValueEngine,
)
from bot.settings import DEFAULT_CONFIG
from bot.state import StateFactory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW_MS = 1_900_000_900_000  # arbitrary future timestamp


def _make_market() -> MarketContext:
    return MarketContext(
        market_id="test-mkt",
        condition_id="0xtest",
        title="Test BTC 15m",
        slug="test-btc-15m",
        start_ts_ms=1_900_000_800_000,
        end_ts_ms=1_900_001_700_000,   # 900s window
        yes_token_id="YES",
        no_token_id="NO",
        clob=ClobMarketInfo(
            tokens=[ClobToken("YES", "Yes"), ClobToken("NO", "No")],
            min_order_size=5.0,
            min_tick_size=0.01,
            maker_base_fee_bps=0,
            taker_base_fee_bps=0,
            taker_delay_enabled=False,
            min_order_age_s=0.0,
            fee_rate=0.0,
            fee_exponent=1.0,
        ),
    )


def _tick(value: float, ts: int = _NOW_MS - 200) -> PriceTick:
    return PriceTick(
        symbol="btc/usd",
        timestamp_ms=ts,
        value=value,
        recv_timestamp_ms=ts + 50,
        sequence_no=1,
    )


def _make_state_with_book(bid: float = 0.46, ask: float = 0.52):
    """State with a valid book and no ticks."""
    state = StateFactory(DEFAULT_CONFIG).create(_make_market())
    # Inject book levels directly via the book dict
    state.yes_book.bids[bid] = 30.0
    state.yes_book.asks[ask] = 25.0
    state.yes_book.best.bid = bid
    state.yes_book.best.ask = ask
    state.yes_book.best.spread = ask - bid
    state.yes_book.timestamp_ms = _NOW_MS - 100
    return state


def _engine() -> FairValueEngine:
    return FairValueEngine(config=DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# signal_adj is bounded by SIGNAL_CAP
# ---------------------------------------------------------------------------

def test_signal_adj_bounded_by_signal_cap_positive():
    """Large positive gap (binance >> chainlink) → signal_adj == SIGNAL_CAP."""
    state = _make_state_with_book()
    # gap_usd = 10_000, sigma_usd = max(~0, 10) = 10 → gap_z = 1000 → capped
    state.last_chainlink = _tick(70_000.0)
    state.last_binance   = _tick(80_000.0)  # gap = +10_000 >> SIGMA_FLOOR_USD
    result = _engine().compute(state, _NOW_MS)
    assert abs(result.signal_adj - SIGNAL_CAP) < 1e-9


def test_signal_adj_bounded_by_signal_cap_negative():
    """Large negative gap → signal_adj == -SIGNAL_CAP."""
    state = _make_state_with_book()
    state.last_chainlink = _tick(80_000.0)
    state.last_binance   = _tick(70_000.0)
    result = _engine().compute(state, _NOW_MS)
    assert abs(result.signal_adj - (-SIGNAL_CAP)) < 1e-9


def test_signal_adj_zero_when_gap_zero():
    """gap_usd = 0 → gap_z = 0 → signal_adj = 0."""
    state = _make_state_with_book()
    state.last_chainlink = _tick(75_000.0)
    state.last_binance   = _tick(75_000.0)
    result = _engine().compute(state, _NOW_MS)
    assert result.signal_adj == 0.0
    assert result.gap_z == 0.0


# ---------------------------------------------------------------------------
# sigma_usd floor
# ---------------------------------------------------------------------------

def test_sigma_usd_never_below_sigma_floor():
    """sigma_usd = max(sigma_60, SIGMA_FLOOR_USD) — even with a single tick."""
    state = _make_state_with_book()
    # Only one binance tick → sigma_60 = 0 → sigma_usd must equal SIGMA_FLOOR_USD
    state.last_chainlink = _tick(75_000.0)
    state.last_binance   = _tick(75_000.0)
    result = _engine().compute(state, _NOW_MS)
    assert result.sigma_usd >= SIGMA_FLOOR_USD


# ---------------------------------------------------------------------------
# book incomplete → RuntimeError
# ---------------------------------------------------------------------------

def test_book_incomplete_raises_runtime_error():
    """No book data → RuntimeError, not a silent return."""
    state = StateFactory(DEFAULT_CONFIG).create(_make_market())
    state.last_chainlink = _tick(75_000.0)
    state.last_binance   = _tick(75_000.0)
    with pytest.raises(RuntimeError, match="book incomplete"):
        _engine().compute(state, _NOW_MS)


# ---------------------------------------------------------------------------
# p_up stays in [0, 1]
# ---------------------------------------------------------------------------

def test_p_up_clamped_to_unit_interval_at_extremes():
    """Even with mid near 0 or 1, p_up remains in [0, 1]."""
    # mid near 1 (deep ask market)
    state_high = _make_state_with_book(bid=0.95, ask=0.99)
    state_high.last_chainlink = _tick(80_000.0)
    state_high.last_binance   = _tick(80_010.0)
    r_high = _engine().compute(state_high, _NOW_MS)
    assert 0.0 <= r_high.p_up <= 1.0

    # mid near 0 (deep bid market)
    state_low = _make_state_with_book(bid=0.01, ask=0.05)
    state_low.last_chainlink = _tick(70_000.0)
    state_low.last_binance   = _tick(70_010.0)
    r_low = _engine().compute(state_low, _NOW_MS)
    assert 0.0 <= r_low.p_up <= 1.0
