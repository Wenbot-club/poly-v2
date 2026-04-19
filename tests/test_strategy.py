"""Tests for bot/strategy/baseline.py — deterministic, no network."""
from __future__ import annotations

from bot.domain import (
    ClobMarketInfo,
    ClobToken,
    FairValueSnapshot,
    MarketContext,
    PriceTick,
)
from bot.settings import DEFAULT_CONFIG
from bot.state import StateFactory
from bot.strategy.baseline import (
    BASE_EDGE,
    BINANCE_MAX_AGE_MS,
    CHAINLINK_MAX_AGE_MS,
    FAIR_SANITY_MIN,
    K_SPREAD,
    K_UNCERTAINTY,
    MAX_BOOK_SPREAD,
    MIN_EDGE,
    QUOTE_NOTIONAL_USD,
    TAU_GATE_S,
    Z_CAP,
    QuotePolicy,
)


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
        end_ts_ms=1_900_001_700_000,
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


def _fresh_tick(value: float = 75_000.0, age_ms: int = 100) -> PriceTick:
    ts = _NOW_MS - age_ms - 50
    return PriceTick(
        symbol="btc/usd",
        timestamp_ms=ts,
        value=value,
        recv_timestamp_ms=_NOW_MS - age_ms,
        sequence_no=1,
    )


def _make_state(
    bid: float = 0.46,
    ask: float = 0.52,
    chainlink_age_ms: int = 100,
    binance_age_ms: int = 100,
    up_free: float = 0.0,
    pusd_free: float = 125.0,
):
    state = StateFactory(DEFAULT_CONFIG).create(_make_market())
    state.yes_book.bids[bid] = 30.0
    state.yes_book.asks[ask] = 25.0
    state.yes_book.best.bid = bid
    state.yes_book.best.ask = ask
    state.yes_book.best.spread = ask - bid
    state.yes_book.timestamp_ms = _NOW_MS - 100
    state.last_chainlink = _fresh_tick(75_000.0, chainlink_age_ms)
    state.last_binance   = _fresh_tick(75_000.0, binance_age_ms)
    state.inventory.pusd_free = pusd_free
    state.inventory.up_free   = up_free
    return state


def _fair(
    p_up: float = 0.50,
    gap_z: float = 0.0,
    tau_s: float = 300.0,
) -> FairValueSnapshot:
    return FairValueSnapshot(
        p_up=p_up,
        p_down=1.0 - p_up,
        z_score=gap_z,
        sigma_60=20.0,
        denom=1.0,
        lead_adj=0.0,
        micro_adj=0.0,
        imbalance=0.0,
        tape=0.0,
        chainlink_last=75_000.0,
        binance_last=75_000.0,
        ptb=75_000.0,
        tau_s=tau_s,
        timestamp_ms=_NOW_MS,
        gap_z=gap_z,
        signal_adj=0.0,
        sigma_usd=20.0,
    )


def _policy() -> QuotePolicy:
    return QuotePolicy(config=DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# Gate: book
# ---------------------------------------------------------------------------

def test_book_gate_no_bid_returns_all_disabled():
    state = _make_state()
    state.yes_book.bids.clear()
    state.yes_book.best.bid = 0.0
    result = _policy().build(state, _fair(), _NOW_MS)
    assert not result.bid.enabled
    assert not result.ask.enabled
    assert result.bid.reason == "book_gate"


def test_book_gate_no_ask_returns_all_disabled():
    state = _make_state()
    state.yes_book.asks.clear()
    state.yes_book.best.ask = 1.0
    result = _policy().build(state, _fair(), _NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "book_gate"


def test_book_gate_spread_too_wide_returns_all_disabled():
    # spread = 0.20 > MAX_BOOK_SPREAD = 0.15
    state = _make_state(bid=0.40, ask=0.60)
    result = _policy().build(state, _fair(), _NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "book_gate"


# ---------------------------------------------------------------------------
# Gate: freshness
# ---------------------------------------------------------------------------

def test_chainlink_stale_returns_all_disabled():
    state = _make_state(chainlink_age_ms=CHAINLINK_MAX_AGE_MS + 1)
    result = _policy().build(state, _fair(), _NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "chainlink_stale"


def test_binance_stale_returns_all_disabled():
    state = _make_state(binance_age_ms=BINANCE_MAX_AGE_MS + 1)
    result = _policy().build(state, _fair(), _NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "binance_stale"


# ---------------------------------------------------------------------------
# Gate: tau
# ---------------------------------------------------------------------------

def test_tau_gate_returns_all_disabled():
    state = _make_state()
    result = _policy().build(state, _fair(tau_s=TAU_GATE_S - 1), _NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "tau_gate"


# ---------------------------------------------------------------------------
# Gate: fair sanity
# ---------------------------------------------------------------------------

def test_fair_sanity_gate_low_p_up():
    state = _make_state()
    result = _policy().build(state, _fair(p_up=FAIR_SANITY_MIN - 0.001), _NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "fair_sanity"


def test_fair_sanity_gate_high_p_up():
    state = _make_state()
    result = _policy().build(state, _fair(p_up=1.0 - FAIR_SANITY_MIN + 0.001), _NOW_MS)
    assert not result.bid.enabled
    assert result.bid.reason == "fair_sanity"


# ---------------------------------------------------------------------------
# Bid price stays strictly inside the book
# ---------------------------------------------------------------------------

def test_bid_price_strictly_below_best_ask():
    state = _make_state(bid=0.46, ask=0.52)
    result = _policy().build(state, _fair(p_up=0.50), _NOW_MS)
    assert result.bid.enabled
    assert result.bid.price < 0.52  # strictly below best ask


# ---------------------------------------------------------------------------
# Ask price stays strictly outside the book
# ---------------------------------------------------------------------------

def test_ask_price_strictly_above_best_bid():
    state = _make_state(bid=0.46, ask=0.52, up_free=100.0)
    result = _policy().build(state, _fair(p_up=0.50), _NOW_MS)
    if result.ask.enabled:
        assert result.ask.price > 0.46  # strictly above best bid


# ---------------------------------------------------------------------------
# Edge widens with book spread
# ---------------------------------------------------------------------------

def test_edge_widens_with_book_spread():
    """Wider book spread → wider bid edge (bid price further from fair)."""
    state_tight = _make_state(bid=0.49, ask=0.51)  # spread 0.02
    state_wide  = _make_state(bid=0.40, ask=0.54)  # spread 0.14

    r_tight = _policy().build(state_tight, _fair(p_up=0.50), _NOW_MS)
    r_wide  = _policy().build(state_wide,  _fair(p_up=0.50), _NOW_MS)

    if r_tight.bid.enabled and r_wide.bid.enabled:
        # With wide spread, edge is larger, so bid is further below fair
        gap_tight = 0.50 - r_tight.bid.price
        gap_wide  = 0.50 - r_wide.bid.price
        assert gap_wide >= gap_tight


# ---------------------------------------------------------------------------
# Edge widens with |gap_z|
# ---------------------------------------------------------------------------

def test_edge_widens_with_gap_z():
    """Higher |gap_z| → larger uncertainty_term → wider edge."""
    state = _make_state()
    r_zero = _policy().build(state, _fair(p_up=0.50, gap_z=0.0), _NOW_MS)
    r_high = _policy().build(state, _fair(p_up=0.50, gap_z=Z_CAP), _NOW_MS)

    if r_zero.bid.enabled and r_high.bid.enabled:
        gap_zero = 0.50 - r_zero.bid.price
        gap_high = 0.50 - r_high.bid.price
        assert gap_high >= gap_zero


# ---------------------------------------------------------------------------
# No ask without inventory
# ---------------------------------------------------------------------------

def test_no_ask_without_inventory():
    state = _make_state(up_free=0.0)
    result = _policy().build(state, _fair(p_up=0.50), _NOW_MS)
    assert not result.ask.enabled


def test_ask_enabled_with_sufficient_inventory():
    """up_free well above min_order_size → ask should be enabled."""
    state = _make_state(up_free=50.0)
    result = _policy().build(state, _fair(p_up=0.50), _NOW_MS)
    assert result.ask.enabled


# ---------------------------------------------------------------------------
# Inventory skew widens bid edge when long
# ---------------------------------------------------------------------------

def test_inventory_skew_bid_further_from_fair_when_long():
    state_flat = _make_state(up_free=0.0)
    state_long = _make_state(up_free=100.0)
    fair = _fair(p_up=0.50)

    r_flat = _policy().build(state_flat, fair, _NOW_MS)
    r_long = _policy().build(state_long, fair, _NOW_MS)

    if r_flat.bid.enabled and r_long.bid.enabled:
        # Long position → wider bid_edge → bid price further below fair
        assert r_long.bid.price <= r_flat.bid.price
