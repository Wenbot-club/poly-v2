"""
BTC M5 strategy tests — 52 tests across 9 groups.

Tests 1-4   : PTB fetching (SSR, API, Chainlink fallback, stale rejection)
Tests 5-15  : Probabilistic entry model (sigma, score, entry gates)
Tests 16-18 : Baseline direction
Tests 19-22 : Hedge trigger + no-second-hedge + cutoff block
Tests 23-24 : price_insane guard
Tests 25-27 : Settlement P&L
Tests 28-29 : Aggregate summary (counters + model diagnostics)
Tests 30-31 : LEG1/HEDGE trace independence + settlement with both legs
Tests 32-34 : Integration (model fields on record, block counts in summary)
Tests 35-41 : Consecutive windows — scheduler, prefetch, resolution
Tests 42-46 : PTB and window audit fields
Tests 47-52 : PTB hardening — SSR context anchor, delta guard, retry, Chainlink last resort
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.m5_session import (
    BtcHistory,
    M5Session,
    M5SignalState,
    PaperFillResult,
    PtbResult,
    fetch_ptb,
    fetch_ptb_ssr,
    fetch_ptb_api,
    fetch_ptb_robust,
    fetch_close_price,
)

from demos.demo_btc_m5 import _first_m5_window_ts

from bot.m5_summary import TradeRecord, aggregate_trades
from bot.settings import M5Config
from bot.strategy.btc_m5 import (
    estimate_sigma_to_close,
    compute_entry_signal,
    baseline_direction,
    should_hedge,
    compute_settlement,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http(responses: dict) -> Any:
    """
    Minimal aiohttp.ClientSession mock.
    responses: { url_substring: {"status": int, "json": dict | None, "text": str | None} }
    """
    class _Resp:
        def __init__(self, cfg):
            self.status = cfg.get("status", 200)
            self._json = cfg.get("json")
            self._text = cfg.get("text", "")

        async def json(self, **_):
            return self._json

        async def text(self):
            return self._text

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    class _Session:
        def get(self, url, **_):
            for substr, cfg in responses.items():
                if substr in url:
                    return _Resp(cfg)
            return _Resp({"status": 404})

    return _Session()


def _make_session(
    signal: Optional[M5SignalState] = None,
    cfg: Optional[M5Config] = None,
    http=None,
    time_fn=None,
    token_prices: Optional[dict] = None,
    btc_history=None,
    prefetched_tokens=None,
) -> M5Session:
    s = M5Session(
        http_session=http or _make_http({}),
        signal_state=signal or M5SignalState(),
        config=cfg or M5Config(),
        time_fn=time_fn or (lambda: 0.0),
        btc_history=btc_history,
        prefetched_tokens=prefetched_tokens,
    )
    if token_prices:
        s._token_prices = token_prices
    return s


# ---------------------------------------------------------------------------
# Tests 1-3: PTB fetching
# ---------------------------------------------------------------------------

def _ssr_html(window_ts: int, open_price: str) -> str:
    """Realistic SSR HTML: includes slug anchor and is > 500 chars."""
    slug = f"btc-updown-5m-{window_ts}"
    body = (
        f'<!DOCTYPE html><html><head><title>BTC UP/DOWN 5m</title></head><body>'
        f'<script id="__NEXT_DATA__">{{"props":{{"pageProps":{{"event":{{'
        f'"slug":"{slug}","openPrice":"{open_price}","closePrice":null}}}}}}}}'
        f'</script></body></html>'
    )
    return body + " " * max(0, 520 - len(body))


def test_ptb_ssr_ok():
    """SSR page contains slug-anchored openPrice — parsed correctly."""
    html = _ssr_html(1_000_000, "84123.45")
    http = _make_http({"/event/btc-updown-5m-": {"status": 200, "text": html}})
    ptb, source = asyncio.run(fetch_ptb(http, 1_000_000, polymarket_base_url="https://fake"))
    assert ptb == pytest.approx(84123.45)
    assert source == "ssr"


def test_ptb_fallback_api_ok():
    """SSR 404, API returns openPrice."""
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 200, "json": {"openPrice": "85000.0"}},
    })
    ptb, source = asyncio.run(fetch_ptb(http, 1_000_000, polymarket_base_url="https://fake"))
    assert ptb == pytest.approx(85000.0)
    assert source == "api"


def test_ptb_fallback_chainlink_ok():
    """SSR 404, API 500, Chainlink price used when fresh enough."""
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 500},
    })
    ptb, source = asyncio.run(
        fetch_ptb(
            http, 1_000_000,
            chainlink_price=84999.0,
            chainlink_age_s=30.0,       # < 60s → accepted
            polymarket_base_url="https://fake",
        )
    )
    assert ptb == pytest.approx(84999.0)
    assert source == "chainlink"


def test_ptb_chainlink_rejected_if_stale():
    """Chainlink older than 60s is not used as PTB fallback."""
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 404},
    })
    ptb, source = asyncio.run(
        fetch_ptb(
            http, 1_000_000,
            chainlink_price=84999.0,
            chainlink_age_s=90.0,       # > 60s → rejected
            polymarket_base_url="https://fake",
        )
    )
    assert ptb is None
    assert source is None


# ---------------------------------------------------------------------------
# Tests 5-15: Probabilistic entry model
# ---------------------------------------------------------------------------

# Helpers for this section
_FLAT = [(i * 1_000, 85_000.0) for i in range(60)]   # 60 flat ticks → sigma = floor
_TAU = 155.0  # window_seconds=300, elapsed=145


def test_sigma_floor_when_insufficient_history():
    """Returns floor when fewer than 3 samples or all-zero variance."""
    assert estimate_sigma_to_close([], tau_s=100.0, sigma_floor_usd=5.0) == 5.0
    assert estimate_sigma_to_close([(0, 100.0), (1_000, 101.0)], tau_s=100.0, sigma_floor_usd=5.0) == 5.0
    # Flat samples: variance = 0 → floor
    assert estimate_sigma_to_close(_FLAT, tau_s=_TAU, sigma_floor_usd=5.0) == 5.0


def test_larger_sigma_reduces_p_model_up():
    """Same gap, higher historical vol → smaller z_gap → p_model_up closer to 0.5."""
    # Low vol: alternating ±0.5 → dp ≈ ±1.0 → var_per_s ≈ 1.0 → sigma ≈ 11.8
    samples_low = [(i * 1_000, 85_000.0 + (0.5 if i % 2 == 0 else -0.5)) for i in range(60)]
    # High vol: alternating ±50 → dp ≈ ±100 → var_per_s ≈ 10000 → sigma ≈ 1183
    samples_high = [(i * 1_000, 85_000.0 + (50.0 if i % 2 == 0 else -50.0)) for i in range(60)]

    kwargs = dict(
        btc=85_100.0, ptb=85_000.0, tau_s=140.0,
        btc_10s=None, btc_30s=None,
        price_up=0.50, price_down=0.50,
        sigma_floor_usd=0.1,   # tiny floor so real vol dominates
        z_gap_min=0.0, p_enter_up_min=0.60, p_enter_down_max=0.40, min_entry_edge=0.0,
    )
    sig_low = compute_entry_signal(btc_samples=samples_low, **kwargs)
    sig_high = compute_entry_signal(btc_samples=samples_high, **kwargs)

    assert sig_low.p_model_up > sig_high.p_model_up
    assert sig_low.p_model_up > 0.99
    assert sig_high.p_model_up < 0.60


def test_larger_gap_increases_p_model_up():
    """Larger positive gap → higher z_gap → higher p_model_up."""
    base = dict(
        ptb=85_000.0, tau_s=_TAU, btc_samples=_FLAT,
        btc_10s=None, btc_30s=None,
        price_up=0.50, price_down=0.50,
        sigma_floor_usd=5.0, z_gap_min=0.0,
        p_enter_up_min=0.60, p_enter_down_max=0.40, min_entry_edge=0.0,
    )
    sig_small = compute_entry_signal(btc=85_010.0, **base)   # gap = 10
    sig_large = compute_entry_signal(btc=85_050.0, **base)   # gap = 50

    assert sig_large.p_model_up > sig_small.p_model_up


def test_price_tokens_do_not_affect_model_probability():
    """Changing price_up/price_down changes edge but never p_model_up."""
    base = dict(
        btc=85_100.0, ptb=85_000.0, tau_s=_TAU, btc_samples=_FLAT,
        btc_10s=None, btc_30s=None,
        sigma_floor_usd=5.0, z_gap_min=0.0,
        p_enter_up_min=0.60, p_enter_down_max=0.40, min_entry_edge=0.0,
    )
    sig_a = compute_entry_signal(price_up=0.50, price_down=0.50, **base)
    sig_b = compute_entry_signal(price_up=0.70, price_down=0.30, **base)

    assert sig_a.p_model_up == pytest.approx(sig_b.p_model_up)
    assert sig_a.edge_up > sig_b.edge_up   # cheap market gives more edge


def test_chainlink_not_in_entry_signal_signature():
    """compute_entry_signal has no chainlink parameter — by design."""
    import inspect
    params = inspect.signature(compute_entry_signal).parameters
    assert "chainlink" not in params


def test_noise_zone_blocks_entry():
    """abs(z_gap) < z_gap_min → direction=None, block_reason='noise_zone'."""
    # gap = 1.0, sigma_floor = 5.0 → z_gap = 0.20 < 0.35
    sig = compute_entry_signal(
        btc=85_001.0, ptb=85_000.0, tau_s=_TAU, btc_samples=_FLAT,
        btc_10s=None, btc_30s=None,
        price_up=0.50, price_down=0.50,
        sigma_floor_usd=5.0, z_gap_min=0.35,
        p_enter_up_min=0.60, p_enter_down_max=0.40, min_entry_edge=0.0,
    )
    assert sig.direction is None
    assert sig.block_reason == "noise_zone"
    assert abs(sig.z_gap) < 0.35


def test_buy_up_on_strong_signal_and_edge():
    """Large positive z_gap + cheap price_up → direction='up'."""
    # gap = 100, sigma = 5.0, z_gap = 20 → p ≈ 1.0; edge = 1.0 - 0.45 = 0.55
    sig = compute_entry_signal(
        btc=85_100.0, ptb=85_000.0, tau_s=_TAU, btc_samples=_FLAT,
        btc_10s=None, btc_30s=None,
        price_up=0.45, price_down=0.55,
        sigma_floor_usd=5.0, z_gap_min=0.35,
        p_enter_up_min=0.60, p_enter_down_max=0.40, min_entry_edge=0.06,
    )
    assert sig.direction == "up"
    assert sig.block_reason is None
    assert sig.p_model_up >= 0.60
    assert sig.edge_up >= 0.06


def test_buy_down_on_strong_signal_and_edge():
    """Large negative z_gap + cheap price_down → direction='down'."""
    # gap = -100, sigma = 5.0, z_gap = -20 → p ≈ 0.0; edge_down = 1.0 - 0.45 = 0.55
    sig = compute_entry_signal(
        btc=84_900.0, ptb=85_000.0, tau_s=_TAU, btc_samples=_FLAT,
        btc_10s=None, btc_30s=None,
        price_up=0.55, price_down=0.45,
        sigma_floor_usd=5.0, z_gap_min=0.35,
        p_enter_up_min=0.60, p_enter_down_max=0.40, min_entry_edge=0.06,
    )
    assert sig.direction == "down"
    assert sig.block_reason is None
    assert sig.p_model_up <= 0.40
    assert sig.edge_down >= 0.06


def test_no_trade_when_edge_insufficient():
    """p_model_up crosses threshold but price_up too close → edge_not_enough."""
    # gap ≈ 2.3 → z_gap ≈ 0.46 → score ≈ 0.62 → p ≈ 0.65 >= 0.60
    # price_up = 0.62 → edge = 0.03 < 0.06
    sig = compute_entry_signal(
        btc=85_002.3, ptb=85_000.0, tau_s=_TAU, btc_samples=_FLAT,
        btc_10s=None, btc_30s=None,
        price_up=0.62, price_down=0.38,
        sigma_floor_usd=5.0, z_gap_min=0.35,
        p_enter_up_min=0.60, p_enter_down_max=0.40, min_entry_edge=0.06,
    )
    assert sig.direction is None
    assert sig.block_reason == "edge_not_enough"
    assert sig.p_model_up >= 0.60
    assert sig.edge_up < 0.06


def test_probability_not_strong_enough_between_thresholds():
    """p_model_up in (0.40, 0.60) → direction=None, block_reason='probability_not_strong_enough'."""
    # gap near 0 → p ≈ 0.50
    sig = compute_entry_signal(
        btc=85_000.5, ptb=85_000.0, tau_s=_TAU, btc_samples=_FLAT,
        btc_10s=None, btc_30s=None,
        price_up=0.44, price_down=0.56,  # edge_up > min but p not strong enough
        sigma_floor_usd=5.0, z_gap_min=0.0,   # z_gap_min=0 so noise zone doesn't block first
        p_enter_up_min=0.60, p_enter_down_max=0.40, min_entry_edge=0.0,
    )
    assert sig.direction is None
    assert sig.block_reason == "probability_not_strong_enough"


# ---------------------------------------------------------------------------
# Tests 7-8: Baseline direction
# ---------------------------------------------------------------------------

def test_baseline_up_when_btc_above_ptb():
    assert baseline_direction(btc=85_001.0, ptb=85_000.0) == "up"


def test_baseline_down_when_btc_below_ptb():
    assert baseline_direction(btc=84_999.0, ptb=85_000.0) == "down"


def test_baseline_none_when_equal():
    assert baseline_direction(btc=85_000.0, ptb=85_000.0) is None


# ---------------------------------------------------------------------------
# Tests 9-10: Hedge trigger
# ---------------------------------------------------------------------------

def test_hedge_triggered_up_to_down():
    """LEG1=UP, btc falls below ptb-1.0 → hedge triggered."""
    assert should_hedge("up", btc=84_998.0, ptb=85_000.0, threshold=1.0) is True


def test_hedge_not_triggered_up_when_within_threshold():
    assert should_hedge("up", btc=84_999.5, ptb=85_000.0, threshold=1.0) is False


def test_hedge_triggered_down_to_up():
    """LEG1=DOWN, btc rises above ptb+1.0 → hedge triggered."""
    assert should_hedge("down", btc=85_001.5, ptb=85_000.0, threshold=1.0) is True


def test_hedge_not_triggered_down_when_within_threshold():
    assert should_hedge("down", btc=85_000.5, ptb=85_000.0, threshold=1.0) is False


# ---------------------------------------------------------------------------
# Test 11: No second hedge
# ---------------------------------------------------------------------------

def test_no_second_hedge():
    """_watch_for_hedge returns immediately if record.hedged is already True."""
    window_ts = 1_000_000
    fixed_time = window_ts + 200.0      # hedge would normally trigger here

    signal = M5SignalState(btc_price=84_998.0)  # would trigger hedge for "up" LEG1
    session = _make_session(
        signal=signal,
        cfg=M5Config(hedge_threshold=1.0, hedge_cutoff_s=250.0, window_seconds=300),
        time_fn=lambda: fixed_time,
        token_prices={"dn": 0.45},
    )

    record = TradeRecord(window_ts=window_ts)
    record.entry_side = "up"
    record.hedged = True        # already hedged

    asyncio.run(session._watch_for_hedge(
        record, ptb=85_000.0,
        up_id="up", down_id="dn",
        window_ts=window_ts,
        window_end_s=window_ts + 300.0,
    ))

    # Still hedged exactly once, method returned without doing anything
    assert record.hedged is True
    assert record.hedge_price is None   # was not re-filled


# ---------------------------------------------------------------------------
# Test 12: Hedge blocked by cutoff
# ---------------------------------------------------------------------------

def test_hedge_blocked_after_cutoff():
    """When elapsed >= hedge_cutoff_s, _watch_for_hedge blocks and returns."""
    window_ts = 1_000_000
    fixed_time = window_ts + 260.0      # > cutoff=250

    signal = M5SignalState(btc_price=84_998.0)
    session = _make_session(
        signal=signal,
        cfg=M5Config(hedge_threshold=1.0, hedge_cutoff_s=250.0, window_seconds=300),
        time_fn=lambda: fixed_time,
        token_prices={"dn": 0.45},
    )

    record = TradeRecord(window_ts=window_ts)
    record.entry_side = "up"

    asyncio.run(session._watch_for_hedge(
        record, ptb=85_000.0,
        up_id="up", down_id="dn",
        window_ts=window_ts,
        window_end_s=window_ts + 300.0,
    ))

    assert session._hedge_blocked_by_cutoff is True
    assert not record.hedged


# ---------------------------------------------------------------------------
# Test 13: price_insane guard
# ---------------------------------------------------------------------------

def test_price_insane_blocks_buy():
    """best_ask >= price_insane_threshold → reject_reason='price_insane'."""
    session = _make_session(cfg=M5Config(price_insane_threshold=0.995))
    result = asyncio.run(session._execute_paper(best_ask=0.995, usd_bet=1.0, is_leg1=True))
    assert result.reject_reason == "price_insane"
    assert result.fill_price is None
    assert session._price_insane_blocks == 1


def test_price_just_below_insane_fills():
    """best_ask = 0.994 < 0.995 → fills normally."""
    session = _make_session(cfg=M5Config(price_insane_threshold=0.995))
    result = asyncio.run(session._execute_paper(best_ask=0.994, usd_bet=1.0, is_leg1=True))
    assert result.reject_reason is None
    assert result.fill_price == pytest.approx(0.994)


# ---------------------------------------------------------------------------
# Tests 14-15: Settlement P&L
# ---------------------------------------------------------------------------

def test_settlement_leg1_win_no_hedge():
    """LEG1=UP wins: pnl = (1-entry)*shares, no hedge."""
    s = compute_settlement(
        close_price=85_100.0, open_price=85_000.0,
        leg1_side="up",
        leg1_entry_price=0.60,
        leg1_shares=1.0 / 0.60,
        leg1_usd_staked=1.00,
    )
    assert s.result == "up"
    assert s.pnl_leg1 == pytest.approx((1.0 - 0.60) * (1.0 / 0.60), rel=1e-6)
    assert s.pnl_hedge == pytest.approx(0.0)
    assert s.net_pnl == pytest.approx(s.pnl_leg1)


def test_settlement_leg1_loss_hedge_win():
    """LEG1=UP loses, HEDGE=DOWN wins: pnl_leg1 = -staked, pnl_hedge positive."""
    shares_hedge = 2.0 / 0.45
    s = compute_settlement(
        close_price=84_900.0, open_price=85_000.0,   # result = "down"
        leg1_side="up",
        leg1_entry_price=0.60,
        leg1_shares=1.0 / 0.60,
        leg1_usd_staked=1.00,
        hedge_side="down",
        hedge_entry_price=0.45,
        hedge_shares=shares_hedge,
        hedge_usd_staked=2.00,
    )
    assert s.result == "down"
    assert s.pnl_leg1 == pytest.approx(-1.00)
    assert s.pnl_hedge == pytest.approx((1.0 - 0.45) * shares_hedge, rel=1e-6)
    assert s.net_pnl == pytest.approx(s.pnl_leg1 + s.pnl_hedge, rel=1e-6)


def test_settlement_both_lose():
    """Both LEG1 and HEDGE on same side, both lose."""
    s = compute_settlement(
        close_price=84_900.0, open_price=85_000.0,   # result = "down"
        leg1_side="up",
        leg1_entry_price=0.60, leg1_shares=1.0/0.60, leg1_usd_staked=1.00,
        hedge_side="up",   # wrong side
        hedge_entry_price=0.55, hedge_shares=2.0/0.55, hedge_usd_staked=2.00,
    )
    assert s.result == "down"
    assert s.pnl_leg1 == pytest.approx(-1.00)
    assert s.pnl_hedge == pytest.approx(-2.00)


# ---------------------------------------------------------------------------
# Test 16: Aggregate summary
# ---------------------------------------------------------------------------

def test_aggregate_summary_correct():
    """aggregate_trades computes all counters correctly."""
    t1 = TradeRecord(
        window_ts=1000,
        entry_mode="early", entry_side="up", entry_price=0.60,
        entry_shares=1.0/0.60, hedged=True,
        hedge_price=0.45, hedge_shares=2.0/0.45, hedge_side="down",
        pnl_leg1=0.25, pnl_hedge=-2.0, net_pnl=-1.75,
    )
    t2 = TradeRecord(
        window_ts=2000,
        entry_mode="baseline", entry_side="down", entry_price=0.40,
        entry_shares=1.0/0.40, hedged=False,
        hedge_blocked_by_cutoff=True,
        pnl_leg1=1.50, pnl_hedge=0.0, net_pnl=1.50,
    )
    t3 = TradeRecord(window_ts=3000, abort_reason="ptb_unavailable")
    t4 = TradeRecord(window_ts=4000, abort_reason="tokens_unavailable", price_insane_block_count=2)

    summary = aggregate_trades([t1, t2, t3, t4])

    assert summary.windows_seen == 4
    assert summary.ptb_fail_count == 1
    assert summary.token_setup_fail_count == 1
    assert summary.leg1_entered_count == 2
    assert summary.early_entry_count == 1
    assert summary.baseline_entry_count == 1
    assert summary.hedge_triggered_count == 1
    assert summary.hedge_blocked_by_cutoff_count == 1
    assert summary.price_insane_block_count == 2
    assert summary.pnl_leg1_total == pytest.approx(0.25 + 1.50)
    assert summary.pnl_hedge_total == pytest.approx(-2.0)
    assert summary.net_pnl_total == pytest.approx(-1.75 + 1.50)
    assert summary.avg_leg1_entry_price == pytest.approx((0.60 + 0.40) / 2)
    assert summary.avg_hedge_entry_price == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_trade_record_stores_model_fields_on_early_entry():
    """After EARLY entry, TradeRecord carries all model observation fields."""
    window_ts = 1_000_000
    # Flat history → sigma = floor = 5.0; gap = 100 → z_gap = 20 >> 0.35
    now_ms = int((window_ts + 145.0) * 1000)
    history = BtcHistory()
    for i in range(60):
        history.record(85_000.0, now_ms - (60 - i) * 1_000)

    signal = M5SignalState(btc_price=85_100.0)
    session = _make_session(
        signal=signal,
        time_fn=lambda: window_ts + 145.0,
        token_prices={"up": 0.45, "dn": 0.55},
        btc_history=history,
    )

    record = TradeRecord(window_ts=window_ts)
    entered = asyncio.run(session._try_early_entry(
        record, ptb=85_000.0, up_id="up", down_id="dn", window_ts=window_ts,
    ))

    assert entered is True
    assert record.entry_mode == "early"
    assert record.entry_side == "up"
    assert record.p_model_up_at_entry is not None
    assert record.p_model_up_at_entry >= 0.60
    assert record.z_gap_at_entry is not None
    assert record.z_gap_at_entry == pytest.approx(100.0 / 5.0, rel=0.05)
    assert record.sigma_to_close_at_entry == pytest.approx(5.0, rel=0.05)
    assert record.edge_up_at_entry is not None
    assert record.edge_up_at_entry >= 0.06
    assert record.entry_block_reason is None


def test_campaign_summary_aggregates_block_counts():
    """M5CampaignSummary counts all three block_reason values and averages model fields."""
    t1 = TradeRecord(window_ts=1000, entry_block_reason="noise_zone")
    t2 = TradeRecord(window_ts=2000, entry_block_reason="edge_not_enough")
    t3 = TradeRecord(window_ts=3000, entry_block_reason="edge_not_enough")
    t4 = TradeRecord(window_ts=4000, entry_block_reason="probability_not_strong_enough")
    t5 = TradeRecord(
        window_ts=5000,
        entry_mode="early", entry_side="up", entry_price=0.60,
        entry_shares=1.0 / 0.60,
        p_model_up_at_entry=0.72, sigma_to_close_at_entry=12.0,
        pnl_leg1=0.50, pnl_hedge=0.0, net_pnl=0.50,
    )

    summary = aggregate_trades([t1, t2, t3, t4, t5])

    assert summary.blocked_by_noise_zone_count == 1
    assert summary.blocked_by_edge_count == 2
    assert summary.blocked_by_probability_count == 1
    assert summary.avg_p_model_up_at_entry == pytest.approx(0.72)
    assert summary.avg_sigma_to_close_at_entry == pytest.approx(12.0)


def test_leg1_and_hedge_fill_traces_are_independent():
    """_apply_fill_trace(role='leg1') and (role='hedge') write to separate fields."""
    session = _make_session()
    record = TradeRecord(window_ts=1_000_000)

    leg1_fill = PaperFillResult(
        fill_price=0.60, shares=round(1.0 / 0.60, 8),
        observed_best_ask=0.60,
        attempted_price=0.601,
        slippage=0.001,
        retries=0, reject_reason=None,
    )
    session._apply_fill_trace(record, leg1_fill, "leg1")

    assert record.leg1_observed_ask == pytest.approx(0.60)
    assert record.leg1_slippage == pytest.approx(0.001)
    assert record.hedge_observed_ask is None   # not yet set

    hedge_fill = PaperFillResult(
        fill_price=0.45, shares=round(2.0 / 0.45, 8),
        observed_best_ask=0.45,
        attempted_price=0.451,
        slippage=0.001,
        retries=1, reject_reason=None,
    )
    session._apply_fill_trace(record, hedge_fill, "hedge")

    assert record.hedge_observed_ask == pytest.approx(0.45)
    assert record.hedge_fill_retries == 1
    # LEG1 fields must be unchanged after hedge trace
    assert record.leg1_observed_ask == pytest.approx(0.60)
    assert record.leg1_fill_retries == 0


def test_settlement_with_both_leg_traces_on_record():
    """A record with both leg1_* and hedge_* traces settles correctly via compute_settlement."""
    session = _make_session()
    record = TradeRecord(window_ts=1_000_000)

    # Simulate LEG1 fill (UP at 0.60)
    leg1_fill = PaperFillResult(
        fill_price=0.60, shares=round(1.0 / 0.60, 8),
        observed_best_ask=0.60, attempted_price=0.601,
        slippage=0.001, retries=0, reject_reason=None,
    )
    session._apply_fill_trace(record, leg1_fill, "leg1")
    record.entry_side = "up"
    record.entry_price = 0.60
    record.entry_shares = leg1_fill.shares

    # Simulate HEDGE fill (DOWN at 0.45)
    hedge_fill = PaperFillResult(
        fill_price=0.45, shares=round(2.0 / 0.45, 8),
        observed_best_ask=0.45, attempted_price=0.451,
        slippage=0.001, retries=0, reject_reason=None,
    )
    session._apply_fill_trace(record, hedge_fill, "hedge")
    record.hedged = True
    record.hedge_side = "down"
    record.hedge_price = 0.45
    record.hedge_shares = hedge_fill.shares

    from bot.strategy.btc_m5 import compute_settlement
    s = compute_settlement(
        close_price=84_900.0, open_price=85_000.0,   # result = "down"
        leg1_side=record.entry_side,
        leg1_entry_price=record.entry_price,
        leg1_shares=record.entry_shares,
        leg1_usd_staked=1.00,
        hedge_side=record.hedge_side,
        hedge_entry_price=record.hedge_price,
        hedge_shares=record.hedge_shares,
        hedge_usd_staked=2.00,
    )

    assert s.result == "down"
    assert s.pnl_leg1 == pytest.approx(-1.00)
    assert s.pnl_hedge == pytest.approx((1.0 - 0.45) * hedge_fill.shares, rel=1e-6)
    # Trace fields are preserved independently
    assert record.leg1_observed_ask == pytest.approx(0.60)
    assert record.hedge_observed_ask == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Tests 35-41: Consecutive windows — scheduler, prefetch, resolution
# ---------------------------------------------------------------------------

def test_first_window_uses_current_when_plenty_of_time():
    """If >80% of the window remains, _first_m5_window_ts returns the current boundary."""
    ws = 300
    wts = 999_900  # valid 300s boundary: 999_900 / 300 = 3333
    # 10s in: 290s remain = 96.7% > 80%
    result = _first_m5_window_ts(M5Config(window_seconds=ws), time_fn=lambda: wts + 10.0)
    assert result == wts


def test_first_window_alignment_advances_when_less_than_80pct_remains():
    """If <80% of the window remains, _first_m5_window_ts returns the next boundary."""
    ws = 300
    wts = 999_900  # valid 300s boundary
    # 70s in: 230s remain = 76.7% < 80%
    result = _first_m5_window_ts(M5Config(window_seconds=ws), time_fn=lambda: wts + 70.0)
    assert result == wts + ws


def test_fetch_close_price_returns_on_first_success():
    """fetch_close_price returns (price, 1, latency) on the first successful call."""
    http = _make_http({
        "/api/crypto/crypto-price": {"status": 200, "json": {"closePrice": "84500.0"}},
    })
    window_ts = 1_000_000
    window_end_s = float(window_ts + 300)
    t = window_end_s + 15.1  # already past first_poll_time
    price, open_price, attempts, latency_s = asyncio.run(
        fetch_close_price(
            http, window_ts,
            polymarket_base_url="https://fake",
            window_end_s=window_end_s,
            settlement_initial_delay_s=15.0,
            settlement_poll_s=0.0,
            settlement_max_attempts=20,
            time_fn=lambda: t,
        )
    )
    assert price == pytest.approx(84500.0)
    assert attempts == 1
    assert latency_s == pytest.approx(t - window_end_s, rel=1e-6)


def test_fetch_close_price_exhausted_returns_none():
    """When all attempts miss closePrice, returns (None, max_attempts, None)."""
    # Response has no closePrice key → all attempts miss
    http = _make_http({
        "/api/crypto/crypto-price": {"status": 200, "json": {"openPrice": "84500.0"}},
    })
    window_ts = 1_000_000
    window_end_s = float(window_ts + 300)
    price, open_price, attempts, latency_s = asyncio.run(
        fetch_close_price(
            http, window_ts,
            polymarket_base_url="https://fake",
            window_end_s=window_end_s,
            settlement_initial_delay_s=0.0,
            settlement_poll_s=0.0,
            settlement_max_attempts=5,
            time_fn=lambda: window_end_s + 1.0,
        )
    )
    assert price is None
    assert open_price is None
    assert attempts == 5
    assert latency_s is None


def test_fetch_close_price_waits_initial_delay():
    """fetch_close_price sleeps until window_end_s + initial_delay before polling."""
    http = _make_http({
        "/api/crypto/crypto-price": {"status": 200, "json": {"closePrice": "84000.0"}},
    })
    window_ts = 1_000_000
    window_end_s = float(window_ts + 300)
    initial_delay = 15.0
    sleep_calls: list = []

    real_sleep = asyncio.sleep

    async def _tracking_sleep(delay, *args, **kwargs):
        sleep_calls.append(delay)
        return await real_sleep(0)

    # time_fn: first call returns value BEFORE first_poll to trigger the initial wait
    times = iter([window_end_s + 5.0, window_end_s + 15.1, window_end_s + 15.2])

    with patch("bot.m5_session.asyncio.sleep", side_effect=_tracking_sleep):
        price, _, attempts, _ = asyncio.run(
            fetch_close_price(
                http, window_ts,
                polymarket_base_url="https://fake",
                window_end_s=window_end_s,
                settlement_initial_delay_s=initial_delay,
                settlement_poll_s=4.0,
                settlement_max_attempts=20,
                time_fn=lambda: next(times, window_end_s + 20.0),
            )
        )

    assert price == pytest.approx(84000.0)
    # First sleep must be positive (initial delay wait)
    assert sleep_calls[0] == pytest.approx(initial_delay - 5.0, rel=1e-3)


def test_prefetched_tokens_sets_used_flag():
    """When prefetched_tokens is provided, record.used_prefetched_tokens is True."""
    window_ts = 1_000_000
    # PTB via API, prefetched tokens, no BTC price → aborts at baseline_no_signal
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 200, "json": {"openPrice": "85000.0"}},
    })
    # Time sequence: past ptb delay, past early scan, past baseline, past window end
    wts = window_ts
    times = iter([
        wts + 15.1,   # ptb fetch
        wts + 170.1,  # early scan elapsed check
        wts + 170.1,  # sleep_until baseline
        wts + 170.1,  # _elapsed inside baseline
        wts + 300.1,  # window end guard
        wts + 300.1,
    ])
    session = _make_session(
        signal=M5SignalState(btc_price=None),
        http=http,
        time_fn=lambda: next(times, wts + 310.0),
        prefetched_tokens=("up_tok", "dn_tok"),
    )
    record = asyncio.run(session.run(window_ts))
    assert record.used_prefetched_tokens is True
    assert record.abort_reason == "baseline_no_signal"


def test_next_tokens_task_launched_after_trading_phases():
    """session.next_tokens_task is set after run() passes the token discovery phase."""
    window_ts = 1_000_000
    # PTB via API + prefetched tokens → skips Gamma fetch
    # No BTC price → aborts at baseline → next_tokens_task still created
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 200, "json": {"openPrice": "85000.0"}},
        # Gamma endpoint for NEXT window prefetch (window_ts + 300)
        f"btc-updown-5m-{window_ts + 300}": {"status": 200, "json": {
            "markets": [{"clobTokenIds": '["next_up", "next_dn"]'}]
        }},
    })
    wts = window_ts
    times = iter([
        wts + 15.1,
        wts + 170.1,
        wts + 170.1,
        wts + 170.1,
        wts + 300.1,
        wts + 300.1,
    ])
    session = _make_session(
        signal=M5SignalState(btc_price=None),
        http=http,
        time_fn=lambda: next(times, wts + 310.0),
        prefetched_tokens=("up_tok", "dn_tok"),
    )
    asyncio.run(session.run(window_ts))
    assert session.next_tokens_task is not None


# ---------------------------------------------------------------------------
# Tests 42-46 : PTB and window audit fields
# ---------------------------------------------------------------------------

def test_window_audit_timestamps_derived_from_window_ts():
    """window_start_utc_iso and window_end_utc_iso are correctly derived from window_ts."""
    import datetime
    window_ts = 1_746_000_000  # a valid unix timestamp
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 200, "json": {"openPrice": "85000.0"}},
    })
    wts = window_ts
    times = iter([
        wts + 15.1,
        wts + 170.1, wts + 170.1, wts + 170.1,
        wts + 300.1, wts + 300.1,
    ])
    session = _make_session(
        signal=M5SignalState(btc_price=None),
        http=http,
        time_fn=lambda: next(times, wts + 310.0),
        prefetched_tokens=("up_tok", "dn_tok"),
    )
    record = asyncio.run(session.run(window_ts))

    expected_start = datetime.datetime.fromtimestamp(window_ts, tz=datetime.timezone.utc).isoformat()
    expected_end = datetime.datetime.fromtimestamp(window_ts + 300, tz=datetime.timezone.utc).isoformat()
    assert record.window_start_utc_iso == expected_start
    assert record.window_end_utc_iso == expected_end


def test_window_audit_event_slug_derived_from_window_ts():
    """event_slug is set to 'btc-updown-5m-{window_ts}' regardless of token discovery."""
    window_ts = 1_746_000_300
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 200, "json": {"openPrice": "85000.0"}},
    })
    wts = window_ts
    times = iter([
        wts + 15.1,
        wts + 170.1, wts + 170.1, wts + 170.1,
        wts + 300.1, wts + 300.1,
    ])
    session = _make_session(
        signal=M5SignalState(btc_price=None),
        http=http,
        time_fn=lambda: next(times, wts + 310.0),
        prefetched_tokens=("up_tok", "dn_tok"),
    )
    record = asyncio.run(session.run(window_ts))
    assert record.event_slug == f"btc-updown-5m-{window_ts}"


def test_fetch_ptb_robust_ssr_takes_priority_within_delta():
    """fetch_ptb_robust selects SSR when both present and |delta| <= threshold."""
    wts = 1_746_000_000
    ssr_price = 84000.0
    api_price = 84005.0   # delta = 5 <= default 10.0
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 200, "text": _ssr_html(wts, str(ssr_price))},
        "/api/crypto/crypto-price": {"status": 200, "json": {"openPrice": str(api_price)}},
    })
    r = asyncio.run(
        fetch_ptb_robust(http, wts, polymarket_base_url="https://fake",
                         ptb_max_attempts=1, ptb_retry_delay_s=0.0)
    )
    assert r.source == "ssr"
    assert r.ptb == pytest.approx(ssr_price)
    assert r.ptb_ssr == pytest.approx(ssr_price)
    assert r.ptb_api == pytest.approx(api_price)
    assert r.ptb_ssr_rejected_for_delta is False
    assert r.ptb_ssr_valid is True
    assert r.ptb_api_valid is True


def test_fetch_ptb_robust_raw_values_for_delta():
    """fetch_ptb_robust exposes raw ssr/api so caller can compute delta."""
    wts = 1_746_000_000
    ssr_price = 84000.0
    api_price = 84100.0   # delta = 100 > threshold → SSR rejected
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 200, "text": _ssr_html(wts, str(ssr_price))},
        "/api/crypto/crypto-price": {"status": 200, "json": {"openPrice": str(api_price)}},
    })
    r = asyncio.run(
        fetch_ptb_robust(http, wts, polymarket_base_url="https://fake",
                         ptb_max_attempts=1, ptb_retry_delay_s=0.0,
                         ptb_max_ssr_api_delta_usd=10.0)
    )
    assert r.ptb_ssr == pytest.approx(ssr_price)
    assert r.ptb_api == pytest.approx(api_price)
    assert round(r.ptb_ssr - r.ptb_api, 2) == pytest.approx(-100.0)


def test_resolution_audit_fields_stored_on_record():
    """resolution_open_price_api and resolution_close_price_api are stored on the record."""
    window_ts = 1_000_000
    open_p = 84000.0
    close_p = 84500.0
    wts = window_ts
    # PTB via API (SSR 404), and same endpoint returns closePrice at settlement
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 200, "json": {
            "openPrice": str(open_p),
            "closePrice": str(close_p),
        }},
    })
    cfg = M5Config(
        entry_scan_start_s=0.0,      # skip early scan
        entry_scan_end_s=0.0,
        baseline_elapsed_s=0.0,      # baseline fires immediately
        hedge_threshold=999.0,       # no hedge
        settlement_initial_delay_s=0.0,
        settlement_poll_s=0.0,
        settlement_max_attempts=5,
    )
    session = _make_session(
        signal=M5SignalState(btc_price=84100.0),  # above ptb → UP direction
        http=http,
        cfg=cfg,
        time_fn=lambda: wts + 300.1,  # always past window end
        prefetched_tokens=("up_tok", "dn_tok"),
        token_prices={"up_tok": 0.60, "dn_tok": 0.40},
    )
    record = asyncio.run(session.run(window_ts))
    assert record.entry_side == "up"  # sanity: entry did happen
    assert record.resolution_close_price_api == pytest.approx(close_p)
    assert record.resolution_open_price_api == pytest.approx(open_p)


# ---------------------------------------------------------------------------
# Tests 47-52 : PTB hardening — SSR context anchor, delta guard, retry
# ---------------------------------------------------------------------------

def test_ssr_extracts_open_price_with_slug_anchor():
    """fetch_ptb_ssr returns openPrice when slug anchor is present in HTML."""
    wts = 1_746_000_300
    html = _ssr_html(wts, "84500.0")
    http = _make_http({"/event/btc-updown-5m-": {"status": 200, "text": html}})
    val = asyncio.run(fetch_ptb_ssr(http, wts, "https://fake"))
    assert val == pytest.approx(84500.0)


def test_ssr_rejects_open_price_without_anchor():
    """fetch_ptb_ssr returns None when openPrice exists but no slug/ISO anchor present."""
    # HTML contains openPrice but NOT the expected slug — global match rejected
    html = '{"openPrice": "84500.0", "someOtherMarket": true}' + " " * 500
    http = _make_http({"/event/btc-updown-5m-": {"status": 200, "text": html}})
    val = asyncio.run(fetch_ptb_ssr(http, 1_746_000_300, "https://fake"))
    assert val is None


def test_ssr_rejects_short_html():
    """fetch_ptb_ssr returns None when HTML is below minimum length."""
    html = '<html><body>{"openPrice": "84000.0"}</body></html>'  # < 500 chars
    assert len(html) < 500
    http = _make_http({"/event/btc-updown-5m-": {"status": 200, "text": html}})
    val = asyncio.run(fetch_ptb_ssr(http, 1_000_000, "https://fake"))
    assert val is None


def test_ptb_robust_rejects_ssr_on_large_delta():
    """When |ssr - api| > threshold, API is selected and ptb_ssr_rejected_for_delta=True."""
    wts = 1_746_000_000
    ssr_price = 84000.0
    api_price = 85000.0   # delta = 1000 >> 10.0
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 200, "text": _ssr_html(wts, str(ssr_price))},
        "/api/crypto/crypto-price": {"status": 200, "json": {"openPrice": str(api_price)}},
    })
    r = asyncio.run(
        fetch_ptb_robust(http, wts, polymarket_base_url="https://fake",
                         ptb_max_attempts=1, ptb_retry_delay_s=0.0,
                         ptb_max_ssr_api_delta_usd=10.0)
    )
    assert r.source == "api"
    assert r.ptb == pytest.approx(api_price)
    assert r.ptb_ssr_rejected_for_delta is True
    assert r.ptb_ssr_valid is True
    assert r.ptb_api_valid is True


def test_ptb_robust_uses_api_when_ssr_absent():
    """When SSR is absent (404), API is selected without delta check."""
    wts = 1_746_000_000
    api_price = 84750.0
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 200, "json": {"openPrice": str(api_price)}},
    })
    r = asyncio.run(
        fetch_ptb_robust(http, wts, polymarket_base_url="https://fake",
                         ptb_max_attempts=1, ptb_retry_delay_s=0.0)
    )
    assert r.source == "api"
    assert r.ptb == pytest.approx(api_price)
    assert r.ptb_ssr_valid is False
    assert r.ptb_api_valid is True
    assert r.ptb_ssr_rejected_for_delta is False


def test_ptb_robust_chainlink_last_resort():
    """When SSR and API both fail, Chainlink is used as last resort."""
    wts = 1_746_000_000
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 500},
    })
    r = asyncio.run(
        fetch_ptb_robust(
            http, wts,
            chainlink_price=84800.0, chainlink_age_s=10.0,
            polymarket_base_url="https://fake",
            ptb_max_attempts=1, ptb_retry_delay_s=0.0,
        )
    )
    assert r.source == "chainlink"
    assert r.ptb == pytest.approx(84800.0)
    assert r.ptb_ssr_valid is False
    assert r.ptb_api_valid is False


def test_ptb_robust_skip_window_if_all_sources_fail():
    """When all sources fail after all retries, ptb=None → session aborts."""
    window_ts = 1_000_000
    wts = window_ts
    http = _make_http({
        "/event/btc-updown-5m-": {"status": 404},
        "/api/crypto/crypto-price": {"status": 500},
    })
    cfg = M5Config(
        ptb_max_attempts=2,
        ptb_retry_delay_s=0.0,
    )
    times = iter([wts + 15.1, wts + 15.2, wts + 15.3, wts + 15.4])
    session = _make_session(
        signal=M5SignalState(btc_price=85000.0),
        http=http,
        cfg=cfg,
        time_fn=lambda: next(times, wts + 20.0),
        prefetched_tokens=("up_tok", "dn_tok"),
    )
    record = asyncio.run(session.run(window_ts))
    assert record.abort_reason == "ptb_unavailable"
    assert record.ptb_attempts == 2
    assert record.ptb is None
