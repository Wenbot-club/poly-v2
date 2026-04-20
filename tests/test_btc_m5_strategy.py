"""
BTC M5 strategy tests — 16 groups matching the spec.

Tests 1-3   : PTB fetching (SSR, API, Chainlink fallback)
Tests 4-6   : EARLY consensus
Tests 7-8   : baseline direction
Tests 9-10  : hedge trigger
Test  11    : no second hedge
Test  12    : hedge blocked by cutoff
Test  13    : price_insane guard
Tests 14-15 : settlement P&L
Test  16    : aggregate summary
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
    fetch_ptb,
    fetch_ptb_ssr,
    fetch_ptb_api,
)
from bot.m5_summary import TradeRecord, aggregate_trades
from bot.settings import M5Config
from bot.strategy.btc_m5 import (
    compute_consensus,
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
) -> M5Session:
    s = M5Session(
        http_session=http or _make_http({}),
        signal_state=signal or M5SignalState(),
        config=cfg or M5Config(),
        time_fn=time_fn or (lambda: 0.0),
    )
    if token_prices:
        s._token_prices = token_prices
    return s


# ---------------------------------------------------------------------------
# Tests 1-3: PTB fetching
# ---------------------------------------------------------------------------

def test_ptb_ssr_ok():
    """SSR page contains openPrice — parsed correctly."""
    html = '<script id="__NEXT_DATA__">{"openPrice": "84123.45"}</script>'
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
# Tests 4-6: EARLY consensus
# ---------------------------------------------------------------------------

def test_consensus_early_up_triggered():
    """Strong UP consensus (score >= 88) → direction='up'."""
    result = compute_consensus(
        btc=85_100.0, ptb=85_000.0,       # +1
        chainlink=85_050.0,               # +1
        btc_10s=84_900.0,                 # btc > 10s ago → +1
        btc_30s=84_800.0,                 # +1
        btc_60s=84_700.0,                 # +1
        price_up=0.65,                    # > 0.55 → +1
        # gap = +100 > 5 → +1, > 20 → +1
        threshold=88.0, min_non_neutral=3,
    )
    assert result.direction == "up"
    assert result.score >= 88.0
    assert result.non_neutral >= 3


def test_consensus_early_down_triggered():
    """Strong DOWN consensus (score <= 12) → direction='down'."""
    result = compute_consensus(
        btc=84_900.0, ptb=85_000.0,       # -1
        chainlink=84_950.0,               # -1
        btc_10s=85_100.0,                 # btc < 10s ago → -1
        btc_30s=85_200.0,                 # -1
        btc_60s=85_300.0,                 # -1
        price_up=0.30,                    # < 0.45 → -1
        # gap = -100 < -5 → -1, < -20 → -1
        threshold=88.0, min_non_neutral=3,
    )
    assert result.direction == "down"
    assert result.score <= 12.0
    assert result.non_neutral >= 3


def test_consensus_no_entry_if_non_neutral_below_minimum():
    """Fewer than 3 non-neutral votes → direction=None."""
    result = compute_consensus(
        btc=85_000.0, ptb=85_000.0,       # 0 (equal)
        chainlink=85_000.0,               # 0
        btc_10s=85_000.0,                 # 0
        btc_30s=None,                     # 0
        btc_60s=None,                     # 0
        price_up=0.50,                    # 0 (neutral zone)
        # gap = 0 → 0, 0
        threshold=88.0, min_non_neutral=3,
    )
    assert result.direction is None
    assert result.reason == "insufficient_non_neutral"
    assert result.non_neutral < 3


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
