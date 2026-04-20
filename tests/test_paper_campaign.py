"""Tests for bot/paper_campaign.py and bot/campaign_report.py — no network."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from bot.campaign_report import (
    BucketStats,
    CampaignSummary,
    compute_campaign_summary,
)
from bot.live_paper import LivePaperSummary
from bot.live_readonly import LiveReadonlySummary
from bot.live_rtds import LiveRTDSSummary
from bot.paper_campaign import CampaignConfig, CampaignRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_market_summary(**kw) -> LiveReadonlySummary:
    defaults = dict(
        market_id="test-mkt",
        yes_token_id="YES",
        no_token_id="NO",
        total_messages=0,
        book_count=0,
        price_change_count=0,
        other_count=0,
        yes_snapshotted=False,
        no_snapshotted=False,
        feed_state_transitions=[],
        final_feed_state="disconnected",
        started_at_ms=0,
        ended_at_ms=0,
    )
    defaults.update(kw)
    return LiveReadonlySummary(**defaults)


def _make_fake_rtds_summary(**kw) -> LiveRTDSSummary:
    defaults = dict(
        symbol="btc/usd",
        source="binance",
        total_ticks=0,
        first_value=None,
        last_value=None,
        min_value=None,
        max_value=None,
        feed_state_transitions=[],
        final_feed_state="disconnected",
        started_at_ms=0,
        ended_at_ms=0,
    )
    defaults.update(kw)
    return LiveRTDSSummary(**defaults)


def _make_fake_summary(**kw) -> LivePaperSummary:
    """LivePaperSummary with zero/None defaults; override with keyword args."""
    defaults: Dict[str, Any] = dict(
        market=_make_fake_market_summary(),
        rtds=_make_fake_rtds_summary(),
        decision_count=0,
        first_decision_ts_ms=None,
        last_decision_ts_ms=None,
        skipped_fair_value_count=0,
        last_fair_value_error=None,
        orders_posted=0,
        orders_cancelled=0,
        orders_rejected=0,
        fills_simulated=0,
        filled_orders=0,
        last_rejection_reason=None,
        bid_orders_posted=0,
        ask_orders_posted=0,
        bid_fills_simulated=0,
        ask_fills_simulated=0,
        fill_rate=None,
        rejection_rate=None,
        cancel_to_post_ratio=None,
        max_up_inventory=0.0,
        max_pusd_reserved=0.0,
        first_fill_ts_ms=None,
        last_fill_ts_ms=None,
        final_pusd_free=100.0,
        final_up_free=0.0,
        final_pusd_reserved=0.0,
        portfolio_value_start=100.0,
        portfolio_value_end_mark=None,
        pnl_total_mark=None,
        pnl_unrealized_mark=None,
        realized_pnl=0.0,
        position_cost_basis=0.0,
        mark_price=None,
        mark_source="yes_book_top_bid",
        decisions_triggered_by_market=0,
        decisions_triggered_by_rtds=0,
        decisions_ptb_locked=0,
        decisions_bid_enabled=0,
        decisions_ask_enabled=0,
        avg_binance_age_ms_at_decision=None,
        max_binance_age_ms_at_decision=None,
        avg_chainlink_age_ms_at_decision=None,
        max_chainlink_age_ms_at_decision=None,
        avg_book_event_age_ms_at_decision=None,
        max_book_event_age_ms_at_decision=None,
        avg_abs_binance_chainlink_gap_at_decision=None,
        max_abs_binance_chainlink_gap_at_decision=None,
        avg_fair_minus_best_bid_at_decision=None,
        avg_best_ask_minus_fair_at_decision=None,
    )
    defaults.update(kw)
    return LivePaperSummary(**defaults)


def _make_decision_event(
    *,
    trigger: str = "market",
    binance_chainlink_gap: Optional[float] = None,
    chainlink_age_ms: Optional[int] = None,
    fair_minus_best_bid: Optional[float] = None,
    best_ask_minus_fair: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "ts_ms": 1000,
        "event": "decision",
        "trigger": trigger,
        "binance_chainlink_gap": binance_chainlink_gap,
        "chainlink_age_ms": chainlink_age_ms,
        "fair_minus_best_bid": fair_minus_best_bid,
        "best_ask_minus_fair": best_ask_minus_fair,
    }


def _default_campaign_summary(**kw) -> CampaignSummary:
    """Call compute_campaign_summary with zero sessions and empty events, override fields."""
    return compute_campaign_summary(
        session_summaries=kw.pop("session_summaries", []),
        all_events=kw.pop("all_events", []),
        session_count_requested=kw.pop("session_count_requested", 1),
        session_duration_s=kw.pop("session_duration_s", 60),
        gap_thresholds_usd=kw.pop("gap_thresholds_usd", (50.0, 200.0)),
        chainlink_age_thresholds_ms=kw.pop("chainlink_age_thresholds_ms", (1000, 5000)),
        campaign_started_at_ms=kw.pop("campaign_started_at_ms", 0),
        campaign_ended_at_ms=kw.pop("campaign_ended_at_ms", 1000),
    )


class _FakeLivePaperSession:
    """Fake LivePaperSession: returns canned summary and events instantly."""

    def __init__(self, summary: LivePaperSummary, events: Optional[List[Dict]] = None):
        self._summary = summary
        self.events: List[Dict] = events or []
        self.state = None

    async def run_for(self, duration: int) -> LivePaperSummary:
        return self._summary


class _FailingLivePaperSession:
    """Fake that raises on run_for(), simulating a mid-campaign failure."""

    def __init__(self):
        self.events: List[Dict] = []
        self.state = None

    async def run_for(self, duration: int) -> LivePaperSummary:
        raise RuntimeError("simulated session failure")


# ---------------------------------------------------------------------------
# Test 1: runner produces N session artifact pairs
# ---------------------------------------------------------------------------

def test_campaign_runner_runs_n_sessions(tmp_path: Path):
    """CampaignRunner calls session_factory N times and writes N jsonl+summary pairs."""
    n = 3
    summaries = [_make_fake_summary() for _ in range(n)]
    idx = [0]

    def factory():
        s = summaries[idx[0]]
        idx[0] += 1
        return _FakeLivePaperSession(s)

    cfg = CampaignConfig(session_count=n, session_duration_s=5, output_dir=tmp_path)
    runner = CampaignRunner()
    asyncio.run(runner.run(factory, cfg))

    for i in range(n):
        assert (tmp_path / f"session_{i:03d}.jsonl").exists()
        assert (tmp_path / f"session_{i:03d}_summary.json").exists()
    assert idx[0] == n


# ---------------------------------------------------------------------------
# Test 2: PnL aggregation
# ---------------------------------------------------------------------------

def test_campaign_summary_aggregates_pnl_correctly():
    """total_realized_pnl == sum of per-session realized_pnl values."""
    summaries = [
        _make_fake_summary(realized_pnl=1.5),
        _make_fake_summary(realized_pnl=-0.3),
        _make_fake_summary(realized_pnl=0.8),
    ]
    result = _default_campaign_summary(
        session_summaries=summaries,
        session_count_requested=3,
    )
    assert abs(result.total_realized_pnl - 2.0) < 1e-10
    assert result.session_count_completed == 3
    assert len(result.pnl_per_session) == 3
    assert result.pnl_per_session[0]["realized_pnl"] == 1.5
    assert result.pnl_per_session[1]["realized_pnl"] == -0.3
    assert result.sessions_profitable_realized == 2  # sessions with pnl > 0


# ---------------------------------------------------------------------------
# Test 3: all artifacts written on success
# ---------------------------------------------------------------------------

def test_campaign_writes_all_artifacts(tmp_path: Path):
    """Successful campaign writes session files, campaign_summary.json, campaign_manifest.json."""
    n = 2
    idx = [0]
    summaries = [_make_fake_summary() for _ in range(n)]

    def factory():
        s = summaries[idx[0]]
        idx[0] += 1
        return _FakeLivePaperSession(s)

    cfg = CampaignConfig(session_count=n, session_duration_s=5, output_dir=tmp_path)
    asyncio.run(CampaignRunner().run(factory, cfg))

    assert (tmp_path / "campaign_summary.json").exists()
    assert (tmp_path / "campaign_manifest.json").exists()

    manifest = json.loads((tmp_path / "campaign_manifest.json").read_text())
    assert "campaign_started_at_ms" in manifest
    assert "campaign_ended_at_ms" in manifest
    assert "files" in manifest
    assert "campaign_manifest.json" in manifest["files"]
    assert "campaign_summary.json" in manifest["files"]
    assert len([f for f in manifest["files"] if f.endswith(".jsonl")]) == n


# ---------------------------------------------------------------------------
# Test 4: bucket breakdown counts decisions correctly
# ---------------------------------------------------------------------------

def test_bucket_breakdown_counts_decisions_correctly():
    """Decision events are assigned to the correct gap bucket."""
    events = [
        _make_decision_event(binance_chainlink_gap=20.0),    # abs=20 → "<50"
        _make_decision_event(binance_chainlink_gap=-30.0),   # abs=30 → "<50"
        _make_decision_event(binance_chainlink_gap=100.0),   # abs=100 → "50-200"
        _make_decision_event(binance_chainlink_gap=-150.0),  # abs=150 → "50-200"
        _make_decision_event(binance_chainlink_gap=500.0),   # abs=500 → ">200"
    ]
    result = _default_campaign_summary(all_events=events, gap_thresholds_usd=(50.0, 200.0))

    assert result.by_gap_bucket["<50"].decision_count == 2
    assert result.by_gap_bucket["50-200"].decision_count == 2
    assert result.by_gap_bucket[">200"].decision_count == 1
    assert result.gap_bucket_excluded_count == 0


# ---------------------------------------------------------------------------
# Test 5: excluded count when gap is None
# ---------------------------------------------------------------------------

def test_gap_bucket_excluded_count_when_gap_missing():
    """Decisions with binance_chainlink_gap=None are counted in gap_bucket_excluded_count."""
    events = [
        # gap present → bucketed; age present → bucketed
        _make_decision_event(binance_chainlink_gap=25.0, chainlink_age_ms=500),
        # gap None → gap excluded; age present → bucketed
        _make_decision_event(binance_chainlink_gap=None, chainlink_age_ms=600),
        # gap None → gap excluded; age None → age excluded
        _make_decision_event(binance_chainlink_gap=None, chainlink_age_ms=None),
    ]
    result = _default_campaign_summary(all_events=events)

    # Events 1 and 2 have gap=None → 2 excluded from gap breakdown
    assert result.gap_bucket_excluded_count == 2
    # Event 2 has age=None → 1 excluded from age breakdown
    assert result.chainlink_age_bucket_excluded_count == 1
    # Events 0 and 1 have age=500/600 → both in "<1000" bucket
    assert result.by_chainlink_age_bucket["<1000"].decision_count == 2
    # Event 0 has gap=25.0 → in "<50" bucket
    assert result.by_gap_bucket["<50"].decision_count == 1


# ---------------------------------------------------------------------------
# Test 6: fail-fast — no campaign_summary.json when session raises
# ---------------------------------------------------------------------------

def test_campaign_runner_does_not_write_campaign_summary_on_failure(tmp_path: Path):
    """If session i raises, campaign_summary.json and manifest are NOT written."""
    call_count = [0]

    def factory():
        call_count[0] += 1
        if call_count[0] == 1:
            return _FakeLivePaperSession(_make_fake_summary())
        return _FailingLivePaperSession()

    cfg = CampaignConfig(session_count=2, session_duration_s=5, output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="simulated session failure"):
        asyncio.run(CampaignRunner().run(factory, cfg))

    # First session's artifacts are present
    assert (tmp_path / "session_000.jsonl").exists()
    assert (tmp_path / "session_000_summary.json").exists()

    # Campaign-level artifacts must NOT be written on partial failure
    assert not (tmp_path / "campaign_summary.json").exists()
    assert not (tmp_path / "campaign_manifest.json").exists()


# ---------------------------------------------------------------------------
# Test 7: window-boundary wait — sleep is called with the correct duration
# ---------------------------------------------------------------------------

def test_window_boundary_wait_sleeps_for_correct_duration(tmp_path: Path):
    """
    When a session ends far earlier than requested (window-clamped), the runner
    sleeps until window_boundary_buffer_s past the start of the new M15 window.

    Mocks:
      _time.monotonic() returns 0.0 for both t0 and t_end of session 0
        → elapsed_s = 0, shortfall = 300 >> window_early_threshold_s (10)
      _time.time() returns 905.0 → 5s into a 900s window
        → time_since_boundary = 5, wait_s = 35 - 5 = 30
    Expected: asyncio.sleep(30.0) called exactly once (not after the last session).
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    sleep_mock = AsyncMock()

    cfg = CampaignConfig(
        session_count=2,
        session_duration_s=300,
        output_dir=tmp_path,
        window_size_s=900,
        window_boundary_buffer_s=35,
        window_early_threshold_s=10,
    )
    fake_summary = _make_fake_summary()

    with patch("asyncio.sleep", sleep_mock), \
         patch("bot.paper_campaign._time") as mock_time:
        # Two sessions × two monotonic calls each (t0 and t_end).
        mock_time.monotonic.side_effect = [0.0, 0.0, 1.0, 1.0]
        mock_time.time.return_value = 905.0

        asyncio.run(
            CampaignRunner().run(
                lambda: _FakeLivePaperSession(fake_summary),
                cfg,
            )
        )

    assert sleep_mock.call_count == 1
    (wait_s,), _ = sleep_mock.call_args
    assert abs(wait_s - 30.0) < 1e-9


def test_window_boundary_wait_not_triggered_when_session_runs_full_duration(tmp_path: Path):
    """No sleep when elapsed ≈ session_duration_s (normal, non-clamped session)."""
    from unittest.mock import AsyncMock, patch

    sleep_mock = AsyncMock()

    cfg = CampaignConfig(
        session_count=1,
        session_duration_s=300,
        output_dir=tmp_path,
        window_early_threshold_s=10,
    )
    fake_summary = _make_fake_summary()

    with patch("asyncio.sleep", sleep_mock), \
         patch("bot.paper_campaign._time") as mock_time:
        # elapsed = 295s → shortfall = 5 < threshold 10 → no sleep
        mock_time.monotonic.side_effect = [0.0, 295.0]
        mock_time.time.return_value = 905.0

        asyncio.run(
            CampaignRunner().run(
                lambda: _FakeLivePaperSession(fake_summary),
                cfg,
            )
        )

    assert sleep_mock.call_count == 0


# ---------------------------------------------------------------------------
# Test 8: gate reason aggregation — by_bid_reason counts bid_reason per decision
# ---------------------------------------------------------------------------

def test_gate_reason_aggregation_counts_bid_reasons():
    """bid_reason values from decision events are tallied into by_bid_reason."""
    events = [
        {**_make_decision_event(), "bid_reason": "chainlink_stale"},
        {**_make_decision_event(), "bid_reason": "chainlink_stale"},
        {**_make_decision_event(), "bid_reason": "book_gate"},
        {**_make_decision_event(), "bid_reason": "quote_bid"},
    ]
    result = _default_campaign_summary(all_events=events)
    assert result.by_bid_reason == {
        "chainlink_stale": 2,
        "book_gate": 1,
        "quote_bid": 1,
    }


def test_gate_reason_aggregation_ignores_missing_bid_reason():
    """Events without bid_reason are excluded from by_bid_reason (no crash, no None key)."""
    events = [
        _make_decision_event(),  # no bid_reason
        {**_make_decision_event(), "bid_reason": "tau_gate"},
    ]
    result = _default_campaign_summary(all_events=events)
    assert result.by_bid_reason == {"tau_gate": 1}
    assert None not in result.by_bid_reason
