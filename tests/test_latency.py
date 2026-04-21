"""
Latency module tests — 10 tests.

Tests 1-4 : LatencyRecord derived properties
Tests 5-7 : LatencyTracker summary stats (avg / p50 / p95 / max)
Tests 8-9 : latency timestamps populated in M5Session (entry + hedge)
Test  10  : WS market message → price_cache via _apply_market_message
"""
from __future__ import annotations

import pytest

from bot.latency import LatencyRecord, LatencyTracker
from demos.demo_btc_m5_live import _apply_market_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec(action="leg1_entry", tick=1000, decision=1020, submit=1021, ack=None):
    return LatencyRecord(
        window_ts=1_746_000_000,
        action=action,
        tick_received_ts_ms=tick,
        decision_ts_ms=decision,
        submit_ts_ms=submit,
        ack_ts_ms=ack,
    )


# ---------------------------------------------------------------------------
# Tests 1-4: LatencyRecord properties
# ---------------------------------------------------------------------------

def test_decision_latency_ms():
    r = _rec(tick=1000, decision=1025)
    assert r.decision_latency_ms == 25


def test_submit_latency_ms():
    r = _rec(decision=1025, submit=1027)
    assert r.submit_latency_ms == 2


def test_end_to_end_no_ack():
    """Without ACK, end-to-end = submit - tick."""
    r = _rec(tick=1000, decision=1025, submit=1027, ack=None)
    assert r.end_to_end_latency_ms == 27


def test_end_to_end_with_ack():
    """With ACK, end-to-end = ack - tick."""
    r = _rec(tick=1000, decision=1025, submit=1027, ack=1090)
    assert r.end_to_end_latency_ms == 90
    assert r.ack_latency_ms == 63


# ---------------------------------------------------------------------------
# Tests 5-7: LatencyTracker summary
# ---------------------------------------------------------------------------

def test_tracker_empty_summary():
    t = LatencyTracker()
    assert t.summary()["orders_attempted"] == 0


def test_tracker_single_record_summary():
    t = LatencyTracker()
    t.add(_rec(tick=1000, decision=1020, submit=1022))
    s = t.summary()
    assert s["orders_attempted"] == 1
    assert s["avg_decision_latency_ms"] == 20.0
    assert s["avg_submit_latency_ms"] == 2.0
    assert s["avg_end_to_end_ms"] == 22.0
    assert s["max_end_to_end_ms"] == 22


def test_tracker_percentiles():
    """p50 and p95 are correct over a known distribution."""
    t = LatencyTracker()
    # decision latencies: 10, 20, 30, 40, 50 ms
    for d_ms in [10, 20, 30, 40, 50]:
        t.add(_rec(tick=1000, decision=1000 + d_ms, submit=1000 + d_ms + 1))
    s = t.summary()
    # p50 index = int(5 * 50/100) = 2 → sorted[2] = 30
    assert s["p50_decision_latency_ms"] == 30.0
    # p95 index = int(5 * 95/100) = 4 → sorted[4] = 50
    assert s["p95_decision_latency_ms"] == 50.0


def test_tracker_records_property():
    t = LatencyTracker()
    r = _rec()
    t.add(r)
    assert len(t.records) == 1
    assert t.records[0].action == "leg1_entry"


# ---------------------------------------------------------------------------
# Tests 8-9: M5Session populates latency timestamps
# ---------------------------------------------------------------------------

def test_m5_session_entry_timestamps_populated(monkeypatch):
    """M5Session._try_early_entry records entry latency timestamps."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    from bot.m5_session import M5Session, M5SignalState, BtcHistory
    from bot.m5_summary import TradeRecord

    state = M5SignalState()
    state.btc_price = 84_100.0
    state.btc_price_ts_ms = 1_746_000_140_000  # tick timestamp

    http_mock = MagicMock()
    tick_calls = {"t": 0.0}

    def fake_time():
        tick_calls["t"] += 0.001  # 1ms increments
        return 1_746_000_140.0 + tick_calls["t"]

    session = M5Session(
        http_session=http_mock,
        signal_state=state,
        config=__import__("bot.settings", fromlist=["DEFAULT_M5_CONFIG"]).DEFAULT_M5_CONFIG,
        time_fn=fake_time,
        btc_history=BtcHistory(),
    )
    session._token_prices = {"up_id": 0.60, "dn_id": 0.42}

    from bot.strategy.btc_m5 import EntrySignal
    mock_sig = MagicMock()
    mock_sig.direction = "up"
    mock_sig.block_reason = None
    mock_sig.p_model_up = 0.75
    mock_sig.edge_up = 0.10
    mock_sig.edge_down = -0.10
    mock_sig.z_gap = 2.0
    mock_sig.sigma_to_close = 5.0

    monkeypatch.setattr(
        "bot.m5_session.compute_entry_signal",
        lambda **kwargs: mock_sig,
    )

    record = TradeRecord(window_ts=1_746_000_000)
    result = asyncio.run(session._try_early_entry(
        record, 84_000.0, "up_id", "dn_id", 1_746_000_000
    ))

    assert result is True
    assert record.entry_tick_ts_ms == 1_746_000_140_000
    assert record.entry_decision_ts_ms is not None
    assert record.entry_submit_ts_ms is not None
    assert record.entry_submit_ts_ms >= record.entry_decision_ts_ms


def test_m5_session_hedge_timestamps_populated(monkeypatch):
    """M5Session._watch_for_hedge records hedge latency timestamps."""
    import asyncio
    from unittest.mock import MagicMock, AsyncMock
    from bot.m5_session import M5Session, M5SignalState, BtcHistory
    from bot.m5_summary import TradeRecord

    state = M5SignalState()
    state.btc_price = 82_000.0  # BTC well below PTB → triggers DOWN hedge
    state.btc_price_ts_ms = 1_746_000_200_000

    call_count = {"n": 0}
    base = 1_746_000_200.0
    window_ts = 1_746_000_000

    def fake_time():
        call_count["n"] += 1
        # Advance past window end after a few calls so the loop terminates
        return base + call_count["n"] * 0.01

    session = M5Session(
        http_session=MagicMock(),
        signal_state=state,
        config=__import__("bot.settings", fromlist=["DEFAULT_M5_CONFIG"]).DEFAULT_M5_CONFIG,
        time_fn=fake_time,
        btc_history=BtcHistory(),
    )
    session._token_prices = {"up_id": 0.60, "dn_id": 0.42}

    monkeypatch.setattr(
        "bot.m5_session.should_hedge",
        lambda *args, **kwargs: True,
    )

    record = TradeRecord(window_ts=window_ts)
    record.entry_side = "up"
    window_end_s = window_ts + 300.0

    asyncio.run(session._watch_for_hedge(
        record, 84_000.0, "up_id", "dn_id", window_ts, window_end_s
    ))

    assert record.hedged
    assert record.hedge_tick_ts_ms == 1_746_000_200_000
    assert record.hedge_decision_ts_ms is not None
    assert record.hedge_submit_ts_ms is not None


# ---------------------------------------------------------------------------
# Test 10: WS price message → price_cache
# ---------------------------------------------------------------------------

def test_apply_market_message_book_event():
    """book event: best ask = min ask price."""
    cache: dict = {}
    msg = {
        "event_type": "book",
        "asset_id": "0xABC",
        "timestamp": 1_700_000_000_000,
        "bids": [{"price": 0.58, "size": 10.0}],
        "asks": [{"price": 0.61, "size": 5.0}, {"price": 0.60, "size": 8.0}],
    }
    _apply_market_message(cache, msg)
    assert cache["0xABC"] == pytest.approx(0.60)


def test_apply_market_message_price_change_event():
    """price_change event with best_ask updates cache."""
    cache: dict = {}
    msg = {
        "event_type": "price_change",
        "timestamp": 1_700_000_000_000,
        "price_changes": [
            {"asset_id": "0xDEF", "price": 0.59, "side": "BUY",
             "size": 3.0, "best_bid": 0.57, "best_ask": 0.59},
        ],
    }
    _apply_market_message(cache, msg)
    assert cache["0xDEF"] == pytest.approx(0.59)


def test_apply_market_message_best_bid_ask_event():
    """best_bid_ask event directly sets best_ask."""
    cache: dict = {}
    msg = {
        "event_type": "best_bid_ask",
        "asset_id": "0xGHI",
        "best_bid": 0.55,
        "best_ask": 0.57,
        "timestamp": 1_700_000_000_000,
    }
    _apply_market_message(cache, msg)
    assert cache["0xGHI"] == pytest.approx(0.57)
