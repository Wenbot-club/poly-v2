"""Tests for bot/live_rtds.py — deterministic, no network."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from bot.live_rtds import LiveRTDSSession, LiveRTDSSummary


# ---------------------------------------------------------------------------
# Fake signal provider
# ---------------------------------------------------------------------------

class FakeSignalProvider:
    """
    Yields a fixed sequence of pre-normalized RTDS ticks.
    feed_state mirrors the real provider contract:
      connecting → live (on connect())
      live → disconnected (on close())
    """

    def __init__(self, ticks: List[Dict[str, Any]]) -> None:
        self._ticks = ticks
        self.feed_state: str = "connecting"
        self.connected_symbol: Optional[str] = None

    async def connect(self, symbol: str) -> None:
        self.connected_symbol = symbol
        self.feed_state = "live"

    async def close(self) -> None:
        self.feed_state = "disconnected"

    async def iter_signals(self) -> AsyncIterator[Dict[str, Any]]:
        for tick in self._ticks:
            yield tick


# ---------------------------------------------------------------------------
# Shared tick builders
# ---------------------------------------------------------------------------

def _binance_tick(value: float, seq: int, ts: int = 1_700_000_000_000) -> Dict[str, Any]:
    return {
        "source": "binance",
        "symbol": "btc/usd",
        "timestamp_ms": ts,
        "recv_timestamp_ms": ts + 50,
        "value": value,
        "sequence_no": seq,
    }


def _make_session(ticks: List[Dict[str, Any]]) -> LiveRTDSSession:
    return LiveRTDSSession(
        signal_provider=FakeSignalProvider(ticks),
    )


# ---------------------------------------------------------------------------
# Tests: no ticks (empty run) — all metrics must be None
# ---------------------------------------------------------------------------

def test_run_for_no_ticks_returns_none_metrics():
    session = _make_session([])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.total_ticks == 0
    assert summary.first_value is None
    assert summary.last_value is None
    assert summary.min_value is None
    assert summary.max_value is None


def test_run_for_no_ticks_final_state_disconnected():
    session = _make_session([])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.final_feed_state == "disconnected"


# ---------------------------------------------------------------------------
# Tests: counters and value tracking
# ---------------------------------------------------------------------------

def test_run_for_single_tick_counted():
    session = _make_session([_binance_tick(42500.0, seq=1)])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.total_ticks == 1


def test_run_for_single_tick_first_last_equal():
    session = _make_session([_binance_tick(42500.0, seq=1)])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.first_value == 42500.0
    assert summary.last_value == 42500.0


def test_run_for_single_tick_min_max_equal():
    session = _make_session([_binance_tick(42500.0, seq=1)])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.min_value == 42500.0
    assert summary.max_value == 42500.0


def test_run_for_multiple_ticks_first_last_correct():
    ticks = [
        _binance_tick(42000.0, seq=1),
        _binance_tick(43000.0, seq=2),
        _binance_tick(41500.0, seq=3),
    ]
    session = _make_session(ticks)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.first_value == 42000.0
    assert summary.last_value == 41500.0


def test_run_for_multiple_ticks_min_max_correct():
    ticks = [
        _binance_tick(42000.0, seq=1),
        _binance_tick(43000.0, seq=2),
        _binance_tick(41500.0, seq=3),
    ]
    session = _make_session(ticks)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.min_value == 41500.0
    assert summary.max_value == 43000.0


def test_run_for_total_ticks_correct():
    ticks = [_binance_tick(float(i), seq=i) for i in range(1, 6)]
    session = _make_session(ticks)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.total_ticks == 5


# ---------------------------------------------------------------------------
# Tests: summary metadata
# ---------------------------------------------------------------------------

def test_run_for_symbol_and_source_in_summary():
    session = _make_session([])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.symbol == "btc/usd"
    assert summary.source == "binance"


def test_run_for_timestamps_set():
    session = _make_session([])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.started_at_ms > 0
    assert summary.ended_at_ms >= summary.started_at_ms


# ---------------------------------------------------------------------------
# Tests: state access
# ---------------------------------------------------------------------------

def test_state_none_before_run():
    session = _make_session([])
    assert session.state is None


def test_state_accessible_after_run():
    session = _make_session([_binance_tick(42500.0, seq=1)])
    asyncio.run(session.run_for(duration=5))
    assert session.state is not None


def test_state_binance_ticks_populated():
    ticks = [_binance_tick(42500.0, seq=i) for i in range(1, 4)]
    session = _make_session(ticks)
    asyncio.run(session.run_for(duration=5))
    assert len(session.state.binance_ticks) == 3


def test_state_last_binance_set():
    session = _make_session([_binance_tick(42500.0, seq=1)])
    asyncio.run(session.run_for(duration=5))
    assert session.state.last_binance is not None
    assert session.state.last_binance.value == 42500.0


def test_external_state_used_when_provided():
    """State passed to constructor is used and populated, not replaced."""
    from bot.live_rtds import _make_rtds_standalone_state
    external_state = _make_rtds_standalone_state()
    session = LiveRTDSSession(
        signal_provider=FakeSignalProvider([_binance_tick(42500.0, seq=1)]),
        state=external_state,
    )
    asyncio.run(session.run_for(duration=5))
    assert session.state is external_state
    assert len(external_state.binance_ticks) == 1


# ---------------------------------------------------------------------------
# Tests: feed_state transitions
# ---------------------------------------------------------------------------

def test_run_for_records_connecting_to_live_transition():
    session = _make_session([])
    summary = asyncio.run(session.run_for(duration=5))
    assert ("connecting", "live") in summary.feed_state_transitions


def test_run_for_records_final_disconnected_transition():
    """Terminal transition to disconnected must always be recorded."""
    session = _make_session([_binance_tick(42500.0, seq=1)])
    summary = asyncio.run(session.run_for(duration=5))
    assert ("live", "disconnected") in summary.feed_state_transitions
    assert summary.final_feed_state == "disconnected"


def test_run_for_final_feed_state_always_disconnected():
    for tick_count in (0, 1, 3):
        ticks = [_binance_tick(float(i), seq=i) for i in range(tick_count)]
        session = _make_session(ticks)
        summary = asyncio.run(session.run_for(duration=5))
        assert summary.final_feed_state == "disconnected", f"failed for tick_count={tick_count}"


# ---------------------------------------------------------------------------
# Tests: provider-exhausted-before-timeout
# ---------------------------------------------------------------------------

def test_run_for_provider_exhausted_before_timeout_returns_normally():
    """
    Provider with finite ticks exhausts before duration — returns valid summary,
    no exception, no hang.
    """
    ticks = [_binance_tick(float(i), seq=i) for i in range(1, 4)]
    session = _make_session(ticks)
    summary = asyncio.run(session.run_for(duration=60))  # long timeout, finishes early
    assert isinstance(summary, LiveRTDSSummary)
    assert summary.total_ticks == 3


# ---------------------------------------------------------------------------
# Tests: connect passes symbol
# ---------------------------------------------------------------------------

def test_connect_passes_symbol_to_provider():
    provider = FakeSignalProvider([])
    session = LiveRTDSSession(signal_provider=provider, symbol="btc/usd")
    asyncio.run(session.run_for(duration=5))
    assert provider.connected_symbol == "btc/usd"
