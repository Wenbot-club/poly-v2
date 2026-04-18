"""
Tests for bot/live_readonly.py — deterministic, no network.

Fakes:
  FakeDiscoveryProvider  — returns a fixed MarketContext
  FakeMarketDataProvider — yields a configurable sequence of normalized messages

The fake provider starts with feed_state="connecting", sets "live" in connect(),
and sets "disconnected" in close() — mirroring the real provider's contract.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List

import pytest

from bot.domain import MarketContext, ClobMarketInfo, ClobToken
from bot.live_readonly import LiveReadonlySession, LiveReadonlySummary
from bot.settings import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_mock_market() -> MarketContext:
    return MarketContext(
        market_id="mkt-btc-15m-demo",
        condition_id="0xbtc15mdemo",
        title="Bitcoin Up or Down - Demo 15m",
        slug="bitcoin-up-or-down-demo-15m",
        start_ts_ms=1_765_000_800_000,
        end_ts_ms=1_765_001_700_000,
        yes_token_id="YES_TOKEN",
        no_token_id="NO_TOKEN",
        clob=ClobMarketInfo(
            tokens=[
                ClobToken(token_id="YES_TOKEN", outcome="Yes"),
                ClobToken(token_id="NO_TOKEN", outcome="No"),
            ],
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


def _book_msg(token_id: str, ts: int = 1_765_000_800_100) -> Dict[str, Any]:
    """Normalized book snapshot for one token."""
    return {
        "event_type": "book",
        "asset_id": token_id,
        "timestamp": ts,
        "bids": [{"price": 0.48, "size": 30.0}, {"price": 0.47, "size": 50.0}],
        "asks": [{"price": 0.52, "size": 25.0}, {"price": 0.53, "size": 40.0}],
    }


def _price_change_msg(token_id: str, ts: int = 1_765_000_800_200) -> Dict[str, Any]:
    """Normalized price_change update for one token."""
    return {
        "event_type": "price_change",
        "timestamp": ts,
        "price_changes": [
            {"asset_id": token_id, "price": 0.49, "side": "BUY", "size": 5.0,
             "best_bid": 0.49, "best_ask": 0.52},
        ],
    }


def _last_trade_msg(token_id: str, ts: int = 1_765_000_800_300) -> Dict[str, Any]:
    return {
        "event_type": "last_trade_price",
        "asset_id": token_id,
        "price": 0.50,
        "side": "BUY",
        "timestamp": ts,
    }


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDiscoveryProvider:
    async def find_active_btc_15m_market(self) -> MarketContext:
        return _make_mock_market()


class FakeMarketDataProvider:
    """
    Yields a fixed sequence of pre-normalized messages.
    feed_state mirrors the real provider contract:
      connecting → live (on connect())
      live → disconnected (on close())
    """

    def __init__(self, messages: List[Dict[str, Any]]) -> None:
        self._messages = messages
        self.feed_state: str = "connecting"
        self.connected_token_ids: List[str] = []

    async def connect(self, token_ids: List[str]) -> None:
        self.connected_token_ids = list(token_ids)
        self.feed_state = "live"

    async def close(self) -> None:
        self.feed_state = "disconnected"

    async def iter_messages(self) -> AsyncIterator[Dict[str, Any]]:
        for msg in self._messages:
            yield msg


def _make_session(messages: List[Dict[str, Any]]) -> LiveReadonlySession:
    return LiveReadonlySession(
        discovery=FakeDiscoveryProvider(),
        provider=FakeMarketDataProvider(messages),
        config=DEFAULT_CONFIG,
    )


# ---------------------------------------------------------------------------
# Tests: counters
# ---------------------------------------------------------------------------

def test_run_for_no_messages():
    session = _make_session([])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.total_messages == 0
    assert summary.book_count == 0
    assert summary.price_change_count == 0
    assert summary.other_count == 0
    assert summary.yes_snapshotted is False
    assert summary.no_snapshotted is False


def test_run_for_book_snapshot_counted():
    session = _make_session([_book_msg("YES_TOKEN")])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.book_count == 1
    assert summary.total_messages == 1
    assert summary.yes_snapshotted is True
    assert summary.no_snapshotted is False


def test_run_for_both_tokens_snapshotted():
    session = _make_session([_book_msg("YES_TOKEN"), _book_msg("NO_TOKEN")])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.yes_snapshotted is True
    assert summary.no_snapshotted is True
    assert summary.book_count == 2


def test_run_for_counters_all_types():
    msgs = [
        _book_msg("YES_TOKEN"),
        _price_change_msg("YES_TOKEN"),
        _last_trade_msg("YES_TOKEN"),
    ]
    session = _make_session(msgs)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.total_messages == 3
    assert summary.book_count == 1
    assert summary.price_change_count == 1
    assert summary.other_count == 1


def test_run_for_market_ids_in_summary():
    session = _make_session([])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.market_id == "mkt-btc-15m-demo"
    assert summary.yes_token_id == "YES_TOKEN"
    assert summary.no_token_id == "NO_TOKEN"


# ---------------------------------------------------------------------------
# Tests: state access and router application
# ---------------------------------------------------------------------------

def test_run_for_state_accessible_after_run():
    session = _make_session([_book_msg("YES_TOKEN")])
    asyncio.run(session.run_for(duration=5))
    assert session.state is not None


def test_run_for_applies_book_to_state():
    session = _make_session([_book_msg("YES_TOKEN")])
    asyncio.run(session.run_for(duration=5))
    state = session.state
    assert state is not None
    # Router applied the book — yes_book should have the bids/asks
    assert 0.48 in state.yes_book.bids
    assert 0.52 in state.yes_book.asks


def test_run_for_state_none_before_run():
    session = _make_session([])
    assert session.state is None


# ---------------------------------------------------------------------------
# Tests: _process_message in isolation
# ---------------------------------------------------------------------------

def test_process_message_returns_event_type():
    session = _make_session([])
    # Manually create state so we can call _process_message directly
    market = _make_mock_market()
    from bot.state import StateFactory
    state = StateFactory(DEFAULT_CONFIG).create(market)

    event_type = session._process_message(state, _book_msg("YES_TOKEN"))
    assert event_type == "book"

    event_type = session._process_message(state, _price_change_msg("YES_TOKEN"))
    assert event_type == "price_change"

    event_type = session._process_message(state, _last_trade_msg("YES_TOKEN"))
    assert event_type == "last_trade_price"


def test_process_message_unknown_returns_unknown():
    session = _make_session([])
    market = _make_mock_market()
    from bot.state import StateFactory
    state = StateFactory(DEFAULT_CONFIG).create(market)
    # Router logs WARN for unknown events — _process_message just returns the type
    event_type = session._process_message(state, {"event_type": "heartbeat"})
    assert event_type == "heartbeat"


# ---------------------------------------------------------------------------
# Tests: feed_state transitions
# ---------------------------------------------------------------------------

def test_run_for_records_connecting_to_live_transition():
    session = _make_session([])
    summary = asyncio.run(session.run_for(duration=5))
    # Fake: connecting → live (in connect()), then live → disconnected (in close())
    assert ("connecting", "live") in summary.feed_state_transitions


def test_run_for_records_final_disconnected_transition():
    """Terminal transition to disconnected must always be recorded."""
    session = _make_session([_book_msg("YES_TOKEN")])
    summary = asyncio.run(session.run_for(duration=5))
    assert ("live", "disconnected") in summary.feed_state_transitions
    assert summary.final_feed_state == "disconnected"


def test_run_for_final_feed_state_is_disconnected():
    session = _make_session([])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.final_feed_state == "disconnected"


def test_run_for_transitions_complete_sequence():
    """Full transition sequence: connecting → live → disconnected."""
    session = _make_session([_book_msg("YES_TOKEN")])
    summary = asyncio.run(session.run_for(duration=5))
    states_seen = [t[0] for t in summary.feed_state_transitions] + [summary.final_feed_state]
    assert "connecting" in states_seen
    assert "live" in states_seen
    assert "disconnected" in states_seen


# ---------------------------------------------------------------------------
# Tests: timing
# ---------------------------------------------------------------------------

def test_run_for_timestamps_set():
    session = _make_session([])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.started_at_ms > 0
    assert summary.ended_at_ms >= summary.started_at_ms


def test_run_for_provider_exhausted_before_timeout_returns_normally():
    """
    If provider exhausts before duration, run_for() returns a valid summary —
    not an exception. This is the core contract for fake-provider tests.
    """
    session = _make_session([_book_msg("YES_TOKEN"), _book_msg("NO_TOKEN")])
    # duration=60 but provider only has 2 messages — should finish immediately
    summary = asyncio.run(session.run_for(duration=60))
    assert isinstance(summary, LiveReadonlySummary)
    assert summary.book_count == 2
