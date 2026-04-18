"""Tests for bot/live_combined.py — deterministic, no network."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from bot.domain import ClobMarketInfo, ClobToken, MarketContext
from bot.live_combined import LiveCombinedSession, LiveCombinedSummary
from bot.settings import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Shared helpers
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
    return {
        "event_type": "book",
        "asset_id": token_id,
        "timestamp": ts,
        "bids": [{"price": 0.48, "size": 30.0}],
        "asks": [{"price": 0.52, "size": 25.0}],
    }


def _binance_tick(value: float, seq: int, ts: int = 1_765_000_800_200) -> Dict[str, Any]:
    return {
        "source": "binance",
        "symbol": "btc/usd",
        "timestamp_ms": ts,
        "recv_timestamp_ms": ts + 50,
        "value": value,
        "sequence_no": seq,
    }


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDiscoveryProvider:
    def __init__(self) -> None:
        self.call_count = 0

    async def find_active_btc_15m_market(self) -> MarketContext:
        self.call_count += 1
        return _make_mock_market()


class FakeMarketDataProvider:
    def __init__(self, messages: List[Dict[str, Any]]) -> None:
        self._messages = messages
        self.feed_state: str = "connecting"

    async def connect(self, token_ids: List[str]) -> None:
        self.feed_state = "live"

    async def close(self) -> None:
        self.feed_state = "disconnected"

    async def iter_messages(self) -> AsyncIterator[Dict[str, Any]]:
        for msg in self._messages:
            yield msg


class FakeSignalProvider:
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


def _make_session(
    market_messages: List[Dict[str, Any]],
    rtds_ticks: List[Dict[str, Any]],
    discovery: Optional[FakeDiscoveryProvider] = None,
) -> LiveCombinedSession:
    return LiveCombinedSession(
        discovery=discovery or FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProvider(market_messages),
        signal_provider=FakeSignalProvider(rtds_ticks),
        config=DEFAULT_CONFIG,
    )


# ---------------------------------------------------------------------------
# Core invariant: discovery called exactly once
# ---------------------------------------------------------------------------

def test_discovery_called_once_in_combined_session():
    """
    Discovery must run exactly once regardless of how many sub-sessions exist.
    This locks the architectural invariant: one market context, one shared state.
    """
    discovery = FakeDiscoveryProvider()
    session = LiveCombinedSession(
        discovery=discovery,
        market_provider=FakeMarketDataProvider([]),
        signal_provider=FakeSignalProvider([]),
    )
    asyncio.run(session.run_for(duration=5))
    assert discovery.call_count == 1


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

def test_run_for_shared_state_populated_by_both_feeds():
    """Market and RTDS both write to the same LocalState instance."""
    market_msgs = [_book_msg("YES_TOKEN")]
    rtds_ticks = [_binance_tick(42500.0, seq=1)]
    session = _make_session(market_msgs, rtds_ticks)
    asyncio.run(session.run_for(duration=5))
    state = session.state
    assert state is not None
    assert 0.48 in state.yes_book.bids          # market loop wrote this
    assert len(state.binance_ticks) == 1         # RTDS loop wrote this
    assert state.last_binance.value == 42500.0   # RTDS loop wrote this


def test_run_for_state_is_single_instance():
    """session.state is the same object used by both sub-sessions (not a copy)."""
    session = _make_session([_book_msg("YES_TOKEN")], [_binance_tick(42000.0, seq=1)])
    asyncio.run(session.run_for(duration=5))
    state = session.state
    # Both feeds populated the same state
    assert 0.48 in state.yes_book.bids
    assert state.last_binance is not None


def test_state_none_before_run():
    session = _make_session([], [])
    assert session.state is None


def test_state_accessible_after_run():
    session = _make_session([], [])
    asyncio.run(session.run_for(duration=5))
    assert session.state is not None


# ---------------------------------------------------------------------------
# Market side summary
# ---------------------------------------------------------------------------

def test_run_for_market_book_count():
    session = _make_session([_book_msg("YES_TOKEN"), _book_msg("NO_TOKEN")], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.market.book_count == 2


def test_run_for_market_yes_snapshotted():
    session = _make_session([_book_msg("YES_TOKEN")], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.market.yes_snapshotted is True
    assert summary.market.no_snapshotted is False


def test_run_for_market_empty():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.market.total_messages == 0
    assert summary.market.yes_snapshotted is False


def test_run_for_market_ids_correct():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.market.market_id == "mkt-btc-15m-demo"
    assert summary.market.yes_token_id == "YES_TOKEN"
    assert summary.market.no_token_id == "NO_TOKEN"


# ---------------------------------------------------------------------------
# RTDS side summary
# ---------------------------------------------------------------------------

def test_run_for_rtds_tick_count():
    ticks = [_binance_tick(float(i), seq=i) for i in range(1, 4)]
    session = _make_session([], ticks)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.rtds.total_ticks == 3


def test_run_for_rtds_min_max_values():
    ticks = [
        _binance_tick(42000.0, seq=1),
        _binance_tick(43000.0, seq=2),
        _binance_tick(41500.0, seq=3),
    ]
    session = _make_session([], ticks)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.rtds.min_value == 41500.0
    assert summary.rtds.max_value == 43000.0


def test_run_for_rtds_empty():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.rtds.total_ticks == 0
    assert summary.rtds.first_value is None
    assert summary.rtds.min_value is None
    assert summary.rtds.max_value is None


# ---------------------------------------------------------------------------
# Feed states
# ---------------------------------------------------------------------------

def test_run_for_both_final_feed_states_disconnected():
    session = _make_session([_book_msg("YES_TOKEN")], [_binance_tick(42500.0, seq=1)])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.market.final_feed_state == "disconnected"
    assert summary.rtds.final_feed_state == "disconnected"


# ---------------------------------------------------------------------------
# Return type and early-exit contract
# ---------------------------------------------------------------------------

def test_run_for_returns_combined_summary_type():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert isinstance(summary, LiveCombinedSummary)


def test_run_for_provider_exhausted_before_timeout_returns_normally():
    """Both providers exhaust before duration — combined summary returned, no exception."""
    market_msgs = [_book_msg("YES_TOKEN"), _book_msg("NO_TOKEN")]
    rtds_ticks = [_binance_tick(42000.0, seq=i) for i in range(1, 4)]
    session = _make_session(market_msgs, rtds_ticks)
    summary = asyncio.run(session.run_for(duration=60))
    assert isinstance(summary, LiveCombinedSummary)
    assert summary.market.book_count == 2
    assert summary.rtds.total_ticks == 3
