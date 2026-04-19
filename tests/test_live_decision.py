"""Tests for bot/live_decision.py — deterministic, no network."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from bot.domain import (
    ClobMarketInfo,
    ClobToken,
    DesiredOrder,
    DesiredQuotes,
    FairValueSnapshot,
    LocalState,
    MarketContext,
)
from bot.live_decision import DecisionSnapshot, LiveDecisionSession, LiveDecisionSummary
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


def _chainlink_tick(value: float, seq: int, ts: int = 1_765_000_800_200) -> Dict[str, Any]:
    return {
        "source": "chainlink",
        "symbol": "btc/usd",
        "timestamp_ms": ts,
        "recv_timestamp_ms": ts + 50,
        "value": value,
        "sequence_no": seq,
    }


def _make_fair_snapshot(p_up: float = 0.52) -> FairValueSnapshot:
    return FairValueSnapshot(
        p_up=p_up,
        p_down=1.0 - p_up,
        z_score=0.0025,
        sigma_60=0.001,
        denom=1.0,
        lead_adj=0.0,
        micro_adj=0.0,
        imbalance=0.0,
        tape=0.0,
        chainlink_last=42000.0,
        binance_last=42000.0,
        ptb=42000.0,
        tau_s=900.0,
        timestamp_ms=1_765_000_800_200,
    )


def _make_desired_quotes(
    bid_price: float = 0.48,
    ask_price: float = 0.52,
    bid_size: float = 10.0,
    ask_size: float = 10.0,
) -> DesiredQuotes:
    return DesiredQuotes(
        bid=DesiredOrder(enabled=True, side="BUY", price=bid_price, size=bid_size, reason="test"),
        ask=DesiredOrder(enabled=True, side="SELL", price=ask_price, size=ask_size, reason="test"),
        mode="passive",
        inventory_skew=0.0,
        timestamp_ms=1_765_000_800_200,
    )


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

    async def connect(self, symbol: str) -> None:
        self.feed_state = "live"

    async def close(self) -> None:
        self.feed_state = "disconnected"

    async def iter_signals(self) -> AsyncIterator[Dict[str, Any]]:
        for tick in self._ticks:
            yield tick


class FakeFairValueEngine:
    """Returns a fixed FairValueSnapshot without requiring last_chainlink."""

    def __init__(self, snapshot: Optional[FairValueSnapshot] = None) -> None:
        self._snapshot = snapshot or _make_fair_snapshot()
        self.call_count = 0

    def compute(self, state: LocalState, now_ms: int) -> FairValueSnapshot:
        self.call_count += 1
        return self._snapshot


class FakeRaisingFairValueEngine:
    """Always raises, simulating missing Chainlink."""

    def __init__(
        self,
        message: str = "chainlink tick required before fair value computation",
    ) -> None:
        self._message = message
        self.call_count = 0

    def compute(self, state: LocalState, now_ms: int) -> FairValueSnapshot:
        self.call_count += 1
        raise RuntimeError(self._message)


class FakeStrategy:
    """Returns fixed DesiredQuotes; tracks call count."""

    def __init__(self, quotes: Optional[DesiredQuotes] = None) -> None:
        self._quotes = quotes or _make_desired_quotes()
        self.call_count = 0

    def build(self, state: LocalState, fair: FairValueSnapshot, now_ms: int) -> DesiredQuotes:
        self.call_count += 1
        return self._quotes


def _make_session(
    market_messages: List[Dict[str, Any]],
    rtds_ticks: List[Dict[str, Any]],
    strategy: Optional[FakeStrategy] = None,
    fair_engine: Optional[Any] = None,
    discovery: Optional[FakeDiscoveryProvider] = None,
    decision_poll_ms: int = 100,
) -> LiveDecisionSession:
    return LiveDecisionSession(
        discovery=discovery or FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProvider(market_messages),
        signal_provider=FakeSignalProvider(rtds_ticks),
        strategy=strategy or FakeStrategy(),
        fair_engine=fair_engine or FakeFairValueEngine(),
        config=DEFAULT_CONFIG,
        decision_poll_ms=decision_poll_ms,
    )


# ---------------------------------------------------------------------------
# Return type and state lifecycle
# ---------------------------------------------------------------------------

def test_run_for_returns_live_decision_summary_type():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert isinstance(summary, LiveDecisionSummary)


def test_state_none_before_run():
    session = _make_session([], [])
    assert session.state is None


def test_state_not_none_after_run():
    session = _make_session([], [])
    asyncio.run(session.run_for(duration=5))
    assert session.state is not None


def test_decisions_empty_before_run():
    session = _make_session([], [])
    assert session.decisions == []


# ---------------------------------------------------------------------------
# Discovery invariant
# ---------------------------------------------------------------------------

def test_discovery_called_exactly_once():
    discovery = FakeDiscoveryProvider()
    session = _make_session([], [], discovery=discovery)
    asyncio.run(session.run_for(duration=5))
    assert discovery.call_count == 1


# ---------------------------------------------------------------------------
# No decisions when no ticks
# ---------------------------------------------------------------------------

def test_no_decisions_when_no_market_or_rtds():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.decision_count == 0
    assert summary.first_decision_ts_ms is None
    assert summary.last_decision_ts_ms is None
    assert summary.last_desired_quotes is None


def test_no_decisions_and_no_skipped_when_feeds_empty():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.decision_count == 0
    assert summary.skipped_fair_value_count == 0


# ---------------------------------------------------------------------------
# Decisions fire when feeds have data
# ---------------------------------------------------------------------------

def test_decisions_stored_when_rtds_fires():
    ticks = [_binance_tick(42000.0, seq=1)]
    session = _make_session([], ticks)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.decision_count >= 1


def test_decisions_stored_when_market_fires():
    msgs = [_book_msg("YES_TOKEN", ts=1_765_000_800_100)]
    session = _make_session(msgs, [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.decision_count >= 1


def test_decisions_list_matches_summary_count():
    ticks = [_binance_tick(42000.0, seq=1)]
    session = _make_session([], ticks)
    summary = asyncio.run(session.run_for(duration=5))
    assert len(session.decisions) == summary.decision_count


# ---------------------------------------------------------------------------
# DecisionSnapshot fields
# ---------------------------------------------------------------------------

def test_decision_snapshot_trigger_rtds_when_binance_fires():
    ticks = [_binance_tick(42000.0, seq=1)]
    session = _make_session([], ticks)
    asyncio.run(session.run_for(duration=5))
    assert session.decisions
    assert session.decisions[0].trigger == "rtds"


def test_decision_snapshot_trigger_market_when_only_book_fires():
    msgs = [_book_msg("YES_TOKEN", ts=1_765_000_800_100)]
    session = _make_session(msgs, [])
    asyncio.run(session.run_for(duration=5))
    assert session.decisions
    assert session.decisions[0].trigger == "market"


def test_decision_snapshot_has_fair_and_quotes():
    ticks = [_binance_tick(42000.0, seq=1)]
    session = _make_session([], ticks)
    asyncio.run(session.run_for(duration=5))
    assert session.decisions
    snap = session.decisions[0]
    assert snap.fair is not None
    assert snap.desired_quotes is not None


def test_decision_snapshot_ptb_not_locked_without_chainlink():
    """With no chainlink ticks, PTBLocker returns locked=False, ptb_value=None."""
    ticks = [_binance_tick(42000.0, seq=1)]
    session = _make_session([], ticks)
    asyncio.run(session.run_for(duration=5))
    assert session.decisions
    snap = session.decisions[0]
    assert snap.ptb_locked is False
    assert snap.ptb_value is None


def test_strategy_called_when_fair_value_available():
    strategy = FakeStrategy()
    ticks = [_binance_tick(42000.0, seq=1)]
    session = _make_session([], ticks, strategy=strategy)
    asyncio.run(session.run_for(duration=5))
    assert strategy.call_count >= 1


def test_summary_timestamps_ordered():
    # Use multiple distinct ticks to maximise chance of >=2 decisions
    ticks = [_binance_tick(float(i), seq=i) for i in range(1, 10)]
    session = _make_session([], ticks)
    summary = asyncio.run(session.run_for(duration=5))
    if summary.decision_count >= 2:
        assert summary.last_decision_ts_ms >= summary.first_decision_ts_ms


def test_last_desired_quotes_matches_last_decision():
    ticks = [_binance_tick(42000.0, seq=1)]
    session = _make_session([], ticks)
    summary = asyncio.run(session.run_for(duration=5))
    if session.decisions:
        assert summary.last_desired_quotes is session.decisions[-1].desired_quotes


# ---------------------------------------------------------------------------
# Deduplication — unit tests on key construction (timing-independent)
# ---------------------------------------------------------------------------

def test_dedup_key_changes_when_bid_price_changes():
    fair = _make_fair_snapshot(p_up=0.52)
    q1 = _make_desired_quotes(bid_price=0.48, ask_price=0.52)
    q2 = _make_desired_quotes(bid_price=0.47, ask_price=0.52)
    k1 = LiveDecisionSession._dedup_key_from(fair, q1)
    k2 = LiveDecisionSession._dedup_key_from(fair, q2)
    assert k1 != k2


def test_dedup_key_changes_when_ask_price_changes():
    fair = _make_fair_snapshot(p_up=0.52)
    q1 = _make_desired_quotes(bid_price=0.48, ask_price=0.52)
    q2 = _make_desired_quotes(bid_price=0.48, ask_price=0.53)
    assert LiveDecisionSession._dedup_key_from(fair, q1) != LiveDecisionSession._dedup_key_from(fair, q2)


def test_dedup_key_changes_when_bid_size_changes():
    fair = _make_fair_snapshot(p_up=0.52)
    q1 = _make_desired_quotes(bid_price=0.48, ask_price=0.52, bid_size=10.0)
    q2 = _make_desired_quotes(bid_price=0.48, ask_price=0.52, bid_size=20.0)
    assert LiveDecisionSession._dedup_key_from(fair, q1) != LiveDecisionSession._dedup_key_from(fair, q2)


def test_dedup_key_changes_when_ask_size_changes():
    fair = _make_fair_snapshot(p_up=0.52)
    q1 = _make_desired_quotes(bid_price=0.48, ask_price=0.52, ask_size=10.0)
    q2 = _make_desired_quotes(bid_price=0.48, ask_price=0.52, ask_size=20.0)
    assert LiveDecisionSession._dedup_key_from(fair, q1) != LiveDecisionSession._dedup_key_from(fair, q2)


def test_dedup_key_changes_when_fair_p_up_changes():
    fair1 = _make_fair_snapshot(p_up=0.52)
    fair2 = _make_fair_snapshot(p_up=0.55)
    q = _make_desired_quotes()
    assert LiveDecisionSession._dedup_key_from(fair1, q) != LiveDecisionSession._dedup_key_from(fair2, q)


def test_dedup_key_stable_for_identical_inputs():
    fair = _make_fair_snapshot(p_up=0.52)
    q = _make_desired_quotes(bid_price=0.48, ask_price=0.52)
    assert LiveDecisionSession._dedup_key_from(fair, q) == LiveDecisionSession._dedup_key_from(fair, q)


# ---------------------------------------------------------------------------
# Deduplication — session-level (via final pass, timing-independent)
# ---------------------------------------------------------------------------

def test_dedup_suppresses_identical_snapshots():
    """Identical fair + identical quotes across all ticks → exactly 1 decision stored."""
    fixed_fair = _make_fair_snapshot(p_up=0.52)
    fixed_quotes = _make_desired_quotes(bid_price=0.48, ask_price=0.52)
    session = _make_session(
        [],
        [_binance_tick(42000.0, seq=i) for i in range(1, 6)],
        strategy=FakeStrategy(quotes=fixed_quotes),
        fair_engine=FakeFairValueEngine(snapshot=fixed_fair),
    )
    asyncio.run(session.run_for(duration=5))
    # Final pass fires once; decision loop may add one more, but all are deduped
    assert len(session.decisions) == 1


# ---------------------------------------------------------------------------
# Skipped fair value / error tracking
# ---------------------------------------------------------------------------

def test_skipped_fair_value_count_increments_when_engine_raises():
    ticks = [_binance_tick(42000.0, seq=1)]
    raising_engine = FakeRaisingFairValueEngine()
    session = _make_session([], ticks, fair_engine=raising_engine)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.skipped_fair_value_count >= 1
    assert summary.decision_count == 0


def test_last_fair_value_error_recorded():
    ticks = [_binance_tick(42000.0, seq=1)]
    raising_engine = FakeRaisingFairValueEngine(
        message="chainlink tick required before fair value computation"
    )
    session = _make_session([], ticks, fair_engine=raising_engine)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.last_fair_value_error is not None
    assert "chainlink" in summary.last_fair_value_error


def test_skipped_zero_when_engine_succeeds():
    ticks = [_binance_tick(42000.0, seq=1)]
    session = _make_session([], ticks, fair_engine=FakeFairValueEngine())
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.skipped_fair_value_count == 0
    assert summary.last_fair_value_error is None


# ---------------------------------------------------------------------------
# Sub-summaries composed
# ---------------------------------------------------------------------------

def test_summary_composes_market_and_rtds_subsummaries():
    msgs = [_book_msg("YES_TOKEN")]
    ticks = [_binance_tick(42000.0, seq=1)]
    session = _make_session(msgs, ticks)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.market.market_id == "mkt-btc-15m-demo"
    assert summary.rtds.total_ticks == 1


def test_provider_exhausted_before_timeout_returns_normally():
    msgs = [_book_msg("YES_TOKEN"), _book_msg("NO_TOKEN")]
    ticks = [_binance_tick(float(i), seq=i) for i in range(1, 4)]
    session = _make_session(msgs, ticks)
    summary = asyncio.run(session.run_for(duration=60))
    assert isinstance(summary, LiveDecisionSummary)
    assert summary.market.book_count == 2
    assert summary.rtds.total_ticks == 3


# ---------------------------------------------------------------------------
# Coinbase anchor unblocking — real FairValueEngine
# ---------------------------------------------------------------------------

def test_real_fair_value_produces_decisions_with_chainlink_anchor():
    """
    Real FairValueEngine (no fake), Binance + Chainlink ticks, valid book.
    Chainlink tick feeds last_chainlink via RTDSMessageRouter → fair value unblocked.
    decision_count >= 1, skipped_fair_value_count == 0.
    """
    from bot.fair_value import FairValueEngine

    msgs = [_book_msg("YES_TOKEN", ts=1_765_000_800_100)]
    ticks = [
        _chainlink_tick(42000.5, seq=1),  # routes to last_chainlink
        _binance_tick(42000.0, seq=1),    # routes to last_binance
    ]
    session = _make_session(
        market_messages=msgs,
        rtds_ticks=ticks,
        fair_engine=FairValueEngine(config=DEFAULT_CONFIG),
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.decision_count >= 1
    assert summary.skipped_fair_value_count == 0


def test_skipped_count_zero_with_chainlink_anchor():
    """Explicit: no skips when anchor is wired — not just decision_count > 0."""
    from bot.fair_value import FairValueEngine

    msgs = [_book_msg("YES_TOKEN", ts=1_765_000_800_100)]
    ticks = [_chainlink_tick(42000.0, seq=1), _binance_tick(42000.0, seq=1)]
    session = _make_session(
        market_messages=msgs,
        rtds_ticks=ticks,
        fair_engine=FairValueEngine(config=DEFAULT_CONFIG),
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.skipped_fair_value_count == 0
    assert summary.last_fair_value_error is None
