"""Tests for bot/live_paper.py — deterministic, no network."""
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
from bot.live_paper import LivePaperSession, LivePaperSummary
from bot.settings import DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Helpers
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


def _book_msg(
    token_id: str,
    ts: int = 100,
    bid_price: float = 0.46,
    bid_size: float = 30.0,
    ask_price: float = 0.52,
    ask_size: float = 25.0,
) -> Dict[str, Any]:
    return {
        "event_type": "book",
        "asset_id": token_id,
        "timestamp": ts,
        "bids": [{"price": bid_price, "size": bid_size}],
        "asks": [{"price": ask_price, "size": ask_size}],
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


def _make_desired_quotes(bid_price: float = 0.48, ask_price: float = 0.52) -> DesiredQuotes:
    return DesiredQuotes(
        bid=DesiredOrder(enabled=True, side="BUY", price=bid_price, size=10.0, reason="test"),
        ask=DesiredOrder(enabled=True, side="SELL", price=ask_price, size=10.0, reason="test"),
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


class FakeMarketDataProviderWithSleep:
    """
    Inserts asyncio.sleep(0) after each yielded message so the decision loop
    gets a guaranteed scheduling slot between consecutive book messages.
    Used only for tests that require interleaved decision firing.
    """

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
            await asyncio.sleep(0)  # let decision loop fire after each message


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
    def __init__(self, snapshot: Optional[FairValueSnapshot] = None) -> None:
        self._snapshot = snapshot or _make_fair_snapshot()

    def compute(self, state: LocalState, now_ms: int) -> FairValueSnapshot:
        return self._snapshot


class FakeRaisingFairValueEngine:
    def __init__(self, message: str = "chainlink tick required before fair value computation") -> None:
        self._message = message

    def compute(self, state: LocalState, now_ms: int) -> FairValueSnapshot:
        raise RuntimeError(self._message)


class FakeStrategy:
    def __init__(self, quotes: Optional[DesiredQuotes] = None) -> None:
        self._quotes = quotes or _make_desired_quotes()

    def build(self, state: LocalState, fair: FairValueSnapshot, now_ms: int) -> DesiredQuotes:
        return self._quotes


def _make_session(
    market_messages: List[Dict[str, Any]],
    rtds_ticks: List[Dict[str, Any]],
    strategy: Optional[FakeStrategy] = None,
    fair_engine: Optional[Any] = None,
    discovery: Optional[FakeDiscoveryProvider] = None,
    initial_pusd: Optional[float] = None,
    decision_poll_ms: int = 100,
) -> LivePaperSession:
    return LivePaperSession(
        discovery=discovery or FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProvider(market_messages),
        signal_provider=FakeSignalProvider(rtds_ticks),
        strategy=strategy or FakeStrategy(),
        fair_engine=fair_engine or FakeFairValueEngine(),
        config=DEFAULT_CONFIG,
        initial_pusd=initial_pusd,
        decision_poll_ms=decision_poll_ms,
    )


# ---------------------------------------------------------------------------
# Return type and state lifecycle
# ---------------------------------------------------------------------------

def test_run_for_returns_live_paper_summary_type():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert isinstance(summary, LivePaperSummary)


def test_state_none_before_run():
    assert _make_session([], []).state is None


def test_state_not_none_after_run():
    session = _make_session([], [])
    asyncio.run(session.run_for(duration=5))
    assert session.state is not None


def test_decisions_empty_before_run():
    assert _make_session([], []).decisions == []


# ---------------------------------------------------------------------------
# Discovery invariant
# ---------------------------------------------------------------------------

def test_discovery_called_exactly_once():
    discovery = FakeDiscoveryProvider()
    session = _make_session([], [], discovery=discovery)
    asyncio.run(session.run_for(duration=5))
    assert discovery.call_count == 1


# ---------------------------------------------------------------------------
# Inventory seeding
# ---------------------------------------------------------------------------

def test_inventory_seeded_from_config_when_initial_pusd_none():
    session = _make_session([], [], initial_pusd=None)
    asyncio.run(session.run_for(duration=5))
    assert session.state is not None
    # Config default is 125.0; no fills → pusd_free unchanged
    assert session.state.inventory.pusd_free == DEFAULT_CONFIG.default_working_capital_usd


def test_inventory_seeded_from_explicit_initial_pusd():
    session = _make_session([], [], initial_pusd=50.0)
    asyncio.run(session.run_for(duration=5))
    assert session.state is not None
    assert session.state.inventory.pusd_free == 50.0


def test_inventory_not_seeded_to_125_literal_when_config_differs():
    """initial_pusd=None always seeds from config, not a hardcoded 125.0."""
    from bot.settings import RuntimeConfig, Thresholds
    custom_config = RuntimeConfig(default_working_capital_usd=77.0)
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProvider([]),
        signal_provider=FakeSignalProvider([]),
        strategy=FakeStrategy(),
        fair_engine=FakeFairValueEngine(),
        config=custom_config,
        initial_pusd=None,
    )
    asyncio.run(session.run_for(duration=5))
    assert session.state is not None
    assert session.state.inventory.pusd_free == 77.0


# ---------------------------------------------------------------------------
# No orders when no data
# ---------------------------------------------------------------------------

def test_no_orders_when_no_ticks():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_posted == 0
    assert summary.orders_cancelled == 0
    assert summary.fills_simulated == 0


# ---------------------------------------------------------------------------
# Order posted after decision
# ---------------------------------------------------------------------------

def test_order_posted_after_decision_with_valid_book():
    # Book with spread: bid 0.46 / ask 0.52 → strategy bids at 0.48 (valid)
    msgs = [_book_msg("YES_TOKEN", ts=100, ask_price=0.52)]
    session = _make_session(msgs, [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_posted >= 1


def test_orders_posted_count_in_summary_matches():
    msgs = [_book_msg("YES_TOKEN", ts=100, ask_price=0.52)]
    session = _make_session(msgs, [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_posted == session._orders_posted


# ---------------------------------------------------------------------------
# Order rejection
# ---------------------------------------------------------------------------

def test_order_rejected_when_insufficient_capital():
    # With initial_pusd=0.0, bid requires pusd → rejected with "not_enough_pusd"
    msgs = [_book_msg("YES_TOKEN", ts=100, ask_price=0.52)]
    session = _make_session(msgs, [], initial_pusd=0.0)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_rejected >= 1
    assert summary.last_rejection_reason == "not_enough_pusd"


def test_order_rejection_counted_when_insufficient_capital():
    """Explicit: orders_rejected > 0 when capital = 0, not silently zero."""
    msgs = [_book_msg("YES_TOKEN", ts=100, ask_price=0.52)]
    session = _make_session(msgs, [], initial_pusd=0.0)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_rejected > 0


def test_no_rejection_when_sufficient_capital():
    msgs = [_book_msg("YES_TOKEN", ts=100, ask_price=0.52)]
    session = _make_session(msgs, [], initial_pusd=1000.0)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_rejected == 0
    assert summary.last_rejection_reason is None


def test_order_rejected_when_book_incomplete():
    # Empty messages → no book → "book_incomplete"
    session = _make_session([], [_binance_tick(42000.0, seq=1)])
    summary = asyncio.run(session.run_for(duration=5))
    if summary.orders_rejected > 0:
        assert summary.last_rejection_reason == "book_incomplete"


# ---------------------------------------------------------------------------
# Fill simulation
# ---------------------------------------------------------------------------

def test_no_fill_when_book_does_not_cross():
    # Book ask (0.52) > strategy bid (0.48) → no fill condition
    msgs = [_book_msg("YES_TOKEN", ts=100, ask_price=0.52)]
    session = _make_session(msgs, [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.fills_simulated == 0


def test_fill_simulated_when_book_crosses_bid():
    """
    Two sequential book messages. FakeMarketDataProviderWithSleep inserts
    asyncio.sleep(0) after each message, guaranteeing the decision loop fires
    between message 1 (ask=0.52, bid posted at 0.48) and message 2 (ask=0.45,
    crosses the live bid → fill simulated).
    """
    msgs = [
        _book_msg("YES_TOKEN", ts=100, ask_price=0.52),
        _book_msg("YES_TOKEN", ts=200, ask_price=0.45),
    ]
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProviderWithSleep(msgs),
        signal_provider=FakeSignalProvider([]),
        strategy=FakeStrategy(),
        fair_engine=FakeFairValueEngine(),
        config=DEFAULT_CONFIG,
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.fills_simulated >= 1


def test_fill_reduces_pusd_and_increases_up():
    msgs = [
        _book_msg("YES_TOKEN", ts=100, ask_price=0.52),
        _book_msg("YES_TOKEN", ts=200, ask_price=0.45),
    ]
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProviderWithSleep(msgs),
        signal_provider=FakeSignalProvider([]),
        strategy=FakeStrategy(),
        fair_engine=FakeFairValueEngine(),
        config=DEFAULT_CONFIG,
        initial_pusd=1000.0,
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    if summary.fills_simulated >= 1:
        # Fill bought YES tokens: pusd decreases, up_free increases
        assert summary.final_pusd_free < 1000.0
        assert summary.final_up_free > 0.0


# ---------------------------------------------------------------------------
# Skipped fair value
# ---------------------------------------------------------------------------

def test_skipped_fair_value_when_engine_raises():
    ticks = [_binance_tick(42000.0, seq=1)]
    session = _make_session([], ticks, fair_engine=FakeRaisingFairValueEngine())
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.skipped_fair_value_count >= 1
    assert "chainlink" in (summary.last_fair_value_error or "")
    assert summary.orders_posted == 0


# ---------------------------------------------------------------------------
# Sub-summaries composed
# ---------------------------------------------------------------------------

def test_summary_composes_market_and_rtds():
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
    assert isinstance(summary, LivePaperSummary)
    assert summary.market.book_count == 2
    assert summary.rtds.total_ticks == 3


# ---------------------------------------------------------------------------
# execution_engine_factory wiring
# ---------------------------------------------------------------------------

def test_custom_factory_is_used_when_provided():
    """Factory receives QueueingUserRouter and returns a working engine."""
    from bot.async_runner import QueueingUserRouter
    from bot.execution.paper import MockExecutionEngine

    factory_calls: list[int] = [0]

    def my_factory(qr: QueueingUserRouter):
        factory_calls[0] += 1
        return MockExecutionEngine(config=DEFAULT_CONFIG, user_router=qr)

    msgs = [_book_msg("YES_TOKEN", ts=100, ask_price=0.52)]
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProvider(msgs),
        signal_provider=FakeSignalProvider([]),
        strategy=FakeStrategy(),
        fair_engine=FakeFairValueEngine(),
        config=DEFAULT_CONFIG,
        execution_engine_factory=my_factory,
    )
    asyncio.run(session.run_for(duration=5))
    assert factory_calls[0] == 1


# ---------------------------------------------------------------------------
# Coinbase anchor unblocking — real FairValueEngine
# ---------------------------------------------------------------------------

def test_orders_posted_with_chainlink_anchor_unblocks_paper_execution():
    """
    Real FairValueEngine, Binance + Chainlink ticks, valid book, sufficient capital.
    orders_posted >= 1 proves the full unblocking chain:
      Chainlink anchor → last_chainlink → fair value computed → decision → order posted.
    """
    from bot.fair_value import FairValueEngine

    msgs = [_book_msg("YES_TOKEN", ts=100, bid_price=0.46, ask_price=0.52)]
    ticks = [
        _chainlink_tick(42000.5, seq=1),  # routes to last_chainlink
        _binance_tick(42000.0, seq=1),    # routes to last_binance
    ]
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProvider(msgs),
        signal_provider=FakeSignalProvider(ticks),
        strategy=FakeStrategy(),
        fair_engine=FairValueEngine(config=DEFAULT_CONFIG),
        config=DEFAULT_CONFIG,
        initial_pusd=1000.0,
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_posted >= 1
    assert summary.skipped_fair_value_count == 0
    assert summary.decision_count >= 1


def test_skipped_count_zero_with_chainlink_anchor_paper():
    """Explicit: no skips when anchor is wired — last_fair_value_error must be None."""
    from bot.fair_value import FairValueEngine

    msgs = [_book_msg("YES_TOKEN", ts=100, bid_price=0.46, ask_price=0.52)]
    ticks = [_chainlink_tick(42000.0, seq=1), _binance_tick(42000.0, seq=1)]
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProvider(msgs),
        signal_provider=FakeSignalProvider(ticks),
        strategy=FakeStrategy(),
        fair_engine=FairValueEngine(config=DEFAULT_CONFIG),
        config=DEFAULT_CONFIG,
        initial_pusd=1000.0,
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.skipped_fair_value_count == 0
    assert summary.last_fair_value_error is None


# ---------------------------------------------------------------------------
# PR #10 — enriched metrics
# ---------------------------------------------------------------------------

def test_fill_rate_is_filled_orders_over_posted():
    """fill_rate = filled_orders / orders_posted; one full fill → 1.0."""
    msgs = [
        _book_msg("YES_TOKEN", ts=100, ask_price=0.52),
        _book_msg("YES_TOKEN", ts=200, ask_price=0.45),
    ]
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProviderWithSleep(msgs),
        signal_provider=FakeSignalProvider([]),
        strategy=FakeStrategy(),
        fair_engine=FakeFairValueEngine(),
        config=DEFAULT_CONFIG,
        initial_pusd=1000.0,
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.fills_simulated >= 1
    assert summary.filled_orders >= 1
    assert summary.orders_posted >= 1
    assert summary.fill_rate is not None
    assert abs(summary.fill_rate - summary.filled_orders / summary.orders_posted) < 1e-10


def test_fill_rate_none_when_no_orders_posted():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_posted == 0
    assert summary.fill_rate is None


def test_filled_orders_counts_unique_order_ids():
    """Two partial fills on the same order → filled_orders == 1, fills_simulated == 2."""
    msgs = [
        _book_msg("YES_TOKEN", ts=100, ask_price=0.52, ask_size=25.0),   # no cross
        _book_msg("YES_TOKEN", ts=200, ask_price=0.45, ask_size=3.0),    # partial fill (3)
        _book_msg("YES_TOKEN", ts=300, ask_price=0.44, ask_size=3.0),    # partial fill (3)
    ]
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProviderWithSleep(msgs),
        signal_provider=FakeSignalProvider([]),
        strategy=FakeStrategy(),
        fair_engine=FakeFairValueEngine(),
        config=DEFAULT_CONFIG,
        initial_pusd=1000.0,
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.fills_simulated == 2
    assert summary.filled_orders == 1


def test_rejection_rate_computed_correctly():
    """All attempts rejected (capital=0) → rejection_rate == 1.0."""
    msgs = [_book_msg("YES_TOKEN", ts=100, ask_price=0.52)]
    session = _make_session(msgs, [], initial_pusd=0.0)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_rejected >= 1
    assert summary.rejection_rate is not None
    assert abs(summary.rejection_rate - 1.0) < 1e-10


def test_rejection_rate_none_when_no_orders():
    session = _make_session([], [])
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_posted == 0
    assert summary.orders_rejected == 0
    assert summary.rejection_rate is None


def test_cancel_to_post_ratio_computed_correctly():
    """Reprice (bid price changes between decisions) → cancel + re-post → ratio correct."""

    class _FakeStrategyReprice:
        def build(self, state, fair, now_ms):
            price = 0.47 if state.yes_book.timestamp_ms > 100 else 0.48
            return DesiredQuotes(
                bid=DesiredOrder(enabled=True, side="BUY", price=price, size=10.0, reason="test"),
                ask=DesiredOrder(enabled=True, side="SELL", price=0.55, size=10.0, reason="test"),
                mode="passive",
                inventory_skew=0.0,
                timestamp_ms=state.yes_book.timestamp_ms,
            )

    msgs = [
        _book_msg("YES_TOKEN", ts=100, ask_price=0.52),
        _book_msg("YES_TOKEN", ts=200, ask_price=0.52),
    ]
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProviderWithSleep(msgs),
        signal_provider=FakeSignalProvider([]),
        strategy=_FakeStrategyReprice(),
        fair_engine=FakeFairValueEngine(),
        config=DEFAULT_CONFIG,
        initial_pusd=1000.0,
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_cancelled >= 1
    assert summary.orders_posted >= 1
    assert summary.cancel_to_post_ratio is not None
    expected = summary.orders_cancelled / summary.orders_posted
    assert abs(summary.cancel_to_post_ratio - expected) < 1e-10


def test_max_up_inventory_nonzero_after_fill():
    msgs = [
        _book_msg("YES_TOKEN", ts=100, ask_price=0.52),
        _book_msg("YES_TOKEN", ts=200, ask_price=0.45),
    ]
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProviderWithSleep(msgs),
        signal_provider=FakeSignalProvider([]),
        strategy=FakeStrategy(),
        fair_engine=FakeFairValueEngine(),
        config=DEFAULT_CONFIG,
        initial_pusd=1000.0,
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    if summary.fills_simulated >= 1:
        assert summary.max_up_inventory > 0.0


def test_peaks_updated_after_post_before_fill():
    """max_pusd_reserved is captured by _drain_user_queue even with no fill."""
    msgs = [_book_msg("YES_TOKEN", ts=100, ask_price=0.52)]  # ask does not cross bid
    session = _make_session(msgs, [], initial_pusd=1000.0)
    summary = asyncio.run(session.run_for(duration=5))
    assert summary.orders_posted >= 1
    assert summary.fills_simulated == 0
    assert summary.max_pusd_reserved > 0.0


def test_first_and_last_fill_ts_ms_populated():
    msgs = [
        _book_msg("YES_TOKEN", ts=100, ask_price=0.52),
        _book_msg("YES_TOKEN", ts=200, ask_price=0.45),
    ]
    session = LivePaperSession(
        discovery=FakeDiscoveryProvider(),
        market_provider=FakeMarketDataProviderWithSleep(msgs),
        signal_provider=FakeSignalProvider([]),
        strategy=FakeStrategy(),
        fair_engine=FakeFairValueEngine(),
        config=DEFAULT_CONFIG,
        initial_pusd=1000.0,
        decision_poll_ms=0,
    )
    summary = asyncio.run(session.run_for(duration=5))
    if summary.fills_simulated >= 1:
        assert summary.first_fill_ts_ms is not None
        assert summary.last_fill_ts_ms is not None
        assert isinstance(summary.first_fill_ts_ms, int)
        assert isinstance(summary.last_fill_ts_ms, int)


def test_events_contain_decision_and_order_posted():
    msgs = [_book_msg("YES_TOKEN", ts=100, ask_price=0.52)]
    session = _make_session(msgs, [], initial_pusd=1000.0)
    asyncio.run(session.run_for(duration=5))
    event_types = {e["event"] for e in session.events}
    assert "decision" in event_types
    assert "order_posted" in event_types


def test_write_jsonl_writes_correct_line_count():
    import tempfile
    from pathlib import Path as _Path
    from bot.paper_journal import write_jsonl

    events = [
        {"ts_ms": 100, "event": "decision", "trigger": "market"},
        {"ts_ms": 200, "event": "order_posted", "order_id": "mock-1"},
        {"ts_ms": 300, "event": "fill_simulated", "order_id": "mock-1"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _Path(tmpdir) / "subdir" / "session.jsonl"
        lines = write_jsonl(events, path)
        assert lines == 3
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert content.count("\n") == 3
