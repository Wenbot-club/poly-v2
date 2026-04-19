"""Tests for bot/providers/composite_signal.py — deterministic, no network."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Literal

from bot.providers.composite_signal import CompositeSignalProvider


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------

class FakeProvider:
    def __init__(
        self,
        source_name: str,
        ticks: List[Dict[str, Any]],
        initial_state: str = "disconnected",
    ) -> None:
        self.source_name = source_name
        self.feed_state: Literal["connecting", "live", "stale", "disconnected"] = initial_state  # type: ignore[assignment]
        self._ticks = ticks
        self.connect_calls: List[str] = []
        self.close_calls: int = 0

    async def connect(self, symbol: str) -> None:
        self.connect_calls.append(symbol)
        self.feed_state = "connecting"

    async def close(self) -> None:
        self.close_calls += 1
        self.feed_state = "disconnected"

    async def iter_signals(self) -> AsyncIterator[Dict[str, Any]]:
        self.feed_state = "live"
        for tick in self._ticks:
            yield tick
        self.feed_state = "disconnected"


def _binance_tick(value: float, seq: int) -> Dict[str, Any]:
    return {
        "source": "binance",
        "symbol": "btc/usd",
        "timestamp_ms": 1_000_000,
        "recv_timestamp_ms": 1_000_050,
        "value": value,
        "sequence_no": seq,
    }


def _coinbase_tick(value: float, seq: int) -> Dict[str, Any]:
    return {
        "source": "coinbase",
        "symbol": "btc/usd",
        "timestamp_ms": 1_000_000,
        "recv_timestamp_ms": 1_000_050,
        "value": value,
        "sequence_no": seq,
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_source_name_is_joined_from_providers():
    p1 = FakeProvider("binance", [])
    p2 = FakeProvider("coinbase", [])
    comp = CompositeSignalProvider([p1, p2])
    assert comp.source_name == "binance+coinbase"


def test_source_name_single_provider():
    p = FakeProvider("binance", [])
    comp = CompositeSignalProvider([p])
    assert comp.source_name == "binance"


def test_empty_providers_raises():
    try:
        CompositeSignalProvider([])
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# connect() / close() forwarding
# ---------------------------------------------------------------------------

def test_connect_calls_all_providers():
    p1 = FakeProvider("binance", [])
    p2 = FakeProvider("coinbase", [])
    comp = CompositeSignalProvider([p1, p2])
    asyncio.run(comp.connect("btc/usd"))
    assert p1.connect_calls == ["btc/usd"]
    assert p2.connect_calls == ["btc/usd"]


def test_close_calls_all_providers():
    p1 = FakeProvider("binance", [])
    p2 = FakeProvider("coinbase", [])
    comp = CompositeSignalProvider([p1, p2])
    asyncio.run(comp.connect("btc/usd"))
    asyncio.run(comp.close())
    assert p1.close_calls == 1
    assert p2.close_calls == 1


# ---------------------------------------------------------------------------
# feed_state aggregation
# ---------------------------------------------------------------------------

def test_feed_state_live_if_any_child_live():
    p1 = FakeProvider("binance", [], initial_state="live")
    p2 = FakeProvider("coinbase", [], initial_state="disconnected")
    comp = CompositeSignalProvider([p1, p2])
    assert comp.feed_state == "live"


def test_feed_state_connecting_if_any_connecting_no_live():
    p1 = FakeProvider("binance", [], initial_state="connecting")
    p2 = FakeProvider("coinbase", [], initial_state="disconnected")
    comp = CompositeSignalProvider([p1, p2])
    assert comp.feed_state == "connecting"


def test_feed_state_stale_if_any_stale_no_live_no_connecting():
    p1 = FakeProvider("binance", [], initial_state="stale")
    p2 = FakeProvider("coinbase", [], initial_state="disconnected")
    comp = CompositeSignalProvider([p1, p2])
    assert comp.feed_state == "stale"


def test_feed_state_disconnected_when_all_disconnected():
    p1 = FakeProvider("binance", [], initial_state="disconnected")
    p2 = FakeProvider("coinbase", [], initial_state="disconnected")
    comp = CompositeSignalProvider([p1, p2])
    assert comp.feed_state == "disconnected"


def test_live_takes_priority_over_stale():
    p1 = FakeProvider("binance", [], initial_state="stale")
    p2 = FakeProvider("coinbase", [], initial_state="live")
    comp = CompositeSignalProvider([p1, p2])
    assert comp.feed_state == "live"


# ---------------------------------------------------------------------------
# iter_signals() merging
# ---------------------------------------------------------------------------

async def _collect(comp: CompositeSignalProvider, symbol: str = "btc/usd") -> List[Dict[str, Any]]:
    await comp.connect(symbol)
    collected = []
    async for tick in comp.iter_signals():
        collected.append(tick)
    return collected


def test_merges_ticks_from_both_providers():
    p1 = FakeProvider("binance", [_binance_tick(42000.0, 1), _binance_tick(42001.0, 2)])
    p2 = FakeProvider("coinbase", [_coinbase_tick(42000.5, 10)])
    comp = CompositeSignalProvider([p1, p2])
    ticks = asyncio.run(_collect(comp))
    assert len(ticks) == 3
    sources = {t["source"] for t in ticks}
    assert sources == {"binance", "coinbase"}


def test_all_ticks_from_single_provider_included():
    p1 = FakeProvider("binance", [_binance_tick(float(i), i) for i in range(5)])
    p2 = FakeProvider("coinbase", [])
    comp = CompositeSignalProvider([p1, p2])
    ticks = asyncio.run(_collect(comp))
    assert len(ticks) == 5


def test_empty_providers_yields_nothing():
    p1 = FakeProvider("binance", [])
    p2 = FakeProvider("coinbase", [])
    comp = CompositeSignalProvider([p1, p2])
    ticks = asyncio.run(_collect(comp))
    assert ticks == []


def test_ticks_have_correct_source_fields():
    p1 = FakeProvider("binance", [_binance_tick(42000.0, 1)])
    p2 = FakeProvider("coinbase", [_coinbase_tick(42000.0, 1)])
    comp = CompositeSignalProvider([p1, p2])
    ticks = asyncio.run(_collect(comp))
    sources = [t["source"] for t in ticks]
    assert "binance" in sources
    assert "coinbase" in sources
