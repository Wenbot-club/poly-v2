# Reconstructed smoke suite — original test file was not present in the upload.
# Covers end-to-end paper run and JSONL replay assertion.
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from bot.async_runner import AsyncLocalRunner
from bot.providers.discovery import DiscoveryService, MockGammaClient
from bot.recorder import JSONLRecorder
from bot.replay import replay_jsonl
from bot.settings import DEFAULT_CONFIG
from bot.state import StateFactory


def _make_state():
    market = DiscoveryService(MockGammaClient(now_ts_ms=1_765_000_800_000)).find_active_btc_15m_market()
    return StateFactory(DEFAULT_CONFIG).create(market)


def _make_script(state) -> List[Dict[str, Any]]:
    start = state.market.start_ts_ms
    yes = state.market.yes_token_id
    no = state.market.no_token_id
    return [
        {"channel": "market", "payload": {"event_type": "book", "asset_id": yes, "timestamp": start + 100, "bids": [{"price": 0.48, "size": 30.0}, {"price": 0.47, "size": 50.0}], "asks": [{"price": 0.52, "size": 25.0}, {"price": 0.53, "size": 40.0}]}},
        {"channel": "market", "payload": {"event_type": "book", "asset_id": no, "timestamp": start + 100, "bids": [{"price": 0.46, "size": 18.0}], "asks": [{"price": 0.54, "size": 20.0}]}},
        {"channel": "rtds", "payload": {"source": "chainlink", "symbol": "btc/usd", "timestamp_ms": start + 120, "recv_timestamp_ms": start + 121, "value": 0.50, "sequence_no": 1}},
        {"channel": "rtds", "payload": {"source": "binance", "symbol": "btc/usd", "timestamp_ms": start + 130, "recv_timestamp_ms": start + 130, "value": 0.53, "sequence_no": 1}},
        {"channel": "control", "payload": {"op": "quote", "now_ms": start + 200, "label": "initial_quote"}},
        {"channel": "control", "payload": {"op": "fill_partial", "now_ms": start + 350, "fill_size": 4.0}},
        {"channel": "control", "payload": {"op": "quote", "now_ms": start + 500, "label": "reprice_after_fill"}},
        {"channel": "control", "payload": {"op": "cancel_final", "now_ms": start + 700, "reason": "demo_complete"}},
    ]


def test_state_creation():
    state = _make_state()
    assert state.market.market_id == "mkt-btc-15m-demo"
    assert state.inventory.pusd_free == 125.0
    assert state.market.yes_token_id == "YES_TOKEN_BTC_15M_DEMO"


def test_async_runner_produces_initial_quote():
    state = _make_state()
    script = _make_script(state)
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    with JSONLRecorder(path) as recorder:
        summary = asyncio.run(AsyncLocalRunner(config=DEFAULT_CONFIG, state=state, event_script=script, recorder=recorder).run())
    assert summary.initial_quote.get("bid_price") is not None
    assert summary.initial_quote.get("bid_enabled") is True
    assert any(a.startswith("post:") for a in summary.actions)


def test_async_runner_fill_and_reprice():
    state = _make_state()
    script = _make_script(state)
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    with JSONLRecorder(path) as recorder:
        summary = asyncio.run(AsyncLocalRunner(config=DEFAULT_CONFIG, state=state, event_script=script, recorder=recorder).run())
    assert any(a.startswith("fill:") for a in summary.actions)
    assert summary.reeval_quote.get("bid_price") is not None


def test_replay_assertion_passes():
    state = _make_state()
    script = _make_script(state)
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    with JSONLRecorder(path) as recorder:
        asyncio.run(AsyncLocalRunner(config=DEFAULT_CONFIG, state=state, event_script=script, recorder=recorder).run())
    result = replay_jsonl(path)
    assert result.assertion_passed is True
    assert result.posted_count >= 1
    assert result.cancel_count >= 1


def test_temporal_consistency_assertions_pass():
    state = _make_state()
    script = _make_script(state)
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = Path(f.name)
    with JSONLRecorder(path) as recorder:
        summary = asyncio.run(AsyncLocalRunner(config=DEFAULT_CONFIG, state=state, event_script=script, recorder=recorder).run())
    for check in summary.temporal_checks:
        assert check["temporal_assertions_passed"] is True
