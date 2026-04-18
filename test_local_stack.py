from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bot.async_runner import AsyncLocalRunner, AsyncRunSummary
from bot.discovery import DiscoveryService, MockGammaClient
from bot.replay import ReplaySummary, replay_jsonl
from bot.recorder import JSONLRecorder
from bot.settings import DEFAULT_CONFIG, RuntimeConfig
from bot.state import StateFactory
from bot.types import LocalState


def build_demo_state(config: RuntimeConfig = DEFAULT_CONFIG) -> LocalState:
    market = DiscoveryService(MockGammaClient(now_ts_ms=1_765_000_800_000)).find_active_btc_15m_market()
    return StateFactory(config).create(market)


def build_demo_event_script(state: LocalState) -> List[Dict[str, Dict[str, Any]]]:
    start = state.market.start_ts_ms
    yes = state.market.yes_token_id
    no = state.market.no_token_id
    return [
        {
            "channel": "market",
            "payload": {
                "event_type": "book",
                "asset_id": yes,
                "timestamp": start + 100,
                "bids": [{"price": 0.48, "size": 30.0}, {"price": 0.47, "size": 50.0}],
                "asks": [{"price": 0.52, "size": 25.0}, {"price": 0.53, "size": 40.0}],
            },
        },
        {
            "channel": "market",
            "payload": {
                "event_type": "book",
                "asset_id": no,
                "timestamp": start + 100,
                "bids": [{"price": 0.46, "size": 18.0}],
                "asks": [{"price": 0.54, "size": 20.0}],
            },
        },
        {
            "channel": "rtds",
            "payload": {
                "source": "chainlink",
                "symbol": "btc/usd",
                "timestamp_ms": start + 120,
                "recv_timestamp_ms": start + 121,
                "value": 0.50,
                "sequence_no": 1,
            },
        },
        {
            "channel": "rtds",
            "payload": {
                "source": "binance",
                "symbol": "btc/usd",
                "timestamp_ms": start + 130,
                "recv_timestamp_ms": start + 130,
                "value": 0.53,
                "sequence_no": 1,
            },
        },
        {"channel": "control", "payload": {"op": "quote", "now_ms": start + 200, "label": "initial_quote"}},
        {"channel": "control", "payload": {"op": "fill_partial", "now_ms": start + 350, "fill_size": 4.0}},
        {"channel": "control", "payload": {"op": "quote", "now_ms": start + 500, "label": "reprice_after_fill"}},
        {"channel": "control", "payload": {"op": "cancel_final", "now_ms": start + 700, "reason": "demo_complete"}},
    ]


def run_async_local_demo(
    *,
    output_path: str | Path = "artifacts/demo_async_runner_local.jsonl",
    config: RuntimeConfig = DEFAULT_CONFIG,
) -> Tuple[AsyncRunSummary, ReplaySummary]:
    state = build_demo_state(config)
    script = build_demo_event_script(state)
    output_path = Path(output_path)
    with JSONLRecorder(output_path) as recorder:
        summary = asyncio.run(AsyncLocalRunner(config=config, state=state, event_script=script, recorder=recorder).run())
    replay = replay_jsonl(output_path)
    return summary, replay


def format_demo_output(summary: AsyncRunSummary, replay: ReplaySummary, output_path: str | Path) -> str:
    lines = [
        "=== local mock paper run ===",
        f"initial quote: bid={summary.initial_quote.get('bid_price')} size={summary.initial_quote.get('bid_size')} actions={summary.initial_quote.get('actions')}",
        f"after partial fill: actions={[a for a in summary.actions if a.startswith('fill:') or a.startswith('status:')]}",
        f"reprice cycle: bid={summary.reeval_quote.get('bid_price')} size={summary.reeval_quote.get('bid_size')} actions={summary.reeval_quote.get('actions')}",
        f"final state: {summary.final_state}",
        f"replay: posted={replay.posted_count} canceled={replay.cancel_count} filled={replay.fill_count} assertion_passed={replay.assertion_passed}",
        f"jsonl: {Path(output_path)}",
    ]
    return "\n".join(lines)


def main() -> None:
    output_path = Path("artifacts/demo_async_runner_local.jsonl")
    summary, replay = run_async_local_demo(output_path=output_path)
    print(format_demo_output(summary, replay, output_path))


if __name__ == "__main__":
    main()
