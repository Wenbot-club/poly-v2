#!/usr/bin/env python3
"""
ETH M5 paper-live runner — optimised for low-latency execution.

Identical architecture to demo_btc_m5_live.py but trades the Polymarket
ETH up/down 5-minute markets using Binance ETH/USDT stream and Chainlink
ETH/USD feed.

Usage:
  python demos/demo_eth_m5_live.py --windows 6 --output-dir m5_out_eth/
  python demos/demo_eth_m5_live.py --windows 6 --live --output-dir m5_out_eth/

Outputs per run:
  <output_dir>/window_NNN.json        — per-window TradeRecord
  <output_dir>/m5_campaign_summary.json
  <output_dir>/latency_summary.json   — per-order latency records
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import time as _time
from pathlib import Path
from typing import Optional

import aiohttp

from bot.latency import LatencyRecord, LatencyTracker
from bot.m5_session import M5Session, M5SignalState, BtcHistory
from bot.m5_summary import TradeRecord, aggregate_trades
from bot.providers.binance_signal import BinanceSignalProvider
from bot.providers.polymarket_chainlink_signal import PolymarketChainlinkSignalProvider
from bot.providers.polymarket_market_data import PolymarketMarketDataProvider
from bot.settings import DEFAULT_ETH_M5_CONFIG, M5Config

_BINANCE_ETH_WS = "wss://stream.binance.com:9443/ws/ethusdt@aggTrade"


# ---------------------------------------------------------------------------
# Background feed tasks
# ---------------------------------------------------------------------------

async def _update_eth_loop(
    http: aiohttp.ClientSession,
    state: M5SignalState,
    price_history: BtcHistory,
) -> None:
    """Stream Binance ETH/USDT aggTrade → state + history."""
    provider = BinanceSignalProvider(session=http, ws_url=_BINANCE_ETH_WS)
    await provider.connect("eth/usd")
    try:
        async for tick in provider.iter_signals():
            state.btc_price = tick["value"]
            state.btc_price_ts_ms = tick["recv_timestamp_ms"]
            price_history.record(tick["value"], tick["recv_timestamp_ms"])
    finally:
        await provider.close()


async def _update_chainlink_loop(
    http: aiohttp.ClientSession,
    state: M5SignalState,
) -> None:
    """Stream Polymarket Chainlink ETH/USD feed → state."""
    provider = PolymarketChainlinkSignalProvider(session=http)
    await provider.connect("eth/usd")
    try:
        async for tick in provider.iter_signals():
            state.chainlink_price = tick["value"]
            state.chainlink_price_ts_ms = tick["recv_timestamp_ms"]
    finally:
        await provider.close()


# ---------------------------------------------------------------------------
# Per-window Polymarket WS token price stream
# ---------------------------------------------------------------------------

def _apply_market_message(price_cache: dict, msg: dict) -> None:
    """Extract best ask from a normalized CLOB WS message, update price_cache."""
    event_type = msg.get("event_type")

    if event_type == "book":
        asks = msg.get("asks", [])
        if asks:
            best_ask = min(a["price"] for a in asks)
            price_cache[msg["asset_id"]] = best_ask

    elif event_type == "price_change":
        for ch in msg.get("price_changes", []):
            if "best_ask" in ch:
                price_cache[ch["asset_id"]] = ch["best_ask"]

    elif event_type == "best_bid_ask":
        price_cache[msg["asset_id"]] = msg["best_ask"]


async def _stream_token_prices_ws(
    http: aiohttp.ClientSession,
    token_ids: list,
    price_cache: dict,
) -> None:
    provider = PolymarketMarketDataProvider(http)
    await provider.connect(token_ids)
    try:
        async for msg in provider.iter_messages():
            _apply_market_message(price_cache, msg)
    except asyncio.CancelledError:
        pass
    finally:
        await provider.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_m5_window_ts(cfg: M5Config, time_fn=_time.time) -> int:
    now = time_fn()
    ws = cfg.window_seconds
    wts = int(math.floor(now / ws) * ws)
    if (wts + ws) - now < ws * 0.8:
        wts += ws
    return wts


def _record_latency(record: TradeRecord, tracker: LatencyTracker) -> None:
    if (record.entry_tick_ts_ms is not None
            and record.entry_decision_ts_ms is not None
            and record.entry_submit_ts_ms is not None):
        tracker.add(LatencyRecord(
            window_ts=record.window_ts,
            action="leg1_entry",
            tick_received_ts_ms=record.entry_tick_ts_ms,
            decision_ts_ms=record.entry_decision_ts_ms,
            submit_ts_ms=record.entry_submit_ts_ms,
        ))

    if (record.hedge_tick_ts_ms is not None
            and record.hedge_decision_ts_ms is not None
            and record.hedge_submit_ts_ms is not None):
        tracker.add(LatencyRecord(
            window_ts=record.window_ts,
            action="hedge",
            tick_received_ts_ms=record.hedge_tick_ts_ms,
            decision_ts_ms=record.hedge_decision_ts_ms,
            submit_ts_ms=record.hedge_submit_ts_ms,
        ))


# ---------------------------------------------------------------------------
# Main campaign runner
# ---------------------------------------------------------------------------

async def run_campaign_live(
    window_count: int,
    output_dir: Path,
    cfg: M5Config = DEFAULT_ETH_M5_CONFIG,
    order_executor=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = M5SignalState()
    price_history = BtcHistory()
    trades: list[TradeRecord] = []
    latency_tracker = LatencyTracker()

    async with aiohttp.ClientSession() as http:
        eth_task = asyncio.create_task(_update_eth_loop(http, state, price_history))
        chainlink_task = asyncio.create_task(_update_chainlink_loop(http, state))

        if order_executor is not None and hasattr(order_executor, "start_heartbeat"):
            order_executor.start_heartbeat()

        await asyncio.sleep(2.0)

        wts = _first_m5_window_ts(cfg)
        next_tokens_cache_window_ts: Optional[int] = None
        next_tokens_cache_task: Optional[asyncio.Task] = None

        try:
            for i in range(window_count):
                now = _time.time()
                wait_s = max(0.0, wts - now)
                if wait_s > 5.0:
                    print(f"\n[live] window {i+1}/{window_count}: ts={wts}"
                          f" — waiting {wait_s:.0f}s")
                    await asyncio.sleep(wait_s)

                prefetched: Optional[tuple] = None
                if (next_tokens_cache_window_ts == wts
                        and next_tokens_cache_task is not None
                        and next_tokens_cache_task.done()
                        and not next_tokens_cache_task.cancelled()):
                    try:
                        prefetched = next_tokens_cache_task.result()
                    except Exception:
                        prefetched = None

                eth_price = state.btc_price  # stored in btc_price field
                print(f"\n[live] window {i+1}/{window_count}  ts={wts}"
                      f"  eth={eth_price}"
                      f"  chainlink={state.chainlink_price}"
                      f"  tokens_prefetch={'hit' if prefetched else 'miss'}")

                token_prices: dict = {}
                ws_price_task: Optional[asyncio.Task] = None
                if prefetched is not None:
                    up_id, down_id = prefetched
                    ws_price_task = asyncio.create_task(
                        _stream_token_prices_ws(http, [up_id, down_id], token_prices)
                    )
                    if order_executor is not None and hasattr(order_executor, "prewarm"):
                        asyncio.create_task(order_executor.prewarm(up_id, down_id))
                    await asyncio.sleep(0.5)

                session = M5Session(
                    http_session=http,
                    signal_state=state,
                    config=cfg,
                    time_fn=_time.time,
                    btc_history=price_history,
                    prefetched_tokens=prefetched,
                    token_prices=token_prices if prefetched is not None else None,
                    order_executor=order_executor,
                    asset="eth",
                )
                record = await session.run(wts)
                trades.append(record)
                _record_latency(record, latency_tracker)

                if ws_price_task is not None:
                    ws_price_task.cancel()
                    try:
                        await ws_price_task
                    except asyncio.CancelledError:
                        pass

                _print_record(record)
                latency_tracker.print_summary()

                next_tokens_cache_window_ts = wts + cfg.window_seconds
                next_tokens_cache_task = session.next_tokens_task

                fname = output_dir / f"window_{i:03d}.json"
                with fname.open("w") as f:
                    json.dump(dataclasses.asdict(record), f, indent=2)

                wts += cfg.window_seconds

        finally:
            eth_task.cancel()
            chainlink_task.cancel()
            for task in (eth_task, chainlink_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    summary = aggregate_trades(trades)
    summary_path = output_dir / "m5_campaign_summary.json"
    with summary_path.open("w") as f:
        json.dump(dataclasses.asdict(summary), f, indent=2)

    lat_records = [dataclasses.asdict(r) for r in latency_tracker.records]
    lat_path = output_dir / "latency_summary.json"
    with lat_path.open("w") as f:
        json.dump({
            "summary": latency_tracker.summary(),
            "records": lat_records,
        }, f, indent=2)

    latency_tracker.print_summary()
    print(f"\n[artifacts] {output_dir}/")


# ---------------------------------------------------------------------------
# Per-window print
# ---------------------------------------------------------------------------

def _print_record(record: TradeRecord) -> None:
    print(f"\n{'=' * 55}")
    if record.window_start_utc_iso and record.window_end_utc_iso:
        print(f"  window : {record.window_start_utc_iso[11:19]}Z"
              f" -> {record.window_end_utc_iso[11:19]}Z")
    if record.abort_reason:
        print(f"  ABORTED: {record.abort_reason}")
        return
    if record.entry_mode is None:
        print(f"  entry  : BLOCKED ({record.entry_block_reason})")
    else:
        print(f"  entry  : mode={record.entry_mode}  side={record.entry_side}"
              f"  t={record.entry_elapsed_s:.1f}s  price={record.entry_price:.4f}")
    if record.hedged:
        print(f"  hedge  : side={record.hedge_side}  t={record.hedge_elapsed_s:.1f}s"
              f"  price={record.hedge_price:.4f}")
    if record.hedge_blocked_by_cutoff:
        print("  hedge  : blocked by cutoff")
    print(f"  result : {record.result}  pnl_leg1={record.pnl_leg1}"
          f"  pnl_hedge={record.pnl_hedge}  net={record.net_pnl}")

    if record.entry_decision_ts_ms and record.entry_tick_ts_ms:
        tick_age = record.entry_decision_ts_ms - record.entry_tick_ts_ms
        s_ms = (record.entry_submit_ts_ms or record.entry_decision_ts_ms) - record.entry_decision_ts_ms
        print(f"  lat(entry): tick_age={tick_age}ms  submit={s_ms}ms")
    if record.hedged and record.hedge_decision_ts_ms and record.hedge_tick_ts_ms:
        tick_age = record.hedge_decision_ts_ms - record.hedge_tick_ts_ms
        s_ms = (record.hedge_submit_ts_ms or record.hedge_decision_ts_ms) - record.hedge_decision_ts_ms
        print(f"  lat(hedge): tick_age={tick_age}ms  submit={s_ms}ms")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="ETH M5 paper-live runner (low-latency)")
    parser.add_argument("--windows", type=int, default=6,
                        help="Number of M5 windows (default: 6)")
    parser.add_argument("--output-dir", type=Path, default=Path("m5_out_eth"),
                        help="Output directory")
    parser.add_argument("--live", action="store_true",
                        help="Post real orders to Polymarket CLOB (default: paper)")
    args = parser.parse_args(argv)

    order_executor = None
    if args.live:
        from bot.trading.live_executor import LiveOrderExecutor
        from bot.trading.credentials import load_credentials
        creds = load_credentials()
        order_executor = LiveOrderExecutor(creds)
        print("[!] LIVE MODE — real ETH orders will be placed on Polymarket", flush=True)
    else:
        print("[paper] dry-run mode — no real orders", flush=True)

    try:
        import uvloop  # type: ignore[import-untyped]
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass

    try:
        asyncio.run(run_campaign_live(args.windows, args.output_dir, order_executor=order_executor))
    except KeyboardInterrupt:
        print("\n[interrupted]")


if __name__ == "__main__":
    main()
