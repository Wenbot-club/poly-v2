"""
BTC M5 PTB consensus demo — runs N consecutive windows.

For each M5 window (300s):
  1. Discovers PTB via SSR/API/Chainlink
  2. Discovers UP/DOWN tokens via Gamma slug btc-updown-5m-{window_ts}
     (prefetched during the previous window's resolution wait when possible)
  3. Scans EARLY consensus [140-170s], falls back to baseline at 170s
  4. Arms hedge if BTCrossed PTB ± 1.0 (cutoff: 250s)
  5. Settles at expiry via closePrice API

Scheduler: deterministic — first window aligns to the next clean 5-min boundary
(or current if >80% of the window remains), then advances strictly by 300s.
No window is ever skipped because resolution ran late.

Writes per-window summaries to <output_dir>/window_NNN.json
and a campaign summary to <output_dir>/m5_campaign_summary.json

Usage:
  python demos/demo_btc_m5.py --windows 3 --output-dir m5_out/
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

from bot.m5_session import M5Session, M5SignalState, BtcHistory, fetch_token_best_ask
from bot.m5_summary import TradeRecord, aggregate_trades
from bot.providers.binance_signal import BinanceSignalProvider
from bot.providers.polymarket_chainlink_signal import PolymarketChainlinkSignalProvider
from bot.settings import DEFAULT_M5_CONFIG, M5Config

_POLY_BASE = "https://polymarket.com"


async def _update_binance_loop(
    http: aiohttp.ClientSession,
    state: M5SignalState,
    btc_history: BtcHistory,
) -> None:
    """Background task: stream Binance BTC/USDT aggTrade and record into history."""
    provider = BinanceSignalProvider(session=http)
    await provider.connect("btc/usd")
    try:
        async for tick in provider.iter_signals():
            price = tick["value"]
            ts_ms = tick["recv_timestamp_ms"]
            state.btc_price = price
            state.btc_price_ts_ms = ts_ms
            btc_history.record(price, ts_ms)
    finally:
        await provider.close()


async def _update_chainlink_loop(
    http: aiohttp.ClientSession,
    state: M5SignalState,
) -> None:
    """Background task: stream Polymarket Chainlink BTC/USD feed."""
    provider = PolymarketChainlinkSignalProvider(session=http)
    await provider.connect("btc/usd")
    try:
        async for tick in provider.iter_signals():
            state.chainlink_price = tick["value"]
            state.chainlink_price_ts_ms = tick["recv_timestamp_ms"]
    finally:
        await provider.close()


def _first_m5_window_ts(cfg: M5Config, time_fn=_time.time) -> int:
    """First window: current boundary if >80% remains, else next."""
    now = time_fn()
    ws = cfg.window_seconds
    wts = int(math.floor(now / ws) * ws)
    if (wts + ws) - now < ws * 0.8:
        wts += ws
    return wts


def _print_record(record: TradeRecord) -> None:
    print(f"\n{'=' * 55}")
    print(f"  Window {record.window_ts}  PTB={record.ptb} ({record.ptb_source})")
    if record.abort_reason:
        print(f"  ABORTED: {record.abort_reason}")
        return
    print(f"  entry  : mode={record.entry_mode}  side={record.entry_side}"
          f"  t={record.entry_elapsed_s:.1f}s  price={record.entry_price:.4f}"
          f"  shares={record.entry_shares:.4f}")
    if record.entry_mode == "early":
        print(f"  model  : p_up={record.p_model_up_at_entry:.3f}"
              f"  z_gap={record.z_gap_at_entry:.2f}"
              f"  sigma={record.sigma_to_close_at_entry:.1f}"
              f"  edge_up={record.edge_up_at_entry:.3f}")
    if record.leg1_slippage is not None:
        print(f"  leg1_trace: ask={record.leg1_observed_ask:.4f}"
              f"  attempted={record.leg1_attempted_price:.4f}"
              f"  slippage={record.leg1_slippage:.4f}")
    if record.hedged:
        print(f"  hedge  : side={record.hedge_side}  t={record.hedge_elapsed_s:.1f}s"
              f"  price={record.hedge_price:.4f}  btc_at_trigger={record.hedge_trigger_btc:.2f}")
        if record.hedge_slippage is not None:
            print(f"  hedge_trace: ask={record.hedge_observed_ask:.4f}"
                  f"  attempted={record.hedge_attempted_price:.4f}"
                  f"  slippage={record.hedge_slippage:.4f}")
    if record.hedge_blocked_by_cutoff:
        print("  hedge  : blocked by cutoff")
    if record.price_insane_block_count:
        print(f"  price_insane blocks: {record.price_insane_block_count}")
    if record.resolution_latency_s is not None:
        print(f"  resolution: attempts={record.resolution_attempts}"
              f"  latency={record.resolution_latency_s:.1f}s")
    else:
        print(f"  resolution: attempts={record.resolution_attempts}  latency=n/a")
    print(f"  result : {record.result}  pnl_leg1={record.pnl_leg1}  "
          f"pnl_hedge={record.pnl_hedge}  net={record.net_pnl}")


async def run_campaign(
    window_count: int,
    output_dir: Path,
    cfg: M5Config = DEFAULT_M5_CONFIG,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = M5SignalState()
    btc_history = BtcHistory()
    trades: list[TradeRecord] = []

    async with aiohttp.ClientSession() as http:
        binance_task = asyncio.create_task(_update_binance_loop(http, state, btc_history))
        chainlink_task = asyncio.create_task(_update_chainlink_loop(http, state))

        # Give feeds a moment to establish before first window
        await asyncio.sleep(2.0)

        # Deterministic scheduler — first boundary alignment, then strict +300s
        wts = _first_m5_window_ts(cfg)
        next_tokens_cache_window_ts: Optional[int] = None
        next_tokens_cache_task: Optional[asyncio.Task] = None

        try:
            for i in range(window_count):
                now = _time.time()
                wait_s = max(0.0, wts - now)

                if wait_s > 5.0:
                    print(f"\n[m5] window {i+1}/{window_count}: window_ts={wts}"
                          f" — waiting {wait_s:.0f}s for window start")
                    await asyncio.sleep(wait_s)

                # Resolve prefetch cache for this window's tokens
                prefetched: Optional[tuple] = None
                if (next_tokens_cache_window_ts == wts
                        and next_tokens_cache_task is not None
                        and next_tokens_cache_task.done()
                        and not next_tokens_cache_task.cancelled()):
                    try:
                        prefetched = next_tokens_cache_task.result()
                    except Exception:
                        prefetched = None

                prefetch_status = "hit" if prefetched is not None else "miss"
                print(f"\n[m5] starting window {i+1}/{window_count}  ts={wts}"
                      f"  btc_now={state.btc_price}"
                      f"  chainlink={state.chainlink_price}"
                      f"  tokens_prefetch={prefetch_status}")

                session = M5Session(
                    http_session=http,
                    signal_state=state,
                    config=cfg,
                    time_fn=_time.time,
                    btc_history=btc_history,
                    prefetched_tokens=prefetched,
                )
                record = await session.run(wts)
                trades.append(record)

                _print_record(record)

                # Capture the prefetch task launched inside run() for the next window
                next_tokens_cache_window_ts = wts + cfg.window_seconds
                next_tokens_cache_task = session.next_tokens_task

                # Write per-window artifact
                fname = output_dir / f"window_{i:03d}.json"
                with fname.open("w") as f:
                    json.dump(dataclasses.asdict(record), f, indent=2)

                # Strict progression — never skip a window
                wts += cfg.window_seconds

        finally:
            binance_task.cancel()
            chainlink_task.cancel()
            for task in (binance_task, chainlink_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    summary = aggregate_trades(trades)
    summary_path = output_dir / "m5_campaign_summary.json"
    with summary_path.open("w") as f:
        json.dump(dataclasses.asdict(summary), f, indent=2)

    print(f"\n{'=' * 55}")
    print("  M5 Campaign summary")
    print(f"  windows_seen        : {summary.windows_seen}")
    print(f"  leg1_entered        : {summary.leg1_entered_count}")
    print(f"    early             : {summary.early_entry_count}")
    print(f"    baseline          : {summary.baseline_entry_count}")
    print(f"  hedge_triggered     : {summary.hedge_triggered_count}")
    print(f"  hedge_cutoff_blocks : {summary.hedge_blocked_by_cutoff_count}")
    print(f"  price_insane_blocks : {summary.price_insane_block_count}")
    print(f"  pnl_leg1_total      : {summary.pnl_leg1_total}")
    print(f"  pnl_hedge_total     : {summary.pnl_hedge_total}")
    print(f"  net_pnl_total       : {summary.net_pnl_total}")
    print(f"  avg_leg1_price      : {summary.avg_leg1_entry_price}")
    print(f"  avg_hedge_price     : {summary.avg_hedge_entry_price}")
    print(f"  avg_leg1_slippage   : {summary.avg_leg1_slippage}")
    print(f"  avg_hedge_slippage  : {summary.avg_hedge_slippage}")
    print(f"  avg_resolution_lat  : {summary.avg_resolution_latency_s}")
    print(f"  prefetch_token_hits : {summary.prefetched_token_hits}")
    print(f"\n[artifacts] {output_dir}/")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="BTC M5 paper trading campaign")
    parser.add_argument("--windows", type=int, default=3, help="Number of M5 windows (default: 3)")
    parser.add_argument("--output-dir", type=Path, default=Path("m5_out"), help="Output dir")
    args = parser.parse_args(argv)

    try:
        asyncio.run(run_campaign(args.windows, args.output_dir))
    except KeyboardInterrupt:
        print("\n[interrupted]")


if __name__ == "__main__":
    main()
