"""
BTC M5 PTB consensus demo — runs N consecutive windows.

For each M5 window (300s):
  1. Discovers PTB via SSR/API/Chainlink
  2. Discovers UP/DOWN tokens via Gamma slug btc-updown-5m-{window_ts}
  3. Scans EARLY consensus [140-170s], falls back to baseline at 170s
  4. Arms hedge if BTCrossed PTB ± 1.0 (cutoff: 250s)
  5. Settles at expiry via closePrice API

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

import aiohttp

from bot.m5_session import M5Session, M5SignalState, fetch_token_best_ask
from bot.m5_summary import TradeRecord, aggregate_trades
from bot.settings import DEFAULT_M5_CONFIG, M5Config

_BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"
_POLY_BASE = "https://polymarket.com"


async def _update_btc_loop(
    http: aiohttp.ClientSession,
    state: M5SignalState,
) -> None:
    """Background task: poll Binance BTC/USDT price every second."""
    while True:
        try:
            async with http.get(
                _BINANCE_PRICE_URL,
                params={"symbol": "BTCUSDT"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    state.btc_price = float(data["price"])
                    state.btc_price_ts_ms = int(_time.time() * 1000)
        except Exception:
            pass
        await asyncio.sleep(1.0)


def _next_m5_window_ts(cfg: M5Config, min_remaining_s: float) -> int:
    """Return window_ts with at least min_remaining_s of life left."""
    now = _time.time()
    ws = cfg.window_seconds
    wts = int(math.floor(now / ws) * ws)
    if (wts + ws) - now < min_remaining_s:
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
        print(f"  consensus: score={record.entry_consensus_score:.1f}"
              f"  non_neutral={record.entry_consensus_non_neutral}")
    if record.hedged:
        print(f"  hedge  : side={record.hedge_side}  t={record.hedge_elapsed_s:.1f}s"
              f"  price={record.hedge_price:.4f}  btc_at_trigger={record.hedge_trigger_btc:.2f}")
    if record.hedge_blocked_by_cutoff:
        print("  hedge  : blocked by cutoff")
    if record.price_insane_block_count:
        print(f"  price_insane blocks: {record.price_insane_block_count}")
    print(f"  result : {record.result}  pnl_leg1={record.pnl_leg1}  "
          f"pnl_hedge={record.pnl_hedge}  net={record.net_pnl}")


async def run_campaign(
    window_count: int,
    output_dir: Path,
    cfg: M5Config = DEFAULT_M5_CONFIG,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = M5SignalState()
    trades: list[TradeRecord] = []

    async with aiohttp.ClientSession() as http:
        btc_task = asyncio.create_task(_update_btc_loop(http, state))

        try:
            for i in range(window_count):
                wts = _next_m5_window_ts(cfg, min_remaining_s=cfg.window_seconds * 0.8)
                window_end = wts + cfg.window_seconds
                now = _time.time()
                wait_s = max(0.0, wts - now)

                if wait_s > 5.0:
                    print(f"\n[m5] window {i+1}/{window_count}: window_ts={wts}"
                          f" — waiting {wait_s:.0f}s for window start")
                    await asyncio.sleep(wait_s)

                print(f"\n[m5] starting window {i+1}/{window_count}  ts={wts}"
                      f"  btc_now={state.btc_price}")

                session = M5Session(
                    http_session=http,
                    signal_state=state,
                    config=cfg,
                    time_fn=_time.time,
                )
                record = await session.run(wts)
                trades.append(record)

                _print_record(record)

                # write per-window artifact
                fname = output_dir / f"window_{i:03d}.json"
                with fname.open("w") as f:
                    json.dump(dataclasses.asdict(record), f, indent=2)

                # if next window would start immediately, sleep until boundary+buffer
                now = _time.time()
                time_to_next_boundary = wts + cfg.window_seconds - now
                if time_to_next_boundary < 5 and i < window_count - 1:
                    await asyncio.sleep(max(0.0, time_to_next_boundary + 5))

        finally:
            btc_task.cancel()
            try:
                await btc_task
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
