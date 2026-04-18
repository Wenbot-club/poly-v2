"""
Read-only live RTDS demo via LiveRTDSSession + BinanceSignalProvider.

Connects to the Binance BTC/USDT aggTrade WebSocket stream, normalizes ticks,
and runs for --duration seconds (default 10).

What this demo does NOT do:
  - no market book feed (no LiveReadonlySession)
  - no fair-value, no strategy, no execution
  - no Chainlink (requires auth — out of scope)
  - no authentication of any kind
"""
from __future__ import annotations

import argparse
import asyncio

import aiohttp

from bot.live_rtds import LiveRTDSSummary, LiveRTDSSession
from bot.providers.binance_signal import BINANCE_WS_URL, BinanceSignalProvider
from bot.settings import DEFAULT_CONFIG


def _print_summary(summary: LiveRTDSSummary) -> None:
    print(f"\n{'=' * 60}")
    print("  LiveRTDSSession summary")
    print(f"{'=' * 60}")
    print(f"  symbol          : {summary.symbol}")
    print(f"  source          : {summary.source}")
    print(f"  total_ticks     : {summary.total_ticks}")
    print(f"  first_value     : {summary.first_value}")
    print(f"  last_value      : {summary.last_value}")
    print(f"  min_value       : {summary.min_value}")
    print(f"  max_value       : {summary.max_value}")
    print(f"  final_feed_state: {summary.final_feed_state}")
    print(f"  feed_transitions: {summary.feed_state_transitions}")
    print(f"  started_at_ms   : {summary.started_at_ms}")
    print(f"  ended_at_ms     : {summary.ended_at_ms}")

    if summary.total_ticks == 0:
        print(
            "\n  WARNING: no ticks received. "
            "Check WS connectivity and Binance stream availability."
        )
    if summary.min_value is not None and summary.max_value is not None:
        spread = round(summary.max_value - summary.min_value, 2)
        print(f"\n  price range (min→max): {summary.min_value} → {summary.max_value}  (spread={spread})")
    print(f"{'=' * 60}")


async def run_demo(duration: int) -> None:
    print(f"{'=' * 60}")
    print("  Binance aggTrade read-only live RTDS demo")
    print(f"  WS: {BINANCE_WS_URL}")
    print(f"  Duration: {duration}s")
    print(f"  Note: Chainlink out of scope (requires auth). Binance only.")
    print(f"{'=' * 60}")

    async with aiohttp.ClientSession() as http_session:
        provider = BinanceSignalProvider(http_session)
        rtds_session = LiveRTDSSession(
            signal_provider=provider,
            config=DEFAULT_CONFIG,
        )

        print("\n[rtds] connecting to Binance aggTrade stream…")
        try:
            summary = await rtds_session.run_for(duration=duration)
        except Exception as exc:
            print(f"[rtds] ERROR: {exc}")
            return

        _print_summary(summary)

        if rtds_session.state is not None:
            state = rtds_session.state
            print(f"\n[state] binance_ticks in deque : {len(state.binance_ticks)}")
            print(f"[state] tape_ewma              : {round(state.tape_ewma, 6)}")
            if state.last_binance is not None:
                lb = state.last_binance
                print(f"[state] last_binance.value     : {lb.value}")
                print(f"[state] last_binance.seq       : {lb.sequence_no}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Binance aggTrade read-only RTDS demo")
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        metavar="SECONDS",
        help="How long to listen (default: 10)",
    )
    args = parser.parse_args(argv)

    try:
        asyncio.run(run_demo(args.duration))
    except KeyboardInterrupt:
        print("\n[interrupted] Ctrl+C — exiting cleanly.")


if __name__ == "__main__":
    main()
