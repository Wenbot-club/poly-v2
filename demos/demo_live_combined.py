"""
Read-only combined live demo: market book feed + Binance RTDS on a shared LocalState.

Connects to both:
  - Polymarket CLOB WebSocket (market book feed)
  - Binance aggTrade WebSocket (BTC price signal)

Runs for --duration seconds (default 10), then prints a combined summary.

What this demo does NOT do:
  - no fair-value, no strategy, no execution
  - no Chainlink (requires auth — out of scope)
  - no authentication of any kind
"""
from __future__ import annotations

import argparse
import asyncio

import aiohttp

from bot.live_combined import LiveCombinedSession, LiveCombinedSummary
from bot.providers.binance_signal import BinanceSignalProvider
from bot.providers.polymarket_discovery import PolymarketDiscoveryProvider
from bot.providers.polymarket_market_data import PolymarketMarketDataProvider
from bot.settings import DEFAULT_CONFIG


def _print_summary(summary: LiveCombinedSummary) -> None:
    m = summary.market
    r = summary.rtds

    print(f"\n{'=' * 60}")
    print("  LiveCombinedSession summary")
    print(f"{'=' * 60}")

    print("\n  [market feed]")
    print(f"  market_id       : {m.market_id}")
    print(f"  yes_token_id    : {m.yes_token_id}")
    print(f"  no_token_id     : {m.no_token_id}")
    print(f"  total_messages  : {m.total_messages}")
    print(f"  book_count      : {m.book_count}")
    print(f"  price_change_cnt: {m.price_change_count}")
    print(f"  yes_snapshotted : {m.yes_snapshotted}")
    print(f"  no_snapshotted  : {m.no_snapshotted}")
    print(f"  final_feed_state: {m.final_feed_state}")
    print(f"  feed_transitions: {m.feed_state_transitions}")

    print("\n  [rtds feed]")
    print(f"  symbol          : {r.symbol}")
    print(f"  source          : {r.source}")
    print(f"  total_ticks     : {r.total_ticks}")
    print(f"  first_value     : {r.first_value}")
    print(f"  last_value      : {r.last_value}")
    print(f"  min_value       : {r.min_value}")
    print(f"  max_value       : {r.max_value}")
    print(f"  final_feed_state: {r.final_feed_state}")
    print(f"  feed_transitions: {r.feed_state_transitions}")

    if r.min_value is not None and r.max_value is not None:
        spread = round(r.max_value - r.min_value, 2)
        print(f"\n  BTC price range  : {r.min_value} → {r.max_value}  (spread={spread})")

    print(f"{'=' * 60}")


async def run_demo(duration: int) -> None:
    print(f"{'=' * 60}")
    print("  Combined read-only live demo")
    print(f"  Market: Polymarket CLOB WebSocket")
    print(f"  Signal: Binance aggTrade WebSocket")
    print(f"  Duration: {duration}s")
    print(f"{'=' * 60}")

    async with aiohttp.ClientSession() as http_session:
        session = LiveCombinedSession(
            discovery=PolymarketDiscoveryProvider(http_session),
            market_provider=PolymarketMarketDataProvider(http_session),
            signal_provider=BinanceSignalProvider(http_session),
            config=DEFAULT_CONFIG,
        )

        print("\n[combined] discovering market and connecting both feeds…")
        try:
            summary = await session.run_for(duration=duration)
        except Exception as exc:
            print(f"[combined] ERROR: {exc}")
            return

        _print_summary(summary)

        if session.state is not None:
            state = session.state
            print(f"\n[state] yes_book bids       : {dict(list(state.yes_book.bids.items())[:3])}")
            print(f"[state] no_book bids        : {dict(list(state.no_book.bids.items())[:3])}")
            print(f"[state] binance_ticks       : {len(state.binance_ticks)}")
            print(f"[state] tape_ewma           : {round(state.tape_ewma, 6)}")
            if state.last_binance is not None:
                lb = state.last_binance
                print(f"[state] last_binance.value  : {lb.value}")
                print(f"[state] last_binance.seq    : {lb.sequence_no}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Combined read-only live demo")
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
