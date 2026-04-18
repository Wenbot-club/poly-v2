"""
Read-only live demo via LiveReadonlySession.

Discovers the active BTC 15m market, connects to the Polymarket CLOB WebSocket,
and runs for --duration seconds (default 10).

What this demo does NOT do:
  - no paper execution, no runner wiring, no authentication
  - no REST book snapshot (WS provides its own on subscribe)

Snapshot-per-token behaviour:
  Each token (YES, NO) snapshots independently. The summary reports per-token
  snapshot status explicitly — no pretence of global consistency before both
  tokens have snapshotted.
"""
from __future__ import annotations

import argparse
import asyncio

import aiohttp

from bot.live_readonly import LiveReadonlySummary, LiveReadonlySession
from bot.providers.polymarket_discovery import (
    AmbiguousMarketError,
    GammaAPIError,
    NoMatchingMarketError,
    PolymarketDiscoveryProvider,
)
from bot.providers.polymarket_market_data import CLOB_WS_URL, PolymarketMarketDataProvider
from bot.settings import DEFAULT_CONFIG


def _print_summary(summary: LiveReadonlySummary) -> None:
    print(f"\n{'=' * 60}")
    print("  LiveReadonlySession summary")
    print(f"{'=' * 60}")
    print(f"  market_id       : {summary.market_id}")
    print(f"  yes_token_id    : {summary.yes_token_id}")
    print(f"  no_token_id     : {summary.no_token_id}")
    print(f"  total_messages  : {summary.total_messages}")
    print(f"  book_count      : {summary.book_count}")
    print(f"  price_change    : {summary.price_change_count}")
    print(f"  other           : {summary.other_count}")
    print(f"  yes_snapshotted : {summary.yes_snapshotted}")
    print(f"  no_snapshotted  : {summary.no_snapshotted}")
    print(f"  final_feed_state: {summary.final_feed_state}")
    print(f"  feed_transitions: {summary.feed_state_transitions}")
    print(f"  started_at_ms   : {summary.started_at_ms}")
    print(f"  ended_at_ms     : {summary.ended_at_ms}")

    if not summary.yes_snapshotted or not summary.no_snapshotted:
        print(
            "\n  NOTE: one or both tokens did not receive a snapshot within the run window."
            "\n  Updates for un-snapshotted tokens are dropped (snapshot-per-token rule)."
        )
    if summary.total_messages == 0:
        print(
            "\n  WARNING: no messages received. Check WS connectivity and market window."
        )
    print(f"{'=' * 60}")


async def run_demo(duration: int) -> None:
    print(f"{'=' * 60}")
    print("  Polymarket CLOB read-only live demo")
    print(f"  WS: {CLOB_WS_URL}")
    print(f"  Duration: {duration}s")
    print(f"{'=' * 60}")

    async with aiohttp.ClientSession() as session:
        discovery = PolymarketDiscoveryProvider(session)
        provider = PolymarketMarketDataProvider(session)

        live_session = LiveReadonlySession(
            discovery=discovery,
            provider=provider,
            config=DEFAULT_CONFIG,
        )

        print("\n[discovery] fetching active BTC 15m market…")
        try:
            summary = await live_session.run_for(duration=duration)
        except NoMatchingMarketError as exc:
            print(f"[discovery] NO ACTIVE MARKET: {exc}")
            print("[discovery] BTC 15m markets are 15-minute windows. Re-run when active.")
            return
        except AmbiguousMarketError as exc:
            print(f"[discovery] AMBIGUOUS MATCH: {exc}")
            return
        except GammaAPIError as exc:
            print(f"[discovery] API ERROR: {exc}")
            return

        _print_summary(summary)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Polymarket CLOB read-only live demo")
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
