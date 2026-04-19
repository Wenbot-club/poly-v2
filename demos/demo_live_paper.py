"""
Conservative paper execution demo via LivePaperSession.

Connects to:
  - Polymarket CLOB WebSocket (market book feed)
  - Binance aggTrade WebSocket (BTC price signal)
  - Coinbase Exchange REST ticker (BTC-USD price anchor, polled every 1 s)

Signal feed: CompositeSignalProvider(Binance + Coinbase).
Coinbase ticks fill the internal price-anchor slot (last_chainlink) via
RTDSMessageRouter, unblocking FairValueEngine and enabling real decisions.

This is a Coinbase price anchor, NOT a Chainlink oracle. It is a practical
no-auth deblocker for this PR — documented here, not hidden.

With both feeds live and the Coinbase anchor providing a price, the decision
layer now computes fair value and posts simulated orders (orders_posted > 0
when market conditions are met). Contrast with the previous behaviour where
decision_count=0, orders_posted=0 because every fair value cycle was skipped.

Simulates order posting and fill via MockExecutionEngine — nothing is posted
to any exchange. All state mutations flow through the existing execution
infrastructure (MockExecutionEngine → QueueingUserRouter → UserMessageRouter).

No real orders, no real fills, no PnL reported.
"""
from __future__ import annotations

import argparse
import asyncio

import aiohttp

from bot.live_paper import LivePaperSession, LivePaperSummary
from bot.providers.binance_signal import BinanceSignalProvider
from bot.providers.coinbase_anchor import CoinbaseAnchorProvider
from bot.providers.composite_signal import CompositeSignalProvider
from bot.providers.polymarket_discovery import PolymarketDiscoveryProvider
from bot.providers.polymarket_market_data import PolymarketMarketDataProvider
from bot.settings import DEFAULT_CONFIG
from bot.strategy.baseline import QuotePolicy


def _print_summary(summary: LivePaperSummary) -> None:
    m = summary.market
    r = summary.rtds

    print(f"\n{'=' * 60}")
    print("  LivePaperSession summary")
    print(f"{'=' * 60}")

    print("\n  [market feed]")
    print(f"  market_id        : {m.market_id}")
    print(f"  total_messages   : {m.total_messages}")
    print(f"  book_count       : {m.book_count}")
    print(f"  yes_snapshotted  : {m.yes_snapshotted}")
    print(f"  final_feed_state : {m.final_feed_state}")

    print("\n  [rtds feed]")
    print(f"  source           : {r.source}")
    print(f"  total_ticks      : {r.total_ticks}")
    print(f"  min_value        : {r.min_value}")
    print(f"  max_value        : {r.max_value}")
    print(f"  final_feed_state : {r.final_feed_state}")

    print("\n  [decision layer]")
    print(f"  decision_count            : {summary.decision_count}")
    print(f"  skipped_fair_value_count  : {summary.skipped_fair_value_count}")
    print(f"  last_fair_value_error     : {summary.last_fair_value_error}")

    print("\n  [paper execution]")
    print(f"  orders_posted    : {summary.orders_posted}")
    print(f"  orders_cancelled : {summary.orders_cancelled}")
    print(f"  orders_rejected  : {summary.orders_rejected}")
    print(f"  fills_simulated  : {summary.fills_simulated}")
    print(f"  last_rejection   : {summary.last_rejection_reason}")

    print("\n  [final inventory]")
    print(f"  pusd_free  : {round(summary.final_pusd_free, 4)}")
    print(f"  up_free    : {round(summary.final_up_free, 6)}")

    if summary.skipped_fair_value_count > 0:
        print(
            f"\n  NOTE: {summary.skipped_fair_value_count} decision cycles skipped —"
            " anchor price not yet received. This is transient at startup."
        )

    print(f"{'=' * 60}")


async def run_demo(duration: int) -> None:
    print(f"{'=' * 60}")
    print("  Conservative paper execution demo")
    print(f"  Market  : Polymarket CLOB WebSocket")
    print(f"  Signal  : Binance aggTrade WebSocket + Coinbase REST anchor")
    print(f"  Anchor  : Coinbase Exchange (BTC-USD, polled every 1 s)")
    print(f"  NOTE    : Coinbase is the price anchor — NOT Chainlink")
    print(f"  Strategy: QuotePolicy (existing, bid-only)")
    print(f"  Engine  : MockExecutionEngine (no real orders)")
    print(f"  Capital : {DEFAULT_CONFIG.default_working_capital_usd} PUSD (from config)")
    print(f"  Duration: {duration}s")
    print(f"{'=' * 60}")

    async with aiohttp.ClientSession() as http_session:
        signal_provider = CompositeSignalProvider([
            BinanceSignalProvider(http_session),
            CoinbaseAnchorProvider(http_session),
        ])
        session = LivePaperSession(
            discovery=PolymarketDiscoveryProvider(http_session),
            market_provider=PolymarketMarketDataProvider(http_session),
            signal_provider=signal_provider,
            strategy=QuotePolicy(config=DEFAULT_CONFIG),
            config=DEFAULT_CONFIG,
        )

        print("\n[paper] discovering market and connecting feeds…")
        try:
            summary = await session.run_for(duration=duration)
        except Exception as exc:
            print(f"[paper] ERROR: {exc}")
            return

        _print_summary(summary)

        if session.state is not None:
            state = session.state
            print(f"\n[state] open_orders      : {len(state.open_orders)}")
            print(f"[state] binance_ticks    : {len(state.binance_ticks)}")
            print(f"[state] chainlink_ticks  : {len(state.chainlink_ticks)}")
            print(f"[state] tape_ewma        : {round(state.tape_ewma, 6)}")
            if state.last_binance is not None:
                print(f"[state] last BTC (Binance) : {state.last_binance.value}")
            if state.last_chainlink is not None:
                print(f"[state] last BTC (Coinbase): {state.last_chainlink.value}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Conservative paper execution demo")
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
