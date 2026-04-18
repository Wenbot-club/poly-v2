"""
Read-only live demo: discover the active BTC 15m market via Gamma REST,
connect to the Polymarket CLOB WebSocket, and print normalized messages for
--duration seconds (default 10).

What this demo does NOT do:
  - no paper execution
  - no runner wiring
  - no REST book snapshot (not confirmed live; WS provides its own on subscribe)
  - no authentication

Snapshot-per-token behaviour (documented):
  Each token (YES, NO) receives its own "book" snapshot independently.
  A token is NOT counted as live until its own snapshot has arrived.
  If only one token has snapshotted by the end of the run, the summary says so.
  This is expected behaviour for this PR — global multi-token consistency is not
  claimed until both tokens have snapshotted.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict, Optional

import aiohttp

from bot.providers.polymarket_discovery import (
    AmbiguousMarketError,
    GammaAPIError,
    NoMatchingMarketError,
    PolymarketDiscoveryProvider,
)
from bot.providers.polymarket_market_data import CLOB_WS_URL, PolymarketMarketDataProvider


def _condensed_book(msg: Dict[str, Any]) -> str:
    bids = msg.get("bids", [])
    asks = msg.get("asks", [])
    top_bid = bids[0] if bids else None
    top_ask = asks[0] if asks else None
    return (
        f"asset={msg['asset_id'][:12]}…  "
        f"top_bid={top_bid['price'] if top_bid else 'n/a'}×{top_bid['size'] if top_bid else ''}  "
        f"top_ask={top_ask['price'] if top_ask else 'n/a'}×{top_ask['size'] if top_ask else ''}  "
        f"levels={len(bids)}b/{len(asks)}a  ts={msg['timestamp']}"
    )


def _condensed_price_change(msg: Dict[str, Any]) -> str:
    changes = msg.get("price_changes", [])
    if not changes:
        return f"ts={msg['timestamp']} (empty)"
    ch = changes[0]
    return (
        f"asset={ch['asset_id'][:12]}…  "
        f"{ch['side']} p={ch['price']} sz={ch['size']}  "
        f"best={ch.get('best_bid', '?')}/{ch.get('best_ask', '?')}  "
        f"ts={msg['timestamp']}  (+{len(changes) - 1} more changes)"
    )


async def run_demo(duration: int) -> None:
    print(f"{'=' * 60}")
    print("  Polymarket CLOB read-only live demo")
    print(f"  WS: {CLOB_WS_URL}")
    print(f"  Duration: {duration}s")
    print(f"{'=' * 60}")

    async with aiohttp.ClientSession() as session:
        # ------------------------------------------------------------------ #
        # 1. Discovery                                                         #
        # ------------------------------------------------------------------ #
        print("\n[discovery] fetching active BTC 15m market from Gamma…")
        discovery = PolymarketDiscoveryProvider(session)
        try:
            market = await discovery.find_active_btc_15m_market()
        except NoMatchingMarketError as exc:
            print(f"[discovery] NO ACTIVE MARKET FOUND: {exc}")
            print("[discovery] BTC 15m markets are time-bounded (15 min windows).")
            print("[discovery] Re-run when a market is active.")
            return
        except AmbiguousMarketError as exc:
            print(f"[discovery] AMBIGUOUS MATCH: {exc}")
            return
        except GammaAPIError as exc:
            print(f"[discovery] API ERROR: {exc}")
            return

        token_ids = [market.yes_token_id, market.no_token_id]
        print(f"[discovery] market   : {market.title}")
        print(f"[discovery] market_id: {market.market_id}")
        print(f"[discovery] yes_token: {market.yes_token_id}")
        print(f"[discovery] no_token : {market.no_token_id}")
        print(f"[discovery] window   : {market.start_ts_ms} → {market.end_ts_ms}")

        # ------------------------------------------------------------------ #
        # 2. Connect                                                           #
        # ------------------------------------------------------------------ #
        provider = PolymarketMarketDataProvider(session)
        await provider.connect(token_ids)
        print(f"\n[ws] connecting…  feed_state={provider.feed_state}")

        # Per-run counters and first-example tracking.
        counts: Dict[str, int] = {"book": 0, "price_change": 0, "other": 0, "total": 0}
        snapshotted: set[str] = set()
        first_book: Dict[str, str] = {}   # token_id → condensed string
        first_price_change: Optional[str] = None
        last_feed_state = provider.feed_state

        async def consume() -> None:
            nonlocal first_price_change, last_feed_state
            async for msg in provider.iter_messages():
                # Feed state change notification.
                if provider.feed_state != last_feed_state:
                    print(f"[ws] feed_state: {last_feed_state} → {provider.feed_state}")
                    last_feed_state = provider.feed_state

                event_type = msg.get("event_type", "unknown")
                counts["total"] += 1

                if event_type == "book":
                    counts["book"] += 1
                    asset_id = str(msg["asset_id"])
                    snapshotted.add(asset_id)
                    if asset_id not in first_book:
                        condensed = _condensed_book(msg)
                        first_book[asset_id] = condensed
                        role = "YES" if asset_id == market.yes_token_id else "NO"
                        print(f"[snapshot/{role}] {condensed}")

                elif event_type == "price_change":
                    counts["price_change"] += 1
                    if first_price_change is None:
                        first_price_change = _condensed_price_change(msg)
                        print(f"[update   ] {first_price_change}")

                else:
                    counts["other"] += 1

                # Progress heartbeat every 25 messages.
                if counts["total"] % 25 == 0:
                    print(
                        f"[progress] total={counts['total']}  "
                        f"book={counts['book']}  "
                        f"price_change={counts['price_change']}  "
                        f"other={counts['other']}  "
                        f"state={provider.feed_state}"
                    )

        # ------------------------------------------------------------------ #
        # 3. Run for `duration` seconds, then stop                            #
        # ------------------------------------------------------------------ #
        try:
            await asyncio.wait_for(consume(), timeout=float(duration))
        except asyncio.TimeoutError:
            pass  # normal exit after duration
        finally:
            await provider.close()

        # ------------------------------------------------------------------ #
        # 4. Summary                                                           #
        # ------------------------------------------------------------------ #
        print(f"\n{'=' * 60}")
        print("  Summary")
        print(f"{'=' * 60}")
        print(f"  duration        : {duration}s")
        print(f"  total messages  : {counts['total']}")
        print(f"  book snapshots  : {counts['book']}")
        print(f"  price_change    : {counts['price_change']}")
        print(f"  other           : {counts['other']}")
        print(f"  feed_state (end): {provider.feed_state}")

        yes_snapshotted = market.yes_token_id in snapshotted
        no_snapshotted = market.no_token_id in snapshotted
        print(f"\n  snapshot status:")
        print(f"    YES token: {'✓ received' if yes_snapshotted else '✗ NOT received — updates for this token were dropped'}")
        print(f"    NO  token: {'✓ received' if no_snapshotted else '✗ NOT received — updates for this token were dropped'}")

        if not yes_snapshotted or not no_snapshotted:
            print(
                "\n  NOTE: snapshot-per-token rule is in effect. "
                "Updates for any token without a snapshot are silently dropped. "
                "This is expected if the WS did not deliver a 'book' event for "
                "that token within the run window."
            )

        if first_price_change is None and counts["total"] > 0:
            print(
                "\n  NOTE: no price_change received. "
                "The market may be inactive during this window."
            )
        if counts["total"] == 0:
            print(
                "\n  WARNING: no messages received at all. "
                "Check WS URL, token IDs, and network connectivity."
            )
        print(f"{'=' * 60}")


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
        print("\n[interrupted] Ctrl+C received — exiting cleanly.")


if __name__ == "__main__":
    main()
