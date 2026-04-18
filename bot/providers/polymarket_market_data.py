"""
Read-only Polymarket CLOB WebSocket market-data provider.

Confirmed (April 2026, Polymarket CLOB WS docs + public captures):
  WS URL    : wss://ws-subscriptions-clob.polymarket.com/ws/market
  Subscribe : {"assets_ids": [...], "type": "market", "custom_feature_enabled": true}
  Snapshot  : event_type="book"  — full order book for one token, sent on subscribe
  Update    : event_type="price_change" — incremental level update for one token
  Other     : "last_trade_price", "tick_size_change", "best_bid_ask" — single-token events

Not confirmed / intentionally excluded:
  REST book snapshot (GET /book?token_id=...) — endpoint shape not confirmed live;
  do not implement until tested.  The WS provides its own book snapshot ("book"
  event) immediately after subscribe, so REST bootstrap is not needed for this PR.

Normalization contract:
  Wire → normalize_market_message() → internal format → emitted by iter_messages()
  Raw wire bytes are never forwarded to callers.

Snapshot-before-update rule (hard):
  Per-token tracking.  Any update (price_change, last_trade_price, best_bid_ask,
  tick_size_change) for a token that has not yet had a "book" snapshot is silently
  dropped.  On reconnect the snapshot set is cleared; all tokens must re-snapshot.

feed_state transitions:
  "disconnected" → connect() called → "connecting"
  "connecting"   → first valid TEXT WS message → "live"
  "live"         → no message within stale_timeout_ms → "stale"
  "stale"        → next valid TEXT WS message → "live"
  any            → close() called → "disconnected"

Reconnect:
  On unexpected WS close or aiohttp error, iter_messages() reconnects automatically
  with exponential backoff (cap: 8 s).  The snapshot set is cleared on each
  reconnect — callers will receive a fresh "book" snapshot before any updates.

close() safety:
  Closing the WS while iter_messages() is blocked on ws.receive() causes aiohttp to
  return a CLOSE frame, which breaks the receive loop cleanly.  No forced
  cancellation is needed.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

import aiohttp

from .normalize import normalize_market_message


CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Reconnect backoff delays in seconds; caps at the last value.
_BACKOFF_DELAYS = (1.0, 2.0, 4.0, 8.0)


class PolymarketMarketDataProvider:
    """
    Read-only live market data feed over the Polymarket CLOB WebSocket.

    Typical usage:
        async with aiohttp.ClientSession() as session:
            provider = PolymarketMarketDataProvider(session)
            await provider.connect(["0xtokenA", "0xtokenB"])
            try:
                async for msg in provider.iter_messages():
                    router.apply(state, msg)
            finally:
                await provider.close()

    iter_messages() runs until close() is called.  All messages are normalized
    (internal format, not wire format).  Updates before snapshot are silently dropped.
    """

    feed_state: Literal["connecting", "live", "stale", "disconnected"]

    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws_url: str = CLOB_WS_URL,
        stale_timeout_ms: int = 3000,
    ) -> None:
        self.feed_state: Literal["connecting", "live", "stale", "disconnected"] = "disconnected"
        self._session = session
        self._ws_url = ws_url
        self._stale_timeout_ms = stale_timeout_ms
        self._token_ids: List[str] = []
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    async def connect(self, token_ids: List[str]) -> None:
        """
        Store token IDs and mark the feed as about to connect.
        Actual WS handshake happens inside iter_messages().
        """
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        self._token_ids = list(token_ids)
        self.feed_state = "connecting"

    async def close(self) -> None:
        """
        Signal stop and close the WS connection if open.
        Any iter_messages() loop will exit on the next receive cycle.
        """
        self.feed_state = "disconnected"
        ws = self._ws
        if ws is not None and not ws.closed:
            await ws.close()
        self._ws = None

    async def iter_messages(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield normalized market messages indefinitely until close() is called.

        Reconnects with exponential backoff on network errors.
        Snapshot set is cleared on each reconnect attempt.
        """
        if not self._token_ids:
            raise RuntimeError("call connect(token_ids) before iter_messages()")

        stale_timeout_s = self._stale_timeout_ms / 1000.0
        backoff_idx = 0

        while self.feed_state != "disconnected":
            try:
                async for msg in self._run_ws_session(stale_timeout_s):
                    backoff_idx = 0  # reset on each successfully yielded message
                    yield msg
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                if self.feed_state == "disconnected":
                    return
                delay = _BACKOFF_DELAYS[min(backoff_idx, len(_BACKOFF_DELAYS) - 1)]
                backoff_idx += 1
                self.feed_state = "connecting"
                await asyncio.sleep(delay)

    async def _run_ws_session(
        self, stale_timeout_s: float
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        One WS connection lifetime: connect → subscribe → receive loop.
        Returns (stops yielding) when the WS closes or feed_state becomes "disconnected".
        """
        subscribe_payload = {
            "assets_ids": self._token_ids,
            "type": "market",
            "custom_feature_enabled": True,
        }
        # Per-connection snapshot tracking.  Cleared on every reconnect.
        snapshotted: set[str] = set()

        async with self._session.ws_connect(self._ws_url) as ws:
            self._ws = ws
            await ws.send_json(subscribe_payload)

            while True:
                if self.feed_state == "disconnected":
                    return

                try:
                    raw_msg = await asyncio.wait_for(
                        ws.receive(), timeout=stale_timeout_s
                    )
                except asyncio.TimeoutError:
                    # No message in stale_timeout_ms — mark stale, keep waiting.
                    if self.feed_state == "live":
                        self.feed_state = "stale"
                    continue

                if raw_msg.type == aiohttp.WSMsgType.TEXT:
                    self.feed_state = "live"
                    try:
                        data = json.loads(raw_msg.data)
                    except json.JSONDecodeError:
                        continue

                    # Polymarket sends individual objects; guard against future arrays.
                    events: List[Any] = data if isinstance(data, list) else [data]

                    for event in events:
                        if not isinstance(event, dict):
                            continue
                        normalized = normalize_market_message(event)
                        if normalized is None:
                            # Unknown or malformed event — silently drop.
                            continue

                        event_type = normalized.get("event_type")

                        if event_type == "book":
                            # Snapshot: always emit, register token as snapshotted.
                            snapshotted.add(str(normalized["asset_id"]))
                            yield normalized

                        elif event_type == "price_change":
                            # Update: drop if any referenced token lacks a snapshot.
                            referenced = {
                                ch["asset_id"]
                                for ch in normalized.get("price_changes", [])
                            }
                            if not referenced.issubset(snapshotted):
                                continue  # update before snapshot — dropped
                            yield normalized

                        else:
                            # last_trade_price, tick_size_change, best_bid_ask:
                            # single-token events — drop if token not yet snapshotted.
                            asset_id = normalized.get("asset_id")
                            if asset_id not in snapshotted:
                                continue  # update before snapshot — dropped
                            yield normalized

                elif raw_msg.type == aiohttp.WSMsgType.PING:
                    # aiohttp auto-replies with PONG; explicit reply here for clarity.
                    await ws.pong(raw_msg.data)

                elif raw_msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    # WS closed or errored — exit this session; outer loop reconnects.
                    break
