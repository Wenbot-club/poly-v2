"""
Read-only Polymarket RTDS Chainlink BTC/USD signal provider over WebSocket.

Confirmed (Polymarket RTDS docs, April 2026):
  WS URL    : wss://ws-live-data.polymarket.com
  Topic     : crypto_prices_chainlink
  Filter    : {"symbol":"btc/usd"}
  Auth      : none required for crypto (Binance + Chainlink) feeds
  Ping      : send WS-level PING frame every 5 s to maintain connection

This is the Polymarket-relayed Chainlink BTC/USD price feed, not an
on-chain Chainlink query or Chainlink Data Streams (which require auth).

iter_signals() yields normalized dicts with source="chainlink" ready for
RTDSMessageRouter.apply() → register_chainlink_tick(). The "chainlink"
path in the router is native — no hacks.

Deduplication: outer envelope timestamp is used as sequence_no (monotone
transport proxy). On reconnect, duplicate sequence_nos are suppressed.

Stale detection:
  No TEXT message within stale_timeout_ms → feed_state → "stale".
  The ping loop keeps the TCP connection alive independently.

Reconnect:
  On WS close or aiohttp error, reconnects with exponential backoff (cap 8 s).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Callable, Dict, Literal, Optional

import aiohttp

from ..domain import utc_now_ms
from .normalize_polymarket_chainlink import normalize_polymarket_chainlink

POLYMARKET_WS_URL = "wss://ws-live-data.polymarket.com"
_BACKOFF_DELAYS = (1.0, 2.0, 4.0, 8.0)


class PolymarketChainlinkSignalProvider:
    """
    Read-only Polymarket RTDS Chainlink BTC/USD price feed.

    Satisfies the SignalProvider Protocol (bot.providers.base).
    connect() accepts "btc/usd" (internal canonical symbol).
    source_name = "chainlink" — routes natively through RTDSMessageRouter.
    """

    source_name: str = "chainlink"

    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws_url: str = POLYMARKET_WS_URL,
        stale_timeout_ms: int = 30_000,
        ping_interval_ms: int = 5_000,
        now_fn: Callable[[], int] = utc_now_ms,
    ) -> None:
        self.feed_state: Literal["connecting", "live", "stale", "disconnected"] = "disconnected"
        self._session = session
        self._ws_url = ws_url
        self._stale_timeout_ms = stale_timeout_ms
        self._ping_interval_ms = ping_interval_ms
        self._now_fn = now_fn
        self._symbol: Optional[str] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    async def connect(self, symbol: str) -> None:
        """Store the target symbol and mark the feed as connecting."""
        self._symbol = symbol
        self.feed_state = "connecting"

    async def close(self) -> None:
        """Signal stop and close the WS connection if open."""
        self.feed_state = "disconnected"
        ws = self._ws
        if ws is not None and not ws.closed:
            await ws.close()
        self._ws = None

    async def iter_signals(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield normalized RTDS dicts indefinitely until close() is called.
        Reconnects with exponential backoff on network errors.
        Deduplicates on sequence_no (outer envelope timestamp).
        """
        if self._symbol is None:
            raise RuntimeError("call connect(symbol) before iter_signals()")

        stale_timeout_s = self._stale_timeout_ms / 1000.0
        backoff_idx = 0
        last_sequence_no: Optional[int] = None

        while self.feed_state != "disconnected":
            try:
                async for tick in self._run_ws_session(self._symbol, stale_timeout_s):
                    seq = tick["sequence_no"]
                    if seq != last_sequence_no:
                        last_sequence_no = seq
                        backoff_idx = 0
                        yield tick
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                if self.feed_state == "disconnected":
                    return
                delay = _BACKOFF_DELAYS[min(backoff_idx, len(_BACKOFF_DELAYS) - 1)]
                backoff_idx += 1
                self.feed_state = "connecting"
                await asyncio.sleep(delay)

    async def _run_ws_session(
        self, symbol: str, stale_timeout_s: float
    ) -> AsyncIterator[Dict[str, Any]]:
        """One WS connection lifetime: connect → subscribe → receive loop → return on close."""
        subscribe_msg = json.dumps({
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": json.dumps({"symbol": symbol}),
            }]
        })

        async with self._session.ws_connect(self._ws_url) as ws:
            self._ws = ws
            await ws.send_str(subscribe_msg)

            ping_task = asyncio.create_task(self._ping_loop(ws))
            try:
                while True:
                    if self.feed_state == "disconnected":
                        return

                    try:
                        raw_msg = await asyncio.wait_for(
                            ws.receive(), timeout=stale_timeout_s
                        )
                    except asyncio.TimeoutError:
                        if self.feed_state == "live":
                            self.feed_state = "stale"
                        continue

                    if raw_msg.type == aiohttp.WSMsgType.TEXT:
                        self.feed_state = "live"
                        try:
                            data = json.loads(raw_msg.data)
                        except json.JSONDecodeError:
                            continue

                        if not isinstance(data, dict):
                            continue

                        normalized = normalize_polymarket_chainlink(
                            data, now_fn=self._now_fn
                        )
                        if normalized is None:
                            continue

                        yield normalized

                    elif raw_msg.type == aiohttp.WSMsgType.PING:
                        await ws.pong(raw_msg.data)

                    elif raw_msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break
            finally:
                ping_task.cancel()
                try:
                    await ping_task
                except asyncio.CancelledError:
                    pass

    async def _ping_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send WS-level PING frames every ping_interval_ms to keep the connection alive."""
        ping_interval_s = self._ping_interval_ms / 1000.0
        while not ws.closed and self.feed_state != "disconnected":
            await asyncio.sleep(ping_interval_s)
            if not ws.closed and self.feed_state != "disconnected":
                await ws.ping()
