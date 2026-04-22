"""
Read-only Binance BTC/USDT aggregate-trade signal provider over WebSocket.

Confirmed (Binance Spot API docs, April 2026):
  WS URL    : wss://stream.binance.com:9443/ws/btcusdt@aggTrade
  No auth   : public market data stream, no API key required
  Subscribe : not needed — stream URL already selects the feed
  Ping-pong : server pings every ~20 s; aiohttp responds automatically

iter_signals() yields normalized dicts (internal RTDS format) ready for
RTDSMessageRouter.apply(). Raw Binance wire bytes are never forwarded.

Stale detection:
  If no aggTrade message arrives within stale_timeout_ms, feed_state → "stale".
  Given BTC/USDT trade frequency, 30 s is a very conservative threshold.

Reconnect:
  On WS close or aiohttp error, reconnects with exponential backoff (cap 8 s).
  feed_state → "connecting" between attempts.

Chainlink (not implemented here):
  RTDSMessageRouter accepts source="chainlink", but Chainlink Data Streams
  requires authentication credentials and is out of scope for this PR.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Callable, Dict, List, Literal, Optional

import aiohttp

from ..domain import utc_now_ms
from .normalize_rtds import normalize_binance_aggtrade


BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
BINANCE_US_WS_URL = "wss://stream.binance.us:9443/ws/btcusdt@aggTrade"
_BACKOFF_DELAYS = (1.0, 2.0, 4.0, 8.0)


class BinanceSignalProvider:
    """
    Read-only live BTC/USD signal feed from Binance aggTrade stream.

    Satisfies the SignalProvider Protocol (bot.providers.base).
    connect() accepts "btc/usd" (internal canonical symbol).
    """

    source_name: str = "binance"
    feed_state: Literal["connecting", "live", "stale", "disconnected"]

    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws_url: str = BINANCE_WS_URL,
        stale_timeout_ms: int = 30_000,
        now_fn: Callable[[], int] = utc_now_ms,
    ) -> None:
        self.feed_state: Literal["connecting", "live", "stale", "disconnected"] = "disconnected"
        self._session = session
        self._ws_url = ws_url
        self._stale_timeout_ms = stale_timeout_ms
        self._now_fn = now_fn
        self._symbol: Optional[str] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    async def connect(self, symbol: str) -> None:
        """
        Store the target symbol and mark the feed as connecting.
        Actual WS connection is established inside iter_signals().

        Only "btc/usd" is supported in this provider; other symbols are
        accepted without error but will still receive BTCUSDT data.
        """
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
        """
        if self._symbol is None:
            raise RuntimeError("call connect(symbol) before iter_signals()")

        stale_timeout_s = self._stale_timeout_ms / 1000.0
        backoff_idx = 0

        while self.feed_state != "disconnected":
            try:
                async for tick in self._run_ws_session(stale_timeout_s):
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
        self, stale_timeout_s: float
    ) -> AsyncIterator[Dict[str, Any]]:
        """One WS connection lifetime: connect → receive loop → return on close."""
        async with self._session.ws_connect(self._ws_url) as ws:
            self._ws = ws

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

                    normalized = normalize_binance_aggtrade(data, now_fn=self._now_fn)
                    if normalized is None:
                        continue

                    yield normalized

                elif raw_msg.type == aiohttp.WSMsgType.PING:
                    # aiohttp auto-replies with PONG; explicit for clarity.
                    await ws.pong(raw_msg.data)

                elif raw_msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
