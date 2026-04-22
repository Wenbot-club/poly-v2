"""
Read-only Coinbase Exchange BTC/USD (or ETH/USD) ticker signal provider.

Coinbase Exchange WebSocket is accessible from AWS datacenter IPs, unlike
Binance (stream.binance.com HTTP 451, stream.binance.us silent from AWS).

WS URL  : wss://ws-feed.exchange.coinbase.com
No auth : public ticker channel requires no API key
Protocol:
  1. Connect → immediately send subscribe message
  2. Server responds with "subscriptions" ack, then "ticker" messages per trade
  3. Server sends no explicit ping — we rely on stale detection (30 s default)

iter_signals() yields normalized dicts compatible with the Binance provider:
  {"source": "coinbase", "symbol": <symbol>, "value": <float price>,
   "recv_timestamp_ms": <int ms>}
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Callable, Dict, Literal, Optional

import aiohttp

from ..domain import utc_now_ms


COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"

_PRODUCT_MAP: Dict[str, str] = {
    "btc/usd": "BTC-USD",
    "eth/usd": "ETH-USD",
}

_BACKOFF_DELAYS = (1.0, 2.0, 4.0, 8.0)


class CoinbaseSignalProvider:
    """
    Read-only live price feed from Coinbase Exchange WebSocket ticker channel.

    Satisfies the SignalProvider Protocol (bot.providers.base).
    connect() accepts "btc/usd" or "eth/usd".
    """

    source_name: str = "coinbase"
    feed_state: Literal["connecting", "live", "stale", "disconnected"]

    def __init__(
        self,
        session: aiohttp.ClientSession,
        ws_url: str = COINBASE_WS_URL,
        stale_timeout_ms: int = 30_000,
        now_fn: Callable[[], int] = utc_now_ms,
    ) -> None:
        self.feed_state: Literal["connecting", "live", "stale", "disconnected"] = "disconnected"
        self._session = session
        self._ws_url = ws_url
        self._stale_timeout_ms = stale_timeout_ms
        self._now_fn = now_fn
        self._symbol: Optional[str] = None
        self._product_id: Optional[str] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    async def connect(self, symbol: str) -> None:
        self._symbol = symbol
        self._product_id = _PRODUCT_MAP.get(symbol.lower(), "BTC-USD")
        self.feed_state = "connecting"

    async def close(self) -> None:
        self.feed_state = "disconnected"
        ws = self._ws
        if ws is not None and not ws.closed:
            await ws.close()
        self._ws = None

    async def iter_signals(self) -> AsyncIterator[Dict[str, Any]]:
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
        async with self._session.ws_connect(self._ws_url) as ws:
            self._ws = ws

            # Subscribe immediately after connect
            await ws.send_json({
                "type": "subscribe",
                "channels": [{"name": "ticker", "product_ids": [self._product_id]}],
            })

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
                    try:
                        data = json.loads(raw_msg.data)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(data, dict):
                        continue

                    # Only process trade-driven ticker updates
                    if data.get("type") != "ticker":
                        continue

                    price_str = data.get("price")
                    if not price_str:
                        continue

                    try:
                        price = float(price_str)
                    except (ValueError, TypeError):
                        continue

                    self.feed_state = "live"
                    yield {
                        "source": "coinbase",
                        "symbol": self._symbol,
                        "value": price,
                        "recv_timestamp_ms": self._now_fn(),
                    }

                elif raw_msg.type == aiohttp.WSMsgType.PING:
                    await ws.pong(raw_msg.data)

                elif raw_msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    break
