"""
Read-only Coinbase Exchange BTC-USD anchor provider via REST polling.

Polls GET https://api.exchange.coinbase.com/products/BTC-USD/ticker
every poll_interval_ms milliseconds (default 1000 ms). No auth required.

Deduplicates on trade_id: if the same last trade is returned by consecutive
polls (no new trade in the interval), nothing is yielded.

Emits normalized dicts with source="coinbase" for RTDSMessageRouter.apply().
RTDSMessageRouter routes "coinbase" ticks to register_chainlink_tick() —
the internal anchor slot currently still named "chainlink" for compatibility.
See ws_rtds.py for routing and the note on naming.

This provider is a practical no-auth alternative to unblock live fair value
computation (FairValueEngine requires an anchor price). It is NOT Chainlink.
It is documented here as a Coinbase price anchor, not a Chainlink oracle.

feed_state transitions:
  connecting → live   : on first successful poll returning a new trade_id
  live → stale        : if no new trade_id is observed within stale_timeout_ms
  any → disconnected  : on close()
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, Dict, Literal, Optional

import aiohttp

from ..domain import utc_now_ms
from .normalize_coinbase import normalize_coinbase_ticker

COINBASE_TICKER_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"


class CoinbaseAnchorProvider:
    """
    Read-only BTC/USD anchor feed from Coinbase Exchange REST API.

    Satisfies the SignalProvider Protocol (bot.providers.base).
    Polls the Coinbase public ticker and deduplicates on trade_id.
    """

    source_name: str = "coinbase"

    def __init__(
        self,
        session: aiohttp.ClientSession,
        ticker_url: str = COINBASE_TICKER_URL,
        poll_interval_ms: int = 1000,
        stale_timeout_ms: int = 30_000,
        now_fn: Callable[[], int] = utc_now_ms,
    ) -> None:
        self.feed_state: Literal["connecting", "live", "stale", "disconnected"] = "disconnected"
        self._session = session
        self._ticker_url = ticker_url
        self._poll_interval_ms = poll_interval_ms
        self._stale_timeout_ms = stale_timeout_ms
        self._now_fn = now_fn
        self._symbol: Optional[str] = None

    async def connect(self, symbol: str) -> None:
        """Store the target symbol and mark the feed as connecting."""
        self._symbol = symbol
        self.feed_state = "connecting"

    async def close(self) -> None:
        """Mark the feed as disconnected; iter_signals() exits cleanly."""
        self.feed_state = "disconnected"

    async def iter_signals(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Yield normalized RTDS dicts until close() is called.

        Polls the Coinbase REST ticker every poll_interval_ms milliseconds.
        Deduplicates on sequence_no (trade_id). On HTTP or network errors,
        waits poll_interval_ms before retrying silently.
        """
        if self._symbol is None:
            raise RuntimeError("call connect(symbol) before iter_signals()")

        last_trade_id: Optional[int] = None
        last_new_tick_ms: Optional[int] = None
        poll_interval_s = self._poll_interval_ms / 1000.0

        while self.feed_state != "disconnected":
            try:
                async with self._session.get(self._ticker_url) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        tick = normalize_coinbase_ticker(data, now_fn=self._now_fn)
                        if tick is not None:
                            seq = tick["sequence_no"]
                            if seq != last_trade_id:
                                last_trade_id = seq
                                last_new_tick_ms = self._now_fn()
                                self.feed_state = "live"
                                yield tick
                            elif (
                                self.feed_state == "live"
                                and last_new_tick_ms is not None
                                and self._now_fn() - last_new_tick_ms > self._stale_timeout_ms
                            ):
                                self.feed_state = "stale"
            except (aiohttp.ClientError, OSError):
                if self.feed_state == "disconnected":
                    return

            await asyncio.sleep(poll_interval_s)
