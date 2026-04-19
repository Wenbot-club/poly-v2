from __future__ import annotations

from typing import AsyncIterator, Dict, Any, List, Literal, Protocol

from ..domain import MarketContext


class DiscoveryProvider(Protocol):
    """Sync discovery — satisfied by MockGammaClient-backed DiscoveryService."""
    def find_active_btc_15m_market(self) -> MarketContext: ...


class AsyncDiscoveryProvider(Protocol):
    """Async discovery — satisfied by PolymarketDiscoveryProvider."""
    async def find_active_btc_15m_market(self) -> MarketContext: ...


class MarketDataProvider(Protocol):
    """
    Read-only live market data feed.

    iter_messages() yields normalized internal-format dicts (not wire payloads).
    Normalization is performed inside the provider via normalize_market_message().

    feed_state transitions:
      connecting → live (on first successful WS message)
      live → stale (no message received within stale_timeout_ms)
      any → disconnected (on close() or unrecoverable error)
    """
    feed_state: Literal["connecting", "live", "stale", "disconnected"]

    async def connect(self, token_ids: List[str]) -> None: ...
    async def close(self) -> None: ...
    async def iter_messages(self) -> AsyncIterator[Dict[str, Any]]: ...


class SignalProvider(Protocol):
    """
    Read-only live price/signal feed (e.g. Binance aggTrade, Coinbase anchor).

    iter_signals() yields normalized internal RTDS dicts ready for
    RTDSMessageRouter.apply() — not raw wire payloads.

    Implemented for this repo:
      BinanceSignalProvider    — btcusdt@aggTrade WebSocket (no auth)
      CoinbaseAnchorProvider   — BTC-USD REST polling (no auth); feeds the
                                 internal price-anchor slot as a practical
                                 Chainlink alternative (not a Chainlink oracle)
      CompositeSignalProvider  — merges multiple providers into one stream

    Not yet implemented:
      Chainlink — Data Streams API requires credentials;
                  on-chain queries require a blockchain RPC node.

    source_name: canonical string identifying the feed source(s).
      e.g. "binance", "coinbase", "binance+coinbase"

    feed_state transitions:
      connecting → live (on first valid signal)
      live → stale (no signal within stale_timeout_ms)
      any → disconnected (on close())
    """
    source_name: str
    feed_state: Literal["connecting", "live", "stale", "disconnected"]

    async def connect(self, symbol: str) -> None: ...
    async def close(self) -> None: ...
    async def iter_signals(self) -> AsyncIterator[Dict[str, Any]]: ...
