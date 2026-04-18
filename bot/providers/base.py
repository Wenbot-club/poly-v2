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
