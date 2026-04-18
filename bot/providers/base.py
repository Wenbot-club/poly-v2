from __future__ import annotations

from typing import Protocol

from ..domain import MarketContext


class DiscoveryProvider(Protocol):
    def find_active_btc_15m_market(self) -> MarketContext: ...


class MarketDataProvider(Protocol):
    """Read-only live market data feed. To be implemented in Phase 2."""
    ...


class SignalProvider(Protocol):
    """External price/signal feed (e.g. Chainlink, Binance). To be implemented in Phase 2."""
    ...
