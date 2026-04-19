"""
Coinbase Exchange public REST ticker → internal RTDS format normalization.

Endpoint: GET https://api.exchange.coinbase.com/products/BTC-USD/ticker
No auth required. Public market data endpoint.

Wire response shape (Coinbase Exchange REST ticker):
  {
    "trade_id": 74,
    "price": "10.00",
    "size": "0.01",
    "time": "2014-11-07T22:19:28.578544Z",
    "bid": "9.90",
    "ask": "10.10",
    "volume": "100.18"
  }

Internal format produced (→ RTDSMessageRouter.apply()):
  {
    "source": "coinbase",
    "symbol": "btc/usd",
    "timestamp_ms": int,        # parsed from ISO-8601 "time" field
    "recv_timestamp_ms": int,   # captured at local receive time via now_fn
    "value": float,             # float(price) — last trade price
    "sequence_no": int,         # int(trade_id) — monotone per product
  }

This tick is routed by RTDSMessageRouter as a price-anchor (feeds
register_chainlink_tick internally). See ws_rtds.py for routing details.

This is a Coinbase anchor, NOT a Chainlink oracle. It is used as a practical
no-auth alternative to unblock fair value computation in live sessions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from ..domain import utc_now_ms

_INTERNAL_SYMBOL = "btc/usd"


def _parse_iso_to_ms(time_str: str) -> int:
    """Parse ISO-8601 UTC timestamp string to milliseconds since epoch."""
    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def normalize_coinbase_ticker(
    raw: Dict[str, Any],
    *,
    now_fn: Callable[[], int] = utc_now_ms,
) -> Optional[Dict[str, Any]]:
    """
    Normalize one Coinbase Exchange REST ticker response to internal RTDS format.

    Returns None for malformed/missing-field payloads. Never raises.
    Required fields: "trade_id", "price", "time".
    """
    try:
        trade_id = raw.get("trade_id")
        price = raw.get("price")
        time_str = raw.get("time")

        if any(v is None for v in (trade_id, price, time_str)):
            return None

        return {
            "source": "coinbase",
            "symbol": _INTERNAL_SYMBOL,
            "timestamp_ms": _parse_iso_to_ms(str(time_str)),
            "recv_timestamp_ms": now_fn(),
            "value": float(price),
            "sequence_no": int(trade_id),
        }
    except (ValueError, TypeError, AttributeError):
        return None
