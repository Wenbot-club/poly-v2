"""
Polymarket RTDS crypto_prices_chainlink → internal RTDS format normalization.

Endpoint  : wss://ws-live-data.polymarket.com
Topic     : crypto_prices_chainlink
Filter    : {"symbol":"btc/usd"}
Auth      : none required for crypto feeds

Live-validated wire format (April 2026 — diag_chainlink.py confirmed):
  {
    "payload": {
      "data": [
        {"timestamp": <price_ms>, "value": <float>},
        ...                                           ← batch of recent ticks
      ]
    }
  }
  Notable: no top-level "type", no top-level "timestamp", no "symbol" per entry.
  The subscription filter guarantees the topic is btc/usd.
  The batch is ordered oldest→newest; we take data[-1] (most recent tick).

Subscription ACK shape (also received, must be silently dropped):
  {"action": "subscribed", ...}  ← no "payload" key → safely returns None.

Documented wire format (official RTDS docs — NOT observed live):
  {
    "topic": "crypto_prices_chainlink",
    "type": "update",
    "timestamp": <server_envelope_ms>,
    "payload": {"symbol": "btc/usd", "timestamp": <price_ms>, "value": <float>}
  }
  Still handled for forward compatibility.

Internal format produced (→ RTDSMessageRouter.apply()):
  {
    "source": "chainlink",
    "symbol": "btc/usd",
    "timestamp_ms": int,        # inner.timestamp — price data timestamp
    "recv_timestamp_ms": int,   # captured at local receive time via now_fn
    "value": float,             # inner.value — Chainlink BTC/USD price
    "sequence_no": int,         # outer["timestamp"] if present, else inner["timestamp"]
  }
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..domain import utc_now_ms

_DEFAULT_SYMBOL = "btc/usd"


def normalize_polymarket_chainlink(
    raw: Dict[str, Any],
    *,
    now_fn: Callable[[], int] = utc_now_ms,
) -> Optional[Dict[str, Any]]:
    """
    Normalize one Polymarket RTDS crypto_prices_chainlink message.

    Handles the live batch format (payload.data list, no type field) and the
    documented single-record format (type="update", payload with symbol).
    Returns None for subscription ACKs, malformed payloads, or missing fields.
    Never raises.
    """
    try:
        # No type check — rely on payload structure alone.
        # Live-validated: snapshot messages use type="subscribe", updates may
        # differ. Subscription ACKs have no "payload" key → fall through to None.
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            return None

        outer_ts = raw.get("timestamp")  # present in documented format, absent live

        # Resolve inner record.
        if "data" in payload:
            # Live format: batch of ticks ordered oldest→newest.
            data = payload["data"]
            if not isinstance(data, list) or not data:
                return None
            inner = data[-1]   # most recent tick
            if not isinstance(inner, dict):
                return None
        else:
            # Documented single-record format.
            inner = payload

        price_ts = inner.get("timestamp")
        value = inner.get("value")

        if price_ts is None or value is None:
            return None

        # symbol: from inner if present (documented format), else subscription
        # filter guarantees btc/usd.
        symbol = str(inner.get("symbol") or raw.get("symbol") or _DEFAULT_SYMBOL)
        sequence_no = int(outer_ts) if outer_ts is not None else int(price_ts)

        return {
            "source": "chainlink",
            "symbol": symbol,
            "timestamp_ms": int(price_ts),
            "recv_timestamp_ms": now_fn(),
            "value": float(value),
            "sequence_no": sequence_no,
        }
    except (ValueError, TypeError, KeyError, IndexError):
        return None
