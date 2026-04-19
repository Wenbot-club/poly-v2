"""
Polymarket RTDS crypto_prices_chainlink → internal RTDS format normalization.

Endpoint  : wss://ws-live-data.polymarket.com
Topic     : crypto_prices_chainlink
Filter    : {"symbol":"btc/usd"}
Auth      : none required for crypto feeds

Documented wire format (official Polymarket RTDS docs):
  {
    "topic": "crypto_prices_chainlink",
    "type": "update",
    "timestamp": <server_envelope_ms>,
    "payload": {
      "symbol": "btc/usd",
      "timestamp": <price_ms>,
      "value": <float>
    }
  }

Empirically observed alternate format (NOT in official RTDS docs; handle
defensively and document as observed, not guaranteed):
  {
    "topic": "crypto_prices_chainlink",
    "type": "update",
    "timestamp": <server_envelope_ms>,
    "payload": {
      "data": [{"symbol": "btc/usd", "timestamp": <price_ms>, "value": <float>}]
    }
  }

Internal format produced (→ RTDSMessageRouter.apply()):
  {
    "source": "chainlink",
    "symbol": "btc/usd",
    "timestamp_ms": int,        # payload.timestamp — price data timestamp
    "recv_timestamp_ms": int,   # captured at local receive time via now_fn
    "value": float,             # payload.value — Chainlink price
    "sequence_no": int,         # outer["timestamp"] — server envelope ms,
                                #   used as a monotone transport-level proxy;
                                #   NOT a guaranteed business sequence number
  }
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..domain import utc_now_ms

_EXPECTED_TYPE = "update"
_EXPECTED_TOPIC = "crypto_prices_chainlink"


def normalize_polymarket_chainlink(
    raw: Dict[str, Any],
    *,
    now_fn: Callable[[], int] = utc_now_ms,
) -> Optional[Dict[str, Any]]:
    """
    Normalize one Polymarket RTDS crypto_prices_chainlink message.

    Accepts both the official payload format (direct payload.symbol/timestamp/value)
    and the empirically observed data[0] variant. Returns None for non-update
    messages, missing/malformed fields, or unsupported payload shapes. Never raises.
    """
    try:
        if raw.get("type") != _EXPECTED_TYPE:
            return None

        outer_ts = raw.get("timestamp")
        payload = raw.get("payload")

        if outer_ts is None or not isinstance(payload, dict):
            return None

        # Resolve inner record: official format or empirical data[0] variant.
        if "data" in payload:
            # Empirically observed — not in official RTDS docs.
            data = payload["data"]
            if not isinstance(data, list) or not data:
                return None
            inner = data[0]
            if not isinstance(inner, dict):
                return None
        else:
            # Documented official format.
            inner = payload

        symbol = inner.get("symbol")
        price_ts = inner.get("timestamp")
        value = inner.get("value")

        if any(v is None for v in (symbol, price_ts, value)):
            return None

        return {
            "source": "chainlink",
            "symbol": str(symbol),
            "timestamp_ms": int(price_ts),
            "recv_timestamp_ms": now_fn(),
            "value": float(value),
            "sequence_no": int(outer_ts),
        }
    except (ValueError, TypeError, KeyError, IndexError):
        return None
