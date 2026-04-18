"""
Binance aggTrade WebSocket wire format → internal RTDS format normalization.

Confirmed endpoint (Binance Spot API, public, no auth):
  wss://stream.binance.com:9443/ws/btcusdt@aggTrade

Wire message shape (Binance aggTrade):
  {
    "e": "aggTrade",
    "E": <event_time_ms>,
    "s": "BTCUSDT",
    "a": <aggregate_trade_id>,
    "p": "<price_string>",
    "q": "<quantity_string>",
    "T": <trade_time_ms>,
    "m": <bool>,
    ...
  }

Internal format produced (→ RTDSMessageRouter.apply()):
  {
    "source": "binance",
    "symbol": "btc/usd",
    "timestamp_ms": int,        # T — trade time
    "recv_timestamp_ms": int,   # captured at local receive time via now_fn
    "value": float,             # float(p) — trade price
    "sequence_no": int,         # a — aggregate trade ID (monotone)
  }

recv_timestamp_ms note:
  Captured at local receive time; expected to be >= timestamp_ms under normal
  clock conditions. Exchange timestamps, clock skew, and network latency can
  in theory violate this — it is not a physical law. The now_fn parameter
  is injectable for deterministic testing.

Chainlink:
  RTDSMessageRouter also accepts source="chainlink", but no Chainlink provider
  exists in this PR. Chainlink Data Streams requires auth credentials; on-chain
  queries require a blockchain RPC node. Both are out of scope for this PR.
  Documented here, not hidden.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..domain import utc_now_ms


_EXPECTED_EVENT_TYPE = "aggTrade"
_INTERNAL_SYMBOL = "btc/usd"


def normalize_binance_aggtrade(
    raw: Dict[str, Any],
    *,
    now_fn: Callable[[], int] = utc_now_ms,
) -> Optional[Dict[str, Any]]:
    """
    Normalize one Binance aggTrade wire message to internal RTDS format.

    Returns None for non-aggTrade events (subscription confirmations, errors,
    unknown shapes) or malformed messages. Never raises.

    Required wire fields: "e" == "aggTrade", "T" (trade time), "p" (price), "a" (trade ID).
    """
    try:
        if raw.get("e") != _EXPECTED_EVENT_TYPE:
            return None

        trade_time = raw.get("T")
        price = raw.get("p")
        agg_id = raw.get("a")

        if any(v is None for v in (trade_time, price, agg_id)):
            return None

        return {
            "source": "binance",
            "symbol": _INTERNAL_SYMBOL,
            "timestamp_ms": int(trade_time),
            "recv_timestamp_ms": now_fn(),
            "value": float(price),
            "sequence_no": int(agg_id),
        }
    except (ValueError, TypeError):
        return None
