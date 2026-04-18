"""
Polymarket CLOB WebSocket → internal domain message normalization.

Wire format reference (as of April 2026, Polymarket CLOB WS docs):
  - "book"         : full order-book snapshot for one token
  - "price_change" : incremental order-book update for one or more tokens
  - "last_trade_price", "tick_size_change", "best_bid_ask": single-token events

All string prices/sizes/timestamps from the wire are coerced to float/int.

The internal format produced here is the format consumed by MarketMessageRouter.
iter_messages() in PolymarketMarketDataProvider returns these normalized dicts,
never the raw wire payloads.

Snapshot vs. update distinction
  is_snapshot_message(raw)  → True for "book" events
  is_update_message(raw)    → True for "price_change" events

REST book snapshot endpoint
  TODO: confirm exact endpoint shape before implementing.
  Do not implement GET /book?token_id=... until tested live.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


_KNOWN_EVENT_TYPES = {
    "book",
    "price_change",
    "last_trade_price",
    "tick_size_change",
    "best_bid_ask",
}


def is_snapshot_message(raw: Dict[str, Any]) -> bool:
    return raw.get("event_type") == "book"


def is_update_message(raw: Dict[str, Any]) -> bool:
    return raw.get("event_type") == "price_change"


def normalize_market_message(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize one raw CLOB WS message to internal format.

    Returns None for unknown or malformed messages that should be discarded.
    Raises nothing — unknown fields are tolerated; missing required fields
    produce None.
    """
    event_type = raw.get("event_type")
    if event_type not in _KNOWN_EVENT_TYPES:
        return None

    try:
        if event_type == "book":
            return _normalize_book(raw)
        if event_type == "price_change":
            return _normalize_price_change(raw)
        if event_type == "last_trade_price":
            return _normalize_last_trade_price(raw)
        if event_type == "tick_size_change":
            return _normalize_tick_size_change(raw)
        if event_type == "best_bid_ask":
            return _normalize_best_bid_ask(raw)
    except (KeyError, ValueError, TypeError):
        return None

    return None  # unreachable, but satisfies type checker


# ---------------------------------------------------------------------------
# Individual event normalizers
# ---------------------------------------------------------------------------

def _normalize_book(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    asset_id = raw.get("asset_id")
    timestamp = raw.get("timestamp")
    if asset_id is None or timestamp is None:
        return None
    bids = _normalize_levels(raw.get("bids", []))
    asks = _normalize_levels(raw.get("asks", []))
    if bids is None or asks is None:
        return None
    return {
        "event_type": "book",
        "asset_id": str(asset_id),
        "timestamp": int(timestamp),
        "bids": bids,
        "asks": asks,
    }


def _normalize_price_change(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Wire shape (Polymarket CLOB WS):

      Single-token form (flat object per token):
        {
          "event_type": "price_change",
          "asset_id": "0x...",
          "timestamp": "1700000000000",   ← string
          "price": "0.52",
          "side": "BUY",
          "size": "10.0",
          "best_bid": "0.48",
          "best_ask": "0.52"
        }

      Multi-token form (array):
        {
          "event_type": "price_change",
          "changes": [{ same fields as above }, ...]
        }

    Both forms are normalised to the internal format expected by
    MarketMessageRouter._apply_price_change:
        {
          "event_type": "price_change",
          "timestamp": int,
          "price_changes": [
            {"asset_id": str, "price": float, "side": str,
             "size": float, "best_bid": float, "best_ask": float}
          ]
        }
    """
    # Determine timestamp — may live at top level or inside each change object.
    ts_raw = raw.get("timestamp")

    changes_raw: List[Dict[str, Any]]
    if "changes" in raw and isinstance(raw["changes"], list):
        changes_raw = raw["changes"]
    elif "asset_id" in raw:
        # Single flat object — wrap it
        changes_raw = [raw]
    else:
        return None

    if not changes_raw:
        return None

    # Resolve timestamp: top-level wins; fall back to first change entry.
    if ts_raw is None:
        ts_raw = changes_raw[0].get("timestamp")
    if ts_raw is None:
        return None
    ts_ms = int(ts_raw)

    normalized_changes: List[Dict[str, Any]] = []
    for ch in changes_raw:
        asset_id = ch.get("asset_id")
        price = ch.get("price")
        side = ch.get("side")
        size = ch.get("size")
        if any(v is None for v in (asset_id, price, side, size)):
            return None  # malformed change → reject whole message
        entry: Dict[str, Any] = {
            "asset_id": str(asset_id),
            "price": float(price),
            "side": str(side).upper(),
            "size": float(size),
        }
        if "best_bid" in ch and "best_ask" in ch:
            entry["best_bid"] = float(ch["best_bid"])
            entry["best_ask"] = float(ch["best_ask"])
        normalized_changes.append(entry)

    return {
        "event_type": "price_change",
        "timestamp": ts_ms,
        "price_changes": normalized_changes,
    }


def _normalize_last_trade_price(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    asset_id = raw.get("asset_id")
    price = raw.get("price")
    side = raw.get("side")
    timestamp = raw.get("timestamp")
    if any(v is None for v in (asset_id, price, side, timestamp)):
        return None
    return {
        "event_type": "last_trade_price",
        "asset_id": str(asset_id),
        "price": float(price),
        "side": str(side).upper(),
        "timestamp": int(timestamp),
    }


def _normalize_tick_size_change(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    asset_id = raw.get("asset_id")
    new_tick_size = raw.get("new_tick_size")
    timestamp = raw.get("timestamp")
    if any(v is None for v in (asset_id, new_tick_size, timestamp)):
        return None
    return {
        "event_type": "tick_size_change",
        "asset_id": str(asset_id),
        "new_tick_size": float(new_tick_size),
        "timestamp": int(timestamp),
    }


def _normalize_best_bid_ask(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    asset_id = raw.get("asset_id")
    best_bid = raw.get("best_bid")
    best_ask = raw.get("best_ask")
    timestamp = raw.get("timestamp")
    if any(v is None for v in (asset_id, best_bid, best_ask, timestamp)):
        return None
    out: Dict[str, Any] = {
        "event_type": "best_bid_ask",
        "asset_id": str(asset_id),
        "best_bid": float(best_bid),
        "best_ask": float(best_ask),
        "timestamp": int(timestamp),
    }
    if "spread" in raw:
        out["spread"] = float(raw["spread"])
    return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_levels(levels: Any) -> Optional[List[Dict[str, float]]]:
    """Convert a list of {price, size} dicts with string values to float."""
    if not isinstance(levels, list):
        return None
    result: List[Dict[str, float]] = []
    for lvl in levels:
        try:
            result.append({"price": float(lvl["price"]), "size": float(lvl["size"])})
        except (KeyError, ValueError, TypeError):
            return None
    return result
