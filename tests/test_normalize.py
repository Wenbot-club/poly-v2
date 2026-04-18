"""Tests for bot/providers/normalize.py — no network, fully deterministic."""
from __future__ import annotations

import pytest

from bot.providers.normalize import (
    is_snapshot_message,
    is_update_message,
    normalize_market_message,
)


# ---------------------------------------------------------------------------
# is_snapshot_message / is_update_message
# ---------------------------------------------------------------------------

def test_is_snapshot_message_book():
    assert is_snapshot_message({"event_type": "book"}) is True


def test_is_snapshot_message_rejects_price_change():
    assert is_snapshot_message({"event_type": "price_change"}) is False


def test_is_update_message_price_change():
    assert is_update_message({"event_type": "price_change"}) is True


def test_is_update_message_rejects_book():
    assert is_update_message({"event_type": "book"}) is False


# ---------------------------------------------------------------------------
# book snapshot — valid
# ---------------------------------------------------------------------------

_BOOK_WIRE = {
    "event_type": "book",
    "asset_id": "0xabc",
    "timestamp": "1700000000000",
    "bids": [{"price": "0.48", "size": "30.0"}, {"price": "0.47", "size": "50.0"}],
    "asks": [{"price": "0.52", "size": "25.0"}, {"price": "0.53", "size": "40.0"}],
}

_BOOK_EXPECTED = {
    "event_type": "book",
    "asset_id": "0xabc",
    "timestamp": 1700000000000,
    "bids": [{"price": 0.48, "size": 30.0}, {"price": 0.47, "size": 50.0}],
    "asks": [{"price": 0.52, "size": 25.0}, {"price": 0.53, "size": 40.0}],
}


def test_book_snapshot_valid():
    result = normalize_market_message(_BOOK_WIRE)
    assert result == _BOOK_EXPECTED


def test_book_snapshot_string_to_float_int():
    result = normalize_market_message(_BOOK_WIRE)
    assert isinstance(result["timestamp"], int)
    assert isinstance(result["bids"][0]["price"], float)
    assert isinstance(result["bids"][0]["size"], float)


def test_book_snapshot_empty_levels():
    raw = {**_BOOK_WIRE, "bids": [], "asks": []}
    result = normalize_market_message(raw)
    assert result is not None
    assert result["bids"] == []
    assert result["asks"] == []


def test_book_snapshot_integer_timestamp():
    raw = {**_BOOK_WIRE, "timestamp": 1700000000000}
    result = normalize_market_message(raw)
    assert result is not None
    assert result["timestamp"] == 1700000000000


# ---------------------------------------------------------------------------
# book snapshot — invalid / rejected
# ---------------------------------------------------------------------------

def test_book_missing_asset_id_rejected():
    raw = {k: v for k, v in _BOOK_WIRE.items() if k != "asset_id"}
    assert normalize_market_message(raw) is None


def test_book_missing_timestamp_rejected():
    raw = {k: v for k, v in _BOOK_WIRE.items() if k != "timestamp"}
    assert normalize_market_message(raw) is None


def test_book_malformed_level_price_rejected():
    raw = {**_BOOK_WIRE, "bids": [{"price": "not_a_float", "size": "1.0"}]}
    assert normalize_market_message(raw) is None


def test_book_level_missing_key_rejected():
    raw = {**_BOOK_WIRE, "asks": [{"price": "0.52"}]}  # no "size"
    assert normalize_market_message(raw) is None


# ---------------------------------------------------------------------------
# price_change (flat / single-token wire form) — valid
# ---------------------------------------------------------------------------

_PRICE_CHANGE_FLAT_WIRE = {
    "event_type": "price_change",
    "asset_id": "0xabc",
    "timestamp": "1700000001000",
    "price": "0.52",
    "side": "BUY",
    "size": "10.0",
    "best_bid": "0.48",
    "best_ask": "0.52",
}

_PRICE_CHANGE_EXPECTED = {
    "event_type": "price_change",
    "timestamp": 1700000001000,
    "price_changes": [
        {
            "asset_id": "0xabc",
            "price": 0.52,
            "side": "BUY",
            "size": 10.0,
            "best_bid": 0.48,
            "best_ask": 0.52,
        }
    ],
}


def test_price_change_flat_wire_valid():
    result = normalize_market_message(_PRICE_CHANGE_FLAT_WIRE)
    assert result == _PRICE_CHANGE_EXPECTED


def test_price_change_flat_no_best_bid_ask():
    raw = {k: v for k, v in _PRICE_CHANGE_FLAT_WIRE.items() if k not in ("best_bid", "best_ask")}
    result = normalize_market_message(raw)
    assert result is not None
    assert "best_bid" not in result["price_changes"][0]
    assert "best_ask" not in result["price_changes"][0]


def test_price_change_side_lowercased_normalized():
    raw = {**_PRICE_CHANGE_FLAT_WIRE, "side": "buy"}
    result = normalize_market_message(raw)
    assert result["price_changes"][0]["side"] == "BUY"


def test_price_change_string_to_float_int():
    result = normalize_market_message(_PRICE_CHANGE_FLAT_WIRE)
    ch = result["price_changes"][0]
    assert isinstance(result["timestamp"], int)
    assert isinstance(ch["price"], float)
    assert isinstance(ch["size"], float)
    assert isinstance(ch["best_bid"], float)
    assert isinstance(ch["best_ask"], float)


# ---------------------------------------------------------------------------
# price_change (multi-token "changes" array form) — valid
# ---------------------------------------------------------------------------

_PRICE_CHANGE_MULTI_WIRE = {
    "event_type": "price_change",
    "timestamp": "1700000002000",
    "changes": [
        {"asset_id": "0xabc", "price": "0.52", "side": "SELL", "size": "5.0", "best_bid": "0.48", "best_ask": "0.52"},
        {"asset_id": "0xdef", "price": "0.48", "side": "BUY", "size": "3.0"},
    ],
}


def test_price_change_multi_wire_valid():
    result = normalize_market_message(_PRICE_CHANGE_MULTI_WIRE)
    assert result is not None
    assert result["event_type"] == "price_change"
    assert result["timestamp"] == 1700000002000
    assert len(result["price_changes"]) == 2
    assert result["price_changes"][0]["asset_id"] == "0xabc"
    assert result["price_changes"][1]["asset_id"] == "0xdef"


# ---------------------------------------------------------------------------
# price_change — invalid / rejected
# ---------------------------------------------------------------------------

def test_price_change_missing_asset_id_rejected():
    raw = {k: v for k, v in _PRICE_CHANGE_FLAT_WIRE.items() if k != "asset_id"}
    assert normalize_market_message(raw) is None


def test_price_change_missing_timestamp_and_no_fallback_rejected():
    raw = {
        "event_type": "price_change",
        "changes": [{"asset_id": "0xabc", "price": "0.52", "side": "BUY", "size": "1.0"}],
        # no timestamp at top level, none in change entry
    }
    assert normalize_market_message(raw) is None


def test_price_change_malformed_price_rejected():
    raw = {**_PRICE_CHANGE_FLAT_WIRE, "price": "not_a_float"}
    assert normalize_market_message(raw) is None


def test_price_change_empty_changes_rejected():
    raw = {"event_type": "price_change", "timestamp": "1700000001000", "changes": []}
    assert normalize_market_message(raw) is None


# ---------------------------------------------------------------------------
# last_trade_price — valid
# ---------------------------------------------------------------------------

def test_last_trade_price_valid():
    raw = {
        "event_type": "last_trade_price",
        "asset_id": "0xabc",
        "price": "0.51",
        "side": "sell",
        "timestamp": "1700000003000",
    }
    result = normalize_market_message(raw)
    assert result == {
        "event_type": "last_trade_price",
        "asset_id": "0xabc",
        "price": 0.51,
        "side": "SELL",
        "timestamp": 1700000003000,
    }


def test_last_trade_price_missing_field_rejected():
    raw = {"event_type": "last_trade_price", "asset_id": "0xabc", "side": "BUY", "timestamp": "100"}
    # missing price
    assert normalize_market_message(raw) is None


# ---------------------------------------------------------------------------
# tick_size_change — valid
# ---------------------------------------------------------------------------

def test_tick_size_change_valid():
    raw = {
        "event_type": "tick_size_change",
        "asset_id": "0xabc",
        "new_tick_size": "0.001",
        "timestamp": "1700000004000",
    }
    result = normalize_market_message(raw)
    assert result == {
        "event_type": "tick_size_change",
        "asset_id": "0xabc",
        "new_tick_size": 0.001,
        "timestamp": 1700000004000,
    }


# ---------------------------------------------------------------------------
# best_bid_ask — valid
# ---------------------------------------------------------------------------

def test_best_bid_ask_valid():
    raw = {
        "event_type": "best_bid_ask",
        "asset_id": "0xabc",
        "best_bid": "0.49",
        "best_ask": "0.51",
        "spread": "0.02",
        "timestamp": "1700000005000",
    }
    result = normalize_market_message(raw)
    assert result == {
        "event_type": "best_bid_ask",
        "asset_id": "0xabc",
        "best_bid": 0.49,
        "best_ask": 0.51,
        "spread": 0.02,
        "timestamp": 1700000005000,
    }


def test_best_bid_ask_no_spread_field():
    raw = {
        "event_type": "best_bid_ask",
        "asset_id": "0xabc",
        "best_bid": "0.49",
        "best_ask": "0.51",
        "timestamp": "1700000005000",
    }
    result = normalize_market_message(raw)
    assert result is not None
    assert "spread" not in result


# ---------------------------------------------------------------------------
# Unknown / unsupported event types
# ---------------------------------------------------------------------------

def test_unknown_event_type_returns_none():
    assert normalize_market_message({"event_type": "heartbeat", "ts": 123}) is None


def test_missing_event_type_returns_none():
    assert normalize_market_message({"asset_id": "0xabc"}) is None


def test_empty_dict_returns_none():
    assert normalize_market_message({}) is None


def test_completely_unrelated_dict_returns_none():
    assert normalize_market_message({"foo": "bar", "baz": 42}) is None
