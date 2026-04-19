"""Tests for bot/providers/normalize_coinbase.py — deterministic, no network."""
from __future__ import annotations

from bot.providers.normalize_coinbase import normalize_coinbase_ticker


_VALID_PAYLOAD = {
    "trade_id": 12345678,
    "price": "95432.10",
    "size": "0.00500",
    "time": "2024-11-07T22:19:28.578544Z",
    "bid": "95430.00",
    "ask": "95434.00",
    "volume": "12345.67",
}

_FIXED_NOW_MS = 1_731_025_200_000


def _now() -> int:
    return _FIXED_NOW_MS


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_ticker_produces_correct_shape():
    result = normalize_coinbase_ticker(_VALID_PAYLOAD, now_fn=_now)
    assert result is not None
    assert result["source"] == "coinbase"
    assert result["symbol"] == "btc/usd"
    assert result["recv_timestamp_ms"] == _FIXED_NOW_MS
    assert isinstance(result["timestamp_ms"], int)
    assert isinstance(result["value"], float)
    assert isinstance(result["sequence_no"], int)


def test_price_string_converted_to_float():
    result = normalize_coinbase_ticker(_VALID_PAYLOAD, now_fn=_now)
    assert result is not None
    assert result["value"] == 95432.10


def test_trade_id_converted_to_int():
    result = normalize_coinbase_ticker(_VALID_PAYLOAD, now_fn=_now)
    assert result is not None
    assert result["sequence_no"] == 12345678


def test_time_iso_parsed_to_timestamp_ms():
    result = normalize_coinbase_ticker(_VALID_PAYLOAD, now_fn=_now)
    assert result is not None
    # "2024-11-07T22:19:28.578544Z" → 1731017968578 ms (verified empirically)
    assert result["timestamp_ms"] == 1_731_017_968_578


def test_source_is_coinbase():
    result = normalize_coinbase_ticker(_VALID_PAYLOAD, now_fn=_now)
    assert result is not None
    assert result["source"] == "coinbase"


def test_symbol_is_btc_usd():
    result = normalize_coinbase_ticker(_VALID_PAYLOAD, now_fn=_now)
    assert result is not None
    assert result["symbol"] == "btc/usd"


def test_recv_timestamp_ms_comes_from_now_fn():
    custom_now = 9_999_999_999_000
    result = normalize_coinbase_ticker(_VALID_PAYLOAD, now_fn=lambda: custom_now)
    assert result is not None
    assert result["recv_timestamp_ms"] == custom_now


# ---------------------------------------------------------------------------
# String price edge cases
# ---------------------------------------------------------------------------

def test_integer_trade_id_as_int_in_payload():
    payload = dict(_VALID_PAYLOAD)
    payload["trade_id"] = 42
    result = normalize_coinbase_ticker(payload, now_fn=_now)
    assert result is not None
    assert result["sequence_no"] == 42


def test_price_with_many_decimal_places():
    payload = dict(_VALID_PAYLOAD)
    payload["price"] = "95432.123456789"
    result = normalize_coinbase_ticker(payload, now_fn=_now)
    assert result is not None
    assert abs(result["value"] - 95432.123456789) < 1e-6


# ---------------------------------------------------------------------------
# Missing fields → None
# ---------------------------------------------------------------------------

def test_missing_trade_id_returns_none():
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "trade_id"}
    assert normalize_coinbase_ticker(payload, now_fn=_now) is None


def test_missing_price_returns_none():
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "price"}
    assert normalize_coinbase_ticker(payload, now_fn=_now) is None


def test_missing_time_returns_none():
    payload = {k: v for k, v in _VALID_PAYLOAD.items() if k != "time"}
    assert normalize_coinbase_ticker(payload, now_fn=_now) is None


# ---------------------------------------------------------------------------
# Malformed payloads → None (never raises)
# ---------------------------------------------------------------------------

def test_empty_dict_returns_none():
    assert normalize_coinbase_ticker({}, now_fn=_now) is None


def test_non_numeric_price_returns_none():
    payload = dict(_VALID_PAYLOAD)
    payload["price"] = "not_a_number"
    assert normalize_coinbase_ticker(payload, now_fn=_now) is None


def test_non_numeric_trade_id_returns_none():
    payload = dict(_VALID_PAYLOAD)
    payload["trade_id"] = "bad_id"
    assert normalize_coinbase_ticker(payload, now_fn=_now) is None


def test_malformed_time_string_returns_none():
    payload = dict(_VALID_PAYLOAD)
    payload["time"] = "not-a-date"
    assert normalize_coinbase_ticker(payload, now_fn=_now) is None


def test_none_price_returns_none():
    payload = dict(_VALID_PAYLOAD)
    payload["price"] = None
    assert normalize_coinbase_ticker(payload, now_fn=_now) is None


def test_none_trade_id_returns_none():
    payload = dict(_VALID_PAYLOAD)
    payload["trade_id"] = None
    assert normalize_coinbase_ticker(payload, now_fn=_now) is None
