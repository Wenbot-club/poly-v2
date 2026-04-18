"""Tests for bot/providers/normalize_rtds.py — deterministic, no network."""
from __future__ import annotations

import pytest

from bot.providers.normalize_rtds import normalize_binance_aggtrade


# Fixed clock for deterministic recv_timestamp_ms
_FIXED_NOW_MS = 1_700_000_001_500
_FIXED_NOW_FN = lambda: _FIXED_NOW_MS  # noqa: E731


# ---------------------------------------------------------------------------
# Valid aggTrade message
# ---------------------------------------------------------------------------

_VALID_WIRE = {
    "e": "aggTrade",
    "E": 1_700_000_001_000,
    "s": "BTCUSDT",
    "a": 12_345_678,
    "p": "42500.00",
    "q": "0.01",
    "f": 100,
    "l": 105,
    "T": 1_700_000_000_500,
    "m": False,
}

_EXPECTED = {
    "source": "binance",
    "symbol": "btc/usd",
    "timestamp_ms": 1_700_000_000_500,
    "recv_timestamp_ms": _FIXED_NOW_MS,
    "value": 42500.00,
    "sequence_no": 12_345_678,
}


def test_valid_aggtrade_returns_expected_shape():
    result = normalize_binance_aggtrade(_VALID_WIRE, now_fn=_FIXED_NOW_FN)
    assert result == _EXPECTED


def test_source_is_binance():
    result = normalize_binance_aggtrade(_VALID_WIRE, now_fn=_FIXED_NOW_FN)
    assert result["source"] == "binance"


def test_symbol_is_internal_canonical():
    result = normalize_binance_aggtrade(_VALID_WIRE, now_fn=_FIXED_NOW_FN)
    assert result["symbol"] == "btc/usd"


def test_timestamp_ms_comes_from_trade_time_T():
    result = normalize_binance_aggtrade(_VALID_WIRE, now_fn=_FIXED_NOW_FN)
    assert result["timestamp_ms"] == _VALID_WIRE["T"]


def test_recv_timestamp_ms_comes_from_now_fn():
    custom_now = 9_999_999_999
    result = normalize_binance_aggtrade(_VALID_WIRE, now_fn=lambda: custom_now)
    assert result["recv_timestamp_ms"] == custom_now


def test_value_is_float_from_price_string():
    result = normalize_binance_aggtrade(_VALID_WIRE, now_fn=_FIXED_NOW_FN)
    assert isinstance(result["value"], float)
    assert result["value"] == 42500.00


def test_sequence_no_is_int_from_agg_id():
    result = normalize_binance_aggtrade(_VALID_WIRE, now_fn=_FIXED_NOW_FN)
    assert isinstance(result["sequence_no"], int)
    assert result["sequence_no"] == 12_345_678


def test_integer_price_coerced_to_float():
    wire = {**_VALID_WIRE, "p": 42500}  # int, not string
    result = normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN)
    assert result is not None
    assert isinstance(result["value"], float)


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

def test_string_trade_time_coerced_to_int():
    wire = {**_VALID_WIRE, "T": "1700000000500"}
    result = normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN)
    assert result is not None
    assert isinstance(result["timestamp_ms"], int)
    assert result["timestamp_ms"] == 1_700_000_000_500


def test_string_agg_id_coerced_to_int():
    wire = {**_VALID_WIRE, "a": "12345678"}
    result = normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN)
    assert result is not None
    assert isinstance(result["sequence_no"], int)


# ---------------------------------------------------------------------------
# Unknown / rejected event types
# ---------------------------------------------------------------------------

def test_non_aggtrade_event_type_returns_none():
    wire = {**_VALID_WIRE, "e": "24hrMiniTicker"}
    assert normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN) is None


def test_missing_event_type_returns_none():
    wire = {k: v for k, v in _VALID_WIRE.items() if k != "e"}
    assert normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN) is None


def test_subscription_confirmation_returns_none():
    # Binance sends {"result": null, "id": 1} on subscription — not an aggTrade
    assert normalize_binance_aggtrade({"result": None, "id": 1}, now_fn=_FIXED_NOW_FN) is None


def test_empty_dict_returns_none():
    assert normalize_binance_aggtrade({}, now_fn=_FIXED_NOW_FN) is None


# ---------------------------------------------------------------------------
# Missing required fields
# ---------------------------------------------------------------------------

def test_missing_trade_time_T_returns_none():
    wire = {k: v for k, v in _VALID_WIRE.items() if k != "T"}
    assert normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN) is None


def test_missing_price_p_returns_none():
    wire = {k: v for k, v in _VALID_WIRE.items() if k != "p"}
    assert normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN) is None


def test_missing_agg_id_a_returns_none():
    wire = {k: v for k, v in _VALID_WIRE.items() if k != "a"}
    assert normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN) is None


# ---------------------------------------------------------------------------
# Malformed fields
# ---------------------------------------------------------------------------

def test_non_parsable_price_returns_none():
    wire = {**_VALID_WIRE, "p": "not_a_float"}
    assert normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN) is None


def test_non_parsable_trade_time_returns_none():
    wire = {**_VALID_WIRE, "T": "not_an_int"}
    assert normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN) is None


def test_none_price_returns_none():
    wire = {**_VALID_WIRE, "p": None}
    assert normalize_binance_aggtrade(wire, now_fn=_FIXED_NOW_FN) is None


# ---------------------------------------------------------------------------
# recv_timestamp_ms contract
# ---------------------------------------------------------------------------

def test_recv_timestamp_ms_from_now_fn_is_used():
    """
    recv_timestamp_ms is captured at local receive time via now_fn.
    Expected to be >= timestamp_ms under normal clock conditions,
    but not a guaranteed invariant (clock skew etc.).
    This test uses a controlled now_fn that is later than T.
    """
    trade_time = 1_700_000_000_500
    recv_time = 1_700_000_001_000   # 500 ms later — normal latency
    wire = {**_VALID_WIRE, "T": trade_time}
    result = normalize_binance_aggtrade(wire, now_fn=lambda: recv_time)
    assert result["recv_timestamp_ms"] == recv_time
    assert result["recv_timestamp_ms"] >= result["timestamp_ms"]
