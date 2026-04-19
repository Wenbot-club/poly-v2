"""Tests for bot/providers/normalize_polymarket_chainlink.py — deterministic, no network."""
from __future__ import annotations

from bot.providers.normalize_polymarket_chainlink import normalize_polymarket_chainlink


_FIXED_NOW_MS = 1_753_400_000_000


def _now() -> int:
    return _FIXED_NOW_MS


# ---------------------------------------------------------------------------
# Documented official format — payload.symbol/timestamp/value
# ---------------------------------------------------------------------------

_VALID_OFFICIAL = {
    "topic": "crypto_prices_chainlink",
    "type": "update",
    "timestamp": 1_753_314_064_237,
    "payload": {
        "symbol": "btc/usd",
        "timestamp": 1_753_314_064_213,
        "value": 95432.10,
    },
}


def test_official_format_produces_correct_shape():
    result = normalize_polymarket_chainlink(_VALID_OFFICIAL, now_fn=_now)
    assert result is not None
    assert result["source"] == "chainlink"
    assert result["symbol"] == "btc/usd"
    assert isinstance(result["timestamp_ms"], int)
    assert isinstance(result["recv_timestamp_ms"], int)
    assert isinstance(result["value"], float)
    assert isinstance(result["sequence_no"], int)


def test_official_format_value_correct():
    result = normalize_polymarket_chainlink(_VALID_OFFICIAL, now_fn=_now)
    assert result is not None
    assert result["value"] == 95432.10


def test_official_format_timestamp_ms_from_payload():
    result = normalize_polymarket_chainlink(_VALID_OFFICIAL, now_fn=_now)
    assert result is not None
    assert result["timestamp_ms"] == 1_753_314_064_213


def test_official_format_recv_timestamp_from_now_fn():
    result = normalize_polymarket_chainlink(_VALID_OFFICIAL, now_fn=_now)
    assert result is not None
    assert result["recv_timestamp_ms"] == _FIXED_NOW_MS


def test_sequence_no_is_outer_timestamp():
    """sequence_no = outer envelope timestamp — monotone transport proxy, not a business ID."""
    result = normalize_polymarket_chainlink(_VALID_OFFICIAL, now_fn=_now)
    assert result is not None
    assert result["sequence_no"] == 1_753_314_064_237


def test_source_is_chainlink():
    result = normalize_polymarket_chainlink(_VALID_OFFICIAL, now_fn=_now)
    assert result is not None
    assert result["source"] == "chainlink"


# ---------------------------------------------------------------------------
# Live wire format — validated April 2026 via diag_chainlink.py
# No "type", no outer "timestamp", no "symbol" in entries, batch of ticks.
# data[-1] (most recent) is used.
# ---------------------------------------------------------------------------

_VALID_LIVE = {
    "payload": {
        "data": [
            {"timestamp": 1_776_607_253_000, "value": 76003.383},
            {"timestamp": 1_776_607_254_000, "value": 76001.11},
            {"timestamp": 1_776_607_259_000, "value": 75997.0},
        ]
    }
}


def test_live_format_accepted():
    result = normalize_polymarket_chainlink(_VALID_LIVE, now_fn=_now)
    assert result is not None
    assert result["source"] == "chainlink"
    assert result["symbol"] == "btc/usd"


def test_live_format_uses_last_batch_entry():
    result = normalize_polymarket_chainlink(_VALID_LIVE, now_fn=_now)
    assert result is not None
    assert result["value"] == 75997.0
    assert result["timestamp_ms"] == 1_776_607_259_000


def test_live_format_sequence_no_falls_back_to_inner_timestamp():
    result = normalize_polymarket_chainlink(_VALID_LIVE, now_fn=_now)
    assert result is not None
    assert result["sequence_no"] == 1_776_607_259_000


# ---------------------------------------------------------------------------
# Empirically observed alternate format — payload.data[0]
# (NOT in official RTDS docs; supported defensively, documented as observed)
# ---------------------------------------------------------------------------

_VALID_EMPIRICAL = {
    "topic": "crypto_prices_chainlink",
    "type": "update",
    "timestamp": 1_753_314_064_500,
    "payload": {
        "data": [
            {
                "symbol": "btc/usd",
                "timestamp": 1_753_314_064_480,
                "value": 95440.0,
            }
        ]
    },
}


def test_empirical_data0_format_produces_correct_shape():
    result = normalize_polymarket_chainlink(_VALID_EMPIRICAL, now_fn=_now)
    assert result is not None
    assert result["source"] == "chainlink"
    assert result["symbol"] == "btc/usd"
    assert result["value"] == 95440.0
    assert result["timestamp_ms"] == 1_753_314_064_480
    assert result["sequence_no"] == 1_753_314_064_500


def test_empirical_empty_data_list_returns_none():
    msg = dict(_VALID_EMPIRICAL)
    msg = {**msg, "payload": {"data": []}}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is None


def test_empirical_data_not_list_returns_none():
    msg = {**_VALID_EMPIRICAL, "payload": {"data": "not_a_list"}}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is None


# ---------------------------------------------------------------------------
# Non-update message types → None
# ---------------------------------------------------------------------------

def test_type_subscribe_is_accepted():
    # Live wire sends type="subscribe" for the initial snapshot batch — must be accepted.
    msg = {**_VALID_OFFICIAL, "type": "subscribe"}
    result = normalize_polymarket_chainlink(msg, now_fn=_now)
    assert result is not None
    assert result["source"] == "chainlink"


def test_type_missing_is_accepted():
    # Live format has no "type" field — must be accepted (not rejected).
    msg = {k: v for k, v in _VALID_OFFICIAL.items() if k != "type"}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is not None


def test_type_none_is_accepted():
    # type=None is treated same as absent — accepted.
    msg = {**_VALID_OFFICIAL, "type": None}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is not None


# ---------------------------------------------------------------------------
# Missing outer fields → None
# ---------------------------------------------------------------------------

def test_missing_outer_timestamp_uses_inner_timestamp_as_sequence_no():
    # Live format has no outer "timestamp" — sequence_no falls back to inner timestamp.
    msg = {k: v for k, v in _VALID_OFFICIAL.items() if k != "timestamp"}
    result = normalize_polymarket_chainlink(msg, now_fn=_now)
    assert result is not None
    assert result["sequence_no"] == _VALID_OFFICIAL["payload"]["timestamp"]


def test_missing_payload_returns_none():
    msg = {k: v for k, v in _VALID_OFFICIAL.items() if k != "payload"}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is None


def test_payload_not_dict_returns_none():
    msg = {**_VALID_OFFICIAL, "payload": "not_a_dict"}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is None


# ---------------------------------------------------------------------------
# Missing inner payload fields → None
# ---------------------------------------------------------------------------

def test_missing_symbol_defaults_to_btc_usd():
    # Live batch entries have no "symbol" — subscription filter guarantees btc/usd.
    payload = {k: v for k, v in _VALID_OFFICIAL["payload"].items() if k != "symbol"}
    msg = {**_VALID_OFFICIAL, "payload": payload}
    result = normalize_polymarket_chainlink(msg, now_fn=_now)
    assert result is not None
    assert result["symbol"] == "btc/usd"


def test_missing_timestamp_in_payload_returns_none():
    payload = {k: v for k, v in _VALID_OFFICIAL["payload"].items() if k != "timestamp"}
    msg = {**_VALID_OFFICIAL, "payload": payload}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is None


def test_missing_value_in_payload_returns_none():
    payload = {k: v for k, v in _VALID_OFFICIAL["payload"].items() if k != "value"}
    msg = {**_VALID_OFFICIAL, "payload": payload}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is None


# ---------------------------------------------------------------------------
# Malformed values → None (never raises)
# ---------------------------------------------------------------------------

def test_empty_dict_returns_none():
    assert normalize_polymarket_chainlink({}, now_fn=_now) is None


def test_non_numeric_value_returns_none():
    payload = {**_VALID_OFFICIAL["payload"], "value": "not_a_number"}
    msg = {**_VALID_OFFICIAL, "payload": payload}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is None


def test_non_numeric_outer_timestamp_returns_none():
    msg = {**_VALID_OFFICIAL, "timestamp": "bad_ts"}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is None


def test_none_value_returns_none():
    payload = {**_VALID_OFFICIAL["payload"], "value": None}
    msg = {**_VALID_OFFICIAL, "payload": payload}
    assert normalize_polymarket_chainlink(msg, now_fn=_now) is None


def test_none_outer_timestamp_uses_inner_timestamp_as_sequence_no():
    # timestamp=None treated same as absent — sequence_no falls back to inner timestamp.
    msg = {**_VALID_OFFICIAL, "timestamp": None}
    result = normalize_polymarket_chainlink(msg, now_fn=_now)
    assert result is not None
    assert result["sequence_no"] == _VALID_OFFICIAL["payload"]["timestamp"]


# ---------------------------------------------------------------------------
# recv_timestamp_ms injection
# ---------------------------------------------------------------------------

def test_recv_timestamp_ms_from_custom_now_fn():
    custom_ts = 9_876_543_210_000
    result = normalize_polymarket_chainlink(_VALID_OFFICIAL, now_fn=lambda: custom_ts)
    assert result is not None
    assert result["recv_timestamp_ms"] == custom_ts
