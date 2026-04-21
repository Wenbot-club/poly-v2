"""
Bootstrap trading module tests — 20 tests.

Tests 1-4   : GeoBlockResult helpers
Tests 5-7   : credentials.load_credentials
Tests 8-10  : clob_client HMAC auth header construction
Tests 11-12 : clob_client dry_run_clob (mocked HTTP)
Tests 13-16 : approvals._encode_call + check_approvals (mocked RPC)
Tests 17-20 : run_bootstrap orchestration (mocked all I/O)
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.trading.approvals import (
    ApprovalStatus,
    _decode_bool,
    _decode_uint256,
    _encode_call,
    _SEL_ALLOWANCE,
    _SEL_IS_APPROVED_FOR_ALL,
    check_approvals,
)
from bot.trading.bootstrap import BootstrapReport, run_bootstrap
from bot.trading.clob_client import ClobAuthError, _build_hmac_headers, dry_run_clob
from bot.trading.credentials import CredentialError, Credentials, load_credentials
from bot.trading.geoblock import EndpointStatus, GeoBlockResult, check_geoblock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_ENV = {
    "POLY_PRIVATE_KEY": "0xdeadbeef" + "00" * 28,
    "POLY_API_KEY": "test-api-key-uuid",
    "POLY_API_SECRET": "dGVzdHNlY3JldA==",  # base64("testsecret")
    "POLY_API_PASSPHRASE": "mypassphrase",
    "POLY_FUNDER_ADDRESS": "0xAbCd1234" + "00" * 16,
    "POLY_RPC_URL": "https://polygon-rpc.com",
}


def _make_creds(**overrides) -> Credentials:
    env = {**_GOOD_ENV, **overrides}
    return load_credentials(env)


# ---------------------------------------------------------------------------
# Tests 1-4: GeoBlockResult helpers
# ---------------------------------------------------------------------------

def test_geo_all_reachable():
    result = GeoBlockResult(endpoints={
        "a": EndpointStatus(url="https://a.com", reachable=True, status_code=200),
        "b": EndpointStatus(url="https://b.com", reachable=True, status_code=200),
    })
    assert result.all_reachable is True
    assert result.blocked_endpoints == []


def test_geo_one_blocked():
    result = GeoBlockResult(endpoints={
        "a": EndpointStatus(url="https://a.com", reachable=True, status_code=200),
        "b": EndpointStatus(url="https://b.com", reachable=False, status_code=451),
    })
    assert result.all_reachable is False
    assert "b" in result.blocked_endpoints


def test_geo_connection_error():
    result = GeoBlockResult(endpoints={
        "clob_api": EndpointStatus(
            url="https://clob.polymarket.com",
            reachable=False,
            error="connection error: ...",
        ),
    })
    assert result.all_reachable is False


def test_geo_check_mocked(monkeypatch):
    """check_geoblock returns correct result for mocked HTTP responses."""
    async def _fake_check(sess, name, url):
        return name, EndpointStatus(url=url, reachable=True, status_code=200)

    monkeypatch.setattr("bot.trading.geoblock._check_endpoint", _fake_check)
    result = asyncio.run(check_geoblock(endpoints={"test": "https://test.example"}))
    assert result.all_reachable is True


# ---------------------------------------------------------------------------
# Tests 5-7: credentials
# ---------------------------------------------------------------------------

def test_credentials_all_present():
    creds = load_credentials(_GOOD_ENV)
    assert creds.api_key == "test-api-key-uuid"
    assert creds.funder_address == _GOOD_ENV["POLY_FUNDER_ADDRESS"].lower()
    assert creds.rpc_url == "https://polygon-rpc.com"


def test_credentials_missing_raises():
    env = {k: v for k, v in _GOOD_ENV.items() if k != "POLY_API_KEY"}
    with pytest.raises(CredentialError) as exc_info:
        load_credentials(env)
    assert "POLY_API_KEY" in str(exc_info.value)


def test_credentials_default_rpc():
    env = {k: v for k, v in _GOOD_ENV.items() if k != "POLY_RPC_URL"}
    creds = load_credentials(env)
    assert creds.rpc_url == "https://polygon-rpc.com"


# ---------------------------------------------------------------------------
# Tests 8-10: HMAC header construction
# ---------------------------------------------------------------------------

def test_hmac_headers_keys_present():
    creds = _make_creds()
    headers = _build_hmac_headers(creds, "GET", "/profile", timestamp=1_700_000_000)
    for key in ("POLY_ADDRESS", "POLY_SIGNATURE", "POLY_TIMESTAMP", "POLY_NONCE",
                "POLY_API_KEY", "POLY_PASSPHRASE"):
        assert key in headers


def test_hmac_headers_timestamp_used():
    creds = _make_creds()
    h1 = _build_hmac_headers(creds, "GET", "/profile", timestamp=1_000)
    h2 = _build_hmac_headers(creds, "GET", "/profile", timestamp=2_000)
    assert h1["POLY_TIMESTAMP"] == "1000"
    assert h2["POLY_TIMESTAMP"] == "2000"
    # Different timestamps → different signatures
    assert h1["POLY_SIGNATURE"] != h2["POLY_SIGNATURE"]


def test_hmac_signature_deterministic():
    creds = _make_creds()
    h1 = _build_hmac_headers(creds, "GET", "/orders", timestamp=999)
    h2 = _build_hmac_headers(creds, "GET", "/orders", timestamp=999)
    assert h1["POLY_SIGNATURE"] == h2["POLY_SIGNATURE"]


# ---------------------------------------------------------------------------
# Tests 11-12: dry_run_clob (mocked HTTP)
# ---------------------------------------------------------------------------

def _mock_session(status: int, body: str) -> MagicMock:
    """Build a minimal aiohttp.ClientSession mock."""
    resp = AsyncMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.get = MagicMock(return_value=resp)
    return session


def test_dry_run_clob_ok():
    creds = _make_creds()
    profile_body = json.dumps({"address": creds.funder_address})
    sess = _mock_session(200, profile_body)
    result = asyncio.run(dry_run_clob(sess, creds))
    assert result["ok"] is True
    assert result["address"] == creds.funder_address


def test_dry_run_clob_auth_fail():
    creds = _make_creds()
    sess = _mock_session(401, '{"error":"unauthorized"}')
    result = asyncio.run(dry_run_clob(sess, creds))
    assert result["ok"] is False
    assert "Authentication rejected" in result["error"]


# ---------------------------------------------------------------------------
# Tests 13-16: approvals
# ---------------------------------------------------------------------------

def test_encode_call_allowance():
    data = _encode_call(
        _SEL_ALLOWANCE,
        "0xAbCd000000000000000000000000000000000001",
        "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    )
    assert data.startswith("0xdd62ed3e")
    assert len(data) == 2 + 8 + 64 + 64  # 0x + selector + 2 args


def test_decode_uint256():
    hex_val = "0x" + "00" * 31 + "0a"  # 10
    assert _decode_uint256(hex_val) == 10


def test_decode_bool_true():
    hex_val = "0x" + "00" * 31 + "01"
    assert _decode_bool(hex_val) is True


def test_decode_bool_false():
    assert _decode_bool("0x" + "00" * 32) is False


def _mock_rpc_session(responses: list[str]) -> MagicMock:
    """Mock aiohttp session returning sequential JSON-RPC results."""
    call_count = {"n": 0}

    class _FakeResp:
        def __init__(self, result: str) -> None:
            self._result = result

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self, **kw):
            return {"jsonrpc": "2.0", "id": 1, "result": self._result}

    def _fake_post(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return _FakeResp(responses[idx])

    session = MagicMock()
    session.post = _fake_post
    return session


def test_check_approvals_all_ok():
    # allowance = 1_000_000_000 (> threshold), both bools True
    allowance_hex = "0x" + hex(1_000_000_000)[2:].zfill(64)
    true_hex = "0x" + "00" * 31 + "01"
    sess = _mock_rpc_session([allowance_hex, true_hex, true_hex])
    result = asyncio.run(check_approvals(sess, "0xfunder", "https://rpc.example"))
    assert result.usdc_approved_ctf is True
    assert result.ctf_approved_neg_risk is True
    assert result.neg_risk_approved_adapter is True
    assert result.ready_to_trade is True
    assert result.error is None


def test_check_approvals_not_ready():
    zero_hex = "0x" + "00" * 32
    sess = _mock_rpc_session([zero_hex, zero_hex, zero_hex])
    result = asyncio.run(check_approvals(sess, "0xfunder", "https://rpc.example"))
    assert result.ready_to_trade is False
    assert result.usdc_approved_ctf is False


# ---------------------------------------------------------------------------
# Tests 17-20: run_bootstrap orchestration
# ---------------------------------------------------------------------------

def _patched_bootstrap(
    geoblock_ok=True,
    creds_env=None,
    clob_ok=True,
    approvals_ready=True,
):
    """Helper: patch all I/O for bootstrap tests."""
    from bot.trading.geoblock import EndpointStatus, GeoBlockResult

    geo = GeoBlockResult(endpoints={
        "polymarket_web": EndpointStatus(
            url="https://polymarket.com",
            reachable=geoblock_ok,
            status_code=200 if geoblock_ok else 451,
        ),
    })

    approval = ApprovalStatus(
        usdc_allowance_ctf=10_000_000 if approvals_ready else 0,
        usdc_approved_ctf=approvals_ready,
        ctf_approved_neg_risk=approvals_ready,
        neg_risk_approved_adapter=approvals_ready,
    )

    clob_result = (
        {"ok": True, "address": "0xfunder", "profile": {}}
        if clob_ok
        else {"ok": False, "error": "auth failed"}
    )

    env = creds_env if creds_env is not None else _GOOD_ENV

    with patch("bot.trading.bootstrap.check_geoblock", AsyncMock(return_value=geo)), \
         patch("bot.trading.bootstrap.dry_run_clob", AsyncMock(return_value=clob_result)), \
         patch("bot.trading.bootstrap.check_approvals", AsyncMock(return_value=approval)):

        async def _run():
            async with __import__("aiohttp").ClientSession() as sess:
                return await run_bootstrap(session=sess, env=env)

        # We need to mock the aiohttp.ClientSession to avoid real network calls
        with patch("aiohttp.ClientSession") as mock_cls:
            mock_sess = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_sess)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            return asyncio.run(run_bootstrap(session=mock_sess, env=env))


def test_bootstrap_all_ok():
    report = _patched_bootstrap()
    assert report.ready is True
    assert report.issues == []


def test_bootstrap_geoblock_fail():
    report = _patched_bootstrap(geoblock_ok=False)
    assert report.ready is False
    assert any("geoblock" in i for i in report.issues)


def test_bootstrap_clob_fail():
    report = _patched_bootstrap(clob_ok=False)
    assert report.ready is False
    assert any("clob" in i for i in report.issues)


def test_bootstrap_approvals_fail():
    report = _patched_bootstrap(approvals_ready=False)
    assert report.ready is False
    assert any("approvals" in i for i in report.issues)


def test_bootstrap_missing_creds():
    bad_env = {k: v for k, v in _GOOD_ENV.items()
               if k not in ("POLY_API_KEY", "POLY_API_SECRET")}
    report = _patched_bootstrap(creds_env=bad_env)
    assert report.ready is False
    assert any("credentials" in i for i in report.issues)
