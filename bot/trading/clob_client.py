"""
Minimal Polymarket CLOB REST client — L2 (API key) authentication only.

Implements only the subset needed for bootstrap validation:
  - HMAC-SHA256 request signing (stdlib hmac + hashlib + base64)
  - GET /data/trades — validates L2 credentials are accepted by the API

No external dependencies beyond aiohttp (already in requirements.txt).

Reference: https://docs.polymarket.com/#authentication
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any, Optional

import aiohttp

from bot.trading.credentials import Credentials

_CLOB_BASE = "https://clob.polymarket.com"
_TIMEOUT = aiohttp.ClientTimeout(total=15.0)


class ClobAuthError(RuntimeError):
    pass


class ClobRequestError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def _build_hmac_headers(
    creds: Credentials,
    method: str,
    path: str,
    body: str = "",
    timestamp: Optional[int] = None,
) -> dict[str, str]:
    """
    Build Polymarket L2 HMAC-SHA256 auth headers.

    Message = str(timestamp) + method.upper() + path + body
    Signature = urlsafe_base64( HMAC-SHA256( urlsafe_b64decode(secret), message ) )

    Matches py-clob-client's create_level_2_headers exactly — POLY_ADDRESS is
    the EOA signer address (derived from the private key), not the funder.
    No POLY_NONCE is sent for L2 auth.
    """
    ts = timestamp if timestamp is not None else int(time.time())
    message = f"{ts}{method.upper()}{path}{body}"

    # Polymarket returns URL-safe base64 secrets, sometimes without padding.
    secret_padded = creds.api_secret + "=" * (-len(creds.api_secret) % 4)
    secret_bytes = base64.urlsafe_b64decode(secret_padded)
    sig_bytes = hmac.new(secret_bytes, message.encode(), hashlib.sha256).digest()
    signature = base64.urlsafe_b64encode(sig_bytes).decode()

    return {
        "POLY_ADDRESS": creds.signer_address,
        "POLY_SIGNATURE": signature,
        "POLY_TIMESTAMP": str(ts),
        "POLY_API_KEY": creds.api_key,
        "POLY_PASSPHRASE": creds.api_passphrase,
        "Content-Type": "application/json",
    }


async def get_trades(
    session: aiohttp.ClientSession,
    creds: Credentials,
) -> Any:
    """
    GET /data/trades — returns the authenticated user's trade history.

    Used as a read-only L2-auth validation endpoint: if auth is correct the
    server returns 200 with a (possibly empty) list. Raises ClobAuthError on
    401/403 and ClobRequestError on other non-2xx responses.
    """
    path = "/data/trades"
    headers = _build_hmac_headers(creds, "GET", path)

    async with session.get(
        f"{_CLOB_BASE}{path}",
        headers=headers,
        timeout=_TIMEOUT,
    ) as resp:
        body = await resp.text()
        if resp.status in (401, 403):
            raise ClobAuthError(f"Authentication rejected (HTTP {resp.status}): {body}")
        if resp.status >= 400:
            raise ClobRequestError(resp.status, body)
        return json.loads(body)


async def dry_run_clob(
    session: aiohttp.ClientSession,
    creds: Credentials,
) -> dict[str, Any]:
    """
    Validate CLOB L2 credentials with a read-only GET /data/trades call.

    Returns {"ok": True, "address": ..., "trade_count": N} on success.
    Returns {"ok": False, "error": "..."} on auth failure or network error.
    """
    try:
        trades = await get_trades(session, creds)
        count = len(trades) if isinstance(trades, list) else 0
        return {"ok": True, "address": creds.funder_address, "trade_count": count}
    except ClobAuthError as e:
        return {"ok": False, "error": str(e)}
    except aiohttp.ClientError as e:
        return {"ok": False, "error": f"network error: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
