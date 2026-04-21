"""
Minimal Polymarket CLOB REST client — L2 (API key) authentication only.

Implements only the subset needed for bootstrap validation:
  - HMAC-SHA256 request signing (stdlib hmac + hashlib + base64)
  - GET /profile — validates credentials are accepted by the API

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
    nonce: int = 0,
) -> dict[str, str]:
    """
    Build Polymarket L2 HMAC-SHA256 auth headers.

    Message = str(timestamp) + method.upper() + path + body
    Signature = base64( HMAC-SHA256( base64decode(secret), message ) )
    """
    ts = timestamp if timestamp is not None else int(time.time())
    message = f"{ts}{method.upper()}{path}{body}"

    # Polymarket returns URL-safe base64 secrets, sometimes without padding.
    secret_padded = creds.api_secret + "=" * (-len(creds.api_secret) % 4)
    secret_bytes = base64.urlsafe_b64decode(secret_padded)
    sig_bytes = hmac.new(secret_bytes, message.encode(), hashlib.sha256).digest()
    signature = base64.urlsafe_b64encode(sig_bytes).decode()

    return {
        "POLY_ADDRESS": creds.funder_address,
        "POLY_SIGNATURE": signature,
        "POLY_TIMESTAMP": str(ts),
        "POLY_NONCE": str(nonce),
        "POLY_API_KEY": creds.api_key,
        "POLY_PASSPHRASE": creds.api_passphrase,
        "Content-Type": "application/json",
    }


async def get_profile(
    session: aiohttp.ClientSession,
    creds: Credentials,
) -> dict[str, Any]:
    """
    GET /profile — returns account profile dict on success.

    Raises ClobAuthError on 401/403.
    Raises ClobRequestError on other non-2xx responses.
    """
    path = "/profile"
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
    Validate CLOB credentials with a read-only GET /profile call.

    Returns {"ok": True, "address": ..., "profile": {...}} on success.
    Returns {"ok": False, "error": "..."} on auth failure or network error.
    """
    try:
        profile = await get_profile(session, creds)
        return {"ok": True, "address": creds.funder_address, "profile": profile}
    except ClobAuthError as e:
        return {"ok": False, "error": str(e)}
    except aiohttp.ClientError as e:
        return {"ok": False, "error": f"network error: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
