"""
Geoblock check — uses the official Polymarket geoblock API.

Calls GET https://polymarket.com/api/geoblock which returns:
  {"blocked": bool, "ip": str, "country": str, "region": str}

`blocked: True`  → this IP/region is not permitted to trade on Polymarket.
`blocked: False` → this IP is eligible.

Reference: https://docs.polymarket.com/api-reference/geoblock
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import aiohttp

_GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
_TIMEOUT = aiohttp.ClientTimeout(total=10.0)


@dataclass
class GeoBlockResult:
    blocked: bool
    ip: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    error: Optional[str] = None

    @property
    def eligible(self) -> bool:
        """True if this IP is permitted to trade (not blocked and no error)."""
        return not self.blocked and self.error is None


async def check_geoblock(
    session: Optional[aiohttp.ClientSession] = None,
) -> GeoBlockResult:
    """
    Call the Polymarket geoblock API and return a GeoBlockResult.

    Fails safe: returns blocked=True with an error message if the API is
    unreachable, so downstream bootstrap checks are not silently skipped.
    """
    async def _run(http: aiohttp.ClientSession) -> GeoBlockResult:
        try:
            async with http.get(_GEOBLOCK_URL, timeout=_TIMEOUT) as resp:
                if resp.status >= 400:
                    return GeoBlockResult(
                        blocked=True,
                        error=f"geoblock API returned HTTP {resp.status}",
                    )
                data = await resp.json(content_type=None)
                return GeoBlockResult(
                    blocked=bool(data.get("blocked", True)),
                    ip=data.get("ip"),
                    country=data.get("country"),
                    region=data.get("region"),
                )
        except asyncio.TimeoutError:
            return GeoBlockResult(blocked=True, error="timeout reaching geoblock API")
        except aiohttp.ClientError as e:
            return GeoBlockResult(blocked=True, error=f"connection error: {e}")
        except Exception as e:
            return GeoBlockResult(blocked=True, error=str(e))

    if session is not None:
        return await _run(session)
    else:
        async with aiohttp.ClientSession() as http:
            return await _run(http)
