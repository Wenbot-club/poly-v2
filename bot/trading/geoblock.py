"""
Geoblock check — verify that Polymarket endpoints are reachable.

Returns a GeoBlockResult indicating accessibility of each endpoint.
HTTP 451 (Unavailable For Legal Reasons) or connection errors are treated
as blocked.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

_ENDPOINTS = {
    "polymarket_web": "https://polymarket.com",
    "clob_api": "https://clob.polymarket.com",
    "gamma_api": "https://gamma-api.polymarket.com",
}

_TIMEOUT = aiohttp.ClientTimeout(total=10.0)
_BLOCKED_STATUSES = {451, 403}


@dataclass
class EndpointStatus:
    url: str
    reachable: bool
    status_code: Optional[int] = None
    error: Optional[str] = None


@dataclass
class GeoBlockResult:
    endpoints: dict[str, EndpointStatus] = field(default_factory=dict)

    @property
    def all_reachable(self) -> bool:
        return all(s.reachable for s in self.endpoints.values())

    @property
    def blocked_endpoints(self) -> list[str]:
        return [name for name, s in self.endpoints.items() if not s.reachable]


async def _check_endpoint(
    session: aiohttp.ClientSession,
    name: str,
    url: str,
) -> tuple[str, EndpointStatus]:
    try:
        async with session.get(url, timeout=_TIMEOUT, allow_redirects=True) as resp:
            reachable = resp.status not in _BLOCKED_STATUSES
            return name, EndpointStatus(url=url, reachable=reachable, status_code=resp.status)
    except aiohttp.ClientConnectionError as e:
        return name, EndpointStatus(url=url, reachable=False, error=f"connection error: {e}")
    except asyncio.TimeoutError:
        return name, EndpointStatus(url=url, reachable=False, error="timeout")
    except Exception as e:
        return name, EndpointStatus(url=url, reachable=False, error=str(e))


async def check_geoblock(
    session: Optional[aiohttp.ClientSession] = None,
    endpoints: Optional[dict[str, str]] = None,
) -> GeoBlockResult:
    """
    Check accessibility of Polymarket endpoints.

    Pass an existing aiohttp.ClientSession to reuse connections.
    Pass custom endpoints dict to override defaults (useful in tests).
    """
    targets = endpoints if endpoints is not None else _ENDPOINTS
    result = GeoBlockResult()

    async def _run(sess: aiohttp.ClientSession) -> None:
        tasks = [_check_endpoint(sess, name, url) for name, url in targets.items()]
        for coro in asyncio.as_completed(tasks):
            name, status = await coro
            result.endpoints[name] = status

    if session is not None:
        await _run(session)
    else:
        async with aiohttp.ClientSession() as sess:
            await _run(sess)

    return result
