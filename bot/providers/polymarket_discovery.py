"""
Read-only Polymarket BTC M15 market discovery via Gamma events API.

Discovery approach (live-validated, April 2026):
  M15 windows are 900-second UTC-aligned slots. The slug is deterministic:
    btc-updown-15m-{window_ts}
  where window_ts = floor(now_utc / 900) * 900.

  Endpoint:
    GET https://gamma-api.polymarket.com/events/slug/btc-updown-15m-{window_ts}

  Response shape (relevant fields):
    {
      "markets": [{
        "id": "...",
        "conditionId": "...",
        "question": "...",
        "clobTokenIds": "[\"0xABC...\", \"0xDEF...\"]",   ← JSON-encoded string
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.01,
        "takerBaseFee": 1000
      }]
    }

  Token ordering (confirmed):
    clobTokenIds[0] = UP token  (YES = "Will BTC go UP?")
    clobTokenIds[1] = DOWN token (NO)

  Timing hazard:
    The market is sometimes not created for the first ~30 s of a new window.
    find_active_btc_15m_market() retries up to LOOKUP_RETRIES times with
    LOOKUP_RETRY_DELAY_S between attempts before raising NoMatchingMarketError.

Fields NOT fetched here — TODO until confirmed:
  maker fee, taker delay, min_order_age_s
  → ClobMarketInfo uses zero/false defaults; correct for paper trading.
"""
from __future__ import annotations

import asyncio
import json
import math
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from ..domain import (
    ClobMarketInfo,
    ClobToken,
    DiscoveryCandidate,
    MarketContext,
    parse_iso_to_ms,
    utc_now_ms,
)


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

_BTC_15M_WINDOW_S: int = 900
_BTC_15M_SLUG_PREFIX: str = "btc-updown-15m-"

LOOKUP_RETRIES: int = 3
LOOKUP_RETRY_DELAY_S: float = 2.0


class GammaAPIError(Exception):
    """Raised when the Gamma API returns an unexpected HTTP status or body."""


class NoMatchingMarketError(Exception):
    """Raised when the BTC 15m market for the current window is not yet available."""


class AmbiguousMarketError(Exception):
    """Kept for API compatibility; not raised by the slug-based lookup."""


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

async def _fetch_event_by_slug(
    session: aiohttp.ClientSession,
    slug: str,
    base_url: str,
) -> Optional[Dict[str, Any]]:
    """
    GET /events/slug/{slug} → dict or None if 404.
    Raises GammaAPIError on non-200/non-404 status.
    """
    url = f"{base_url.rstrip('/')}/events/slug/{slug}"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        if resp.status == 404:
            return None
        if resp.status != 200:
            text = await resp.text()
            raise GammaAPIError(
                f"GET /events/slug/{slug} returned HTTP {resp.status}: {text[:200]}"
            )
        return await resp.json(content_type=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_list_field(value: Any) -> List[str]:
    """
    Gamma encodes some list fields as JSON strings (e.g. clobTokenIds).
    Accept both a real list and a JSON-encoded string; return [] on failure.
    """
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(v) for v in decoded]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_btc_15m_event(
    raw_event: Dict[str, Any],
    slug: str,
    window_ts: int,
) -> MarketContext:
    """
    Parse a Gamma event dict into a MarketContext for a BTC M15 market.

    Token ordering (live-validated):
      clobTokenIds[0] → UP  → yes_token_id  (YES = BTC goes up)
      clobTokenIds[1] → DOWN → no_token_id

    Raises ValueError if the event structure is missing required fields.
    """
    markets = raw_event.get("markets")
    if not isinstance(markets, list) or not markets:
        raise ValueError(f"Event {slug!r}: missing or empty 'markets' in response")

    raw_mkt = markets[0]

    token_ids = _decode_list_field(raw_mkt.get("clobTokenIds"))
    if len(token_ids) < 2:
        raise ValueError(
            f"Event {slug!r}: expected ≥2 clobTokenIds, got {token_ids!r}"
        )

    yes_token_id = token_ids[0]   # UP
    no_token_id = token_ids[1]    # DOWN

    clob_tokens = [
        ClobToken(token_id=yes_token_id, outcome="Yes"),
        ClobToken(token_id=no_token_id, outcome="No"),
    ]

    min_order_size = float(raw_mkt.get("orderMinSize") or 5.0)
    min_tick_size = float(raw_mkt.get("orderPriceMinTickSize") or 0.01)
    taker_fee_bps = int(raw_mkt.get("takerBaseFee") or 0)

    clob = ClobMarketInfo(
        tokens=clob_tokens,
        min_order_size=min_order_size,
        min_tick_size=min_tick_size,
        maker_base_fee_bps=0,
        taker_base_fee_bps=taker_fee_bps,
        taker_delay_enabled=False,
        min_order_age_s=0.0,
        fee_rate=0.0,
        fee_exponent=1.0,
    )

    question = str(
        raw_mkt.get("question") or raw_event.get("title") or slug
    )
    market_id = str(raw_mkt.get("id") or slug)
    condition_id = str(raw_mkt.get("conditionId") or raw_mkt.get("id") or slug)

    return MarketContext(
        market_id=market_id,
        condition_id=condition_id,
        title=question,
        slug=slug,
        start_ts_ms=window_ts * 1000,
        end_ts_ms=(window_ts + _BTC_15M_WINDOW_S) * 1000,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        clob=clob,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class PolymarketDiscoveryProvider:
    """
    Async, read-only BTC M15 market discovery via Gamma events API.

    Uses a deterministic slug (btc-updown-15m-{window_ts}) instead of
    scanning all active markets — faster and unambiguous.

    Caller is responsible for creating and closing the aiohttp.ClientSession.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str = GAMMA_BASE_URL,
        now_fn: Callable[[], int] = utc_now_ms,
    ) -> None:
        self._session = session
        self._base_url = base_url
        self._now_fn = now_fn

    async def find_active_btc_15m_market(self) -> MarketContext:
        """
        Look up the BTC M15 market for the current 900-second UTC window.

        Retries up to LOOKUP_RETRIES times (with LOOKUP_RETRY_DELAY_S between
        attempts) to tolerate the ~30 s publication delay at window start.

        Raises:
          GammaAPIError         — HTTP error from Gamma API
          NoMatchingMarketError — market not yet published after all retries
          ValueError            — event found but response is malformed
        """
        now_s = self._now_fn() / 1000.0
        window_ts = int(math.floor(now_s / _BTC_15M_WINDOW_S) * _BTC_15M_WINDOW_S)
        slug = f"{_BTC_15M_SLUG_PREFIX}{window_ts}"

        for attempt in range(LOOKUP_RETRIES):
            raw_event = await _fetch_event_by_slug(self._session, slug, self._base_url)
            if raw_event is not None:
                return _parse_btc_15m_event(raw_event, slug, window_ts)
            if attempt < LOOKUP_RETRIES - 1:
                await asyncio.sleep(LOOKUP_RETRY_DELAY_S)

        raise NoMatchingMarketError(
            f"BTC 15m market not found for window_ts={window_ts} (slug={slug!r}) "
            f"after {LOOKUP_RETRIES} attempts ({LOOKUP_RETRY_DELAY_S}s between each). "
            "The market may not be created yet — wait a few seconds and retry."
        )
