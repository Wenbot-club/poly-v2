"""
Read-only Polymarket market discovery via Gamma REST API.

Confirmed endpoint (April 2026, Polymarket Gamma docs):
  GET https://gamma-api.polymarket.com/markets?active=true&closed=false

Confirmed response fields used here:
  id, conditionId, title, slug, startDate, endDate,
  active, closed, tokens[].tokenId, tokens[].outcome

Fields used defensively (present in observed responses, not in public spec):
  minimumOrderSize, minimumTickSize

Fields NOT fetched here — TODO until confirmed endpoint:
  fee rates, taker delay, min_order_age_s
  → ClobMarketInfo uses zero/false defaults for these; they are correct for
    paper trading but must be filled before live execution is ever wired.

BTC 15m selection heuristic:
  This implementation has NO stable market-ID anchor. It matches by:
    1. active=true, closed=false (server-side filter)
    2. start_ts_ms <= now_ms < end_ts_ms (active time window)
    3. title or slug contains a BTC keyword AND a 15-minute keyword (text heuristic)
  If 0 or >1 markets pass all filters, an explicit error is raised.
  The heuristic is intentionally conservative; broaden _15M_KEYWORDS or
  _BTC_KEYWORDS only after live validation.
"""
from __future__ import annotations

import re
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

# Text-match sets — deliberately narrow to avoid false positives.
# Expand only after verifying live market titles.
_BTC_KEYWORDS: frozenset[str] = frozenset({"bitcoin", "btc"})
_15M_KEYWORDS: frozenset[str] = frozenset({
    "15m", "15-m", "15min", "15-min", "15 min", "15 minute", "15 minutes",
})


class GammaAPIError(Exception):
    """Raised when the Gamma API returns an unexpected HTTP status or body."""


class NoMatchingMarketError(Exception):
    """Raised when no active BTC 15m market is found after all filters."""


class AmbiguousMarketError(Exception):
    """Raised when more than one market passes the BTC 15m filter."""


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

async def _fetch_active_markets(
    session: aiohttp.ClientSession,
    base_url: str,
) -> List[Dict[str, Any]]:
    """
    GET /markets?active=true&closed=false

    Pagination: not implemented. The active+closed filter is narrow enough
    that a single page should cover all live 15m BTC markets at any given
    moment. Add pagination if this proves wrong in production.
    """
    url = f"{base_url.rstrip('/')}/markets"
    params = {"active": "true", "closed": "false"}
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise GammaAPIError(f"GET /markets returned HTTP {resp.status}: {text[:200]}")
        data = await resp.json(content_type=None)
    if not isinstance(data, list):
        raise GammaAPIError(f"Expected list from GET /markets, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------------------
# Selection heuristic
# ---------------------------------------------------------------------------

def _normalize_text(value: str) -> str:
    return re.sub(r"[_\-/]", " ", value.lower())


def _matches_btc_15m(raw: Dict[str, Any], now_ms: int) -> bool:
    """
    Three-gate filter (all must pass):
      1. active=true, closed=false — already enforced server-side, re-checked here
      2. start_ts_ms <= now_ms < end_ts_ms — market must be in its active window
      3. BTC keyword AND 15m keyword present in title or slug
    """
    if not raw.get("active", False) or raw.get("closed", False):
        return False

    try:
        start_ms = parse_iso_to_ms(str(raw["startDate"]))
        end_ms = parse_iso_to_ms(str(raw["endDate"]))
    except (KeyError, ValueError):
        return False

    if not (start_ms <= now_ms < end_ms):
        return False

    title_text = _normalize_text(str(raw.get("title", "")))
    slug_text = _normalize_text(str(raw.get("slug", "")))
    combined = f"{title_text} {slug_text}"

    has_btc = any(kw in combined for kw in _BTC_KEYWORDS)
    has_15m = any(kw in combined for kw in _15M_KEYWORDS)
    return has_btc and has_15m


# ---------------------------------------------------------------------------
# Parsing — pure functions, no I/O
# ---------------------------------------------------------------------------

def _extract_tokens(raw: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """
    Extract (yes_token_id, no_token_id) from Gamma market dict.

    Confirmed field: tokens[].tokenId, tokens[].outcome
    Outcome values are "Yes"/"No" (case-insensitive match used for safety).
    Returns (None, None) if the structure is missing or malformed.
    """
    tokens = raw.get("tokens")
    if not isinstance(tokens, list) or len(tokens) < 2:
        return None, None
    yes_id: Optional[str] = None
    no_id: Optional[str] = None
    for tok in tokens:
        outcome = str(tok.get("outcome", "")).strip().lower()
        token_id = tok.get("tokenId") or tok.get("token_id")  # accept both casings
        if token_id is None:
            return None, None
        if outcome == "yes":
            yes_id = str(token_id)
        elif outcome == "no":
            no_id = str(token_id)
    return yes_id, no_id


def _parse_clob_info(raw: Dict[str, Any]) -> ClobMarketInfo:
    """
    Build ClobMarketInfo from Gamma market dict.

    minimumOrderSize and minimumTickSize are present in observed Gamma
    responses but are not in the published spec — used defensively with
    fallbacks. All fee fields default to zero: confirmed correct for paper
    trading; must be sourced from CLOB API before live execution.
    """
    min_order_size = float(raw.get("minimumOrderSize") or 5.0)
    min_tick_size = float(raw.get("minimumTickSize") or 0.01)

    tokens = raw.get("tokens", [])
    clob_tokens = [
        ClobToken(token_id=str(t.get("tokenId") or t.get("token_id", "")), outcome=str(t.get("outcome", "")))
        for t in tokens
        if (t.get("tokenId") or t.get("token_id"))
    ]

    return ClobMarketInfo(
        tokens=clob_tokens,
        min_order_size=min_order_size,
        min_tick_size=min_tick_size,
        # TODO: source from CLOB API (/markets/{condition_id} or similar) once endpoint confirmed
        maker_base_fee_bps=0,
        taker_base_fee_bps=0,
        taker_delay_enabled=False,
        min_order_age_s=0.0,
        fee_rate=0.0,
        fee_exponent=1.0,
    )


def parse_gamma_market(raw: Dict[str, Any]) -> MarketContext:
    """
    Pure function: one Gamma market dict → MarketContext.
    Raises ValueError with a descriptive message on any missing required field.
    """
    required = ("id", "conditionId", "title", "slug", "startDate", "endDate")
    missing = [f for f in required if not raw.get(f)]
    if missing:
        raise ValueError(f"Gamma market missing required fields: {missing!r}  raw_id={raw.get('id')!r}")

    yes_id, no_id = _extract_tokens(raw)
    if yes_id is None or no_id is None:
        raise ValueError(
            f"Could not extract YES/NO token IDs from market {raw.get('id')!r}; "
            f"tokens field: {raw.get('tokens')!r}"
        )

    try:
        start_ms = parse_iso_to_ms(str(raw["startDate"]))
        end_ms = parse_iso_to_ms(str(raw["endDate"]))
    except ValueError as exc:
        raise ValueError(f"Could not parse dates for market {raw.get('id')!r}: {exc}") from exc

    return MarketContext(
        market_id=str(raw["id"]),
        condition_id=str(raw["conditionId"]),
        title=str(raw["title"]),
        slug=str(raw["slug"]),
        start_ts_ms=start_ms,
        end_ts_ms=end_ms,
        yes_token_id=yes_id,
        no_token_id=no_id,
        clob=_parse_clob_info(raw),
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class PolymarketDiscoveryProvider:
    """
    Async, read-only Polymarket market discovery via Gamma REST.

    Caller is responsible for creating and closing the aiohttp.ClientSession.
    This provider does not own the session lifetime.

    Not registered as DiscoveryProvider (sync Protocol) — the async interface
    is intentionally distinct. See AsyncDiscoveryProvider in base.py.
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
        Fetch active markets from Gamma and return the single BTC 15m market.

        Selection criteria (see _matches_btc_15m for full detail):
          - active=true, closed=false
          - start_ts_ms <= now < end_ts_ms
          - title/slug contains BTC keyword AND 15m keyword (text heuristic)

        Raises:
          GammaAPIError         — HTTP error or unexpected response shape
          NoMatchingMarketError — 0 candidates after all filters
          AmbiguousMarketError  — >1 candidates (caller must refine criteria)
          ValueError            — required fields missing in winning candidate
        """
        now_ms = self._now_fn()
        raw_markets = await _fetch_active_markets(self._session, self._base_url)
        candidates = [m for m in raw_markets if _matches_btc_15m(m, now_ms)]

        if len(candidates) == 0:
            raise NoMatchingMarketError(
                f"No active BTC 15m market found at now_ms={now_ms}. "
                f"Total markets returned by Gamma: {len(raw_markets)}."
            )
        if len(candidates) > 1:
            titles = [m.get("title", m.get("id", "?")) for m in candidates]
            raise AmbiguousMarketError(
                f"{len(candidates)} markets matched BTC 15m filter at now_ms={now_ms}: {titles!r}. "
                "Tighten _BTC_KEYWORDS/_15M_KEYWORDS or add a slug prefix after live validation."
            )

        return parse_gamma_market(candidates[0])
