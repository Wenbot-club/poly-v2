"""
BTC M5 session orchestrator.

M5Session.run(window_ts) executes one complete 5-minute window:
  1. Wait for PTB fetch time (window_ts + ptb_fetch_delay_s)
  2. Fetch PTB (SSR → API → Chainlink fallback)
  3. Discover UP/DOWN token IDs via Gamma slug btc-updown-5m-{window_ts}
  4. Start background token price polling
  5. EARLY scan: [140s, 170s) every 500ms — enter on consensus >= 88
  6. BASELINE: at 170s if no LEG1 — enter on btc vs ptb
  7. HEDGE watch: Binance vs PTB ± threshold, cutoff at 250s
  8. Wait for window expiry
  9. Poll closePrice and settle

Paper execution model (simplified):
  - LIMITATION: fills at best_ask with no partial fills or book depth modelling.
  - Attempted price = best_ask + offset, capped by config max.
  - Slippage = attempted - observed is traced per trade.
  - price_insane guard: refuse if best_ask >= price_insane_threshold (0.995).
  - Real FAK retry logic is a no-op in paper: first attempt always succeeds
    unless price_insane.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import math
import re
import time as _time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import aiohttp

from .m5_summary import TradeRecord
from .settings import M5Config, DEFAULT_M5_CONFIG
from .strategy.btc_m5 import (
    EntrySignal,
    compute_entry_signal,
    baseline_direction,
    should_hedge,
    compute_settlement,
)

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_POLY_BASE = "https://polymarket.com"
_CLOB_BASE = "https://clob.polymarket.com"


# ---------------------------------------------------------------------------
# Shared signal state (updated by caller / background tasks)
# ---------------------------------------------------------------------------

@dataclass
class M5SignalState:
    btc_price: Optional[float] = None
    btc_price_ts_ms: Optional[int] = None
    chainlink_price: Optional[float] = None
    chainlink_price_ts_ms: Optional[int] = None


# ---------------------------------------------------------------------------
# Rolling BTC price history
# ---------------------------------------------------------------------------

class BtcHistory:
    """Rolling buffer of (ts_ms, price) pairs for 10s/30s/60s lookbacks."""

    _MAX_TOLERANCE_MS = 30_000

    def __init__(self) -> None:
        self._samples: deque = deque()

    def record(self, price: float, ts_ms: int) -> None:
        self._samples.append((ts_ms, price))
        cutoff = ts_ms - 120_000
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def price_n_secs_ago(self, n_s: float, now_ms: int) -> Optional[float]:
        target = now_ms - int(n_s * 1000)
        best: Optional[float] = None
        best_dist = float("inf")
        for ts, price in self._samples:
            dist = abs(ts - target)
            if dist < best_dist:
                best_dist = dist
                best = price
        return best if best_dist <= self._MAX_TOLERANCE_MS else None

    def recent_samples(self, since_ms: int) -> list:
        """Return [(ts_ms, price)] for all samples at or after since_ms."""
        return [(ts, p) for ts, p in self._samples if ts >= since_ms]


# ---------------------------------------------------------------------------
# PTB fetching (3 sources)
# ---------------------------------------------------------------------------

async def fetch_ptb_ssr(
    http: aiohttp.ClientSession,
    window_ts: int,
    base_url: str = _POLY_BASE,
    *,
    asset: str = "btc",
    ptb_sanity_floor: float = 50.0,
    min_html_len: int = 500,
) -> Optional[float]:
    """
    Scrape openPrice from Polymarket event page, anchored to the specific window.

    Uses the event slug or UTC ISO timestamp as a context anchor so that a global
    openPrice that belongs to a different market/event is never picked up.
    Returns None when HTML is too short, no anchor found, or value fails sanity check.
    """
    url = f"{base_url}/event/{asset}-updown-5m-{window_ts}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; polybot/1.0)"}
    try:
        async with http.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            html = await resp.text()
            if len(html) < min_html_len:
                return None

            # Anchors — try slug first, then ISO UTC variants
            slug = f"{asset}-updown-5m-{window_ts}"
            dt = datetime.datetime.fromtimestamp(window_ts, tz=datetime.timezone.utc)
            iso_base = dt.strftime("%Y-%m-%dT%H:%M:%S")
            anchors = [slug, iso_base + "Z", iso_base + ".000Z", iso_base + "+00:00"]

            for anchor in anchors:
                idx = html.find(anchor)
                if idx == -1:
                    continue
                context = html[max(0, idx - 200):idx + 1000]
                m = re.search(r'"openPrice"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)"?', context)
                if m:
                    val = float(m.group(1))
                    if val > ptb_sanity_floor:
                        return val
    except Exception:
        pass
    return None


async def fetch_ptb_api(
    http: aiohttp.ClientSession,
    window_ts: int,
    base_url: str = _POLY_BASE,
    *,
    asset: str = "btc",
) -> Optional[float]:
    """Fetch openPrice from Polymarket crypto-price API."""
    url = f"{base_url}/api/crypto/crypto-price"
    params = {"symbol": asset.upper(), "eventStartTime": window_ts, "variant": "fiveminute"}
    try:
        async with http.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
            v = data.get("openPrice")
            if v is not None:
                return float(v)
    except Exception:
        pass
    return None


async def fetch_ptb(
    http: aiohttp.ClientSession,
    window_ts: int,
    chainlink_price: Optional[float] = None,
    chainlink_age_s: Optional[float] = None,
    *,
    polymarket_base_url: str = _POLY_BASE,
    chainlink_max_age_s: float = 60.0,
) -> tuple:  # (Optional[float], Optional[str])
    """
    Fetch PTB using 3-source priority: SSR → API → Chainlink.
    Returns (ptb_value, source_name) or (None, None).
    """
    ptb = await fetch_ptb_ssr(http, window_ts, polymarket_base_url)
    if ptb is not None:
        return ptb, "ssr"

    ptb = await fetch_ptb_api(http, window_ts, polymarket_base_url)
    if ptb is not None:
        return ptb, "api"

    if (chainlink_price is not None
            and chainlink_age_s is not None
            and chainlink_age_s < chainlink_max_age_s):
        return chainlink_price, "chainlink"

    return None, None


@dataclass
class PtbResult:
    """Result of fetch_ptb_robust() — carries all audit and diagnostic fields."""
    ptb: Optional[float]
    source: Optional[str]           # "ssr" | "api" | "chainlink" | None
    ptb_ssr: Optional[float]
    ptb_api: Optional[float]
    ptb_chainlink: Optional[float]
    ptb_ssr_valid: bool
    ptb_api_valid: bool
    ptb_ssr_rejected_for_delta: bool
    attempts: int
    latency_s: Optional[float]


async def fetch_ptb_robust(
    http: aiohttp.ClientSession,
    window_ts: int,
    chainlink_price: Optional[float] = None,
    chainlink_age_s: Optional[float] = None,
    *,
    asset: str = "btc",
    ptb_sanity_floor: float = 50.0,
    polymarket_base_url: str = _POLY_BASE,
    chainlink_max_age_s: float = 60.0,
    ptb_max_attempts: int = 10,
    ptb_retry_delay_s: float = 3.0,
    ptb_max_ssr_api_delta_usd: float = 10.0,
    time_fn: Callable = _time.time,
) -> PtbResult:
    """
    Robust PTB fetch with retries, hardened SSR, and SSR/API delta guard.

    Selection:
    - Both present, |delta| <= threshold → SSR (more direct source)
    - Both present, |delta| > threshold  → API, ptb_ssr_rejected_for_delta=True
    - SSR only → SSR
    - API only → API
    - Neither after all attempts → Chainlink last resort, or None
    """
    t_start = time_fn()

    ptb_cl: Optional[float] = None
    if (chainlink_price is not None
            and chainlink_age_s is not None
            and chainlink_age_s < chainlink_max_age_s):
        ptb_cl = chainlink_price

    ptb_ssr_last: Optional[float] = None
    ptb_api_last: Optional[float] = None

    for attempt in range(1, ptb_max_attempts + 1):
        ptb_ssr = await fetch_ptb_ssr(http, window_ts, polymarket_base_url,
                                      asset=asset, ptb_sanity_floor=ptb_sanity_floor)
        ptb_api = await fetch_ptb_api(http, window_ts, polymarket_base_url, asset=asset)
        ptb_ssr_last = ptb_ssr
        ptb_api_last = ptb_api

        ssr_valid = ptb_ssr is not None
        api_valid = ptb_api is not None

        if not ssr_valid and not api_valid:
            if attempt < ptb_max_attempts:
                await asyncio.sleep(ptb_retry_delay_s)
            continue

        latency_s = time_fn() - t_start

        if ssr_valid and api_valid:
            delta = abs(ptb_ssr - ptb_api)
            if delta > ptb_max_ssr_api_delta_usd:
                return PtbResult(
                    ptb=ptb_api, source="api",
                    ptb_ssr=ptb_ssr, ptb_api=ptb_api, ptb_chainlink=ptb_cl,
                    ptb_ssr_valid=True, ptb_api_valid=True,
                    ptb_ssr_rejected_for_delta=True,
                    attempts=attempt, latency_s=latency_s,
                )
            return PtbResult(
                ptb=ptb_ssr, source="ssr",
                ptb_ssr=ptb_ssr, ptb_api=ptb_api, ptb_chainlink=ptb_cl,
                ptb_ssr_valid=True, ptb_api_valid=True,
                ptb_ssr_rejected_for_delta=False,
                attempts=attempt, latency_s=latency_s,
            )

        if ssr_valid:
            return PtbResult(
                ptb=ptb_ssr, source="ssr",
                ptb_ssr=ptb_ssr, ptb_api=None, ptb_chainlink=ptb_cl,
                ptb_ssr_valid=True, ptb_api_valid=False,
                ptb_ssr_rejected_for_delta=False,
                attempts=attempt, latency_s=latency_s,
            )

        return PtbResult(
            ptb=ptb_api, source="api",
            ptb_ssr=None, ptb_api=ptb_api, ptb_chainlink=ptb_cl,
            ptb_ssr_valid=False, ptb_api_valid=True,
            ptb_ssr_rejected_for_delta=False,
            attempts=attempt, latency_s=latency_s,
        )

    # All attempts exhausted: Chainlink last resort
    if ptb_cl is not None:
        return PtbResult(
            ptb=ptb_cl, source="chainlink",
            ptb_ssr=ptb_ssr_last, ptb_api=ptb_api_last, ptb_chainlink=ptb_cl,
            ptb_ssr_valid=False, ptb_api_valid=False,
            ptb_ssr_rejected_for_delta=False,
            attempts=ptb_max_attempts, latency_s=time_fn() - t_start,
        )

    return PtbResult(
        ptb=None, source=None,
        ptb_ssr=ptb_ssr_last, ptb_api=ptb_api_last, ptb_chainlink=ptb_cl,
        ptb_ssr_valid=False, ptb_api_valid=False,
        ptb_ssr_rejected_for_delta=False,
        attempts=ptb_max_attempts, latency_s=None,
    )


# ---------------------------------------------------------------------------
# Token discovery
# ---------------------------------------------------------------------------

async def fetch_m5_tokens(
    http: aiohttp.ClientSession,
    window_ts: int,
    gamma_base_url: str = _GAMMA_BASE,
    *,
    asset: str = "btc",
) -> Optional[tuple]:  # (up_token_id, down_token_id) or None
    """Return (up_token_id, down_token_id) for the M5 window, or None."""
    slug = f"{asset}-updown-5m-{window_ts}"
    url = f"{gamma_base_url}/events/slug/{slug}"
    try:
        async with http.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status in (404, 204):
                return None
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
    except Exception:
        return None

    markets = data.get("markets")
    if not isinstance(markets, list) or not markets:
        return None

    raw = markets[0].get("clobTokenIds")
    if isinstance(raw, str):
        try:
            token_ids = json.loads(raw)
        except Exception:
            return None
    elif isinstance(raw, list):
        token_ids = raw
    else:
        return None

    if len(token_ids) < 2:
        return None
    return str(token_ids[0]), str(token_ids[1])


# ---------------------------------------------------------------------------
# Token price polling
# ---------------------------------------------------------------------------

async def fetch_token_best_ask(
    http: aiohttp.ClientSession,
    token_id: str,
    clob_base_url: str = _CLOB_BASE,
) -> Optional[float]:
    """Fetch best ask for a token from the CLOB price endpoint."""
    url = f"{clob_base_url}/price"
    params = {"token_id": token_id, "side": "buy"}
    try:
        async with http.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json(content_type=None)
            v = data.get("price")
            if v is not None:
                return float(v)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Settlement price polling
# ---------------------------------------------------------------------------

async def fetch_close_price(
    http: aiohttp.ClientSession,
    window_ts: int,
    *,
    polymarket_base_url: str = _POLY_BASE,
    window_end_s: float,
    settlement_initial_delay_s: float = 15.0,
    settlement_poll_s: float = 4.0,
    settlement_max_attempts: int = 20,
    time_fn: Callable = _time.time,
) -> tuple:  # (Optional[float], int, Optional[float]) = (price, attempts, latency_s)
    """
    Poll for closePrice after window_end_s + settlement_initial_delay_s.
    Returns (price, attempt_count, latency_s_from_window_end).
    latency_s is None if all attempts exhausted without a result.
    """
    url = f"{polymarket_base_url}/api/crypto/crypto-price"
    params = {"symbol": "BTC", "eventStartTime": window_ts, "variant": "fiveminute"}

    first_poll_time = window_end_s + settlement_initial_delay_s
    now = time_fn()
    if now < first_poll_time:
        await asyncio.sleep(first_poll_time - now)

    for attempt in range(1, settlement_max_attempts + 1):
        try:
            async with http.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    v = data.get("closePrice")
                    if v is not None:
                        open_v = data.get("openPrice")
                        latency_s = time_fn() - window_end_s
                        open_price = float(open_v) if open_v is not None else None
                        return float(v), open_price, attempt, latency_s
        except Exception:
            pass
        if attempt < settlement_max_attempts:
            await asyncio.sleep(settlement_poll_s)

    return None, None, settlement_max_attempts, None


# ---------------------------------------------------------------------------
# Paper fill result
# ---------------------------------------------------------------------------

@dataclass
class PaperFillResult:
    fill_price: Optional[float]
    shares: Optional[float]
    observed_best_ask: float
    attempted_price: float
    slippage: float         # attempted - observed
    retries: int
    reject_reason: Optional[str]   # "price_insane" | None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class M5Session:
    """
    Orchestrates one BTC M5 window.

    Inject time_fn for testing (defaults to time.time).
    Inject gamma_base_url / clob_base_url / polymarket_base_url to redirect
    HTTP calls to mocks in tests.
    """

    def __init__(
        self,
        http_session: aiohttp.ClientSession,
        signal_state: M5SignalState,
        config: M5Config = DEFAULT_M5_CONFIG,
        time_fn: Callable[[], float] = _time.time,
        gamma_base_url: str = _GAMMA_BASE,
        clob_base_url: str = _CLOB_BASE,
        polymarket_base_url: str = _POLY_BASE,
        btc_history: Optional[BtcHistory] = None,
        prefetched_tokens: Optional[tuple] = None,
        token_prices: Optional[dict] = None,
        order_executor: Optional[Callable[[str, float, float], Awaitable["PaperFillResult"]]] = None,
        asset: str = "btc",
    ) -> None:
        self._http = http_session
        self._signals = signal_state
        self._cfg = config
        self._time_fn = time_fn
        self._gamma_url = gamma_base_url
        self._clob_url = clob_base_url
        self._poly_url = polymarket_base_url
        self._asset = asset

        # External token price cache injected by caller (e.g. WS stream).
        # If provided the internal HTTP poller is skipped for that window.
        if token_prices is not None:
            self._token_prices: dict = token_prices
            self._external_prices = True
        else:
            self._token_prices = {}
            self._external_prices = False
        self._btc_history = btc_history if btc_history is not None else BtcHistory()
        self._price_insane_blocks: int = 0
        self._hedge_blocked_by_cutoff: bool = False
        self._prefetched_tokens = prefetched_tokens
        self._order_executor = order_executor
        self.next_tokens_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, window_ts: int) -> TradeRecord:
        record = TradeRecord(window_ts=window_ts)

        # Populate window audit fields immediately
        ws = self._cfg.window_seconds
        wstart = datetime.datetime.fromtimestamp(window_ts, tz=datetime.timezone.utc)
        wend = wstart + datetime.timedelta(seconds=ws)
        record.window_start_utc_iso = wstart.isoformat()
        record.window_end_utc_iso = wend.isoformat()
        record.window_start_local_iso = datetime.datetime.fromtimestamp(window_ts).isoformat()
        record.window_end_local_iso = datetime.datetime.fromtimestamp(window_ts + ws).isoformat()
        record.event_slug = f"{self._asset}-updown-5m-{window_ts}"

        # Phase 1: wait until PTB fetch is safe
        await self._sleep_until(window_ts + self._cfg.ptb_fetch_delay_s)

        # Phase 2: fetch PTB (robust — retries, delta guard, full audit)
        cl_price = self._signals.chainlink_price
        cl_age_s: Optional[float] = None
        if cl_price is not None and self._signals.chainlink_price_ts_ms is not None:
            cl_age_s = (self._time_fn() - self._signals.chainlink_price_ts_ms / 1000.0)

        pr = await fetch_ptb_robust(
            self._http, window_ts, cl_price, cl_age_s,
            asset=self._asset,
            ptb_sanity_floor=self._cfg.ptb_sanity_floor_usd,
            polymarket_base_url=self._poly_url,
            ptb_max_attempts=self._cfg.ptb_max_attempts,
            ptb_retry_delay_s=self._cfg.ptb_retry_delay_s,
            ptb_max_ssr_api_delta_usd=self._cfg.ptb_max_ssr_api_delta_usd,
            time_fn=self._time_fn,
        )
        record.ptb_ssr = pr.ptb_ssr
        record.ptb_api = pr.ptb_api
        record.ptb_chainlink = pr.ptb_chainlink
        record.ptb_ssr_api_delta_usd = (
            round(pr.ptb_ssr - pr.ptb_api, 2)
            if pr.ptb_ssr is not None and pr.ptb_api is not None else None
        )
        record.ptb_attempts = pr.attempts
        record.ptb_retrieval_latency_s = pr.latency_s
        record.ptb_ssr_valid = pr.ptb_ssr_valid
        record.ptb_api_valid = pr.ptb_api_valid
        record.ptb_ssr_rejected_for_delta = pr.ptb_ssr_rejected_for_delta

        if pr.ptb is None:
            record.abort_reason = "ptb_unavailable"
            return record
        record.ptb = pr.ptb
        record.ptb_source = pr.source
        record.ptb_selected = pr.ptb
        record.ptb_selected_source = pr.source
        record.ptb_selected_api_delta_usd = (
            round(pr.ptb - pr.ptb_api, 2) if pr.ptb_api is not None else None
        )

        # Phase 3: token discovery (use prefetch cache if available)
        if self._prefetched_tokens is not None:
            tokens = self._prefetched_tokens
            record.used_prefetched_tokens = True
        else:
            tokens = await fetch_m5_tokens(self._http, window_ts, self._gamma_url,
                                           asset=self._asset)
        if tokens is None:
            record.abort_reason = "tokens_unavailable"
            return record
        up_id, down_id = tokens

        # Phase 4: token prices — WS stream (external) or HTTP polling (fallback)
        poll_task = (
            None if self._external_prices
            else asyncio.create_task(self._poll_token_prices(up_id, down_id))
        )
        try:
            await self._trading_phases(record, pr.ptb, window_ts, up_id, down_id)
        finally:
            if poll_task is not None:
                poll_task.cancel()
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass

        # Propagate session-level counters to record
        record.price_insane_block_count = self._price_insane_blocks
        record.hedge_blocked_by_cutoff = self._hedge_blocked_by_cutoff

        # Launch prefetch for next window (runs concurrently with settlement wait)
        self.next_tokens_task = asyncio.create_task(
            fetch_m5_tokens(self._http, window_ts + self._cfg.window_seconds,
                            self._gamma_url, asset=self._asset)
        )

        # Phase 8: settlement — fetch_close_price handles the window_end wait internally
        window_end_s = window_ts + self._cfg.window_seconds

        if record.entry_side is not None and record.entry_price is not None:
            close_price, open_price_api, attempts, latency_s = await fetch_close_price(
                self._http, window_ts,
                polymarket_base_url=self._poly_url,
                window_end_s=window_end_s,
                settlement_initial_delay_s=self._cfg.settlement_initial_delay_s,
                settlement_poll_s=self._cfg.settlement_poll_s,
                settlement_max_attempts=self._cfg.settlement_max_attempts,
                time_fn=self._time_fn,
            )
            record.resolution_attempts = attempts
            record.resolution_source = "api" if close_price is not None else None
            record.resolution_latency_s = latency_s
            record.resolution_open_price_api = open_price_api
            record.resolution_close_price_api = close_price
            if close_price is not None:
                s = compute_settlement(
                    close_price=close_price,
                    open_price=pr.ptb,
                    leg1_side=record.entry_side,
                    leg1_entry_price=record.entry_price,
                    leg1_shares=record.entry_shares or 0.0,
                    leg1_usd_staked=self._cfg.leg1_bet_usd,
                    hedge_side=record.hedge_side,
                    hedge_entry_price=record.hedge_price,
                    hedge_shares=record.hedge_shares,
                    hedge_usd_staked=self._cfg.hedge_bet_usd if record.hedged else None,
                )
                record.result = s.result
                record.pnl_leg1 = s.pnl_leg1
                record.pnl_hedge = s.pnl_hedge
                record.net_pnl = s.net_pnl
                # Trailing stop exited the hedge early — use actual exit pnl
                if (record.hedge_limit_trailing_stop_triggered
                        and record.hedge_limit_exit_pnl is not None):
                    record.pnl_hedge = record.hedge_limit_exit_pnl
                    record.net_pnl = round(record.pnl_leg1 + record.pnl_hedge, 6)
        else:
            # No position — still wait for window end before returning
            now = self._time_fn()
            if now < window_end_s:
                await asyncio.sleep(window_end_s - now)

        return record

    # ------------------------------------------------------------------
    # Trading phases
    # ------------------------------------------------------------------

    async def _trading_phases(
        self,
        record: TradeRecord,
        ptb: float,
        window_ts: int,
        up_id: str,
        down_id: str,
    ) -> None:
        cfg = self._cfg
        window_end_s = window_ts + cfg.window_seconds

        # EARLY scan: wait for window start then poll every 500ms
        await self._sleep_until(window_ts + cfg.entry_scan_start_s)

        while self._elapsed(window_ts) < cfg.entry_scan_end_s:
            if await self._try_early_entry(record, ptb, up_id, down_id, window_ts):
                break
            await asyncio.sleep(cfg.early_poll_interval_s)

        # BASELINE: one-shot at 170s if no LEG1 yet
        if record.entry_side is None:
            await self._sleep_until(window_ts + cfg.baseline_elapsed_s)
            await self._try_baseline_entry(record, ptb, up_id, down_id, window_ts)

        # HEDGE watch: only if LEG1 was taken
        if record.entry_side is not None:
            self._launch_hedge_presign(record.entry_side, up_id, down_id)
            entry_price = record.entry_price or 0.0
            if 0.45 <= entry_price <= 0.55:
                print(
                    f"[m5] hedge disabled: leg1 price={entry_price:.4f} in noise zone [0.45, 0.55]",
                    flush=True,
                )
            elif cfg.hedge_use_limit_approach:
                _live = (self._order_executor is not None
                         and hasattr(self._order_executor, "post_limit_buy"))
                if _live:
                    await self._watch_for_hedge_limit_live(
                        record, ptb, up_id, down_id, window_ts, window_end_s
                    )
                else:
                    await self._watch_for_hedge_limit_paper(
                        record, ptb, up_id, down_id, window_ts, window_end_s
                    )
            else:
                await self._watch_for_hedge(record, ptb, up_id, down_id, window_ts, window_end_s)

    # ------------------------------------------------------------------
    # EARLY entry
    # ------------------------------------------------------------------

    async def _try_early_entry(
        self,
        record: TradeRecord,
        ptb: float,
        up_id: str,
        down_id: str,
        window_ts: int,
    ) -> bool:
        btc = self._signals.btc_price
        if btc is None:
            record.entry_block_reason = "no_btc_price"
            return False

        elapsed_s = self._elapsed(window_ts)
        tau_s = self._cfg.window_seconds - elapsed_s
        now_ms = int(self._time_fn() * 1000)
        since_ms = now_ms - int(self._cfg.sigma_lookback_s * 1000)

        sig = compute_entry_signal(
            btc=btc, ptb=ptb, tau_s=tau_s,
            btc_samples=self._btc_history.recent_samples(since_ms),
            btc_10s=self._btc_history.price_n_secs_ago(10, now_ms),
            btc_30s=self._btc_history.price_n_secs_ago(30, now_ms),
            price_up=self._token_prices.get(up_id),
            price_down=self._token_prices.get(down_id),
            sigma_floor_usd=self._cfg.sigma_floor_usd,
            z_gap_min=self._cfg.z_gap_min,
            p_enter_up_min=self._cfg.p_enter_up_min,
            p_enter_down_max=self._cfg.p_enter_down_max,
            min_entry_edge=self._cfg.min_entry_edge,
        )

        if sig.direction is None:
            record.entry_block_reason = sig.block_reason
            return False

        token_id = up_id if sig.direction == "up" else down_id
        best_ask = self._token_prices.get(token_id)
        if best_ask is None:
            record.entry_block_reason = "no_token_price"
            return False

        record.entry_tick_ts_ms = self._signals.btc_price_ts_ms
        record.entry_decision_ts_ms = int(self._time_fn() * 1000)
        fill = await self._execute_paper(token_id, best_ask, self._cfg.leg1_bet_usd, is_leg1=True)
        record.entry_submit_ts_ms = int(self._time_fn() * 1000)
        self._apply_fill_trace(record, fill, "leg1")
        if fill.reject_reason is not None:
            record.entry_block_reason = fill.reject_reason
            return False

        record.entry_mode = "early"
        record.entry_side = sig.direction
        record.entry_elapsed_s = elapsed_s
        record.entry_price = fill.fill_price
        record.entry_shares = fill.shares
        record.p_model_up_at_entry = sig.p_model_up
        record.edge_up_at_entry = sig.edge_up
        record.edge_down_at_entry = sig.edge_down
        record.z_gap_at_entry = sig.z_gap
        record.sigma_to_close_at_entry = sig.sigma_to_close
        record.entry_block_reason = None
        return True

    # ------------------------------------------------------------------
    # BASELINE entry
    # ------------------------------------------------------------------

    async def _try_baseline_entry(
        self,
        record: TradeRecord,
        ptb: float,
        up_id: str,
        down_id: str,
        window_ts: int,
    ) -> bool:
        btc = self._signals.btc_price
        if btc is None:
            record.abort_reason = "baseline_no_signal"
            return False

        direction = baseline_direction(btc, ptb)
        if direction is None:
            record.abort_reason = "baseline_no_direction"
            return False

        token_id = up_id if direction == "up" else down_id
        best_ask = self._token_prices.get(token_id)
        if best_ask is None:
            record.abort_reason = "baseline_no_price"
            return False

        record.entry_tick_ts_ms = self._signals.btc_price_ts_ms
        record.entry_decision_ts_ms = int(self._time_fn() * 1000)
        fill = await self._execute_paper(token_id, best_ask, self._cfg.leg1_bet_usd, is_leg1=True)
        record.entry_submit_ts_ms = int(self._time_fn() * 1000)
        self._apply_fill_trace(record, fill, "leg1")
        if fill.reject_reason is not None:
            record.entry_block_reason = fill.reject_reason
            return False

        record.entry_mode = "baseline"
        record.entry_side = direction
        record.entry_elapsed_s = self._elapsed(window_ts)
        record.entry_price = fill.fill_price
        record.entry_shares = fill.shares
        return True

    # ------------------------------------------------------------------
    # HEDGE watch
    # ------------------------------------------------------------------

    async def _watch_for_hedge(
        self,
        record: TradeRecord,
        ptb: float,
        up_id: str,
        down_id: str,
        window_ts: int,
        window_end_s: float,
    ) -> None:
        if record.hedged:
            return

        cfg = self._cfg
        while self._time_fn() < window_end_s:
            elapsed_s = self._elapsed(window_ts)

            btc = self._signals.btc_price
            if btc is not None and record.entry_side is not None:
                if should_hedge(record.entry_side, btc, ptb, cfg.hedge_threshold):
                    hedge_side = "down" if record.entry_side == "up" else "up"
                    token_id = down_id if hedge_side == "down" else up_id
                    best_ask = self._token_prices.get(token_id)
                    if best_ask is not None:
                        record.hedge_tick_ts_ms = self._signals.btc_price_ts_ms
                        record.hedge_decision_ts_ms = int(self._time_fn() * 1000)
                        executor = self._order_executor
                        if executor is not None and hasattr(executor, "post_presigned_hedge"):
                            fill = await executor.post_presigned_hedge(best_ask)
                        else:
                            fill = await self._execute_paper(
                                token_id, best_ask, cfg.hedge_bet_usd, is_leg1=False
                            )
                        record.hedge_submit_ts_ms = int(self._time_fn() * 1000)
                        self._apply_fill_trace(record, fill, "hedge")
                        if fill.reject_reason is None:
                            record.hedged = True
                            record.hedge_elapsed_s = elapsed_s
                            record.hedge_side = hedge_side
                            record.hedge_price = fill.fill_price
                            record.hedge_shares = fill.shares
                            record.hedge_trigger_btc = btc
                            break

            await asyncio.sleep(0.2)

    def _launch_hedge_presign(self, entry_side: str, up_id: str, down_id: str) -> None:
        if self._order_executor is None or not hasattr(self._order_executor, "presign_hedge"):
            return
        hedge_token_id = down_id if entry_side == "up" else up_id
        asyncio.create_task(
            self._order_executor.presign_hedge(hedge_token_id, self._cfg.hedge_bet_usd)
        )

    # ------------------------------------------------------------------
    # Limit-order hedge — paper simulation
    # ------------------------------------------------------------------

    async def _watch_for_hedge_limit_paper(
        self,
        record: TradeRecord,
        ptb: float,
        up_id: str,
        down_id: str,
        window_ts: int,
        window_end_s: float,
    ) -> None:
        """
        Anticipatory limit hedge simulation.

        Strategy: watch BTC approach PTB and fill the hedge BEFORE the trigger
        fires, while the token is still fairly priced.

        Fill logic:
          - Each tick, check if BTC has entered the "anticipation zone":
              UP entry: btc < ptb + hedge_anticipation_buffer_usd
              DOWN entry: btc > ptb - hedge_anticipation_buffer_usd
          - If yes and best_ask <= hedge_limit_max_price → paper fill at best_ask.
          - If ask > max_price → keep watching (price may improve next tick).
          - If real trigger fires with no fill yet → record as "skipped" (no FOK).

        Post-fill: trailing stop (peak * ratio) and late cut (underwater + BTC
        reversed) are both available to exit early.

        All sim results are written to record.hedge_limit_* fields.
        """
        if record.hedged:
            return

        cfg = self._cfg
        entry_side = record.entry_side
        hedge_token_id = down_id if entry_side == "up" else up_id
        hedge_side = "down" if entry_side == "up" else "up"

        sim_filled: Optional[float] = None
        sim_shares: Optional[float] = None
        peak_price: float = 0.0
        fill_elapsed: Optional[float] = None
        fill_source: Optional[str] = None
        sl_armed: bool = False

        def _in_anticipation_zone(btc: float, now_ms: int) -> tuple[bool, str]:
            """
            Returns (should_place, reason).
            Velocity-based: fire if ETA to trigger < hedge_anticipation_seconds.
            Safety floor: always fire if BTC within min_buffer of PTB.
            Fallback: static buffer if not enough samples for velocity.
            """
            # Safety floor — always fire if very close to PTB
            min_buf = cfg.hedge_anticipation_min_buffer_usd
            if entry_side == "up" and btc < ptb + min_buf:
                return True, "safety_floor"
            if entry_side == "down" and btc > ptb - min_buf:
                return True, "safety_floor"

            # Velocity-based ETA
            btc_past = self._btc_history.price_n_secs_ago(
                cfg.hedge_velocity_window_s, now_ms
            )
            if btc_past is not None:
                velocity = (btc - btc_past) / cfg.hedge_velocity_window_s  # USD/s, signed
                if entry_side == "up":
                    trigger_level = ptb - cfg.hedge_threshold
                    distance = btc - trigger_level
                    if velocity < 0 and distance > 0:
                        eta_s = distance / abs(velocity)
                        if eta_s < cfg.hedge_anticipation_seconds:
                            return True, f"velocity(eta={eta_s:.1f}s,v={velocity:+.2f})"
                else:
                    trigger_level = ptb + cfg.hedge_threshold
                    distance = trigger_level - btc
                    if velocity > 0 and distance > 0:
                        eta_s = distance / velocity
                        if eta_s < cfg.hedge_anticipation_seconds:
                            return True, f"velocity(eta={eta_s:.1f}s,v={velocity:+.2f})"
                return False, ""

            # Fallback: static buffer (not enough samples yet)
            buf = cfg.hedge_anticipation_buffer_usd
            if entry_side == "up" and btc < ptb + buf:
                return True, "static_fallback"
            if entry_side == "down" and btc > ptb - buf:
                return True, "static_fallback"
            return False, ""

        while self._time_fn() < window_end_s:
            elapsed_s = self._elapsed(window_ts)

            btc = self._signals.btc_price
            best_ask = self._token_prices.get(hedge_token_id)

            # ── PRE-FILL: wait for BTC to enter anticipation zone ────────
            if sim_filled is None:
                triggered = (btc is not None
                             and should_hedge(entry_side, btc, ptb, cfg.hedge_threshold))

                should_place = False
                reason = ""
                if btc is not None:
                    now_ms = int(self._time_fn() * 1000)
                    should_place, reason = _in_anticipation_zone(btc, now_ms)

                if should_place:
                    if best_ask is not None:
                        # Fill immediately — we're ahead of the trigger
                        sim_filled = best_ask
                        sim_shares = cfg.hedge_bet_usd / sim_filled
                        fill_elapsed = elapsed_s
                        fill_source = "anticipatory"
                        peak_price = best_ask
                        record.hedge_limit_initial_bid = best_ask
                        record.hedge_trigger_btc = btc
                        print(
                            f"[m5] limit_hedge: anticipatory fill t={elapsed_s:.1f}s"
                            f"  btc={btc:.2f}  ptb={ptb:.2f}"
                            f"  ask={best_ask:.3f}"
                            f"  trig={reason}",
                            flush=True,
                        )

                if triggered and sim_filled is None:
                    fill_source = "skipped"
                    record.hedge_trigger_elapsed_s = elapsed_s
                    print(
                        f"[m5] limit_hedge: trigger fired, no fill yet — hedge skipped",
                        flush=True,
                    )
                    break

            # ── POST-FILL: track trigger lag + trailing stop ────────────
            else:
                # Record when real trigger fires (even if we're already filled)
                if (btc is not None
                        and record.hedge_trigger_elapsed_s is None
                        and should_hedge(entry_side, btc, ptb, cfg.hedge_threshold)):
                    record.hedge_trigger_elapsed_s = elapsed_s
                    lag_s = elapsed_s - fill_elapsed
                    print(
                        f"[m5] limit_hedge: real trigger at t={elapsed_s:.1f}s"
                        f"  lag={lag_s:+.1f}s after anticipatory fill",
                        flush=True,
                    )

                if best_ask is not None:
                    peak_price = max(peak_price, best_ask)
                    arm_level = sim_filled + cfg.hedge_limit_sl_arm_threshold_usd
                    sl_level = sim_filled + cfg.hedge_limit_sl_offset_usd

                    # Arm the SL once the token has gained enough
                    if not sl_armed and best_ask >= arm_level:
                        sl_armed = True
                        print(
                            f"[m5] limit_hedge: SL armed t={elapsed_s:.1f}s"
                            f"  fill={sim_filled:.3f}  ask={best_ask:.3f}"
                            f"  → SL at {sl_level:.3f}",
                            flush=True,
                        )

                    # If SL armed and price falls back to SL level → exit
                    if sl_armed and best_ask <= sl_level:
                        exit_pnl = round(sim_shares * (best_ask - sim_filled), 4)
                        record.hedge_limit_trailing_stop_triggered = True
                        record.hedge_limit_exit_elapsed_s = elapsed_s
                        record.hedge_limit_exit_price = best_ask
                        record.hedge_limit_exit_pnl = exit_pnl
                        print(
                            f"[m5] limit_hedge: SL hit t={elapsed_s:.1f}s"
                            f"  fill={sim_filled:.3f}  peak={peak_price:.3f}"
                            f"  exit={best_ask:.3f}  pnl={exit_pnl:+.3f}",
                            flush=True,
                        )
                        break

                    # Late cut: hedge underwater + BTC reversed toward leg1
                    if (elapsed_s > cfg.hedge_limit_late_cut_elapsed_s
                            and best_ask < sim_filled
                            and btc is not None
                            and baseline_direction(btc, ptb) == entry_side):
                        exit_pnl = round(sim_shares * (best_ask - sim_filled), 4)
                        record.hedge_limit_trailing_stop_triggered = True
                        record.hedge_limit_exit_elapsed_s = elapsed_s
                        record.hedge_limit_exit_price = best_ask
                        record.hedge_limit_exit_pnl = exit_pnl
                        print(
                            f"[m5] limit_hedge: late cut t={elapsed_s:.1f}s"
                            f"  fill={sim_filled:.3f}  exit={best_ask:.3f}"
                            f"  pnl={exit_pnl:+.3f}",
                            flush=True,
                        )
                        break

            await asyncio.sleep(0.2)

        # ── Finalise record ──────────────────────────────────────────────
        record.hedge_limit_reprices = 0
        record.hedge_limit_fill_elapsed_s = fill_elapsed
        record.hedge_limit_fill_source = fill_source
        record.hedge_limit_fill_price = sim_filled
        record.hedge_limit_peak_bid = round(peak_price, 4) if peak_price else None

        if sim_filled is not None and not record.hedged:
            fill = PaperFillResult(
                fill_price=sim_filled,
                shares=sim_shares,
                observed_best_ask=sim_filled,
                attempted_price=sim_filled,
                slippage=0.0,
                retries=0,
                reject_reason=None,
            )
            self._apply_fill_trace(record, fill, "hedge")
            record.hedged = True
            record.hedge_elapsed_s = fill_elapsed
            record.hedge_side = hedge_side
            record.hedge_price = sim_filled
            record.hedge_shares = sim_shares
            # hedge_trigger_btc already set at fill time for anticipatory fills

    # ------------------------------------------------------------------
    # Limit-order hedge — live execution
    # ------------------------------------------------------------------

    async def _watch_for_hedge_limit_live(
        self,
        record: TradeRecord,
        ptb: float,
        up_id: str,
        down_id: str,
        window_ts: int,
        window_end_s: float,
    ) -> None:
        """
        Live limit-order hedge execution.

        Flow:
          1. Watch BTC for anticipation zone (velocity/safety-floor).
          2. Place GTC limit buy at hedge_limit_max_price — fills immediately
             as a taker since max_price (0.65) >> typical ask (0.30-0.40).
          3. If not immediately filled, poll get_order_status each tick.
          4. Once filled: monitor token price for SL arm (+arm_threshold).
          5. When armed: place GTC limit sell (SL) at fill + sl_offset.
          6. On window end: cancel any open buy or sell orders.
        """
        if record.hedged:
            return

        cfg = self._cfg
        executor = self._order_executor
        entry_side = record.entry_side
        hedge_token_id = down_id if entry_side == "up" else up_id
        hedge_side = "down" if entry_side == "up" else "up"

        sl_order_id: Optional[str] = None
        fill_price: Optional[float] = None
        fill_shares: Optional[float] = None
        fill_elapsed: Optional[float] = None
        fill_source: Optional[str] = None
        sl_armed: bool = False
        sl_triggered: bool = False
        peak_price: float = 0.0
        # FAK accumulation state
        accumulating: bool = False
        total_cost: float = 0.0
        total_shares_acc: float = 0.0
        remaining_usd: float = cfg.hedge_bet_usd

        def _in_anticipation_zone(btc: float, now_ms: int) -> "tuple[bool, str]":
            min_buf = cfg.hedge_anticipation_min_buffer_usd
            if entry_side == "up" and btc < ptb + min_buf:
                return True, "safety_floor"
            if entry_side == "down" and btc > ptb - min_buf:
                return True, "safety_floor"
            btc_past = self._btc_history.price_n_secs_ago(
                cfg.hedge_velocity_window_s, now_ms
            )
            if btc_past is not None:
                velocity = (btc - btc_past) / cfg.hedge_velocity_window_s
                if entry_side == "up":
                    trigger_level = ptb - cfg.hedge_threshold
                    distance = btc - trigger_level
                    if velocity < 0 and distance > 0:
                        eta_s = distance / abs(velocity)
                        if eta_s < cfg.hedge_anticipation_seconds:
                            return True, f"velocity(eta={eta_s:.1f}s,v={velocity:+.2f})"
                else:
                    trigger_level = ptb + cfg.hedge_threshold
                    distance = trigger_level - btc
                    if velocity > 0 and distance > 0:
                        eta_s = distance / velocity
                        if eta_s < cfg.hedge_anticipation_seconds:
                            return True, f"velocity(eta={eta_s:.1f}s,v={velocity:+.2f})"
                return False, ""
            buf = cfg.hedge_anticipation_buffer_usd
            if entry_side == "up" and btc < ptb + buf:
                return True, "static_fallback"
            if entry_side == "down" and btc > ptb - buf:
                return True, "static_fallback"
            return False, ""

        while self._time_fn() < window_end_s:
            elapsed_s = self._elapsed(window_ts)


            btc = self._signals.btc_price
            best_ask = self._token_prices.get(hedge_token_id)

            # ── PRE-FILL ───────────────────────────────────────────────
            if fill_price is None:
                triggered = (btc is not None
                             and should_hedge(entry_side, btc, ptb, cfg.hedge_threshold))

                if True:
                    # Start accumulation when in anticipation zone
                    if not accumulating and btc is not None:
                        now_ms = int(self._time_fn() * 1000)
                        should_place, reason = _in_anticipation_zone(btc, now_ms)
                        if should_place and best_ask is not None:
                            accumulating = True
                            record.hedge_limit_initial_bid = best_ask
                            record.hedge_trigger_btc = btc
                            print(
                                f"[m5] limit_hedge_live: start accumulation t={elapsed_s:.1f}s"
                                f"  btc={btc:.2f}  ptb={ptb:.2f}"
                                f"  ask={best_ask:.3f}  target={cfg.hedge_bet_usd:.2f}"
                                f"  trig={reason}",
                                flush=True,
                            )

                    # FAK loop: keep buying until target reached
                    if accumulating and best_ask is not None and remaining_usd >= 0.10:
                        filled_usd, fp, fs = await executor.post_limit_buy(
                            hedge_token_id, best_ask + 0.02, remaining_usd, best_ask
                        )
                        if filled_usd > 0:
                            total_cost += filled_usd
                            total_shares_acc += fs
                            remaining_usd -= filled_usd
                            print(
                                f"[m5] limit_hedge_live: +{filled_usd:.2f} filled"
                                f"  total={total_cost:.2f}/{cfg.hedge_bet_usd:.2f}"
                                f"  avg={total_cost/total_shares_acc:.4f}",
                                flush=True,
                            )
                        if remaining_usd < 0.10 and total_cost > 0:
                            fill_price = total_cost / total_shares_acc
                            fill_shares = total_shares_acc
                            fill_elapsed = elapsed_s
                            fill_source = "anticipatory"
                            peak_price = fill_price
                            print(
                                f"[m5] limit_hedge_live: fully accumulated t={elapsed_s:.1f}s"
                                f"  fill={fill_price:.4f}  shares={fill_shares:.4f}"
                                f"  cost={total_cost:.2f}",
                                flush=True,
                            )

                    # Trigger fired — accept partial accumulation or skip
                    if triggered and fill_price is None:
                        record.hedge_trigger_elapsed_s = elapsed_s
                        if total_cost > 0:
                            fill_price = total_cost / total_shares_acc
                            fill_shares = total_shares_acc
                            fill_elapsed = elapsed_s
                            fill_source = "anticipatory"
                            peak_price = fill_price
                            print(
                                f"[m5] limit_hedge_live: trigger fired, partial fill"
                                f"  cost={total_cost:.2f}  accepted",
                                flush=True,
                            )
                        else:
                            fill_source = "skipped"
                            print(
                                f"[m5] limit_hedge_live: trigger fired, no fill — skipped",
                                flush=True,
                            )
                            break

            # ── POST-FILL ──────────────────────────────────────────────
            else:
                # Record when real trigger fires (lag metric)
                if (btc is not None
                        and record.hedge_trigger_elapsed_s is None
                        and should_hedge(entry_side, btc, ptb, cfg.hedge_threshold)):
                    record.hedge_trigger_elapsed_s = elapsed_s
                    lag_s = elapsed_s - fill_elapsed
                    print(
                        f"[m5] limit_hedge_live: real trigger t={elapsed_s:.1f}s"
                        f"  lag={lag_s:+.1f}s after anticipatory fill",
                        flush=True,
                    )

                if best_ask is not None:
                    peak_price = max(peak_price, best_ask)
                    arm_level = fill_price + cfg.hedge_limit_sl_arm_threshold_usd
                    sl_level = fill_price + cfg.hedge_limit_sl_offset_usd

                    # Place SL sell once token has gained enough
                    if not sl_armed and best_ask >= arm_level:
                        sl_armed = True
                        print(
                            f"[m5] limit_hedge_live: arming SL t={elapsed_s:.1f}s"
                            f"  fill={fill_price:.3f}  ask={best_ask:.3f}"
                            f"  → limit sell at {sl_level:.4f}",
                            flush=True,
                        )
                        # Wait for CLOB to settle hedge fill balance before selling
                        await asyncio.sleep(3.0)
                        sl_order_id = await executor.post_limit_sell(
                            hedge_token_id, sl_level, fill_shares
                        )
                        if sl_order_id:
                            print(
                                f"[m5] limit_hedge_live: SL order placed {sl_order_id}",
                                flush=True,
                            )

                    # Check if SL sell has filled
                    if sl_armed and sl_order_id is not None:
                        sl_order = await executor.get_order_status(sl_order_id)
                        sl_status = sl_order.get("status", "")
                        sl_making = float(sl_order.get("makingAmount") or 0)
                        sl_taking = float(sl_order.get("takingAmount") or 0)
                        sl_size_filled = float(
                            sl_order.get("sizeFilled") or sl_order.get("size_filled") or 0
                        )
                        if sl_making > 0 and sl_taking > 0:
                            exit_price = sl_making / sl_taking
                        elif sl_size_filled > 0 or sl_status in ("MATCHED", "FILLED", "SETTLED"):
                            exit_price = sl_level
                        else:
                            exit_price = None
                        if exit_price is not None:
                            exit_pnl = round(fill_shares * (exit_price - fill_price), 4)
                            sl_triggered = True
                            record.hedge_limit_trailing_stop_triggered = True
                            record.hedge_limit_exit_elapsed_s = elapsed_s
                            record.hedge_limit_exit_price = exit_price
                            record.hedge_limit_exit_pnl = exit_pnl
                            print(
                                f"[m5] limit_hedge_live: SL hit t={elapsed_s:.1f}s"
                                f"  fill={fill_price:.3f}  peak={peak_price:.3f}"
                                f"  exit={exit_price:.3f}  pnl={exit_pnl:+.3f}",
                                flush=True,
                            )
                            break

            await asyncio.sleep(0.5)

        # ── Cleanup: cancel open SL sell if not triggered ───────────────
        if sl_order_id is not None and not sl_triggered:
            print(
                f"[m5] limit_hedge_live: cancelling open SL sell {sl_order_id}",
                flush=True,
            )
            await executor.cancel_order(sl_order_id)

        # ── Finalise record ──────────────────────────────────────────────
        record.hedge_limit_reprices = 0
        record.hedge_limit_fill_elapsed_s = fill_elapsed
        record.hedge_limit_fill_source = fill_source
        record.hedge_limit_fill_price = fill_price
        record.hedge_limit_peak_bid = round(peak_price, 4) if peak_price else None

        if fill_price is not None and fill_shares is not None and not record.hedged:
            fill = PaperFillResult(
                fill_price=fill_price,
                shares=fill_shares,
                observed_best_ask=fill_price,
                attempted_price=fill_price,
                slippage=0.0,
                retries=0,
                reject_reason=None,
            )
            self._apply_fill_trace(record, fill, "hedge")
            record.hedged = True
            record.hedge_elapsed_s = fill_elapsed
            record.hedge_side = hedge_side
            record.hedge_price = fill_price
            record.hedge_shares = fill_shares

    # ------------------------------------------------------------------
    # Paper execution
    # ------------------------------------------------------------------

    async def _execute_paper(
        self,
        token_id: str,
        best_ask: float,
        usd_bet: float,
        is_leg1: bool,
    ) -> PaperFillResult:
        """
        Execute an order: routes to the real CLOB if order_executor is set,
        otherwise simulates a FOK fill at best_ask (paper mode).
        """
        cfg = self._cfg

        # price_insane guard: applies to both paper and live
        if best_ask >= cfg.price_insane_threshold:
            self._price_insane_blocks += 1
            return PaperFillResult(
                fill_price=None, shares=None,
                observed_best_ask=best_ask,
                attempted_price=best_ask,
                slippage=0.0,
                retries=0,
                reject_reason="price_insane",
            )

        fok_offset = cfg.fok_price_offset
        fok_max = cfg.fok_max_price_leg1 if is_leg1 else cfg.fak_max_price
        attempted_price = min(best_ask + fok_offset, fok_max)

        if self._order_executor is not None:
            return await self._order_executor(token_id, attempted_price, usd_bet)

        fill_price = best_ask
        shares = usd_bet / fill_price if fill_price > 0 else 0.0
        return PaperFillResult(
            fill_price=fill_price,
            shares=round(shares, 8),
            observed_best_ask=best_ask,
            attempted_price=attempted_price,
            slippage=round(attempted_price - best_ask, 6),
            retries=0,
            reject_reason=None,
        )

    # ------------------------------------------------------------------
    # Background token price polling
    # ------------------------------------------------------------------

    async def _poll_token_prices(self, up_id: str, down_id: str) -> None:
        while True:
            for tid in (up_id, down_id):
                price = await fetch_token_best_ask(self._http, tid, self._clob_url)
                if price is not None:
                    self._token_prices[tid] = price
            await asyncio.sleep(self._cfg.token_price_refresh_s)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _sleep_until(self, target_s: float) -> None:
        now = self._time_fn()
        if now < target_s:
            await asyncio.sleep(target_s - now)

    def _elapsed(self, window_ts: int) -> float:
        return self._time_fn() - window_ts

    def _apply_fill_trace(self, record: TradeRecord, fill: PaperFillResult, role: str) -> None:
        if role == "leg1":
            record.leg1_observed_ask = fill.observed_best_ask
            record.leg1_attempted_price = fill.attempted_price
            record.leg1_slippage = fill.slippage
            record.leg1_fill_retries = fill.retries
        else:
            record.hedge_observed_ask = fill.observed_best_ask
            record.hedge_attempted_price = fill.attempted_price
            record.hedge_slippage = fill.slippage
            record.hedge_fill_retries = fill.retries
