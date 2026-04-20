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
import json
import math
import re
import time as _time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import aiohttp

from .m5_summary import TradeRecord
from .settings import M5Config, DEFAULT_M5_CONFIG
from .strategy.btc_m5 import (
    ConsensusResult,
    compute_consensus,
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


# ---------------------------------------------------------------------------
# PTB fetching (3 sources)
# ---------------------------------------------------------------------------

async def fetch_ptb_ssr(
    http: aiohttp.ClientSession,
    window_ts: int,
    base_url: str = _POLY_BASE,
) -> Optional[float]:
    """Scrape openPrice from Polymarket event page (__NEXT_DATA__ JSON)."""
    url = f"{base_url}/event/btc-updown-5m-{window_ts}"
    try:
        async with http.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            html = await resp.text()
            m = re.search(r'"openPrice"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)"?', html)
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return None


async def fetch_ptb_api(
    http: aiohttp.ClientSession,
    window_ts: int,
    base_url: str = _POLY_BASE,
) -> Optional[float]:
    """Fetch openPrice from Polymarket crypto-price API."""
    url = f"{base_url}/api/crypto/crypto-price"
    params = {"symbol": "BTC", "eventStartTime": window_ts, "variant": "fiveminute"}
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


# ---------------------------------------------------------------------------
# Token discovery
# ---------------------------------------------------------------------------

async def fetch_m5_tokens(
    http: aiohttp.ClientSession,
    window_ts: int,
    gamma_base_url: str = _GAMMA_BASE,
) -> Optional[tuple]:  # (up_token_id, down_token_id) or None
    """Return (up_token_id, down_token_id) for the M5 window, or None."""
    slug = f"btc-updown-5m-{window_ts}"
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
    timeout_s: float = 60.0,
    poll_interval_s: float = 5.0,
) -> Optional[float]:
    """Poll for closePrice after window expiry."""
    url = f"{polymarket_base_url}/api/crypto/crypto-price"
    params = {"symbol": "BTC", "eventStartTime": window_ts, "variant": "fiveminute"}
    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        try:
            async with http.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    v = data.get("closePrice")
                    if v is not None:
                        return float(v)
        except Exception:
            pass
        await asyncio.sleep(poll_interval_s)
    return None


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
    ) -> None:
        self._http = http_session
        self._signals = signal_state
        self._cfg = config
        self._time_fn = time_fn
        self._gamma_url = gamma_base_url
        self._clob_url = clob_base_url
        self._poly_url = polymarket_base_url

        self._token_prices: dict = {}   # token_id → best_ask
        self._btc_history = btc_history if btc_history is not None else BtcHistory()
        self._price_insane_blocks: int = 0
        self._hedge_blocked_by_cutoff: bool = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, window_ts: int) -> TradeRecord:
        record = TradeRecord(window_ts=window_ts)

        # Phase 1: wait until PTB fetch is safe
        await self._sleep_until(window_ts + self._cfg.ptb_fetch_delay_s)

        # Phase 2: fetch PTB
        cl_price = self._signals.chainlink_price
        cl_age_s: Optional[float] = None
        if cl_price is not None and self._signals.chainlink_price_ts_ms is not None:
            cl_age_s = (self._time_fn() - self._signals.chainlink_price_ts_ms / 1000.0)

        ptb, ptb_source = await fetch_ptb(
            self._http, window_ts, cl_price, cl_age_s,
            polymarket_base_url=self._poly_url,
        )
        if ptb is None:
            record.abort_reason = "ptb_unavailable"
            return record
        record.ptb = ptb
        record.ptb_source = ptb_source

        # Phase 3: token discovery
        tokens = await fetch_m5_tokens(self._http, window_ts, self._gamma_url)
        if tokens is None:
            record.abort_reason = "tokens_unavailable"
            return record
        up_id, down_id = tokens

        # Phase 4: start background token price polling
        poll_task = asyncio.create_task(self._poll_token_prices(up_id, down_id))

        try:
            await self._trading_phases(record, ptb, window_ts, up_id, down_id)
        finally:
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass

        # Propagate session-level counters to record
        record.price_insane_block_count = self._price_insane_blocks
        record.hedge_blocked_by_cutoff = self._hedge_blocked_by_cutoff

        # Phase 8: settlement
        window_end_s = window_ts + self._cfg.window_seconds
        now = self._time_fn()
        if now < window_end_s:
            await asyncio.sleep(window_end_s - now)

        if record.entry_side is not None and record.entry_price is not None:
            close_price = await fetch_close_price(
                self._http, window_ts,
                polymarket_base_url=self._poly_url,
                timeout_s=self._cfg.settlement_timeout_s,
                poll_interval_s=self._cfg.settlement_poll_s,
            )
            if close_price is not None:
                s = compute_settlement(
                    close_price=close_price,
                    open_price=ptb,
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
        await self._sleep_until(window_ts + cfg.early_window_start_s)

        while self._elapsed(window_ts) < cfg.early_window_end_s:
            if await self._try_early_entry(record, ptb, up_id, down_id, window_ts):
                break
            await asyncio.sleep(cfg.early_poll_interval_s)

        # BASELINE: one-shot at 170s if no LEG1 yet
        if record.entry_side is None:
            await self._sleep_until(window_ts + cfg.baseline_elapsed_s)
            await self._try_baseline_entry(record, ptb, up_id, down_id, window_ts)

        # HEDGE watch: only if LEG1 was taken
        if record.entry_side is not None:
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
            return False

        now_ms = int(self._time_fn() * 1000)
        consensus = compute_consensus(
            btc=btc, ptb=ptb,
            chainlink=self._signals.chainlink_price,
            btc_10s=self._btc_history.price_n_secs_ago(10, now_ms),
            btc_30s=self._btc_history.price_n_secs_ago(30, now_ms),
            btc_60s=self._btc_history.price_n_secs_ago(60, now_ms),
            price_up=self._token_prices.get(up_id),
            threshold=self._cfg.early_consensus_threshold,
            min_non_neutral=self._cfg.early_min_non_neutral,
        )
        if consensus.direction is None:
            return False

        token_id = up_id if consensus.direction == "up" else down_id
        best_ask = self._token_prices.get(token_id)
        if best_ask is None:
            return False

        fill = await self._execute_paper(best_ask, self._cfg.leg1_bet_usd, is_leg1=True)
        self._apply_fill_trace(record, fill, "leg1")
        if fill.reject_reason is not None:
            return False

        record.entry_mode = "early"
        record.entry_side = consensus.direction
        record.entry_elapsed_s = self._elapsed(window_ts)
        record.entry_price = fill.fill_price
        record.entry_shares = fill.shares
        record.entry_consensus_score = consensus.score
        record.entry_consensus_non_neutral = consensus.non_neutral
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

        fill = await self._execute_paper(best_ask, self._cfg.leg1_bet_usd, is_leg1=True)
        self._apply_fill_trace(record, fill, "leg1")
        if fill.reject_reason is not None:
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

            if elapsed_s >= cfg.hedge_cutoff_s:
                self._hedge_blocked_by_cutoff = True
                print(f"[m5] hedge_cutoff at {elapsed_s:.1f}s — no hedge posted", flush=True)
                break

            btc = self._signals.btc_price
            if btc is not None and record.entry_side is not None:
                if should_hedge(record.entry_side, btc, ptb, cfg.hedge_threshold):
                    hedge_side = "down" if record.entry_side == "up" else "up"
                    token_id = down_id if hedge_side == "down" else up_id
                    best_ask = self._token_prices.get(token_id)
                    if best_ask is not None:
                        fill = await self._execute_paper(
                            best_ask, cfg.hedge_bet_usd, is_leg1=False
                        )
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

    # ------------------------------------------------------------------
    # Paper execution
    # ------------------------------------------------------------------

    async def _execute_paper(
        self,
        best_ask: float,
        usd_bet: float,
        is_leg1: bool,
    ) -> PaperFillResult:
        """
        Simulates FOK order. Fills at best_ask.
        LIMITATION: no partial fills, no FAK retries in paper mode.
        In real trading the attempted_price (best_ask + offset) would be sent
        to the exchange; here it is only traced for slippage analysis.
        """
        cfg = self._cfg

        # price_insane guard: refuse before any offset logic
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
