"""
Paper bot pour la strategie leg1_stop_0.55_C.

Logique :
  - Aligne sur les windows M5 (multiples de 300s UTC)
  - Recupere PTB depuis api.polymarket.com
  - Loop seconde par seconde de s=10 a s=300
  - A chaque tick : poll btc + up + down
  - s=140-170 : scan signal via compute_entry_signal()
  - s=170 (si pas de signal early) : baseline_direction
  - Apres entree leg1 : arme le stop-buy hedge a partir de s_signal + 11
  - Stop fire si hedge_token_ask >= 0.55 -> simule fill au prix marche
  - Apres s=300 : recupere close price -> settle
  - Log tout dans paper_log.jsonl

Lance : python paper_bot_stop055_C.py
Stoppe : Ctrl+C
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

_STATS_LOCK = threading.Lock()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bot.settings import DEFAULT_M5_CONFIG as CFG
from bot.strategy.btc_m5 import compute_entry_signal, baseline_direction

# === Strategy parameters ===
LEG1_STAKE = 1.0
HEDGE_STAKE = 2.0
STOP_LEVEL = 0.55
MIN_ARM_DELAY_S = 11

# === API endpoints ===
BINANCE_SPOT = "https://api.binance.com/api/v3/ticker/price"
CLOB_PRICE = "https://clob.polymarket.com/price"
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CRYPTO_PRICE = "https://polymarket.com/api/crypto/crypto-price"

LOG_FILE = ROOT / "paper_log.jsonl"

# Cumulative state for live display
_STATS = {"n_trades": 0, "n_wins": 0, "pnl_total": 0.0}


def _now_str():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def log(payload: dict):
    """Write JSON to file and print human-readable summary to console."""
    payload["log_ts"] = time.time()
    line = json.dumps(payload)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    # Human-readable console output
    e = payload.get("event")
    s = payload.get("s")
    s_str = f"s={s:>3}" if s is not None else "        "
    prefix = f"[{_now_str()}] {s_str}"

    if e == "window_start":
        ts = payload.get("window_ts", 0)
        utc = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M")
        print(f"\n{prefix} === Window {utc} UTC start ===", flush=True)
    elif e == "ptb_ok":
        print(f"{prefix} PTB = {payload['ptb']:,.2f}", flush=True)
    elif e == "tokens_ok":
        print(f"{prefix} Tokens UP/DOWN OK", flush=True)
    elif e == "leg1_entry":
        src = payload.get("source", "?")
        d = payload.get("direction")
        price = payload.get("leg1_price")
        btc = payload.get("btc")
        ptb = payload.get("ptb")
        gap = btc - ptb if btc and ptb else 0
        print(f"{prefix} >> LEG1 {d.upper():<4} @ {price:.3f}  "
              f"(src={src}, BTC={btc:,.2f}, gap={gap:+.2f}, hedge_arme_a_s={payload.get('hedge_armed_at_s')})",
              flush=True)
    elif e == "signal_rejected_price":
        print(f"{prefix} signal {payload['direction']} rejete : prix {payload['price']} hors limites", flush=True)
    elif e == "hedge_filled":
        delay = payload.get("delay_since_signal")
        side = payload.get("hedge_side")
        price = payload.get("hedge_fill_price")
        print(f"{prefix} >> HEDGE {side.upper():<4} fill @ {price:.3f}  (delay={delay}s)", flush=True)
    elif e == "settle":
        r = payload["result"]
        pl = payload["pnl_leg1"]
        ph = payload["pnl_hedge"]
        pt = payload["pnl_total"]
        emoji = "WIN " if pt > 0 else "LOSS"
        d = payload.get("direction") or "?"
        leg1_p = payload.get("leg1_price")
        h_filled = payload.get("hedge_filled")
        h_price = payload.get("hedge_fill_price")
        leg1_str = f"leg1={d}@{leg1_p:.3f}" if leg1_p else "leg1=NONE"
        hedge_str = f"hedge@{h_price:.3f}" if h_filled else "no_hedge"
        print(f"{prefix} ## SETTLE {emoji}  result={r.upper()}  {leg1_str}  {hedge_str}  "
              f"|  pnl_leg1=${pl:+.3f}  pnl_hedge=${ph:+.3f}  TOTAL=${pt:+.3f}",
              flush=True)
        with _STATS_LOCK:
            _STATS["n_trades"] += 1
            _STATS["pnl_total"] += pt
            if pt > 0:
                _STATS["n_wins"] += 1
            n = _STATS["n_trades"]
            w = _STATS["n_wins"]
            tot = _STATS["pnl_total"]
        wr = w / n
        avg = tot / n
        print(f"{'':>11}    >>> CUMUL: {n} trades  |  "
              f"WR={wr:.1%}  |  P&L=${tot:+.2f}  |  avg=${avg:+.4f}/trade <<<",
              flush=True)
    elif e == "skip_no_ptb":
        print(f"{prefix} SKIP : PTB unavailable", flush=True)
    elif e == "skip_no_tokens":
        print(f"{prefix} SKIP : tokens unavailable", flush=True)
    elif e == "settle_skip_no_close":
        print(f"{prefix} SKIP : api close unavailable", flush=True)
    elif e == "window_end":
        pass  # silent
    elif e in ("binance_error", "ptb_error", "gamma_error"):
        print(f"{prefix} ERR {e}: {payload.get('err')}", flush=True)
    elif e == "error_window":
        print(f"{prefix} ERR window: {payload.get('err')}", flush=True)


def get_btc_spot() -> Optional[float]:
    try:
        r = requests.get(BINANCE_SPOT, params={"symbol": "BTCUSDT"}, timeout=3)
        if r.status_code == 200:
            return float(r.json()["price"])
    except Exception as e:
        log({"event": "binance_error", "err": str(e)})
    return None


def get_clob_price(token_id: str, side: str = "BUY") -> Optional[float]:
    try:
        r = requests.get(CLOB_PRICE, params={"token_id": token_id, "side": side}, timeout=4)
        if r.status_code == 200:
            return float(r.json().get("price", 0))
    except Exception:
        pass
    return None


def get_event(slug: str) -> Optional[dict]:
    try:
        r = requests.get(GAMMA_EVENTS, params={"slug": slug}, timeout=8)
        if r.status_code == 200:
            ev = r.json()
            return ev[0] if ev else None
    except Exception as e:
        log({"event": "gamma_error", "err": str(e), "slug": slug})
    return None


def get_token_ids(event: dict) -> tuple[Optional[str], Optional[str]]:
    if not event or not event.get("markets"):
        return None, None
    market = event["markets"][0]
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        try:
            tokens = json.loads(raw)
        except Exception:
            return None, None
    else:
        tokens = raw
    if not tokens or len(tokens) < 2:
        return None, None
    return tokens[0], tokens[1]


def get_ptb_or_close(window_ts: int) -> tuple[Optional[float], Optional[float]]:
    """Returns (openPrice, closePrice). closePrice may be None until after window."""
    dt_start = datetime.fromtimestamp(window_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dt_end = datetime.fromtimestamp(window_ts + 300, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        r = requests.get(CRYPTO_PRICE, params={
            "symbol": "BTC",
            "eventStartTime": dt_start,
            "variant": "fiveminute",
            "endDate": dt_end,
        }, timeout=10)
        if r.status_code == 200:
            d = r.json()
            op = d.get("openPrice")
            cp = d.get("closePrice")
            return (float(op) if op else None, float(cp) if cp else None)
    except Exception as e:
        log({"event": "ptb_error", "err": str(e), "window_ts": window_ts})
    return None, None


def fetch_ptb(window_ts: int) -> Optional[float]:
    for _ in range(5):
        op, _ = get_ptb_or_close(window_ts)
        if op:
            return op
        time.sleep(2)
    return None


def fetch_close(window_ts: int) -> Optional[float]:
    """Available a few seconds after window end. Patience up to ~5 min."""
    for _ in range(60):
        _, cp = get_ptb_or_close(window_ts)
        if cp:
            return cp
        time.sleep(5)
    return None


# ============== Trade state ==============

class WindowState:
    def __init__(self, window_ts: int):
        self.window_ts = window_ts
        self.ptb: Optional[float] = None
        self.token_up: Optional[str] = None
        self.token_down: Optional[str] = None

        self.s_signal: Optional[int] = None
        self.direction: Optional[str] = None
        self.signal_source: Optional[str] = None
        self.leg1_price: Optional[float] = None
        self.leg1_filled = False

        self.hedge_armed_at_s: Optional[int] = None
        self.hedge_filled = False
        self.hedge_fill_price: Optional[float] = None
        self.hedge_fill_s: Optional[int] = None

        self.btc_history: list[tuple[int, float]] = []

    def hedge_side(self) -> str:
        return "down" if self.direction == "up" else "up"


def try_signal(state: WindowState, s: int, btc: float, up: float, down: float) -> bool:
    if state.leg1_filled:
        return False

    samples = [(ms, p) for (ms, p) in state.btc_history
               if ms >= (s * 1000 - int(CFG.sigma_lookback_s * 1000))]
    if len(samples) < 3:
        return False

    btc_10s = next((p for ms, p in reversed(state.btc_history) if ms <= (s - 10) * 1000), None)
    btc_30s = next((p for ms, p in reversed(state.btc_history) if ms <= (s - 30) * 1000), None)

    if CFG.entry_scan_start_s <= s < CFG.entry_scan_end_s:
        sig = compute_entry_signal(
            btc=btc, ptb=state.ptb, tau_s=300.0 - s,
            btc_samples=samples,
            btc_10s=btc_10s, btc_30s=btc_30s,
            price_up=up, price_down=down,
            sigma_floor_usd=CFG.sigma_floor_usd,
            z_gap_min=CFG.z_gap_min,
            p_enter_up_min=CFG.p_enter_up_min,
            p_enter_down_max=CFG.p_enter_down_max,
            min_entry_edge=CFG.min_entry_edge,
        )
        if sig.direction is not None:
            price = up if sig.direction == "up" else down
            if price is None or price <= 0 or price >= CFG.fok_max_price_leg1:
                log({"event": "signal_rejected_price", "s": s,
                     "direction": sig.direction, "price": price})
                return False
            state.s_signal = s
            state.direction = sig.direction
            state.signal_source = "early"
            state.leg1_price = price
            state.leg1_filled = True
            state.hedge_armed_at_s = s + MIN_ARM_DELAY_S
            log({
                "event": "leg1_entry", "window_ts": state.window_ts, "s": s,
                "source": "early", "direction": sig.direction, "leg1_price": price,
                "btc": btc, "ptb": state.ptb, "z_gap": sig.z_gap,
                "p_model_up": sig.p_model_up, "edge_up": sig.edge_up,
                "edge_down": sig.edge_down, "hedge_armed_at_s": state.hedge_armed_at_s,
            })
            return True
        return False

    if s == int(CFG.baseline_elapsed_s):
        d = baseline_direction(btc, state.ptb)
        if d is not None:
            price = up if d == "up" else down
            if price is None or price <= 0 or price >= CFG.fok_max_price_leg1:
                log({"event": "baseline_rejected_price", "s": s, "direction": d, "price": price})
                return False
            state.s_signal = s
            state.direction = d
            state.signal_source = "baseline"
            state.leg1_price = price
            state.leg1_filled = True
            state.hedge_armed_at_s = s + MIN_ARM_DELAY_S
            log({
                "event": "leg1_entry", "window_ts": state.window_ts, "s": s,
                "source": "baseline", "direction": d, "leg1_price": price,
                "btc": btc, "ptb": state.ptb, "hedge_armed_at_s": state.hedge_armed_at_s,
            })
            return True
    return False


def try_hedge(state: WindowState, s: int, up: float, down: float):
    if not state.leg1_filled or state.hedge_filled:
        return
    if state.hedge_armed_at_s is None or s < state.hedge_armed_at_s:
        return
    hedge_side = state.hedge_side()
    hedge_price = down if hedge_side == "down" else up
    if hedge_price is None:
        return
    if hedge_price >= STOP_LEVEL:
        state.hedge_filled = True
        state.hedge_fill_price = hedge_price
        state.hedge_fill_s = s
        log({
            "event": "hedge_filled", "window_ts": state.window_ts, "s": s,
            "delay_since_signal": s - state.s_signal if state.s_signal else None,
            "hedge_side": hedge_side, "hedge_fill_price": hedge_price,
            "leg1_direction": state.direction,
        })


def settle(state: WindowState, api_open: float, api_close: float) -> dict:
    result = "up" if api_close >= api_open else "down"
    pnl_leg1 = 0.0
    pnl_hedge = 0.0
    if state.leg1_filled and state.leg1_price and state.leg1_price > 0:
        shares_l = LEG1_STAKE / state.leg1_price
        if result == state.direction:
            pnl_leg1 = (1.0 - state.leg1_price) * shares_l
        else:
            pnl_leg1 = -LEG1_STAKE
    if state.hedge_filled and state.hedge_fill_price and state.hedge_fill_price > 0:
        shares_h = HEDGE_STAKE / state.hedge_fill_price
        if result == state.hedge_side():
            pnl_hedge = (1.0 - state.hedge_fill_price) * shares_h
        else:
            pnl_hedge = -HEDGE_STAKE
    total = pnl_leg1 + pnl_hedge
    return {
        "event": "settle", "window_ts": state.window_ts,
        "api_open": api_open, "api_close": api_close, "result": result,
        "leg1_filled": state.leg1_filled, "direction": state.direction,
        "leg1_price": state.leg1_price, "hedge_filled": state.hedge_filled,
        "hedge_fill_price": state.hedge_fill_price, "hedge_fill_s": state.hedge_fill_s,
        "pnl_leg1": round(pnl_leg1, 4), "pnl_hedge": round(pnl_hedge, 4),
        "pnl_total": round(total, 4),
    }


# ============== Main loop ==============

def run_window(window_ts: int) -> None:
    dt = datetime.fromtimestamp(window_ts, tz=timezone.utc)
    log({"event": "window_start", "window_ts": window_ts, "utc": dt.isoformat()})

    state = WindowState(window_ts)

    wait_until = window_ts + 15
    while time.time() < wait_until:
        time.sleep(0.5)

    state.ptb = fetch_ptb(window_ts)
    if not state.ptb:
        log({"event": "skip_no_ptb", "window_ts": window_ts})
        return
    log({"event": "ptb_ok", "window_ts": window_ts, "ptb": state.ptb})

    slug = f"btc-updown-5m-{window_ts}"
    event = get_event(slug)
    state.token_up, state.token_down = get_token_ids(event) if event else (None, None)
    if not state.token_up or not state.token_down:
        log({"event": "skip_no_tokens", "window_ts": window_ts, "slug": slug})
        return
    log({"event": "tokens_ok", "window_ts": window_ts})

    while True:
        now = time.time()
        s = int(now - window_ts)
        if s >= 300:
            break
        if s < 10:
            time.sleep(0.5)
            continue

        btc = get_btc_spot()
        in_entry_zone = CFG.entry_scan_start_s - 5 <= s < CFG.entry_scan_end_s + 5
        in_hedge_zone = state.leg1_filled and not state.hedge_filled
        poll_clob = (s % 2 == 0) or in_entry_zone or in_hedge_zone
        up_p = down_p = None
        if poll_clob:
            up_p = get_clob_price(state.token_up, "BUY")
            down_p = get_clob_price(state.token_down, "BUY")

        if btc:
            state.btc_history.append((s * 1000, btc))

        if btc and up_p is not None and down_p is not None:
            if not state.leg1_filled:
                try_signal(state, s, btc, up_p, down_p)
            try_hedge(state, s, up_p, down_p)

        next_s_target = window_ts + s + 1
        delay = max(0.0, next_s_target - time.time())
        if delay > 0:
            time.sleep(delay)

    log({"event": "window_end", "window_ts": window_ts})
    threading.Thread(target=_settle_in_background, args=(state,), daemon=True).start()


def _settle_in_background(state: WindowState):
    api_open = state.ptb
    api_close = fetch_close(state.window_ts)
    if api_close is None:
        log({"event": "settle_skip_no_close", "window_ts": state.window_ts})
        return
    res = settle(state, api_open, api_close)
    log(res)


def next_window_ts(now: float) -> int:
    n = int(now)
    return ((n + 299) // 300) * 300


def main():
    print(f"\n=== Paper Bot leg1_stop_0.55_C ===", flush=True)
    print(f"Strategy: leg1 + stop-buy hedge a {STOP_LEVEL}, delai {MIN_ARM_DELAY_S}s", flush=True)
    print(f"Stakes: leg1=${LEG1_STAKE}, hedge=${HEDGE_STAKE}", flush=True)
    print(f"Log file: {LOG_FILE}", flush=True)
    print(f"Stoppe avec Ctrl+C\n", flush=True)

    while True:
        now = time.time()
        wts = next_window_ts(now)
        wait = wts - now
        dt_next = datetime.fromtimestamp(wts, tz=timezone.utc)
        print(f"\n[{_now_str()}] Next window: {dt_next.strftime('%H:%M:%S')} UTC (in {wait:.0f}s)",
              flush=True)
        time.sleep(max(0, wait - 1))

        try:
            run_window(wts)
        except KeyboardInterrupt:
            print("\nStoppe.", flush=True)
            break
        except Exception as e:
            log({"event": "error_window", "err": str(e), "window_ts": wts})
            print(f"Error: {e}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.", flush=True)
