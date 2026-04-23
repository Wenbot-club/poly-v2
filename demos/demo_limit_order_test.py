#!/usr/bin/env python3
"""
Minibot de test pour les ordres limites Polymarket.

Place un ordre limite GTC à best_ask + 0.01 sur les tokens UP et/ou DOWN
du marché BTC M5 courant. Observe si l'ordre fill et à quel prix.
Annule automatiquement après --timeout secondes si non rempli.

Usage:
  python demos/demo_limit_order_test.py --live --usd 0.10
  python demos/demo_limit_order_test.py --live --usd 0.10 --side up
  python demos/demo_limit_order_test.py --live --usd 0.10 --timeout 60
"""
from __future__ import annotations

import argparse
import asyncio
import time as _time
from pathlib import Path

import aiohttp

import sys
sys.stdout.reconfigure(line_buffering=True)


GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"


def _current_window_ts() -> int:
    """Timestamp arrondi à la fenêtre M5 courante (ou prochaine si <10s)."""
    import time
    now = int(time.time())
    boundary = (now // 300) * 300
    # Si on est dans les 10 dernières secondes de la fenêtre, prendre la suivante
    if now - boundary >= 290:
        boundary += 300
    return boundary


async def fetch_tokens(http: aiohttp.ClientSession, window_ts: int):
    slug = f"btc-updown-5m-{window_ts}"
    url  = f"{GAMMA_BASE}/events/slug/{slug}"
    async with http.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status != 200:
            return None
        data = await r.json(content_type=None)
    markets = data.get("markets") or []
    if not markets:
        return None
    import json
    raw = markets[0].get("clobTokenIds")
    ids = json.loads(raw) if isinstance(raw, str) else raw
    if not ids or len(ids) < 2:
        return None
    return str(ids[0]), str(ids[1])  # up, down


async def fetch_best_ask(http: aiohttp.ClientSession, token_id: str) -> float | None:
    url = f"{CLOB_BASE}/price"
    async with http.get(url, params={"token_id": token_id, "side": "buy"},
                        timeout=aiohttp.ClientTimeout(total=5)) as r:
        if r.status != 200:
            return None
        data = await r.json(content_type=None)
        v = data.get("price")
        return float(v) if v is not None else None


async def poll_until_filled(executor, order_id: str, timeout_s: float, label: str, t_sent: float):
    """Poll toutes les 0.5s pour détecter le fill avec précision ~500ms."""
    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        await asyncio.sleep(0.5)
        status = await executor.get_order_status(order_id)
        size_matched = float(status.get("size_matched") or 0)
        size = float(status.get("size") or 0)
        order_status = status.get("status") or status.get("order_status") or "?"
        price = status.get("price") or status.get("avg_price") or "?"
        elapsed_ms = (_time.time() - t_sent) * 1000
        print(f"  [{label}] +{elapsed_ms:.0f}ms  status={order_status}  matched={size_matched}/{size}  avg_price={price}")
        if size_matched > 0 or order_status in ("MATCHED", "matched", "FILLED", "filled"):
            return True, size_matched, price
        if order_status in ("CANCELLED", "cancelled", "CANCELED", "canceled"):
            return False, 0, None
    return False, 0, None


async def run_test(args):
    from bot.trading.credentials import load_credentials
    from bot.trading.live_executor import LiveOrderExecutor

    creds = load_credentials()
    executor = LiveOrderExecutor(creds)
    executor.start_heartbeat()

    async with aiohttp.ClientSession() as http:
        wts = _current_window_ts()
        print(f"\n=== Limit Order Test — window ts={wts} ===")
        print(f"    usd={args.usd}  offset=+0.01  timeout={args.timeout}s\n")

        tokens = await fetch_tokens(http, wts)
        if tokens is None:
            print(f"[!] Aucun token trouvé pour la fenêtre {wts}. Essai fenêtre suivante.")
            wts += 300
            tokens = await fetch_tokens(http, wts)
        if tokens is None:
            print("[!] Impossible de récupérer les tokens — abort.")
            return

        up_id, down_id = tokens
        print(f"  UP   token_id={up_id[:16]}...")
        print(f"  DOWN token_id={down_id[:16]}...")

        sides = []
        if args.side in ("up", "both"):
            sides.append(("UP", up_id))
        if args.side in ("down", "both"):
            sides.append(("DOWN", down_id))

        for label, token_id in sides:
            ask_before = await fetch_best_ask(http, token_id)
            if ask_before is None:
                print(f"\n[{label}] Impossible de lire le best_ask — skip.")
                continue

            limit_price = round(ask_before + 0.01, 2)
            limit_price = min(limit_price, 0.99)
            print(f"\n[{label}]  best_ask(avant)={ask_before:.4f}  order_price={limit_price:.2f}  usd={args.usd}")

            if not args.live:
                print(f"[{label}]  DRY-RUN — aucun ordre envoyé (ajoute --live pour envoyer réellement)")
                continue

            t_sent = _time.time()
            order_id, fill_price, fill_shares = await executor.post_limit_buy(
                token_id=token_id,
                max_price=limit_price,
                usd_amount=args.usd,
                observed_ask=ask_before,
            )
            t_posted = _time.time()
            ask_after = await fetch_best_ask(http, token_id)
            latency_post_ms = (t_posted - t_sent) * 1000

            drift = None if ask_after is None else ask_after - ask_before
            print(f"[{label}]  best_ask(après)={ask_after}  drift={drift:+.4f}" if drift is not None else f"[{label}]  best_ask(après)=N/A")
            print(f"[{label}]  latence envoi→réponse API : {latency_post_ms:.1f} ms")

            if order_id is None:
                print(f"[{label}]  ERREUR — ordre non posté")
                continue

            print(f"[{label}]  ordre posté  order_id={order_id}")

            if fill_price is not None:
                print(f"[{label}]  FILL IMMÉDIAT (dans la réponse post_order)  price={fill_price:.4f}  shares={fill_shares}")
                print(f"[{label}]  latence totale envoi→fill : {latency_post_ms:.1f} ms")
                continue

            print(f"[{label}]  En attente de fill ({args.timeout}s)...")
            filled, matched, avg_price = await poll_until_filled(
                executor, order_id, args.timeout, label, t_sent
            )

            if filled:
                total_ms = (_time.time() - t_sent) * 1000
                print(f"[{label}]  FILLED  avg_price={avg_price}  shares={matched}")
                print(f"[{label}]  latence totale envoi→fill détecté : {total_ms:.1f} ms (précision ±2000ms poll)")
            else:
                print(f"[{label}]  Timeout — annulation de l'ordre {order_id}")
                cancelled = await executor.cancel_order(order_id)
                print(f"[{label}]  Annulation {'OK' if cancelled else 'ECHEC'}")


def main():
    parser = argparse.ArgumentParser(description="Test d'ordres limites Polymarket")
    parser.add_argument("--live", action="store_true",
                        help="Envoyer de vrais ordres (sans ce flag = dry-run)")
    parser.add_argument("--usd", type=float, default=1.0,
                        help="Montant USD par ordre (min Polymarket = $1, défaut: 1.0)")
    parser.add_argument("--side", choices=["up", "down", "both"], default="both",
                        help="Côté(s) à tester (défaut: both)")
    parser.add_argument("--timeout", type=float, default=30,
                        help="Secondes avant annulation si non rempli (défaut: 30)")
    args = parser.parse_args()
    asyncio.run(run_test(args))


if __name__ == "__main__":
    main()
