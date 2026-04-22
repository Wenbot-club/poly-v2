"""
Live order executor — posts real FOK BUY orders to the Polymarket CLOB.

Uses py-clob-client for EIP-712 order signing.  All synchronous
py-clob-client calls are wrapped in asyncio.to_thread so the main
event loop stays non-blocking.

Returns PaperFillResult for drop-in compatibility with the paper path.
"""
from __future__ import annotations

import asyncio
import time as _time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from bot.m5_session import PaperFillResult
    from bot.trading.credentials import Credentials


class LiveOrderExecutor:
    """
    One instance per trading session; ClobClient is created once on init.
    """

    _MARKET_PRICE_CAP = 0.99  # FOK limit — fills at best ask, up to this cap
    _HEARTBEAT_IDLE_S = 15.0
    _HEARTBEAT_HOT_S = 2.0
    _HOT_DURATION_S = 150.0  # stay hot for ~one post-entry window

    def __init__(self, creds: "Credentials") -> None:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        from py_clob_client.constants import POLYGON

        # signature_type: 0=EOA, 1=POLY_PROXY, 2=POLY_GNOSIS_SAFE.
        # When signer == funder the account is a plain EOA (sig_type=0).
        # Otherwise the funder is a Polymarket Gnosis Safe proxy (sig_type=2).
        same = creds.signer_address.lower() == creds.funder_address.lower()
        self._client = ClobClient(
            host="https://clob.polymarket.com",
            key="0x" + creds.private_key.removeprefix("0x"),
            chain_id=POLYGON,
            creds=ApiCreds(
                api_key=creds.api_key,
                api_secret=creds.api_secret,
                api_passphrase=creds.api_passphrase,
            ),
            **({} if same else {"funder": creds.funder_address, "signature_type": 2}),
        )
        self._hot_until: float = 0.0
        self._heartbeat_task: Optional[asyncio.Task] = None
        # Pre-sign state for hedge order
        self._hedge_presign_task: Optional[asyncio.Task] = None
        self._hedge_presign_token_id: Optional[str] = None
        self._hedge_presign_usd: float = 0.0

    def start_heartbeat(self) -> None:
        """
        Launch a background task that pings the CLOB periodically to keep
        the TCP/TLS connection warm.  Rate is adaptive: 15s idle, 2s during
        the ~150s post-entry critical window.
        """
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while True:
            hot = _time.time() < self._hot_until
            await asyncio.sleep(self._HEARTBEAT_HOT_S if hot else self._HEARTBEAT_IDLE_S)
            try:
                await asyncio.to_thread(self._client.get_ok)
            except Exception:
                pass

    async def prewarm(self, *token_ids: str) -> None:
        """
        Prime the internal caches (neg_risk, tick_size) for the given tokens.

        Call this ahead of entry so the hot path skips ~300ms of HTTP fetches.
        Runs in a thread so we don't block the event loop; errors are swallowed.
        """
        def _warm():
            for tid in token_ids:
                try:
                    self._client.get_neg_risk(tid)
                    self._client.get_tick_size(tid)
                except Exception:
                    pass
        await asyncio.to_thread(_warm)

    async def presign_hedge(self, token_id: str, usd_amount: float) -> None:
        """
        Pre-sign the hedge order right after entry fills.  Runs create_market_order
        in a background thread so the hedge hot path only needs to call post_order.
        """
        self._hedge_presign_token_id = token_id
        self._hedge_presign_usd = usd_amount
        self._hedge_presign_task = asyncio.create_task(
            asyncio.to_thread(self._do_presign, token_id, usd_amount)
        )

    def _do_presign(self, token_id: str, usd_amount: float) -> object:
        from py_clob_client.clob_types import MarketOrderArgs
        args = MarketOrderArgs(
            token_id=token_id,
            amount=usd_amount,
            price=self._MARKET_PRICE_CAP,
            side="BUY",
        )
        return self._client.create_market_order(args)

    async def post_presigned_hedge(self, observed_ask: float) -> "PaperFillResult":
        """
        Post the hedge order using the pre-signed bytes when available.
        Falls back to the full sign+post path if presign failed or wasn't done.
        """
        from bot.m5_session import PaperFillResult
        from py_clob_client.clob_types import OrderType

        self._hot_until = _time.time() + self._HOT_DURATION_S
        order_price = self._MARKET_PRICE_CAP
        token_id = self._hedge_presign_token_id or ""
        usd_amount = self._hedge_presign_usd

        signed = None
        if self._hedge_presign_task is not None:
            try:
                signed = await self._hedge_presign_task
            except Exception:
                signed = None

        if signed is None:
            print("[live] hedge presign=MISS — signing on hot path", flush=True)
            try:
                return await asyncio.to_thread(
                    self._post_fok, token_id, order_price, usd_amount, observed_ask
                )
            except Exception as exc:
                return PaperFillResult(
                    fill_price=None, shares=None, observed_best_ask=observed_ask,
                    attempted_price=order_price, slippage=0.0, retries=0,
                    reject_reason=str(exc)[:200],
                )

        print("[live] hedge presign=HIT — skipping sign", flush=True)
        try:
            return await asyncio.to_thread(
                self._post_presigned, signed, observed_ask, order_price
            )
        except Exception as exc:
            return PaperFillResult(
                fill_price=None, shares=None, observed_best_ask=observed_ask,
                attempted_price=order_price, slippage=0.0, retries=0,
                reject_reason=str(exc)[:200],
            )

    def _post_presigned(
        self,
        signed: object,
        observed_ask: float,
        order_price: float,
    ) -> "PaperFillResult":
        from py_clob_client.clob_types import OrderType
        from bot.m5_session import PaperFillResult

        resp = self._client.post_order(signed, OrderType.FOK)
        success = bool(resp.get("success", False))
        making = float(resp.get("makingAmount", 0)) if success else 0.0
        taking = float(resp.get("takingAmount", 0)) if success else 0.0
        fill_price = (making / taking) if (success and taking > 0) else None
        fill_shares = taking if success else None
        return PaperFillResult(
            fill_price=fill_price,
            shares=fill_shares,
            observed_best_ask=observed_ask,
            attempted_price=order_price,
            slippage=round((fill_price or order_price) - observed_ask, 6),
            retries=0,
            reject_reason=None if success else (resp.get("errorMsg") or "fok_rejected"),
        )

    async def __call__(
        self,
        token_id: str,
        price: float,
        usd_bet: float,
    ) -> "PaperFillResult":
        from bot.m5_session import PaperFillResult

        # Enter hot heartbeat mode so the connection stays warm for the hedge.
        self._hot_until = _time.time() + self._HOT_DURATION_S

        order_price = self._MARKET_PRICE_CAP
        try:
            return await asyncio.to_thread(
                self._post_fok, token_id, order_price, usd_bet, price
            )
        except Exception as exc:
            return PaperFillResult(
                fill_price=None,
                shares=None,
                observed_best_ask=price,
                attempted_price=order_price,
                slippage=0.0,
                retries=0,
                reject_reason=str(exc)[:200],
            )

    # ------------------------------------------------------------------
    # Limit-order hedge — live execution
    # ------------------------------------------------------------------

    async def post_limit_buy(
        self,
        token_id: str,
        max_price: float,
        usd_amount: float,
        observed_ask: float,
    ) -> "tuple[Optional[str], Optional[float], Optional[float]]":
        """
        Post a GTC limit buy at max_price (immediately marketable since max_price >> ask).
        Returns (order_id, fill_price, fill_shares).  fill_price/fill_shares are set if the
        order was immediately matched in the post_order response; otherwise poll get_order_status.
        """
        self._hot_until = _time.time() + self._HOT_DURATION_S
        try:
            return await asyncio.to_thread(
                self._do_post_limit_buy, token_id, max_price, usd_amount, observed_ask
            )
        except Exception as exc:
            print(f"[live] post_limit_buy error: {exc}", flush=True)
            return None, None, None

    def _do_post_limit_buy(
        self,
        token_id: str,
        max_price: float,
        usd_amount: float,
        observed_ask: float,
    ) -> "tuple[Optional[str], Optional[float], Optional[float]]":
        from py_clob_client.clob_types import MarketOrderArgs, OrderType

        args = MarketOrderArgs(
            token_id=token_id,
            amount=usd_amount,
            price=max_price,
            side="BUY",
        )
        signed = self._client.create_market_order(args)
        resp = self._client.post_order(signed, OrderType.GTC)
        print(f"[live] GTC buy resp: {resp}", flush=True)

        order_id = resp.get("orderID") or resp.get("order_id")
        making = float(resp.get("makingAmount") or 0)
        taking = float(resp.get("takingAmount") or 0)
        if making > 0 and taking > 0:
            fill_price = making / taking
            return order_id, fill_price, taking
        return order_id, None, None

    async def get_order_status(self, order_id: str) -> dict:
        """Fetch current order state from CLOB."""
        try:
            result = await asyncio.to_thread(self._client.get_order, order_id)
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            print(f"[live] get_order_status error: {exc}", flush=True)
            return {}

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        try:
            await asyncio.to_thread(self._client.cancel_order, order_id)
            print(f"[live] cancelled order {order_id}", flush=True)
            return True
        except Exception as exc:
            print(f"[live] cancel_order error: {exc}", flush=True)
            return False

    async def post_limit_sell(
        self,
        token_id: str,
        sell_price: float,
        shares: float,
    ) -> "Optional[str]":
        """Post a GTC limit sell at sell_price. Returns order_id or None on error."""
        try:
            return await asyncio.to_thread(
                self._do_post_limit_sell, token_id, sell_price, shares
            )
        except Exception as exc:
            print(f"[live] post_limit_sell error: {exc}", flush=True)
            return None

    def _do_post_limit_sell(
        self,
        token_id: str,
        sell_price: float,
        shares: float,
    ) -> "Optional[str]":
        from py_clob_client.clob_types import OrderArgs, OrderType

        args = OrderArgs(
            token_id=token_id,
            price=sell_price,
            size=shares,
            side="SELL",
        )
        signed = self._client.create_order(args)
        resp = self._client.post_order(signed, OrderType.GTC)
        print(f"[live] GTC sell (SL) resp: {resp}", flush=True)
        return resp.get("orderID") or resp.get("order_id")

    def _post_fok(
        self,
        token_id: str,
        order_price: float,
        usd_amount: float,
        observed_ask: float,
    ) -> "PaperFillResult":
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from bot.m5_session import PaperFillResult

        # create_market_order handles maker/taker amount precision
        # (maker: 2 decimals, taker: 4 decimals) which create_order does not.
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=usd_amount,
            price=order_price,
            side="BUY",
        )
        signed = self._client.create_market_order(order_args)
        resp = self._client.post_order(signed, OrderType.FOK)

        success = bool(resp.get("success", False))
        making = float(resp.get("makingAmount", 0)) if success else 0.0
        taking = float(resp.get("takingAmount", 0)) if success else 0.0
        fill_price = (making / taking) if (success and taking > 0) else None
        fill_shares = taking if success else None
        return PaperFillResult(
            fill_price=fill_price,
            shares=fill_shares,
            observed_best_ask=observed_ask,
            attempted_price=order_price,
            slippage=round((fill_price or order_price) - observed_ask, 6),
            retries=0,
            reject_reason=None if success else (resp.get("errorMsg") or "fok_rejected"),
        )
