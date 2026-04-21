"""
Live order executor — posts real FOK BUY orders to the Polymarket CLOB.

Uses py-clob-client for EIP-712 order signing.  All synchronous
py-clob-client calls are wrapped in asyncio.to_thread so the main
event loop stays non-blocking.

Returns PaperFillResult for drop-in compatibility with the paper path.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.m5_session import PaperFillResult
    from bot.trading.credentials import Credentials


class LiveOrderExecutor:
    """
    One instance per trading session.

    A fresh ClobClient is created for every order because reusing a
    long-lived client produced 'invalid signature' errors in the service
    while identical one-shot scripts succeeded.
    """

    def __init__(self, creds: "Credentials") -> None:
        self._creds = creds

    _MARKET_PRICE_CAP = 0.99  # FOK limit — fills at best ask, up to this cap

    def _make_client(self):
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        from py_clob_client.constants import POLYGON

        creds = self._creds
        same = creds.signer_address.lower() == creds.funder_address.lower()
        return ClobClient(
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

    async def __call__(
        self,
        token_id: str,
        price: float,
        usd_bet: float,
    ) -> "PaperFillResult":
        from bot.m5_session import PaperFillResult

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

    def _post_fok(
        self,
        token_id: str,
        order_price: float,
        usd_amount: float,
        observed_ask: float,
    ) -> "PaperFillResult":
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from bot.m5_session import PaperFillResult

        # Fresh client per order — the shared long-lived client in the service
        # produced invalid_signature errors while one-shot scripts succeeded.
        client = self._make_client()
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=usd_amount,
            price=order_price,
            side="BUY",
        )
        signed = client.create_market_order(order_args)
        resp = client.post_order(signed, OrderType.FOK)

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
