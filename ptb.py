from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .types import ClobMarketInfo, ClobToken, DiscoveryCandidate, MarketContext


@dataclass(slots=True)
class MockGammaClient:
    now_ts_ms: int

    def list_candidates(self) -> List[DiscoveryCandidate]:
        return [
            DiscoveryCandidate(
                market_id="mkt-btc-15m-demo",
                condition_id="0xbtc15mdemo",
                title="Bitcoin Up or Down - Demo 15m",
                slug="bitcoin-up-or-down-demo-15m",
                start_ts_ms=1_765_000_800_000,
                end_ts_ms=1_765_001_700_000,
                active=True,
                closed=False,
                clob_token_ids=["YES_TOKEN_BTC_15M_DEMO", "NO_TOKEN_BTC_15M_DEMO"],
            )
        ]


@dataclass(slots=True)
class DiscoveryService:
    gamma_client: MockGammaClient

    def find_active_btc_15m_market(self) -> MarketContext:
        candidates = self.gamma_client.list_candidates()
        if not candidates:
            raise RuntimeError("No mock BTC 15m market available")
        candidate = candidates[0]
        return MarketContext(
            market_id=candidate.market_id,
            condition_id=candidate.condition_id,
            title=candidate.title,
            slug=candidate.slug,
            start_ts_ms=candidate.start_ts_ms,
            end_ts_ms=candidate.end_ts_ms,
            yes_token_id="YES_TOKEN_BTC_15M_DEMO",
            no_token_id="NO_TOKEN_BTC_15M_DEMO",
            clob=ClobMarketInfo(
                tokens=[
                    ClobToken(token_id="YES_TOKEN_BTC_15M_DEMO", outcome="Yes"),
                    ClobToken(token_id="NO_TOKEN_BTC_15M_DEMO", outcome="No"),
                ],
                min_order_size=5.0,
                min_tick_size=0.01,
                maker_base_fee_bps=0,
                taker_base_fee_bps=0,
                taker_delay_enabled=False,
                min_order_age_s=0.0,
                fee_rate=0.0,
                fee_exponent=1.0,
            ),
        )
