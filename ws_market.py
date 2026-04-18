from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .settings import RuntimeConfig
from .state import chainlink_age_ms, market_book_age_ms
from .types import LocalState


@dataclass(slots=True)
class RiskDecision:
    allow_quotes: bool
    reason: str
    chainlink_age_ms: int | None
    market_book_age_ms: int
    actions: List[str]
    decision_ts_ms: int


@dataclass
class RiskManager:
    config: RuntimeConfig

    def evaluate(self, state: LocalState, now_ms: int) -> RiskDecision:
        state.set_clock(now_ms)
        ch_age = chainlink_age_ms(state, now_ms)
        book_age = market_book_age_ms(state, now_ms)

        if ch_age is None:
            decision = RiskDecision(
                allow_quotes=False,
                reason="chainlink_missing",
                chainlink_age_ms=None,
                market_book_age_ms=book_age,
                actions=[],
                decision_ts_ms=now_ms,
            )
            state.log(
                "WARN",
                "risk_evaluated",
                ts_ms=now_ms,
                allow_quotes=decision.allow_quotes,
                reason=decision.reason,
                chainlink_age_ms=decision.chainlink_age_ms,
                market_book_age_ms=decision.market_book_age_ms,
            )
            return decision

        if ch_age > int(self.config.thresholds.freshness.chainlink_max_age_ms.value):
            decision = RiskDecision(
                allow_quotes=False,
                reason="stale_chainlink",
                chainlink_age_ms=ch_age,
                market_book_age_ms=book_age,
                actions=[],
                decision_ts_ms=now_ms,
            )
            state.log(
                "WARN",
                "risk_evaluated",
                ts_ms=now_ms,
                allow_quotes=decision.allow_quotes,
                reason=decision.reason,
                chainlink_age_ms=decision.chainlink_age_ms,
                market_book_age_ms=decision.market_book_age_ms,
            )
            return decision

        if book_age > int(self.config.thresholds.freshness.market_book_max_age_ms.value):
            decision = RiskDecision(
                allow_quotes=False,
                reason="stale_market_book",
                chainlink_age_ms=ch_age,
                market_book_age_ms=book_age,
                actions=[],
                decision_ts_ms=now_ms,
            )
            state.log(
                "WARN",
                "risk_evaluated",
                ts_ms=now_ms,
                allow_quotes=decision.allow_quotes,
                reason=decision.reason,
                chainlink_age_ms=decision.chainlink_age_ms,
                market_book_age_ms=decision.market_book_age_ms,
            )
            return decision

        decision = RiskDecision(
            allow_quotes=True,
            reason="ok",
            chainlink_age_ms=ch_age,
            market_book_age_ms=book_age,
            actions=[],
            decision_ts_ms=now_ms,
        )
        state.log(
            "INFO",
            "risk_evaluated",
            ts_ms=now_ms,
            allow_quotes=decision.allow_quotes,
            reason=decision.reason,
            chainlink_age_ms=decision.chainlink_age_ms,
            market_book_age_ms=decision.market_book_age_ms,
        )
        return decision
