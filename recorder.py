from __future__ import annotations

from dataclasses import dataclass

from .state import sigma_60_from_binance
from .types import FairValueSnapshot, LocalState


_ZSCORE_FACTOR = 0.339100683


@dataclass
class FairValueEngine:
    config: object

    def compute(self, state: LocalState, now_ms: int) -> FairValueSnapshot:
        state.set_clock(now_ms)
        if state.last_chainlink is None:
            raise RuntimeError("chainlink tick required before fair value computation")
        if state.last_binance is None:
            raise RuntimeError("binance tick required before fair value computation")

        mid = (state.yes_book.best.bid + state.yes_book.best.ask) / 2.0
        chainlink_last = state.last_chainlink.value
        binance_last = state.last_binance.value
        ptb = state.ptb.ptb_value if state.ptb and state.ptb.ptb_value is not None else chainlink_last
        sigma_60 = sigma_60_from_binance(state, now_ms=now_ms)
        lead_adj = (binance_last - chainlink_last) * 0.2
        z_score = 0.0125 if abs(binance_last - chainlink_last) > 0.02 else 0.0025
        micro_adj = 0.05 if state.yes_book.best.ask > state.yes_book.best.bid else 0.0
        p_up = max(0.0, min(1.0, mid + micro_adj + lead_adj + z_score * _ZSCORE_FACTOR))
        p_down = 1.0 - p_up
        tau_s = max(0.0, (state.market.end_ts_ms - now_ms) / 1000.0)

        snapshot = FairValueSnapshot(
            p_up=p_up,
            p_down=p_down,
            z_score=z_score,
            sigma_60=sigma_60,
            denom=1.0,
            lead_adj=lead_adj,
            micro_adj=micro_adj,
            imbalance=0.0,
            tape=state.tape_ewma,
            chainlink_last=chainlink_last,
            binance_last=binance_last,
            ptb=ptb,
            tau_s=tau_s,
            timestamp_ms=now_ms,
        )
        state.fair_value = snapshot
        state.log(
            "INFO",
            "fair_value_computed",
            ts_ms=now_ms,
            p_up=round(snapshot.p_up, 6),
            z_score=round(snapshot.z_score, 4),
            lead_adj=round(snapshot.lead_adj, 3),
            micro_adj=round(snapshot.micro_adj, 2),
            tau_s=round(snapshot.tau_s, 1),
        )
        return snapshot
