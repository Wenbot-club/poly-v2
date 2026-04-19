from __future__ import annotations

from dataclasses import dataclass

from .state import sigma_60_from_binance
from .domain import FairValueSnapshot, LocalState


# ---------------------------------------------------------------------------
# Signal normalization constants
# ---------------------------------------------------------------------------

SIGMA_FLOOR_USD: float = 10.0  # minimum effective sigma — prevents /0 and division by noise
K_SIGNAL: float        = 0.02  # p_up adjustment per unit of gap_z (2 prob-points per sigma)
SIGNAL_CAP: float      = 0.04  # hard cap on |signal_adj| in probability space


@dataclass
class FairValueEngine:
    config: object

    def compute(self, state: LocalState, now_ms: int) -> FairValueSnapshot:
        state.set_clock(now_ms)
        if state.last_chainlink is None:
            raise RuntimeError("chainlink tick required before fair value computation")
        if state.last_binance is None:
            raise RuntimeError("binance tick required before fair value computation")

        # Book mid in probability space (0..1) — no unit conversion needed.
        # Raises RuntimeError when book is incomplete so the caller can count the skip.
        top_bid = state.yes_book.top_bid()
        top_ask = state.yes_book.top_ask()
        if top_bid is None or top_ask is None:
            raise RuntimeError("book incomplete — cannot compute fair value")
        mid = (top_bid.price + top_ask.price) / 2.0

        chainlink_last = state.last_chainlink.value
        binance_last   = state.last_binance.value
        ptb = (
            state.ptb.ptb_value
            if state.ptb and state.ptb.ptb_value is not None
            else chainlink_last
        )

        # sigma in USD — floored so we never divide by noise or zero.
        sigma_60  = sigma_60_from_binance(state, now_ms=now_ms)
        sigma_usd = max(sigma_60, SIGMA_FLOOR_USD)

        # Normalise the Binance–Chainlink gap into a dimensionless z-score,
        # then bound the resulting probability adjustment.
        gap_usd    = binance_last - chainlink_last
        gap_z      = gap_usd / sigma_usd
        signal_adj = max(-SIGNAL_CAP, min(SIGNAL_CAP, K_SIGNAL * gap_z))

        p_up  = max(0.0, min(1.0, mid + signal_adj))
        p_down = 1.0 - p_up
        tau_s  = max(0.0, (state.market.end_ts_ms - now_ms) / 1000.0)

        snapshot = FairValueSnapshot(
            p_up=p_up,
            p_down=p_down,
            # Legacy fields kept for JSONL serialisation compatibility.
            z_score=gap_z,       # was quasi-constant; now carries the normalised gap
            sigma_60=sigma_60,
            denom=1.0,
            lead_adj=signal_adj, # was USD-confused; now carries the bounded prob adjustment
            micro_adj=0.0,       # removed: was arbitrary +0.05 when spread existed
            imbalance=0.0,
            tape=state.tape_ewma,
            chainlink_last=chainlink_last,
            binance_last=binance_last,
            ptb=ptb,
            tau_s=tau_s,
            timestamp_ms=now_ms,
            # New semantic fields used by strategy.
            gap_z=gap_z,
            signal_adj=signal_adj,
            sigma_usd=sigma_usd,
        )
        state.fair_value = snapshot
        state.log(
            "INFO",
            "fair_value_computed",
            ts_ms=now_ms,
            p_up=round(snapshot.p_up, 6),
            gap_z=round(snapshot.gap_z, 4),
            signal_adj=round(snapshot.signal_adj, 4),
            sigma_usd=round(snapshot.sigma_usd, 2),
            tau_s=round(snapshot.tau_s, 1),
        )
        return snapshot
