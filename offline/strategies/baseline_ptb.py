"""
Baseline PTB strategy: enter at 170s based on BTC vs PTB, hedge same as live.

No EARLY scan. One-shot entry at baseline_sec.
"""
from __future__ import annotations

from ..strategy_interface import Action, M5Strategy, TickState


class BaselinePtbStrategy(M5Strategy):

    def __init__(
        self,
        entry_sec: int = 170,
        hedge_threshold: float = 1.0,
        hedge_cutoff_sec: int = 250,
    ) -> None:
        self._entry_sec = entry_sec
        self._hedge_threshold = hedge_threshold
        self._hedge_cutoff_sec = hedge_cutoff_sec
        self._entry_fired = False

    def reset(self) -> None:
        self._entry_fired = False

    def on_tick(self, state: TickState) -> Action:
        # One-shot baseline entry
        if not state.entry_taken and not self._entry_fired and state.sec >= self._entry_sec:
            self._entry_fired = True
            if state.binance is None:
                return Action.NOOP
            if state.binance > state.ptb_api:
                return Action.BUY_UP_LEG1
            if state.binance < state.ptb_api:
                return Action.BUY_DOWN_LEG1

        # Hedge watch: fire once if BTC crosses PTB against leg1 direction
        if (state.entry_taken and not state.hedge_taken
                and state.sec < self._hedge_cutoff_sec
                and state.binance is not None):
            if state.entry_side == "up" and state.binance <= state.ptb_api - self._hedge_threshold:
                return Action.BUY_DOWN_HEDGE
            if state.entry_side == "down" and state.binance >= state.ptb_api + self._hedge_threshold:
                return Action.BUY_UP_HEDGE

        return Action.NOOP
