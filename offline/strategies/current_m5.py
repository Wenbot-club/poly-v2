"""
Offline replica of the live BTC M5 strategy.

Replicates the live decision logic (EARLY scan + BASELINE + HEDGE) synchronously,
using the same bot.strategy.btc_m5 functions and BtcHistory rolling buffer.
"""
from __future__ import annotations

from bot.m5_session import BtcHistory
from bot.settings import DEFAULT_M5_CONFIG, M5Config
from bot.strategy.btc_m5 import (
    baseline_direction,
    compute_entry_signal,
    should_hedge,
)
from ..strategy_interface import Action, M5Strategy, TickState


class CurrentM5Strategy(M5Strategy):

    def __init__(self, cfg: M5Config = DEFAULT_M5_CONFIG) -> None:
        self._cfg = cfg
        self._history = BtcHistory()
        self._baseline_fired = False

    def reset(self) -> None:
        self._history = BtcHistory()
        self._baseline_fired = False

    def on_tick(self, state: TickState) -> Action:
        # Feed Binance price into rolling history
        if state.binance is not None:
            ts_ms = int((state.window_ts + state.sec) * 1000)
            self._history.record(state.binance, ts_ms)

        # EARLY scan [140s, 170s)
        if (not state.entry_taken
                and self._cfg.entry_scan_start_s <= state.sec < self._cfg.entry_scan_end_s
                and state.binance is not None
                and state.price_up_ask is not None
                and state.price_down_ask is not None):
            now_ms = int((state.window_ts + state.sec) * 1000)
            since_ms = now_ms - int(self._cfg.sigma_lookback_s * 1000)
            sig = compute_entry_signal(
                btc=state.binance,
                ptb=state.ptb_api,
                tau_s=state.tau_s,
                btc_samples=self._history.recent_samples(since_ms),
                btc_10s=self._history.price_n_secs_ago(10, now_ms),
                btc_30s=self._history.price_n_secs_ago(30, now_ms),
                price_up=state.price_up_ask,
                price_down=state.price_down_ask,
                sigma_floor_usd=self._cfg.sigma_floor_usd,
                z_gap_min=self._cfg.z_gap_min,
                p_enter_up_min=self._cfg.p_enter_up_min,
                p_enter_down_max=self._cfg.p_enter_down_max,
                min_entry_edge=self._cfg.min_entry_edge,
            )
            if sig.direction == "up":
                return Action.BUY_UP_LEG1
            if sig.direction == "down":
                return Action.BUY_DOWN_LEG1

        # BASELINE (one-shot at first tick >= 170s)
        if (not state.entry_taken
                and not self._baseline_fired
                and state.sec >= int(self._cfg.baseline_elapsed_s)
                and state.binance is not None):
            self._baseline_fired = True
            direction = baseline_direction(state.binance, state.ptb_api)
            if direction == "up":
                return Action.BUY_UP_LEG1
            if direction == "down":
                return Action.BUY_DOWN_LEG1

        # HEDGE watch
        if (state.entry_taken and not state.hedge_taken
                and state.sec < int(self._cfg.hedge_cutoff_s)
                and state.binance is not None):
            if should_hedge(state.entry_side, state.binance, state.ptb_api, self._cfg.hedge_threshold):
                if state.entry_side == "up":
                    return Action.BUY_DOWN_HEDGE
                return Action.BUY_UP_HEDGE

        return Action.NOOP
