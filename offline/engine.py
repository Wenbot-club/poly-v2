"""
BTC M5 offline backtest engine.

Replays WindowData tick-by-tick, calling strategy.on_tick() each second.

Execution model (V1 — compare ideas, not simulate CLOB):
  - Fill at the ask price visible at the tick where the action is taken.
  - LEG1 stake  = leg1_usd_stake  (default 1.0 USD)
  - HEDGE stake = hedge_usd_stake (default 2.0 USD)
  - At most one LEG1 and one HEDGE per window (engine silently ignores extras).
  - No partial fills, no depth modelling, no retry logic.
  - Settlement uses compute_settlement() — same formula as the live bot.
"""
from __future__ import annotations

from .data_types import TickData, TradeResult, WindowData
from .strategy_interface import Action, M5Strategy, TickState
from bot.strategy.btc_m5 import compute_settlement


class BacktestEngine:

    def __init__(
        self,
        leg1_usd_stake: float = 1.0,
        hedge_usd_stake: float = 2.0,
        baseline_sec: int = 170,
    ) -> None:
        self._leg1_stake = leg1_usd_stake
        self._hedge_stake = hedge_usd_stake
        self._baseline_sec = baseline_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_window(self, window: WindowData, strategy: M5Strategy) -> TradeResult:
        """Replay one window. Returns a TradeResult with full PnL attribution."""
        strategy.reset()
        strategy.on_window_start(window.window_ts, window.ptb_api)

        trade = TradeResult(window_ts=window.window_ts)

        for tick in window.ticks:
            state = TickState(
                sec=tick.sec,
                tau_s=float(max(0, 300 - tick.sec)),
                window_ts=window.window_ts,
                ptb_api=window.ptb_api,
                binance=tick.binance,
                chainlink=tick.chainlink,
                price_up_ask=tick.price_up_ask,
                price_down_ask=tick.price_down_ask,
                entry_taken=trade.entry_taken,
                entry_side=trade.entry_side,
                hedge_taken=trade.hedged,
            )
            action = strategy.on_tick(state)
            self._apply_action(trade, action, tick)

        self._settle(trade, window)
        strategy.on_resolution(window.result, window.close_price)
        return trade

    def run_campaign(
        self, windows: list[WindowData], strategy: M5Strategy
    ) -> list[TradeResult]:
        """Replay all windows sequentially. Returns one TradeResult per window."""
        return [self.run_window(w, strategy) for w in windows]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, trade: TradeResult, action: Action, tick: TickData) -> None:
        if action == Action.BUY_UP_LEG1 and not trade.entry_taken:
            if tick.price_up_ask is None:
                return
            trade.entry_taken = True
            trade.entry_side = "up"
            trade.entry_sec = tick.sec
            trade.entry_price = tick.price_up_ask
            trade.entry_mode = "early" if tick.sec < self._baseline_sec else "baseline"

        elif action == Action.BUY_DOWN_LEG1 and not trade.entry_taken:
            if tick.price_down_ask is None:
                return
            trade.entry_taken = True
            trade.entry_side = "down"
            trade.entry_sec = tick.sec
            trade.entry_price = tick.price_down_ask
            trade.entry_mode = "early" if tick.sec < self._baseline_sec else "baseline"

        elif action == Action.BUY_UP_HEDGE and not trade.hedged:
            if tick.price_up_ask is None:
                return
            trade.hedged = True
            trade.hedge_side = "up"
            trade.hedge_sec = tick.sec
            trade.hedge_price = tick.price_up_ask

        elif action == Action.BUY_DOWN_HEDGE and not trade.hedged:
            if tick.price_down_ask is None:
                return
            trade.hedged = True
            trade.hedge_side = "down"
            trade.hedge_sec = tick.sec
            trade.hedge_price = tick.price_down_ask

    def _settle(self, trade: TradeResult, window: WindowData) -> None:
        if not trade.entry_taken or trade.entry_price is None:
            trade.result = window.result
            trade.pnl_leg1 = None
            trade.pnl_hedge = None
            trade.net_pnl = None
            return

        s = compute_settlement(
            close_price=window.close_price,
            open_price=window.ptb_api,
            leg1_side=trade.entry_side,
            leg1_entry_price=trade.entry_price,
            leg1_shares=self._leg1_stake / trade.entry_price,
            leg1_usd_staked=self._leg1_stake,
            hedge_side=trade.hedge_side,
            hedge_entry_price=trade.hedge_price,
            hedge_shares=(self._hedge_stake / trade.hedge_price if trade.hedge_price else None),
            hedge_usd_staked=(self._hedge_stake if trade.hedged else None),
        )
        trade.result = s.result
        trade.pnl_leg1 = s.pnl_leg1
        trade.pnl_hedge = s.pnl_hedge
        trade.net_pnl = s.net_pnl
