"""
Strategy interface for the BTC M5 offline backtest engine.

Implement M5Strategy to create a new trading strategy.
The engine calls: reset() → on_window_start() → on_tick() × N → on_resolution()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Action(Enum):
    NOOP = "noop"
    BUY_UP_LEG1 = "buy_up_leg1"
    BUY_DOWN_LEG1 = "buy_down_leg1"
    BUY_UP_HEDGE = "buy_up_hedge"
    BUY_DOWN_HEDGE = "buy_down_hedge"


@dataclass
class TickState:
    """State exposed to the strategy at each tick."""
    sec: int                            # seconds elapsed since window_ts
    tau_s: float                        # seconds remaining in window
    window_ts: int
    ptb_api: float                      # window open price (strike price)
    binance: Optional[float]            # current Binance BTC/USDT price
    chainlink: Optional[float]          # current Chainlink BTC/USD feed
    price_up_ask: Optional[float]       # UP token best ask
    price_down_ask: Optional[float]     # DOWN token best ask
    # Current position state (read-only — engine manages fills)
    entry_taken: bool
    entry_side: Optional[str]           # "up" | "down" | None
    hedge_taken: bool


class M5Strategy(ABC):
    """
    Abstract base for all BTC M5 backtest strategies.

    Lifecycle per window:
      reset() → on_window_start() → on_tick() × N → on_resolution()
    """

    def reset(self) -> None:
        """Called before each window. Override to reset per-window state."""

    def on_window_start(self, window_ts: int, ptb_api: float) -> None:
        """Called once at the start of each window."""

    @abstractmethod
    def on_tick(self, state: TickState) -> Action:
        """Called once per tick (second). Return an action or NOOP."""

    def on_resolution(self, result: str, close_price: float) -> None:
        """Called after settlement with the final result."""
