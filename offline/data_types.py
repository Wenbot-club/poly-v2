"""
Core data types for the BTC M5 offline backtest engine.

WindowData  — one replay-able M5 window (ticks + settlement truth)
TickData    — one-second price snapshot inside a window
TradeResult — per-window outcome produced by the engine
BacktestReport — aggregated results for one strategy across all windows

Expected window JSON format:
{
  "window_ts": 1746000000,
  "ptb_api": 84000.0,
  "close_price": 84500.0,
  "result": "up",
  "ticks": [
    {"sec": 0, "binance": 84010.0, "chainlink": null,
     "price_up_ask": 0.52, "price_down_ask": 0.50},
    ...
  ]
}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TickData:
    """Price snapshot at one second inside an M5 window."""
    sec: int                            # seconds elapsed since window_ts
    binance: Optional[float] = None
    chainlink: Optional[float] = None
    price_up_ask: Optional[float] = None
    price_down_ask: Optional[float] = None


@dataclass
class WindowData:
    """One complete M5 window ready for replay."""
    window_ts: int
    ptb_api: float                      # open price from Polymarket API
    close_price: float                  # settlement price from Polymarket API
    result: str                         # "up" | "down"
    ticks: list[TickData] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> WindowData:
        ticks = [TickData(**t) for t in d.get("ticks", [])]
        return cls(
            window_ts=int(d["window_ts"]),
            ptb_api=float(d["ptb_api"]),
            close_price=float(d["close_price"]),
            result=str(d["result"]),
            ticks=ticks,
        )


@dataclass
class TradeResult:
    """Per-window outcome produced by BacktestEngine.run_window()."""
    window_ts: int
    entry_taken: bool = False
    entry_mode: Optional[str] = None    # "early" | "baseline"
    entry_side: Optional[str] = None    # "up" | "down"
    entry_sec: Optional[int] = None
    entry_price: Optional[float] = None
    hedged: bool = False
    hedge_sec: Optional[int] = None
    hedge_side: Optional[str] = None    # "up" | "down"
    hedge_price: Optional[float] = None
    result: Optional[str] = None        # "up" | "down"
    pnl_leg1: Optional[float] = None
    pnl_hedge: Optional[float] = None
    net_pnl: Optional[float] = None


@dataclass
class BacktestReport:
    """Aggregated results across all windows for one strategy."""
    strategy_name: str
    windows_total: int
    trades_taken: int
    early_count: int
    baseline_count: int
    hedge_triggered_count: int
    pnl_leg1_total: float
    pnl_hedge_total: float
    net_pnl_total: float
    avg_entry_price: Optional[float]
    win_rate: Optional[float]           # wins / trades_taken
    avg_net_per_trade: Optional[float]
    # Splits: hedge attribution
    hedged_pnl_leg1: float
    hedged_pnl_hedge: float
    hedged_net_pnl: float
    non_hedged_net_pnl: float
    # Splits: entry mode
    early_net_pnl: float
    baseline_net_pnl: float
    # Splits: direction
    up_entry_net_pnl: float
    down_entry_net_pnl: float
    trades: list = field(default_factory=list)  # list[TradeResult]
