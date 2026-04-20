#!/usr/bin/env python3
"""
CLI: replay archived M5 windows through one or more strategies.

Usage:
  python offline/run_backtest.py --data-dir path/to/windows/ [--strategy baseline_ptb current_m5]

Expected window JSON format (one file per window):
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

import argparse
import json
from pathlib import Path

from offline.data_types import WindowData
from offline.engine import BacktestEngine
from offline.reporting import aggregate_results, compare_strategies, print_report
from offline.strategies.baseline_ptb import BaselinePtbStrategy
from offline.strategies.current_m5 import CurrentM5Strategy

_STRATEGIES: dict = {
    "baseline_ptb": BaselinePtbStrategy,
    "current_m5": CurrentM5Strategy,
}


def load_windows(data_dir: Path) -> list[WindowData]:
    windows = []
    for path in sorted(data_dir.glob("*.json")):
        try:
            windows.append(WindowData.from_dict(json.loads(path.read_text())))
        except Exception as e:
            print(f"[warn] skipping {path.name}: {e}")
    return windows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="BTC M5 offline backtest")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory of window JSON files")
    parser.add_argument("--strategy", nargs="+",
                        choices=list(_STRATEGIES), default=list(_STRATEGIES),
                        help="Strategies to run (default: all)")
    args = parser.parse_args(argv)

    windows = load_windows(args.data_dir)
    if not windows:
        print(f"[error] No windows found in {args.data_dir}")
        return
    print(f"Loaded {len(windows)} windows from {args.data_dir}")

    engine = BacktestEngine()
    reports = []
    for name in args.strategy:
        strategy = _STRATEGIES[name]()
        results = engine.run_campaign(windows, strategy)
        report = aggregate_results(results, strategy_name=name)
        print_report(report)
        reports.append(report)

    if len(reports) > 1:
        compare_strategies(reports)


if __name__ == "__main__":
    main()
