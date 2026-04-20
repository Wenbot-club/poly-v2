"""
Aggregation and reporting for backtest results.

aggregate_results()    — build BacktestReport from list[TradeResult]
print_report()         — human-readable single-strategy report
compare_strategies()   — side-by-side table across multiple strategies
"""
from __future__ import annotations

from .data_types import BacktestReport, TradeResult


def aggregate_results(
    results: list[TradeResult], *, strategy_name: str = "unnamed"
) -> BacktestReport:
    entered = [r for r in results if r.entry_taken]
    hedged = [r for r in entered if r.hedged]
    non_hedged = [r for r in entered if not r.hedged]
    early = [r for r in entered if r.entry_mode == "early"]
    baseline = [r for r in entered if r.entry_mode == "baseline"]
    up_entries = [r for r in entered if r.entry_side == "up"]
    down_entries = [r for r in entered if r.entry_side == "down"]
    wins = [r for r in entered if r.result is not None and r.result == r.entry_side]

    pnl_leg1 = sum(r.pnl_leg1 for r in entered if r.pnl_leg1 is not None)
    pnl_hedge = sum(r.pnl_hedge for r in entered if r.pnl_hedge is not None)

    return BacktestReport(
        strategy_name=strategy_name,
        windows_total=len(results),
        trades_taken=len(entered),
        early_count=len(early),
        baseline_count=len(baseline),
        hedge_triggered_count=len(hedged),
        pnl_leg1_total=round(pnl_leg1, 6),
        pnl_hedge_total=round(pnl_hedge, 6),
        net_pnl_total=round(pnl_leg1 + pnl_hedge, 6),
        avg_entry_price=(
            sum(r.entry_price for r in entered) / len(entered) if entered else None
        ),
        win_rate=(len(wins) / len(entered) if entered else None),
        avg_net_per_trade=(
            round((pnl_leg1 + pnl_hedge) / len(entered), 6) if entered else None
        ),
        hedged_pnl_leg1=round(
            sum(r.pnl_leg1 for r in hedged if r.pnl_leg1 is not None), 6
        ),
        hedged_pnl_hedge=round(
            sum(r.pnl_hedge for r in hedged if r.pnl_hedge is not None), 6
        ),
        hedged_net_pnl=round(
            sum(r.net_pnl for r in hedged if r.net_pnl is not None), 6
        ),
        non_hedged_net_pnl=round(
            sum(r.net_pnl for r in non_hedged if r.net_pnl is not None), 6
        ),
        early_net_pnl=round(
            sum(r.net_pnl for r in early if r.net_pnl is not None), 6
        ),
        baseline_net_pnl=round(
            sum(r.net_pnl for r in baseline if r.net_pnl is not None), 6
        ),
        up_entry_net_pnl=round(
            sum(r.net_pnl for r in up_entries if r.net_pnl is not None), 6
        ),
        down_entry_net_pnl=round(
            sum(r.net_pnl for r in down_entries if r.net_pnl is not None), 6
        ),
        trades=results,
    )


def print_report(report: BacktestReport) -> None:
    def _f(v, fmt=".4f") -> str:
        return f"{v:{fmt}}" if v is not None else "n/a"

    def _pnl(v) -> str:
        return f"{v:+.4f}" if v is not None else "n/a"

    wr = f"{report.win_rate:.1%}" if report.win_rate is not None else "n/a"

    print(f"\n{'═' * 60}")
    print(f"  Backtest: {report.strategy_name}")
    print(f"{'─' * 60}")
    print(f"  windows_total        : {report.windows_total}")
    print(f"  trades_taken         : {report.trades_taken}")
    print(f"    early              : {report.early_count}")
    print(f"    baseline           : {report.baseline_count}")
    print(f"  hedge_triggered      : {report.hedge_triggered_count}")
    print()
    print(f"  pnl_leg1_total       : {_pnl(report.pnl_leg1_total)}")
    print(f"  pnl_hedge_total      : {_pnl(report.pnl_hedge_total)}")
    print(f"  net_pnl_total        : {_pnl(report.net_pnl_total)}")
    print(f"  win_rate             : {wr}")
    print(f"  avg_net_per_trade    : {_pnl(report.avg_net_per_trade)}")
    print()
    print(f"  ── hedge attribution ──────────────────────────────────")
    print(f"  hedged trades ({report.hedge_triggered_count})")
    print(f"    leg1 pnl (standalone)  : {_pnl(report.hedged_pnl_leg1)}")
    print(f"    hedge pnl (incremental): {_pnl(report.hedged_pnl_hedge)}")
    print(f"    net                    : {_pnl(report.hedged_net_pnl)}")
    print(f"  non-hedged net pnl       : {_pnl(report.non_hedged_net_pnl)}")
    print()
    print(f"  ── entry mode ─────────────────────────────────────────")
    print(f"  early    ({report.early_count:>3}) net pnl : {_pnl(report.early_net_pnl)}")
    print(f"  baseline ({report.baseline_count:>3}) net pnl : {_pnl(report.baseline_net_pnl)}")
    print()
    print(f"  ── direction ──────────────────────────────────────────")
    print(f"  up   entries net pnl : {_pnl(report.up_entry_net_pnl)}")
    print(f"  down entries net pnl : {_pnl(report.down_entry_net_pnl)}")


def compare_strategies(reports: list[BacktestReport]) -> None:
    if not reports:
        return
    print(f"\n{'═' * 68}")
    print(f"  Strategy Comparison")
    print(f"{'─' * 68}")
    header = (
        f"  {'Strategy':<22} {'Trades':>7} {'Win%':>7} "
        f"{'Leg1':>9} {'Hedge':>9} {'Net':>9}"
    )
    print(header)
    print(f"  {'-' * 63}")
    for r in reports:
        wr = f"{r.win_rate:.1%}" if r.win_rate is not None else "  n/a"
        print(
            f"  {r.strategy_name:<22} {r.trades_taken:>7} {wr:>7} "
            f"{r.pnl_leg1_total:>+9.4f} {r.pnl_hedge_total:>+9.4f} "
            f"{r.net_pnl_total:>+9.4f}"
        )
