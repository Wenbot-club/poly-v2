"""
Attribution report for BTC M5 live campaigns.

compute_attribution(trades)       → AttributionReport
print_attribution(report)         → terminal block
export_attribution(report, path)  → per-window CSV
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BucketStats:
    count: int = 0
    hedge_triggered_count: int = 0
    pnl_leg1_total: float = 0.0
    pnl_hedge_total: float = 0.0
    net_pnl_total: float = 0.0
    avg_entry_price: Optional[float] = None
    win_rate: Optional[float] = None


@dataclass
class BankrollSimulation:
    flat_final: float = 100.0
    compound_final: float = 100.0
    flat_return_pct: float = 0.0
    compound_return_pct: float = 0.0
    max_drawdown_flat: float = 0.0
    max_drawdown_compound: float = 0.0


@dataclass
class PerWindowRow:
    window_ts: int
    entry_mode: Optional[str]
    entry_side: Optional[str]
    entry_price: Optional[float]
    hedged: bool
    hedge_price: Optional[float]
    result: Optional[str]
    pnl_leg1: Optional[float]
    pnl_hedge: Optional[float]
    net_pnl: Optional[float]
    flat_bankroll_after: float
    compound_bankroll_after: float


@dataclass
class AttributionReport:
    # Core: leg1 vs hedge split
    leg1_standalone_net_total: float
    hedge_incremental_total: float
    net_pnl_total: float

    # Splits
    early: BucketStats
    baseline: BucketStats
    hedged: BucketStats
    non_hedged: BucketStats
    up: BucketStats
    down: BucketStats

    # Bankroll simulation
    bankroll: BankrollSimulation
    initial_bankroll: float

    # Per-window export rows
    rows: list = field(default_factory=list)  # list[PerWindowRow]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bucket(trades_subset: list) -> BucketStats:
    """Aggregate a pre-filtered list of TradeRecord into a BucketStats."""
    entered = [t for t in trades_subset
               if t.entry_side is not None and t.net_pnl is not None]
    if not entered:
        return BucketStats()

    wins = [t for t in entered
            if t.result is not None and t.result == t.entry_side]
    pnl_leg1 = sum(t.pnl_leg1 for t in entered if t.pnl_leg1 is not None)
    pnl_hedge = sum(t.pnl_hedge for t in entered if t.pnl_hedge is not None)
    net_pnl = sum(t.net_pnl for t in entered)
    prices = [t.entry_price for t in entered if t.entry_price is not None]

    return BucketStats(
        count=len(entered),
        hedge_triggered_count=sum(1 for t in entered if t.hedged),
        pnl_leg1_total=round(pnl_leg1, 6),
        pnl_hedge_total=round(pnl_hedge, 6),
        net_pnl_total=round(net_pnl, 6),
        avg_entry_price=(sum(prices) / len(prices) if prices else None),
        win_rate=(len(wins) / len(entered)),
    )


def _max_drawdown(curve: list[float]) -> float:
    """Max peak-to-trough drawdown (absolute USD, always positive)."""
    if len(curve) < 2:
        return 0.0
    peak = curve[0]
    max_dd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_attribution(
    trades: list,
    initial_bankroll: float = 100.0,
) -> AttributionReport:
    """Build full AttributionReport from a list of TradeRecord."""
    entered = [t for t in trades
               if t.entry_side is not None and t.net_pnl is not None]

    leg1_standalone = sum(t.pnl_leg1 for t in entered if t.pnl_leg1 is not None)
    hedge_incremental = sum(t.pnl_hedge for t in entered if t.pnl_hedge is not None)

    # Bankroll curves (one point per window, including non-entered)
    flat_curve: list[float] = [initial_bankroll]
    compound_curve: list[float] = [initial_bankroll]
    flat_br = initial_bankroll
    compound_br = initial_bankroll

    rows: list[PerWindowRow] = []
    for t in trades:
        if t.entry_side is not None and t.net_pnl is not None:
            flat_br += t.net_pnl
            # Compound: scale pnl proportionally to current bankroll
            compound_br += t.net_pnl * (compound_br / initial_bankroll)
        flat_curve.append(flat_br)
        compound_curve.append(compound_br)
        rows.append(PerWindowRow(
            window_ts=t.window_ts,
            entry_mode=t.entry_mode,
            entry_side=t.entry_side,
            entry_price=t.entry_price,
            hedged=t.hedged,
            hedge_price=t.hedge_price,
            result=t.result,
            pnl_leg1=t.pnl_leg1,
            pnl_hedge=t.pnl_hedge,
            net_pnl=t.net_pnl,
            flat_bankroll_after=round(flat_br, 6),
            compound_bankroll_after=round(compound_br, 6),
        ))

    flat_ret = (flat_br - initial_bankroll) / initial_bankroll * 100.0
    compound_ret = (compound_br - initial_bankroll) / initial_bankroll * 100.0

    bankroll = BankrollSimulation(
        flat_final=round(flat_br, 4),
        compound_final=round(compound_br, 4),
        flat_return_pct=round(flat_ret, 4),
        compound_return_pct=round(compound_ret, 4),
        max_drawdown_flat=round(_max_drawdown(flat_curve), 4),
        max_drawdown_compound=round(_max_drawdown(compound_curve), 4),
    )

    return AttributionReport(
        leg1_standalone_net_total=round(leg1_standalone, 6),
        hedge_incremental_total=round(hedge_incremental, 6),
        net_pnl_total=round(leg1_standalone + hedge_incremental, 6),
        early=_make_bucket([t for t in trades if t.entry_mode == "early"]),
        baseline=_make_bucket([t for t in trades if t.entry_mode == "baseline"]),
        hedged=_make_bucket([t for t in entered if t.hedged]),
        non_hedged=_make_bucket([t for t in entered if not t.hedged]),
        up=_make_bucket([t for t in entered if t.entry_side == "up"]),
        down=_make_bucket([t for t in entered if t.entry_side == "down"]),
        bankroll=bankroll,
        initial_bankroll=initial_bankroll,
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_attribution(report: AttributionReport) -> None:
    def _pnl(v: float) -> str:
        return f"{v:>+9.4f}"

    def _wr(v: Optional[float]) -> str:
        return f"{v:.1%}" if v is not None else "  n/a"

    def _px(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "  n/a"

    def _bucket(name: str, b: BucketStats) -> None:
        print(
            f"  {name:<10}: count={b.count:>3}  hedged={b.hedge_triggered_count:>3}"
            f"  leg1={_pnl(b.pnl_leg1_total)}  hedge={_pnl(b.pnl_hedge_total)}"
            f"  net={_pnl(b.net_pnl_total)}  win={_wr(b.win_rate)}"
            f"  avg_px={_px(b.avg_entry_price)}"
        )

    bk = report.bankroll
    print(f"\n{'═' * 70}")
    print("  [attribution]")
    print(f"  leg1_standalone_net_total : {_pnl(report.leg1_standalone_net_total)}")
    print(f"  hedge_incremental_total   : {_pnl(report.hedge_incremental_total)}")
    print(f"  net_pnl_total             : {_pnl(report.net_pnl_total)}")
    print()
    print("  [by entry mode]")
    _bucket("early", report.early)
    _bucket("baseline", report.baseline)
    print()
    print("  [by hedge status]")
    _bucket("hedged", report.hedged)
    _bucket("non_hedged", report.non_hedged)
    print()
    print("  [by entry side]")
    _bucket("up", report.up)
    _bucket("down", report.down)
    print()
    print("  [flat vs compound]")
    print(f"  initial_bankroll         : {report.initial_bankroll:.2f} USD")
    print(f"  flat_final_bankroll      : {bk.flat_final:>+.4f}")
    print(f"  compound_final_bankroll  : {bk.compound_final:>+.4f}")
    print(f"  flat_return_pct          : {bk.flat_return_pct:>+.4f}%")
    print(f"  compound_return_pct      : {bk.compound_return_pct:>+.4f}%")
    print(f"  max_drawdown_flat        : {bk.max_drawdown_flat:.4f}")
    print(f"  max_drawdown_compound    : {bk.max_drawdown_compound:.4f}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "window_ts", "entry_mode", "entry_side", "entry_price",
    "hedged", "hedge_price", "result",
    "pnl_leg1", "pnl_hedge", "net_pnl",
    "flat_bankroll_after", "compound_bankroll_after",
]


def export_attribution(report: AttributionReport, path: Path) -> None:
    """Write per-window attribution rows to a CSV file."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in report.rows:
            writer.writerow({
                "window_ts": row.window_ts,
                "entry_mode": row.entry_mode or "",
                "entry_side": row.entry_side or "",
                "entry_price": "" if row.entry_price is None else row.entry_price,
                "hedged": row.hedged,
                "hedge_price": "" if row.hedge_price is None else row.hedge_price,
                "result": row.result or "",
                "pnl_leg1": "" if row.pnl_leg1 is None else row.pnl_leg1,
                "pnl_hedge": "" if row.pnl_hedge is None else row.pnl_hedge,
                "net_pnl": "" if row.net_pnl is None else row.net_pnl,
                "flat_bankroll_after": row.flat_bankroll_after,
                "compound_bankroll_after": row.compound_bankroll_after,
            })
