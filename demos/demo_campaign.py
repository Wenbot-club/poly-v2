"""
Paper trading campaign demo.

Runs N sequential LivePaperSession rounds and writes a structured report to disk:
  <output_dir>/session_000.jsonl
  <output_dir>/session_000_summary.json
  ...
  <output_dir>/campaign_summary.json
  <output_dir>/campaign_manifest.json

Signal feed : Binance aggTrade + Polymarket RTDS Chainlink (no auth)
Execution   : MockExecutionEngine (no real orders)
Capital     : DEFAULT_CONFIG.default_working_capital_usd per session

For single-session runs use demos/demo_live_paper.py instead.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import aiohttp

from bot.campaign_report import CampaignSummary
from bot.live_paper import LivePaperSession
from bot.paper_campaign import CampaignConfig, CampaignRunner
from bot.providers.binance_signal import BinanceSignalProvider
from bot.providers.composite_signal import CompositeSignalProvider
from bot.providers.polymarket_chainlink_signal import PolymarketChainlinkSignalProvider
from bot.providers.polymarket_discovery import PolymarketDiscoveryProvider
from bot.providers.polymarket_market_data import PolymarketMarketDataProvider
from bot.settings import DEFAULT_CONFIG
from bot.strategy.directional_v2 import DirectionalPolicyV2


def _print_campaign_summary(cs: CampaignSummary) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Campaign summary")
    print(sep)

    print("\n  [campaign metadata]")
    print(f"  sessions_requested  : {cs.session_count_requested}")
    print(f"  sessions_completed  : {cs.session_count_completed}")
    print(f"  session_duration_s  : {cs.session_duration_s}")
    print(f"  started_at_ms       : {cs.campaign_started_at_ms}")
    print(f"  ended_at_ms         : {cs.campaign_ended_at_ms}")

    print("\n  [totals]")
    print(f"  total_realized_pnl      : {round(cs.total_realized_pnl, 4)}")
    tpm = cs.total_pnl_mark
    print(f"  total_pnl_mark          : {round(tpm, 4) if tpm is not None else None}")
    print(f"  total_orders_posted     : {cs.total_orders_posted}")
    print(f"  total_orders_rejected   : {cs.total_orders_rejected}")
    print(f"  total_orders_cancelled  : {cs.total_orders_cancelled}")
    print(f"  total_bid_fills         : {cs.total_bid_fills}")
    print(f"  total_ask_fills         : {cs.total_ask_fills}")
    print(f"  fill_rate               : {cs.fill_rate}")
    print(f"  rejection_rate          : {cs.rejection_rate}")
    print(f"  cancel_to_post_ratio    : {cs.cancel_to_post_ratio}")
    print(f"  max_up_inventory_peak   : {round(cs.max_up_inventory_peak, 6)}")

    print("\n  [session profitability]")
    print(f"  sessions_with_open_position   : {cs.sessions_with_open_position}")
    print(f"  sessions_profitable_realized  : {cs.sessions_profitable_realized}")
    print(f"  sessions_profitable_mark      : {cs.sessions_profitable_mark}")

    print("\n  [per-session PnL]")
    for row in cs.pnl_per_session:
        pnl_m = row["pnl_total_mark"]
        print(
            f"  session {row['session_index']:03d} "
            f"realized={round(row['realized_pnl'], 4):>8}  "
            f"mark={round(pnl_m, 4) if pnl_m is not None else 'N/A':>8}  "
            f"posted={row['orders_posted']}  fills={row['fills_simulated']}  "
            f"up_free={round(row['final_up_free'], 4)}"
        )

    print("\n  [gap bucket breakdown  abs(binance−chainlink) USD]")
    print(f"  excluded (gap=None): {cs.gap_bucket_excluded_count}")
    for label, stats in cs.by_gap_bucket.items():
        fmb = f"{round(stats.avg_fair_minus_best_bid, 5)}" if stats.avg_fair_minus_best_bid is not None else "N/A"
        bamf = f"{round(stats.avg_best_ask_minus_fair, 5)}" if stats.avg_best_ask_minus_fair is not None else "N/A"
        print(f"  {label:>10}: decisions={stats.decision_count}  avg_fair−bid={fmb}  avg_ask−fair={bamf}")

    print("\n  [chainlink age bucket breakdown  ms]")
    print(f"  excluded (age=None): {cs.chainlink_age_bucket_excluded_count}")
    for label, stats in cs.by_chainlink_age_bucket.items():
        fmb = f"{round(stats.avg_fair_minus_best_bid, 5)}" if stats.avg_fair_minus_best_bid is not None else "N/A"
        bamf = f"{round(stats.avg_best_ask_minus_fair, 5)}" if stats.avg_best_ask_minus_fair is not None else "N/A"
        print(f"  {label:>12}: decisions={stats.decision_count}  avg_fair−bid={fmb}  avg_ask−fair={bamf}")

    print("\n  [trigger breakdown]")
    for trigger, stats in cs.by_trigger.items():
        print(f"  {trigger:>6}: decisions={stats.decision_count}")

    print("\n  [strategy state breakdown]")
    total_d = cs.total_decisions_in_flat + cs.total_decisions_in_long
    flat_pct = 100.0 * cs.total_decisions_in_flat / total_d if total_d else 0.0
    long_pct = 100.0 * cs.total_decisions_in_long / total_d if total_d else 0.0
    print(f"  {'flat':>6}: {cs.total_decisions_in_flat:>6}  ({flat_pct:.1f}%)")
    print(f"  {'long':>6}: {cs.total_decisions_in_long:>6}  ({long_pct:.1f}%)")

    print("\n  [gate reason breakdown  bid_reason per decision]")
    total_decisions = sum(cs.by_bid_reason.values())
    for reason, count in sorted(cs.by_bid_reason.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * count / total_decisions if total_decisions else 0.0
        print(f"  {reason:>20}: {count:>5}  ({pct:.1f}%)")

    print(f"\n{sep}")


async def run_campaign(
    session_count: int,
    session_duration: int,
    output_dir: Path,
) -> None:
    cfg = CampaignConfig(
        session_count=session_count,
        session_duration_s=session_duration,
        output_dir=output_dir,
    )

    print(f"\n{'=' * 60}")
    print("  Paper campaign")
    print(f"  Sessions : {session_count} × {session_duration}s")
    print(f"  Signal   : Binance aggTrade + Polymarket RTDS Chainlink")
    print(f"  Strategy : DirectionalPolicyV2 (FLAT/LONG state machine)")
    print(f"  Engine   : MockExecutionEngine (no real orders)")
    print(f"  Capital  : {DEFAULT_CONFIG.default_working_capital_usd} PUSD / session")
    print(f"  Output   : {output_dir}")
    print(f"{'=' * 60}\n")

    async with aiohttp.ClientSession() as http_session:
        def session_factory() -> LivePaperSession:
            signal_provider = CompositeSignalProvider([
                BinanceSignalProvider(http_session),
                PolymarketChainlinkSignalProvider(http_session),
            ])
            return LivePaperSession(
                discovery=PolymarketDiscoveryProvider(
                    http_session,
                    min_remaining_s=session_duration,
                ),
                market_provider=PolymarketMarketDataProvider(http_session),
                signal_provider=signal_provider,
                strategy=DirectionalPolicyV2(config=DEFAULT_CONFIG),
                config=DEFAULT_CONFIG,
            )

        runner = CampaignRunner()
        try:
            campaign_summary = await runner.run(session_factory, cfg)
        except Exception as exc:
            print(f"[campaign] ERROR: {exc}")
            return

    _print_campaign_summary(campaign_summary)
    print(f"\n[artifacts] written to {output_dir}/")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paper trading campaign")
    parser.add_argument(
        "--session-count",
        type=int,
        default=3,
        metavar="N",
        help="Number of sessions to run (default: 3)",
    )
    parser.add_argument(
        "--session-duration",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Duration of each session in seconds (default: 30)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("campaign_out"),
        metavar="DIR",
        help="Directory for artifacts (default: campaign_out/)",
    )
    args = parser.parse_args(argv)

    try:
        asyncio.run(run_campaign(args.session_count, args.session_duration, args.output_dir))
    except KeyboardInterrupt:
        print("\n[interrupted] Ctrl+C — partial artifacts may remain on disk.")


if __name__ == "__main__":
    main()
