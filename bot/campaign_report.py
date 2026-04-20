"""
Aggregate performance report for paper trading campaigns.

compute_campaign_summary() is a pure function — no I/O.
All file writing lives in CampaignRunner (paper_campaign.py).

Bucket semantics (borders documented here, enforced in _bucket_label):
  Given sorted thresholds (T1, T2, ...):
    value < T1                  → "<{T1}"
    T_i <= value <= T_{i+1}     → "{T_i}-{T_{i+1}}"   (inclusive on both sides)
    value > T_last              → ">{T_last}"
  Labels use :g float format (strips trailing zeros: 50.0 → "50", 1000 → "1000").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .live_paper import LivePaperSummary


@dataclass(slots=True)
class BucketStats:
    decision_count: int
    avg_fair_minus_best_bid: Optional[float]
    avg_best_ask_minus_fair: Optional[float]


@dataclass(slots=True)
class CampaignSummary:
    # Campaign metadata
    session_count_requested: int
    session_count_completed: int
    session_duration_s: int
    campaign_started_at_ms: int
    campaign_ended_at_ms: int
    # Cross-session totals
    total_realized_pnl: float
    total_pnl_mark: Optional[float]   # None if any session had no mark (open inventory, no bid)
    total_orders_posted: int
    total_orders_rejected: int
    total_orders_cancelled: int
    total_bid_fills: int
    total_ask_fills: int
    fill_rate: Optional[float]          # total_filled_orders / total_orders_posted
    rejection_rate: Optional[float]     # total_rejected / (total_posted + total_rejected)
    cancel_to_post_ratio: Optional[float]
    max_up_inventory_peak: float
    sessions_with_open_position: int
    sessions_profitable_realized: int   # sessions where realized_pnl > 0
    sessions_profitable_mark: int       # sessions where pnl_total_mark > 0
    # Per-session detail (for histograms and dispersion analysis)
    pnl_per_session: list              # list[dict]: see compute_campaign_summary docstring
    # Decision-level breakdowns
    by_gap_bucket: dict                # str → BucketStats; abs(binance_chainlink_gap)
    by_chainlink_age_bucket: dict      # str → BucketStats; chainlink_age_ms
    by_trigger: dict                   # "market" | "rtds" → BucketStats
    # Excluded counts (decisions where the field was None, so not bucketed)
    gap_bucket_excluded_count: int
    chainlink_age_bucket_excluded_count: int
    # Gate reason breakdown: bid_reason → decision count across all sessions
    by_bid_reason: dict
    # Strategy state counters across all sessions
    total_decisions_in_flat: int
    total_decisions_in_long: int
    # Round-trip aggregates
    total_completed_round_trips: int
    total_forced_exits: int
    avg_holding_time_s: Optional[float]
    by_exit_reason: dict                   # exit strategy_reason → fill count across all sessions


def _bucket_label(value: float, thresholds: tuple) -> str:
    """
    Assign a bucket label for value given sorted thresholds.

    Bucket semantics (inclusive upper bound on intermediate intervals):
      thresholds = (T1, T2):
        value < T1          → "<{T1}"
        T1 <= value <= T2   → "{T1}-{T2}"
        value > T2          → ">{T2}"
    """
    if value < thresholds[0]:
        return f"<{thresholds[0]:g}"
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= value <= thresholds[i + 1]:
            return f"{thresholds[i]:g}-{thresholds[i + 1]:g}"
    return f">{thresholds[-1]:g}"


def _ordered_labels(thresholds: tuple) -> list:
    """Ordered list of all bucket labels for the given thresholds."""
    labels = [f"<{thresholds[0]:g}"]
    for i in range(len(thresholds) - 1):
        labels.append(f"{thresholds[i]:g}-{thresholds[i + 1]:g}")
    labels.append(f">{thresholds[-1]:g}")
    return labels


def _to_bucket_stats(samples: list) -> BucketStats:
    fmb = [s["fair_minus_best_bid"] for s in samples if s["fair_minus_best_bid"] is not None]
    bamf = [s["best_ask_minus_fair"] for s in samples if s["best_ask_minus_fair"] is not None]
    return BucketStats(
        decision_count=len(samples),
        avg_fair_minus_best_bid=sum(fmb) / len(fmb) if fmb else None,
        avg_best_ask_minus_fair=sum(bamf) / len(bamf) if bamf else None,
    )


def compute_campaign_summary(
    *,
    session_summaries: list,          # list[LivePaperSummary]
    all_events: list,                 # list[dict] — all events from all sessions
    session_count_requested: int,
    session_duration_s: int,
    gap_thresholds_usd: tuple,
    chainlink_age_thresholds_ms: tuple,
    campaign_started_at_ms: int,
    campaign_ended_at_ms: int,
) -> CampaignSummary:
    """
    Compute a CampaignSummary from completed session summaries and their events.

    pnl_per_session shape per entry:
      {
        "session_index": int,
        "realized_pnl": float,
        "pnl_total_mark": float | None,
        "orders_posted": int,
        "fills_simulated": int,
        "final_up_free": float,
      }

    total_pnl_mark: None if any session's pnl_total_mark is None (incomplete mark data).
    """
    n = len(session_summaries)

    # --- Totals ---
    total_realized_pnl = sum(s.realized_pnl for s in session_summaries)

    mark_values = [s.pnl_total_mark for s in session_summaries]
    total_pnl_mark: Optional[float] = (
        sum(mark_values)  # type: ignore[arg-type]
        if all(v is not None for v in mark_values) and n > 0
        else None
    )

    total_posted = sum(s.orders_posted for s in session_summaries)
    total_rejected = sum(s.orders_rejected for s in session_summaries)
    total_cancelled = sum(s.orders_cancelled for s in session_summaries)
    total_bid_fills = sum(s.bid_fills_simulated for s in session_summaries)
    total_ask_fills = sum(s.ask_fills_simulated for s in session_summaries)
    total_filled = sum(s.filled_orders for s in session_summaries)
    total_fills_simulated = sum(s.fills_simulated for s in session_summaries)

    fill_rate = total_filled / total_posted if total_posted > 0 else None
    rejection_rate = (
        total_rejected / (total_posted + total_rejected)
        if (total_posted + total_rejected) > 0
        else None
    )
    cancel_to_post_ratio = total_cancelled / total_posted if total_posted > 0 else None

    max_up_peak = max((s.max_up_inventory for s in session_summaries), default=0.0)
    sessions_with_open = sum(1 for s in session_summaries if s.final_up_free > 0)
    sessions_profitable_realized = sum(1 for s in session_summaries if s.realized_pnl > 0)
    sessions_profitable_mark = sum(
        1 for s in session_summaries
        if s.pnl_total_mark is not None and s.pnl_total_mark > 0
    )

    pnl_per_session = [
        {
            "session_index": i,
            "realized_pnl": s.realized_pnl,
            "pnl_total_mark": s.pnl_total_mark,
            "orders_posted": s.orders_posted,
            "fills_simulated": s.fills_simulated,
            "final_up_free": s.final_up_free,
        }
        for i, s in enumerate(session_summaries)
    ]

    # --- Bucket analysis from decision events ---
    gap_thresholds_f = tuple(float(t) for t in gap_thresholds_usd)
    age_thresholds_f = tuple(float(t) for t in chainlink_age_thresholds_ms)

    gap_samples: dict = {label: [] for label in _ordered_labels(gap_thresholds_f)}
    age_samples: dict = {label: [] for label in _ordered_labels(age_thresholds_f)}
    trigger_samples: dict = {"market": [], "rtds": []}

    gap_excluded = 0
    age_excluded = 0

    for ev in all_events:
        if ev.get("event") != "decision":
            continue

        sample: dict = {
            "fair_minus_best_bid": ev.get("fair_minus_best_bid"),
            "best_ask_minus_fair": ev.get("best_ask_minus_fair"),
        }

        # Gap bucket: abs(binance_chainlink_gap)
        raw_gap = ev.get("binance_chainlink_gap")
        if raw_gap is not None:
            label = _bucket_label(abs(raw_gap), gap_thresholds_f)
            gap_samples[label].append(sample)
        else:
            gap_excluded += 1

        # Chainlink age bucket
        age = ev.get("chainlink_age_ms")
        if age is not None:
            label = _bucket_label(float(age), age_thresholds_f)
            age_samples[label].append(sample)
        else:
            age_excluded += 1

        # Trigger split
        trigger = ev.get("trigger")
        if trigger in trigger_samples:
            trigger_samples[trigger].append(sample)

    by_gap = {label: _to_bucket_stats(samples) for label, samples in gap_samples.items()}
    by_age = {label: _to_bucket_stats(samples) for label, samples in age_samples.items()}
    by_trigger = {label: _to_bucket_stats(samples) for label, samples in trigger_samples.items()}

    # Gate reason counts: bid_reason from each decision event.
    bid_reason_counts: dict[str, int] = {}
    for ev in all_events:
        if ev.get("event") != "decision":
            continue
        reason = ev.get("bid_reason")
        if reason is not None:
            bid_reason_counts[reason] = bid_reason_counts.get(reason, 0) + 1

    total_decisions_in_flat = sum(s.decisions_in_flat for s in session_summaries)
    total_decisions_in_long = sum(s.decisions_in_long for s in session_summaries)

    total_completed_round_trips = sum(s.completed_round_trips for s in session_summaries)
    total_forced_exits = sum(s.forced_exit_count for s in session_summaries)

    # Weighted average holding time across all round trips
    weighted_sum = sum(
        s.avg_holding_time_s * s.completed_round_trips
        for s in session_summaries
        if s.avg_holding_time_s is not None and s.completed_round_trips > 0
    )
    total_with_holding = sum(
        s.completed_round_trips
        for s in session_summaries
        if s.avg_holding_time_s is not None and s.completed_round_trips > 0
    )
    avg_holding_time_s: Optional[float] = (
        weighted_sum / total_with_holding if total_with_holding > 0 else None
    )

    # Exit reason tally from fill_simulated events with intent=="exit"
    by_exit_reason: dict = {}
    for ev in all_events:
        if ev.get("event") == "fill_simulated" and ev.get("intent") == "exit":
            reason = ev.get("strategy_reason")
            if reason:
                by_exit_reason[reason] = by_exit_reason.get(reason, 0) + 1

    return CampaignSummary(
        session_count_requested=session_count_requested,
        session_count_completed=n,
        session_duration_s=session_duration_s,
        campaign_started_at_ms=campaign_started_at_ms,
        campaign_ended_at_ms=campaign_ended_at_ms,
        total_realized_pnl=total_realized_pnl,
        total_pnl_mark=total_pnl_mark,
        total_orders_posted=total_posted,
        total_orders_rejected=total_rejected,
        total_orders_cancelled=total_cancelled,
        total_bid_fills=total_bid_fills,
        total_ask_fills=total_ask_fills,
        fill_rate=fill_rate,
        rejection_rate=rejection_rate,
        cancel_to_post_ratio=cancel_to_post_ratio,
        max_up_inventory_peak=max_up_peak,
        sessions_with_open_position=sessions_with_open,
        sessions_profitable_realized=sessions_profitable_realized,
        sessions_profitable_mark=sessions_profitable_mark,
        pnl_per_session=pnl_per_session,
        by_gap_bucket=by_gap,
        by_chainlink_age_bucket=by_age,
        by_trigger=by_trigger,
        gap_bucket_excluded_count=gap_excluded,
        chainlink_age_bucket_excluded_count=age_excluded,
        by_bid_reason=bid_reason_counts,
        total_decisions_in_flat=total_decisions_in_flat,
        total_decisions_in_long=total_decisions_in_long,
        total_completed_round_trips=total_completed_round_trips,
        total_forced_exits=total_forced_exits,
        avg_holding_time_s=avg_holding_time_s,
        by_exit_reason=by_exit_reason,
    )
