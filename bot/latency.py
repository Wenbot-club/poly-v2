"""
Latency tracking for BTC M5 live/paper-live orders.

LatencyRecord  — per-order timestamps + derived latencies
LatencyTracker — accumulates records and computes summary stats
print_latency_summary(tracker) — terminal block
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional


@dataclass
class LatencyRecord:
    window_ts: int
    action: str                      # "leg1_entry" | "hedge"
    tick_received_ts_ms: int         # btc_price_ts_ms at decision moment
    decision_ts_ms: int              # wall-clock when strategy decided
    submit_ts_ms: int                # wall-clock after paper fill submitted
    ack_ts_ms: Optional[int] = None  # real-exchange ACK (None in paper mode)

    @property
    def decision_latency_ms(self) -> int:
        """Time from tick arrival to strategy decision."""
        return self.decision_ts_ms - self.tick_received_ts_ms

    @property
    def submit_latency_ms(self) -> int:
        """Time from strategy decision to order submit (paper fill overhead)."""
        return self.submit_ts_ms - self.decision_ts_ms

    @property
    def ack_latency_ms(self) -> Optional[int]:
        """Round-trip from submit to exchange ACK (None in paper mode)."""
        if self.ack_ts_ms is None:
            return None
        return self.ack_ts_ms - self.submit_ts_ms

    @property
    def end_to_end_latency_ms(self) -> int:
        """Full latency from tick to ACK (or submit if no ACK)."""
        end = self.ack_ts_ms if self.ack_ts_ms is not None else self.submit_ts_ms
        return end - self.tick_received_ts_ms


def _pctile(values: list, p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * p / 100)))
    return float(s[idx])


class LatencyTracker:
    def __init__(self) -> None:
        self._records: list[LatencyRecord] = []

    def add(self, record: LatencyRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> list[LatencyRecord]:
        return list(self._records)

    def summary(self) -> dict:
        if not self._records:
            return {"orders_attempted": 0}

        decision_ms = [r.decision_latency_ms for r in self._records]
        submit_ms = [r.submit_latency_ms for r in self._records]
        e2e_ms = [r.end_to_end_latency_ms for r in self._records]

        return {
            "orders_attempted": len(self._records),
            "avg_decision_latency_ms": round(statistics.mean(decision_ms), 1),
            "p50_decision_latency_ms": _pctile(decision_ms, 50),
            "p95_decision_latency_ms": _pctile(decision_ms, 95),
            "avg_submit_latency_ms": round(statistics.mean(submit_ms), 1),
            "p95_submit_latency_ms": _pctile(submit_ms, 95),
            "avg_end_to_end_ms": round(statistics.mean(e2e_ms), 1),
            "p50_end_to_end_ms": _pctile(e2e_ms, 50),
            "p95_end_to_end_ms": _pctile(e2e_ms, 95),
            "max_end_to_end_ms": max(e2e_ms),
        }

    def print_summary(self) -> None:
        s = self.summary()
        if s.get("orders_attempted", 0) == 0:
            print("  [latency] no orders recorded")
            return
        print(f"\n{'─' * 55}")
        print("  [latency]")
        print(f"  orders_attempted        : {s['orders_attempted']}")
        print(f"  avg_decision_latency_ms : {s['avg_decision_latency_ms']}")
        print(f"  p50_decision_latency_ms : {s['p50_decision_latency_ms']}")
        print(f"  p95_decision_latency_ms : {s['p95_decision_latency_ms']}")
        print(f"  avg_submit_latency_ms   : {s['avg_submit_latency_ms']}")
        print(f"  p95_submit_latency_ms   : {s['p95_submit_latency_ms']}")
        print(f"  avg_end_to_end_ms       : {s['avg_end_to_end_ms']}")
        print(f"  p50_end_to_end_ms       : {s['p50_end_to_end_ms']}")
        print(f"  p95_end_to_end_ms       : {s['p95_end_to_end_ms']}")
        print(f"  max_end_to_end_ms       : {s['max_end_to_end_ms']}")
