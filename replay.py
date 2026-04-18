from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .types import LocalState


@dataclass(slots=True)
class HeartbeatStatus:
    healthy: bool
    reason: str
    age_ms: Optional[int]
    checked_ts_ms: int


@dataclass
class HeartbeatMonitor:
    miss_timeout_ms: int
    last_confirmed_ms: Optional[int] = None

    def confirm_cycle(self, state: LocalState, now_ms: int) -> None:
        state.set_clock(now_ms)
        self.last_confirmed_ms = now_ms
        state.log("INFO", "heartbeat_cycle_confirmed", ts_ms=now_ms, last_confirmed_ms=now_ms)

    def evaluate(self, state: LocalState, now_ms: int) -> HeartbeatStatus:
        state.set_clock(now_ms)
        if self.last_confirmed_ms is None:
            status = HeartbeatStatus(False, "heartbeat_missing", None, now_ms)
            state.log("WARN", "heartbeat_evaluated", ts_ms=now_ms, healthy=False, reason=status.reason, age_ms=None)
            return status

        age_ms = now_ms - self.last_confirmed_ms
        if age_ms > self.miss_timeout_ms:
            status = HeartbeatStatus(False, "heartbeat_missed", age_ms, now_ms)
            state.log("WARN", "heartbeat_evaluated", ts_ms=now_ms, healthy=False, reason=status.reason, age_ms=age_ms)
            return status

        status = HeartbeatStatus(True, "ok", age_ms, now_ms)
        state.log("INFO", "heartbeat_evaluated", ts_ms=now_ms, healthy=True, reason=status.reason, age_ms=age_ms)
        return status
