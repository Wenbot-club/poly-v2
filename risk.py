from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .settings import RuntimeConfig
from .types import LocalState, PTBDecision, PriceTick


@dataclass
class PTBLocker:
    config: RuntimeConfig

    def try_lock(self, state: LocalState, now_ms: int) -> PTBDecision:
        state.set_clock(now_ms)
        if state.ptb and state.ptb.locked:
            return state.ptb

        start_ms = state.market.start_ts_ms
        prestart_grace_ms = int(self.config.thresholds.ptb.prestart_grace_ms.value)
        max_delay_ms = int(self.config.thresholds.ptb.max_delay_ms.value)
        collision_window_ms = int(self.config.thresholds.ptb.collision_window_ms.value)

        ticks = [t for t in state.chainlink_ticks if t.symbol == "btc/usd"]
        post_start = [t for t in ticks if start_ms <= t.timestamp_ms <= start_ms + max_delay_ms]
        prestart = [t for t in ticks if start_ms - prestart_grace_ms <= t.timestamp_ms < start_ms]

        used_prestart_grace = False
        candidates: List[PriceTick]
        if post_start:
            candidates = sorted(post_start, key=_poststart_sort_key)
            reason = "locked_post_start_tick"
        elif prestart and now_ms >= start_ms + max_delay_ms:
            candidates = sorted(prestart, key=_prestart_fallback_sort_key)
            used_prestart_grace = True
            reason = "locked_prestart_nearest_open_fallback"
        else:
            decision = PTBDecision(
                locked=False,
                ptb_value=None,
                selected_tick=None,
                used_prestart_grace=False,
                collision_detected=False,
                collision_ticks=[],
                reason="waiting_for_chainlink_open_tick",
                decision_ts_ms=now_ms,
            )
            state.ptb = decision
            return decision

        selected = candidates[0]
        collision_ticks = [t for t in candidates[1:] if abs(t.timestamp_ms - selected.timestamp_ms) <= collision_window_ms]
        collision_detected = any(abs(t.value - selected.value) > 1e-9 for t in collision_ticks)

        decision = PTBDecision(
            locked=True,
            ptb_value=selected.value,
            selected_tick=selected,
            used_prestart_grace=used_prestart_grace,
            collision_detected=collision_detected,
            collision_ticks=collision_ticks,
            reason=reason,
            decision_ts_ms=now_ms,
        )
        state.ptb = decision
        state.log(
            "INFO" if not collision_detected else "WARN",
            "ptb_locked",
            ts_ms=now_ms,
            ptb_value=selected.value,
            payload_timestamp_ms=selected.timestamp_ms,
            recv_timestamp_ms=selected.recv_timestamp_ms,
            sequence_no=selected.sequence_no,
            used_prestart_grace=used_prestart_grace,
            collision_detected=collision_detected,
            collision_ticks=[_tick_to_dict(t) for t in collision_ticks],
            selection_mode="post_start_earliest" if not used_prestart_grace else "prestart_nearest_open",
        )
        return decision


def _poststart_sort_key(tick: PriceTick) -> tuple[int, int, int]:
    return (tick.timestamp_ms, tick.recv_timestamp_ms, tick.sequence_no)


def _prestart_fallback_sort_key(tick: PriceTick) -> tuple[int, int, int]:
    return (-tick.timestamp_ms, tick.recv_timestamp_ms, tick.sequence_no)


def _tick_to_dict(tick: PriceTick) -> dict:
    return {
        "symbol": tick.symbol,
        "timestamp_ms": tick.timestamp_ms,
        "recv_timestamp_ms": tick.recv_timestamp_ms,
        "value": tick.value,
        "sequence_no": tick.sequence_no,
    }
