from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .discovery import DiscoveryService, MockGammaClient
from .settings import DEFAULT_CONFIG
from .state import StateFactory
from .ws_user import UserMessageRouter


@dataclass(slots=True)
class ReplaySummary:
    actions: list[str]
    posted_count: int
    cancel_count: int
    fill_count: int
    final_state: dict
    recorded_final_state: dict
    assertion_passed: bool


def replay_jsonl(path: str | Path) -> ReplaySummary:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    header = _single(rows, "run_header")
    recorded_final_state = _single(rows, "final_state")["payload"]["state"]

    state = StateFactory(DEFAULT_CONFIG).create(
        DiscoveryService(MockGammaClient(now_ts_ms=int(header["payload"].get("initial_clock_ms") or 0))).find_active_btc_15m_market()
    )
    inv = header["payload"]["initial_inventory"]
    state.inventory.up_free = float(inv["up_free"])
    state.inventory.down_free = float(inv["down_free"])
    state.inventory.pusd_free = float(inv["pusd_free"])
    state.inventory.up_target = float(inv["up_target"])

    router = UserMessageRouter()
    actions: list[str] = []
    for row in rows:
        if row["event_type"] == "user_message":
            router.apply(state, row["payload"])
        elif row["event_type"] == "execution_action":
            actions.append(str(row["payload"]["action"]))

    final_state = _snapshot_state(state)
    assert final_state == recorded_final_state, {"replayed_final_state": final_state, "recorded_final_state": recorded_final_state}
    return ReplaySummary(
        actions=actions,
        posted_count=sum(1 for a in actions if a.startswith("post:")),
        cancel_count=sum(1 for a in actions if a.startswith("cancel:")),
        fill_count=sum(1 for a in actions if a.startswith("fill:")),
        final_state=final_state,
        recorded_final_state=recorded_final_state,
        assertion_passed=True,
    )


def _single(rows: list[dict], event_type: str) -> dict:
    matches = [row for row in rows if row["event_type"] == event_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {event_type!r}, got {len(matches)}")
    return matches[0]


def _snapshot_state(state) -> dict:
    return {
        "open_orders": {oid: {"side": order.side, "price": order.price, "remaining": order.remaining, "status": order.status} for oid, order in state.open_orders.items()},
        "live_bid_order_id": state.live_bid_order_id,
        "live_ask_order_id": state.live_ask_order_id,
        "pusd_free": round(state.inventory.pusd_free, 6),
        "up_free": round(state.inventory.up_free, 6),
        "pusd_reserved_for_bids": round(state.inventory.pusd_reserved_for_bids, 6),
        "up_reserved_for_asks": round(state.inventory.up_reserved_for_asks, 6),
        "available_pusd": round(state.inventory.available_pusd(), 6),
        "available_up": round(state.inventory.available_up(), 6),
    }
