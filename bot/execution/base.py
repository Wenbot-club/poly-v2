from __future__ import annotations

from typing import List, Optional, Protocol

from ..domain import LocalState


class ExecutionGateway(Protocol):
    def post_order(
        self,
        state: LocalState,
        *,
        asset_id: str,
        side: str,
        price: float,
        size: float,
        now_ms: int,
        slot: str,
    ) -> Optional[str]: ...

    def cancel_order(
        self, state: LocalState, order_id: str, now_ms: int, reason: str
    ) -> Optional[str]: ...

    def cancel_all(
        self, state: LocalState, now_ms: int, reason: str
    ) -> List[str]: ...

    def simulate_fill(
        self,
        state: LocalState,
        *,
        order_id: str,
        fill_size: float,
        now_ms: int,
    ) -> List[str]: ...
