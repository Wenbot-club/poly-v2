from __future__ import annotations

from typing import Protocol

from ..domain import DesiredQuotes, FairValueSnapshot, LocalState


class Strategy(Protocol):
    def build(self, state: LocalState, fair: FairValueSnapshot, now_ms: int) -> DesiredQuotes: ...
