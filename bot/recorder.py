from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class JSONLRecorder:
    path: str | Path
    sequence_no: int = 0
    _fh: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._fh = p.open("w", encoding="utf-8")

    def record(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.sequence_no += 1
        row = {"seq": self.sequence_no, "event_type": event_type, "payload": payload}
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if getattr(self, "_fh", None) is not None:
            self._fh.close()

    def __enter__(self) -> "JSONLRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
