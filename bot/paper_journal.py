"""
JSONL session journal writer for LivePaperSession.

Usage:
    from bot.paper_journal import write_jsonl

    summary = await session.run_for(duration=60)
    lines = write_jsonl(session.events, "logs/session.jsonl")

write_jsonl() is a pure function with no side effects beyond writing to disk.
It creates parent directories as needed. LivePaperSession never calls this
automatically — the caller decides whether and where to persist the event log.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(events: list[dict], path: str | Path) -> int:
    """
    Write events as newline-delimited JSON. Returns the number of lines written.

    Parent directories are created if they do not exist.
    Each event is serialised as compact JSON (no extra spaces).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = 0
    with p.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
            lines += 1
    return lines
