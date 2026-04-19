"""
Paper trading campaign runner.

CampaignRunner orchestrates N sequential LivePaperSession runs, writes per-session
artifacts, and produces a campaign summary and manifest.

Fail-fast contract:
  If session i raises, the exception propagates immediately.
  Artifacts from sessions 0..i-1 remain on disk.
  campaign_summary.json and campaign_manifest.json are NOT written on failure.

Artifact layout (inside CampaignConfig.output_dir):
  session_000.jsonl              — JSONL event log (one JSON object per line)
  session_000_summary.json       — LivePaperSummary serialised as JSON
  session_001.jsonl
  session_001_summary.json
  ...
  campaign_summary.json          — CampaignSummary (aggregated metrics + breakdowns)
  campaign_manifest.json         — config + file list + start/end timestamps
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .campaign_report import CampaignSummary, compute_campaign_summary
from .domain import utc_now_ms
from .paper_journal import write_jsonl


@dataclass
class CampaignConfig:
    session_count: int = 5
    session_duration_s: int = 60
    output_dir: Path = field(default_factory=lambda: Path("campaign_out"))
    # Bucket thresholds for decision-level breakdowns.
    # Gap buckets: abs(binance_chainlink_gap) in USD.
    #   (50.0, 200.0) → "<50" / "50-200" / ">200"
    # Age buckets: chainlink_age_ms in milliseconds.
    #   (1000, 5000) → "<1000" / "1000-5000" / ">5000"
    gap_thresholds_usd: tuple = (50.0, 200.0)
    chainlink_age_thresholds_ms: tuple = (1000, 5000)


class CampaignRunner:
    """Run N paper sessions sequentially, write artifacts, return CampaignSummary."""

    async def run(
        self,
        session_factory: Callable,
        campaign_config: CampaignConfig,
    ) -> CampaignSummary:
        cfg = campaign_config
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        started_at_ms = utc_now_ms()
        session_summaries: list = []
        all_events: list = []
        written_files: list[str] = []

        for i in range(cfg.session_count):
            session = session_factory()
            # Exception here propagates; partial artifacts remain on disk.
            summary = await session.run_for(cfg.session_duration_s)

            jsonl_name = f"session_{i:03d}.jsonl"
            summary_name = f"session_{i:03d}_summary.json"

            write_jsonl(session.events, out / jsonl_name)
            _write_json(out / summary_name, dataclasses.asdict(summary))

            written_files.extend([jsonl_name, summary_name])
            session_summaries.append(summary)
            all_events.extend(session.events)

        ended_at_ms = utc_now_ms()

        campaign_summary = compute_campaign_summary(
            session_summaries=session_summaries,
            all_events=all_events,
            session_count_requested=cfg.session_count,
            session_duration_s=cfg.session_duration_s,
            gap_thresholds_usd=cfg.gap_thresholds_usd,
            chainlink_age_thresholds_ms=cfg.chainlink_age_thresholds_ms,
            campaign_started_at_ms=started_at_ms,
            campaign_ended_at_ms=ended_at_ms,
        )

        campaign_summary_name = "campaign_summary.json"
        _write_json(out / campaign_summary_name, dataclasses.asdict(campaign_summary))
        written_files.append(campaign_summary_name)

        manifest_name = "campaign_manifest.json"
        manifest: dict[str, Any] = {
            "session_count": cfg.session_count,
            "session_duration_s": cfg.session_duration_s,
            "gap_thresholds_usd": list(cfg.gap_thresholds_usd),
            "chainlink_age_thresholds_ms": list(cfg.chainlink_age_thresholds_ms),
            "output_dir": str(cfg.output_dir),
            "campaign_started_at_ms": started_at_ms,
            "campaign_ended_at_ms": ended_at_ms,
            "files": written_files + [manifest_name],
        }
        _write_json(out / manifest_name, manifest)

        return campaign_summary


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
