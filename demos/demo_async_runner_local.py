# reconstituted demo — original not present in upload
# runs the same pipeline as demo_paper_local.py with a distinct output path
from __future__ import annotations

from pathlib import Path

from demos.demo_paper_local import format_demo_output, run_async_local_demo


def main() -> None:
    output_path = Path("artifacts/demo_async_runner_local.jsonl")
    summary, replay = run_async_local_demo(output_path=output_path)
    print(format_demo_output(summary, replay, output_path))


if __name__ == "__main__":
    main()
