from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _build_run_dir(output_root: Path) -> Path:
    branch = os.environ.get("AUTORESEARCH_BRANCH", "detached").replace("/", "_")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / branch / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _compute_aggregate_score(results_path: Path) -> int:
    total_score = 0
    with results_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            score = record.get("score")
            if score is not None:
                total_score += int(score)
    return total_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Thin autoresearch wrapper around python -m evals.gdpval")
    parser.add_argument("--task-ids-file", required=True)
    parser.add_argument("--output-root", default="logs/autoresearch")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--grading-model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--grading-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--grading-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    run_dir = _build_run_dir(output_root)
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    command = [
        sys.executable,
        "-m",
        "evals.gdpval",
        "--local",
        "--grade",
        "--task-ids-file",
        args.task_ids_file,
        "--output-dir",
        str(run_dir),
        "--concurrency",
        str(args.concurrency),
        "--model",
        args.model,
        "--grading-model",
        args.grading_model,
    ]
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    if args.grading_base_url:
        command.extend(["--grading-base-url", args.grading_base_url])
    if args.grading_api_key:
        command.extend(["--grading-api-key", args.grading_api_key])

    log_path = run_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if process.returncode != 0:
        print(f"adapter_failed_returncode={process.returncode}", file=sys.stderr)
        return process.returncode

    if not results_path.exists():
        print("adapter_missing_results_jsonl", file=sys.stderr)
        return 2

    aggregate_score = _compute_aggregate_score(results_path)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"run_dir={run_dir}\n")
        log_file.write(f"log_path={log_path}\n")
        log_file.write(f"metric={aggregate_score}\n")
    print(f"run_dir={run_dir}")
    print(f"log_path={log_path}")
    print(f"metric={aggregate_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
