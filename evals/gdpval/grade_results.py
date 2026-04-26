"""Re-grade existing GDPVal outputs without re-running agents.

Usage:
    python -m evals.gdpval.grade_results --results-dir ./gdpval_outputs [options]

Examples:
    # Re-grade all tasks in test_output using gpt-4o
    python -m evals.gdpval.grade_results --results-dir ./test_output --model gpt-4o

    # Re-grade specific tasks
    python -m evals.gdpval.grade_results --results-dir ./test_output --task-ids task_001 task_002

    # Use a different grading model
    python -m evals.gdpval.grade_results --results-dir ./test_output --model claude-3-5-sonnet-20241022 --base-url https://...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _generate_report(graded_results: list[dict[str, Any]], report_path: Path) -> None:
    """Generate a human-readable markdown grading report."""
    lines: list[str] = []
    lines.append("# GDPVal Grading Report\n")

    # Summary
    scored = [r for r in graded_results if r.get("score") is not None]
    total_score = sum(r["score"] for r in scored)
    total_max = sum(r["max_score"] for r in scored)
    pct = 100 * total_score / total_max if total_max else 0

    lines.append(f"**Tasks graded:** {len(scored)} / {len(graded_results)}")
    lines.append(f"**Total score:** {total_score} / {total_max} ({pct:.1f}%)\n")
    lines.append("---\n")

    # Per-task details
    for result in graded_results:
        task_id = result.get("task_id", "unknown")
        score = result.get("score")
        max_score = result.get("max_score")
        rubric_results = result.get("rubric_results")

        if score is None:
            lines.append(f"## Task `{task_id[:12]}...`  —  SKIPPED\n")
            continue

        task_pct = 100 * score / max_score if max_score else 0
        lines.append(f"## Task `{task_id[:12]}...`  —  {score}/{max_score} ({task_pct:.0f}%)\n")
        lines.append(f"- **Sector:** {result.get('sector', 'N/A')}")
        lines.append(f"- **Occupation:** {result.get('occupation', 'N/A')}\n")

        if rubric_results:
            lines.append("| # | Score | Criterion | Judgment |")
            lines.append("|---|-------|-----------|---------|")
            for i, rr in enumerate(rubric_results, 1):
                awarded = rr.get("awarded_score", 0)
                max_s = rr.get("max_score", 0)
                mark = "PASS" if awarded > 0 else "FAIL"
                criterion = rr.get("criterion", "")
                # Truncate long criteria for table readability
                if len(criterion) > 120:
                    criterion = criterion[:117] + "..."
                # Escape pipes for markdown table
                criterion = criterion.replace("|", "\\|")
                judgment = (rr.get("judgment") or "").replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {i} | {mark} {awarded}/{max_s} | {criterion} | {judgment} |")
            lines.append("")

        lines.append("---\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Grading report written to %s", report_path)


async def grade_existing_results(
    results_dir: Path,
    *,
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    concurrency: int = 1,
    output_path: Path | None = None,
    task_ids: list[str] | None = None,
) -> None:
    """Load results.jsonl, grade each task, write results incrementally."""
    from .grader import grade_task
    from .loader import download_deliverable_files, load_gdpval_tasks

    results_path = results_dir / "results.jsonl"
    if not results_path.exists():
        logger.error("results.jsonl not found at %s", results_path)
        return

    # Load existing results
    results: list[dict[str, Any]] = []
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if task_ids:
        task_id_set = set(task_ids)
        results = [r for r in results if r.get("task_id") in task_id_set]

    logger.info("Loaded %d results to grade", len(results))

    # Load dataset to get rubric_json and deliverable URLs
    all_tasks = load_gdpval_tasks()
    task_map = {t["task_id"]: t for t in all_tasks}

    # Output file — line-buffered, write incrementally
    out_path = output_path or (results_dir / "graded_results.jsonl")
    out_file = out_path.open("w", buffering=1)
    write_lock = asyncio.Lock()

    semaphore = asyncio.Semaphore(concurrency)
    all_graded: list[dict[str, Any]] = []
    graded_lock = asyncio.Lock()

    async def _grade_one(result: dict[str, Any]) -> None:
        task_id = result.get("task_id", "unknown")
        task = task_map.get(task_id)

        if not task:
            logger.warning("Task %s not found in dataset, skipping grading", task_id)
            result["score"] = None
            result["max_score"] = None
            result["rubric_results"] = None
            result["grading_token_usage"] = None
        else:
            task_output_dir = results_dir / task_id
            if not task_output_dir.exists():
                logger.warning("Task %s output dir not found, skipping", task_id)
                result["score"] = None
                result["max_score"] = None
                result["rubric_results"] = None
                result["grading_token_usage"] = None
            else:
                async with semaphore:
                    deliverable_dir = task_output_dir / "deliverables"
                    try:
                        await download_deliverable_files(task, deliverable_dir)
                    except Exception as exc:
                        logger.warning("Task %s: failed to download deliverable files: %s", task_id, exc)

                    try:
                        grade_result = await grade_task(
                            task=task,
                            output_dir=task_output_dir,
                            model=model,
                            api_key=api_key,
                            base_url=base_url,
                            deliverable_dir=deliverable_dir,
                        )
                        result["score"] = grade_result.score
                        result["max_score"] = grade_result.max_score
                        result["rubric_results"] = [r.model_dump() for r in grade_result.rubric_results]
                        result["grading_token_usage"] = grade_result.grading_token_usage
                    except Exception as exc:
                        logger.exception("Task %s: grading failed: %s", task_id, exc)
                        result["score"] = None
                        result["max_score"] = None
                        result["rubric_results"] = None
                        result["grading_token_usage"] = None

        # Write incrementally
        async with write_lock:
            out_file.write(json.dumps(result) + "\n")
        async with graded_lock:
            all_graded.append(result)

        score_str = f"{result['score']}/{result['max_score']}" if result.get("score") is not None else "SKIP"
        logger.info("Graded task %s: %s", task_id, score_str)

    await asyncio.gather(*(_grade_one(r) for r in results))
    out_file.close()

    # Summary
    scored = [r for r in all_graded if r.get("score") is not None]
    total_score = sum(r["score"] for r in scored)
    total_max = sum(r["max_score"] for r in scored)

    logger.info("Graded results written to %s", out_path)
    if scored:
        logger.info(
            "Grading summary: %d/%d (%.1f%%) across %d tasks",
            total_score, total_max, 100 * total_score / total_max if total_max else 0, len(scored),
        )

    # Generate readable report
    report_path = out_path.with_suffix(".md")
    _generate_report(all_graded, report_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-grade existing GDPVal outputs using LLM-as-Judge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", required=True, help="Directory containing results.jsonl and task output dirs")
    parser.add_argument("--model", default="gpt-4o", help="LLM model for grading")
    parser.add_argument("--api-key", default=None, help="API key (defaults to OPENAI_API_KEY env var)")
    parser.add_argument("--base-url", default=None, help="API base URL for OpenAI-compatible providers")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of tasks to grade in parallel")
    parser.add_argument("--output", default=None, help="Output file path (defaults to <results-dir>/graded_results.jsonl)")
    parser.add_argument("--task-ids", nargs="+", default=None, help="Grade only these task IDs")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No API key provided. Set OPENAI_API_KEY or use --api-key.")

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else None

    asyncio.run(
        grade_existing_results(
            results_dir=results_dir,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            concurrency=args.concurrency,
            output_path=output_path,
            task_ids=args.task_ids,
        )
    )


if __name__ == "__main__":
    main()
