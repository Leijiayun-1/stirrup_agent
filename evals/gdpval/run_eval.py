"""CLI entry point for running GDPVal evaluation.

Usage:
    python -m evals.gdpval [options]

Examples:
    # Run 1 task for debugging
    OPENAI_API_KEY=sk-... python -m evals.gdpval --max-tasks 1 --output-dir ./test_output

    # Run specific tasks
    python -m evals.gdpval --task-ids task_001 task_002

    # Run all 220 tasks with concurrency 4
    python -m evals.gdpval --output-dir ./gdpval_outputs --concurrency 4
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


def _load_completed_task_ids(results_path: Path) -> set[str]:
    """Load already-completed task IDs from an existing results.jsonl file."""
    completed: set[str] = set()
    if not results_path.exists():
        return completed
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                task_id = record.get("task_id")
                if task_id:
                    completed.add(task_id)
            except json.JSONDecodeError:
                pass
    return completed


async def _run_all(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    api_key: str | None,
    base_url: str | None,
    docker_image: str,
    use_local: bool,
    concurrency: int,
    results_path: Path,
    visualize: bool = False,
    max_tokens: int | None = None,
) -> None:
    """Run all tasks with bounded concurrency, appending results to results.jsonl."""
    from .runner import run_task

    semaphore = asyncio.Semaphore(concurrency)
    results_lock = asyncio.Lock()
    results_file = results_path.open("a", buffering=1)  # line-buffered

    try:
        async def _run_one(task: dict[str, Any]) -> None:
            task_id = task.get("task_id", "unknown")
            async with semaphore:
                logger.info("Starting task %s", task_id)
                result = await run_task(
                    task=task,
                    output_base_dir=output_dir,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    docker_image=docker_image,
                    use_local=use_local,
                    visualize=visualize,
                    max_tokens=max_tokens,
                )
                async with results_lock:
                    results_file.write(json.dumps(result) + "\n")
                status = "OK" if result["success"] else "FAIL"
                logger.info("Task %s finished [%s]", task_id, status)

        await asyncio.gather(*(_run_one(t) for t in tasks))
    finally:
        results_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GDPVal evaluation benchmark using Stirrup + Docker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="gpt-4o", help="LLM model identifier")
    parser.add_argument("--output-dir", default="./gdpval_outputs", help="Base output directory")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks (for debugging)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max output tokens per LLM call. Reduce for models with small context windows (e.g., 8192 for deepseek-chat).",
    )
    parser.add_argument("--task-ids", nargs="+", default=None, help="Run only these task IDs")
    parser.add_argument(
        "--task-ids-file",
        default=None,
        help="Path to a text file with one task ID per line",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL for OpenAI-compatible providers (e.g., https://api.deepseek.com/v1)",
    )
    parser.add_argument("--docker-image", default="python:3.12-slim", help="Docker image for code execution")
    parser.add_argument("--local", action="store_true", help="Use local execution instead of Docker")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of tasks to run in parallel")
    parser.add_argument("--split", default="train", help="Dataset split to use")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable web visualization of agent tools and LLM outputs",
    )
    args = parser.parse_args()

    task_ids: list[str] | None = args.task_ids
    if args.task_ids_file:
        file_ids = Path(args.task_ids_file).read_text().splitlines()
        file_ids = [l.strip() for l in file_ids if l.strip() and not l.startswith("#")]
        task_ids = (task_ids or []) + file_ids

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Resolve API key
    api_key: str | None = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "No API key provided. Set OPENAI_API_KEY env var or use --api-key. "
            "Proceeding anyway (may fail at inference time)."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    # Load already-completed tasks for resume support
    completed_ids = _load_completed_task_ids(results_path)
    if completed_ids:
        logger.info("Resuming: skipping %d already-completed tasks", len(completed_ids))

    # Load tasks from HuggingFace
    from .loader import load_gdpval_tasks

    tasks = load_gdpval_tasks(split=args.split, task_ids=task_ids, max_tasks=args.max_tasks)

    # Filter out completed tasks
    pending_tasks = [t for t in tasks if t.get("task_id") not in completed_ids]
    logger.info("%d tasks pending (total=%d, skipped=%d)", len(pending_tasks), len(tasks), len(tasks) - len(pending_tasks))

    if not pending_tasks:
        logger.info("All tasks already completed. Results at: %s", results_path)
        return

    server = None
    if args.visualize:
        try:
            from visualize import start_visualizer_server

            server = start_visualizer_server(open_browser=True)
        except Exception as exc:
            logger.error("Failed to start visualizer server: %s", exc)

    try:
        asyncio.run(
            _run_all(
                tasks=pending_tasks,
                output_dir=output_dir,
                model=args.model,
                api_key=api_key,
                base_url=args.base_url,
                docker_image=args.docker_image,
                use_local=args.local,
                concurrency=args.concurrency,
                results_path=results_path,
                visualize=args.visualize,
                max_tokens=args.max_tokens,
            )
        )
    finally:
        if server is not None:
            server.shutdown()

    # Print summary
    completed_count = sum(1 for _ in _load_completed_task_ids(results_path))
    logger.info("Done. %d tasks recorded in %s", completed_count, results_path)


if __name__ == "__main__":
    main()
