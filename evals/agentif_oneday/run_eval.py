"""CLI entry point for running AgentIF-OneDay tasks with local Stirrup."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .loader import (
    DEFAULT_DATA_ROOT,
    SUITE_DEFAULTS,
    build_attachment_index,
    load_question_ids_file,
    load_tasks,
    parse_question_ids,
    resolve_input_files,
    resolve_suite_paths,
)
from .runner import DEFAULT_SYSTEM_PROMPT, parse_bool, run_task

logger = logging.getLogger(__name__)


def _load_dotenv_if_available(env_file: Path | None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=env_file, override=False)


def _load_completed_question_ids(results_path: Path) -> set[str]:
    completed: set[str] = set()
    if not results_path.exists():
        return completed
    with results_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("question_id") and record.get("status") == "completed":
                completed.add(str(record["question_id"]))
    return completed


async def _run_all(
    *,
    tasks: list[Any],
    attachment_dir: Path,
    attachment_search_root: Path,
    output_dir: Path,
    results_path: Path,
    model: str,
    api_key: str | None,
    base_url: str | None,
    concurrency: int,
    max_turns: int,
    client_timeout_seconds: int,
    web_timeout_seconds: int,
    brave_api_key: str | None,
    system_prompt: str | None,
    include_score_criteria: bool,
    browser_headless: bool,
    browser_executable_path: str | None,
    browser_cdp_url: str | None,
    browser_profile_dir: Path | None,
    browser_user_agent: str | None,
    browser_timezone: str | None,
    cf_retry_attempts: int,
    cf_retry_wait_seconds: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    semaphore = asyncio.Semaphore(max(concurrency, 1))
    results_lock = asyncio.Lock()
    attachment_index = build_attachment_index(attachment_search_root)

    with results_path.open("a", encoding="utf-8", buffering=1) as results_file:

        async def _run_one(task: Any) -> None:
            async with semaphore:
                logger.info("Starting %s", task.question_id)
                try:
                    input_files = resolve_input_files(task, attachment_dir, attachment_index)
                    result = await run_task(
                        task=task,
                        input_files=input_files,
                        output_base_dir=output_dir,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        max_turns=max_turns,
                        client_timeout_seconds=client_timeout_seconds,
                        web_timeout_seconds=web_timeout_seconds,
                        brave_api_key=brave_api_key,
                        system_prompt=system_prompt,
                        include_score_criteria=include_score_criteria,
                        browser_headless=browser_headless,
                        browser_executable_path=browser_executable_path,
                        browser_cdp_url=browser_cdp_url,
                        browser_profile_dir=browser_profile_dir,
                        browser_user_agent=browser_user_agent,
                        browser_timezone=browser_timezone,
                        cf_retry_attempts=cf_retry_attempts,
                        cf_retry_wait_seconds=cf_retry_wait_seconds,
                        overwrite=overwrite,
                        dry_run=dry_run,
                    )
                except Exception as exc:
                    result = {
                        "question_id": task.question_id,
                        "title": task.title,
                        "status": "error",
                        "success": False,
                        "error": str(exc),
                        "output_dir": str(output_dir / task.question_id),
                    }

                async with results_lock:
                    results_file.write(json.dumps(result, ensure_ascii=False) + "\n")

                logger.info("Finished %s [%s]", task.question_id, result.get("status"))

        await asyncio.gather(*(_run_one(task) for task in tasks))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AgentIF-OneDay tasks using the local modified Stirrup agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", default=".env", help="Optional .env path loaded before reading defaults")
    parser.add_argument("--suite", default="all", choices=sorted(SUITE_DEFAULTS), help="Task suite to run")
    parser.add_argument("--data-root", default=None, help="AgentIF-OneDay data root")
    parser.add_argument("--task-jsonl", default=None, help="Override task JSONL path")
    parser.add_argument("--attachment-dir", default=None, help="Override primary attachment directory")
    parser.add_argument("--attachment-search-root", default=None, help="Recursive fallback attachment search root")
    parser.add_argument("--output-dir", default="./agentif_oneday_outputs", help="Base output directory")
    parser.add_argument("--question-ids", default="", help="Comma/space separated task ids, e.g. taskif_1,taskif_2")
    parser.add_argument("--question-ids-file", default=None, help="Text file with one or more task ids per line")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit task count when ids are not specified")
    parser.add_argument("--model", default=None, help="Model id, or STIRRUP_MODEL/MODEL_NAME env")
    parser.add_argument("--api-key", default=None, help="API key, or STIRRUP_API_KEY/MODEL_API_KEY/OPENAI_API_KEY env")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, or STIRRUP_BASE_URL/MODEL_BASE_URL env")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of tasks to run concurrently")
    parser.add_argument("--max-turns", type=int, default=None, help="Agent max turns")
    parser.add_argument("--client-timeout-seconds", type=int, default=1800, help="LLM request timeout")
    parser.add_argument("--web-timeout-seconds", type=int, default=180, help="Web tool timeout")
    parser.add_argument("--brave-api-key", default=None, help="Optional Brave Search API key")
    parser.add_argument("--browser-headless", default=None, help="Browser headless mode: true/false")
    parser.add_argument("--browser-executable-path", default=None, help="Chrome/Chromium executable path")
    parser.add_argument("--browser-cdp-url", default=None, help="Optional existing Chrome CDP URL")
    parser.add_argument("--browser-profile-dir", default=None, help="Persistent browser profile directory")
    parser.add_argument("--browser-user-agent", default=None, help="Browser User-Agent override")
    parser.add_argument("--browser-timezone", default=None, help="Browser timezone identifier")
    parser.add_argument("--cf-retry-attempts", type=int, default=None, help="Cloudflare retry count")
    parser.add_argument("--cf-retry-wait-seconds", type=int, default=None, help="Wait seconds per Cloudflare retry")
    parser.add_argument("--include-score-criteria", action="store_true", help="Append scoring criteria to the prompt")
    parser.add_argument("--system-prompt-file", default=None, help="Use a custom system prompt text file")
    parser.add_argument("--no-system-prompt", action="store_true", help="Do not use the AgentIF execution policy prompt")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing task output directory before rerun")
    parser.add_argument("--dry-run", action="store_true", help="Write mapped payloads without calling the LLM")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    args = parser.parse_args()

    env_file = Path(args.env_file).resolve() if args.env_file else None
    _load_dotenv_if_available(env_file)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    data_root = Path(args.data_root or os.getenv("AGENTIF_ONEDAY_DATA_ROOT") or DEFAULT_DATA_ROOT).resolve()
    task_jsonl, attachment_dir, default_search_root = resolve_suite_paths(
        suite=args.suite,
        data_root=data_root,
        task_jsonl=Path(args.task_jsonl) if args.task_jsonl else None,
        attachment_dir=Path(args.attachment_dir) if args.attachment_dir else None,
    )
    attachment_search_root = Path(args.attachment_search_root).resolve() if args.attachment_search_root else default_search_root

    question_ids = parse_question_ids(args.question_ids)
    if args.question_ids_file:
        question_ids.extend(load_question_ids_file(Path(args.question_ids_file)))

    model = args.model or os.getenv("STIRRUP_MODEL") or os.getenv("MODEL_NAME") or ""
    api_key = (
        args.api_key
        or os.getenv("STIRRUP_API_KEY")
        or os.getenv("MODEL_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or None
    )
    base_url = args.base_url or os.getenv("STIRRUP_BASE_URL") or os.getenv("MODEL_BASE_URL") or None
    brave_api_key = args.brave_api_key or os.getenv("BRAVE_API_KEY") or None

    if not args.dry_run and not model.strip():
        raise SystemExit("--model is required, or set STIRRUP_MODEL / MODEL_NAME in .env")

    browser_headless = parse_bool(
        args.browser_headless or os.getenv("STIRRUP_BROWSER_HEADLESS"),
        default=False,
    )
    browser_executable_path = (
        args.browser_executable_path or os.getenv("STIRRUP_BROWSER_EXECUTABLE_PATH") or ""
    ).strip() or None
    browser_cdp_url = (args.browser_cdp_url or os.getenv("STIRRUP_BROWSER_CDP_URL") or "").strip() or None
    browser_profile_dir_raw = (
        args.browser_profile_dir
        or os.getenv("STIRRUP_BROWSER_PROFILE_DIR")
        or str((Path.cwd() / ".browser_profile").resolve())
    )
    browser_profile_dir = Path(browser_profile_dir_raw).resolve() if browser_profile_dir_raw.strip() else None
    browser_user_agent = (
        args.browser_user_agent
        or os.getenv("STIRRUP_BROWSER_USER_AGENT")
        or "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    browser_timezone = args.browser_timezone or os.getenv("STIRRUP_BROWSER_TIMEZONE") or "America/Los_Angeles"
    max_turns = args.max_turns or int(os.getenv("STIRRUP_MAX_TURNS", "30"))
    cf_retry_attempts = args.cf_retry_attempts
    if cf_retry_attempts is None:
        cf_retry_attempts = int(os.getenv("STIRRUP_CF_RETRY_ATTEMPTS", "2"))
    cf_retry_wait_seconds = args.cf_retry_wait_seconds
    if cf_retry_wait_seconds is None:
        cf_retry_wait_seconds = int(os.getenv("STIRRUP_CF_RETRY_WAIT_SECONDS", "8"))

    if args.no_system_prompt:
        system_prompt = None
    elif args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    else:
        system_prompt = os.getenv("STIRRUP_AGENTIF_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "data_root": str(data_root),
        "task_jsonl": str(task_jsonl),
        "attachment_dir": str(attachment_dir),
        "attachment_search_root": str(attachment_search_root),
        "output_dir": str(output_dir),
        "results_path": str(results_path),
        "question_ids": question_ids,
        "max_tasks": args.max_tasks,
        "model": model,
        "base_url": base_url or "",
        "concurrency": args.concurrency,
        "max_turns": max_turns,
        "client_timeout_seconds": args.client_timeout_seconds,
        "web_timeout_seconds": args.web_timeout_seconds,
        "include_score_criteria": args.include_score_criteria,
        "browser_headless": browser_headless,
        "browser_executable_path": browser_executable_path or "",
        "browser_cdp_url": browser_cdp_url or "",
        "browser_profile_dir": str(browser_profile_dir) if browser_profile_dir else "",
        "browser_timezone": browser_timezone,
        "cf_retry_attempts": cf_retry_attempts,
        "cf_retry_wait_seconds": cf_retry_wait_seconds,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "python": {"executable": sys.executable, "version": sys.version},
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    completed_ids = set() if args.overwrite else _load_completed_question_ids(results_path)
    tasks = load_tasks(task_jsonl, question_ids=question_ids or None, max_tasks=args.max_tasks)
    pending_tasks = [task for task in tasks if task.question_id not in completed_ids]
    logger.info("Loaded %d tasks from %s; pending=%d", len(tasks), task_jsonl, len(pending_tasks))

    if not pending_tasks:
        logger.info("No pending tasks. Results at %s", results_path)
        return

    asyncio.run(
        _run_all(
            tasks=pending_tasks,
            attachment_dir=attachment_dir,
            attachment_search_root=attachment_search_root,
            output_dir=output_dir,
            results_path=results_path,
            model=model,
            api_key=api_key,
            base_url=base_url,
            concurrency=args.concurrency,
            max_turns=max_turns,
            client_timeout_seconds=args.client_timeout_seconds,
            web_timeout_seconds=args.web_timeout_seconds,
            brave_api_key=brave_api_key,
            system_prompt=system_prompt,
            include_score_criteria=args.include_score_criteria,
            browser_headless=browser_headless,
            browser_executable_path=browser_executable_path,
            browser_cdp_url=browser_cdp_url,
            browser_profile_dir=browser_profile_dir,
            browser_user_agent=browser_user_agent,
            browser_timezone=browser_timezone,
            cf_retry_attempts=cf_retry_attempts,
            cf_retry_wait_seconds=cf_retry_wait_seconds,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    )
    logger.info("Done. Results at %s", results_path)


if __name__ == "__main__":
    main()
