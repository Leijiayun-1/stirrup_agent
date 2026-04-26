"""GDPVal single-task runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .client import EvalClient
from stirrup.core.agent import Agent
from stirrup.core.models import TokenUsage
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.tools.finish import SIMPLE_FINISH_TOOL
from stirrup.tools.web import WebToolProvider

from .loader import download_reference_files

logger = logging.getLogger(__name__)


async def run_task(
    task: dict[str, Any],
    output_base_dir: Path,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    docker_image: str = "python:3.12-slim",
    use_local: bool = False,
    visualize: bool = False,
    max_tokens: int | None = None,
    grade: bool = False,
    grading_model: str | None = None,
    grading_api_key: str | None = None,
    grading_base_url: str | None = None,
) -> dict[str, Any]:
    """Run a single GDPVal task and return a structured result.

    Args:
        task: Task dict from the GDPVal dataset.
        output_base_dir: Base directory for task outputs. Each task gets its own subdir.
        model: LLM model identifier (e.g., "gpt-4o", "deepseek-chat").
        api_key: API key for the LLM provider. If None, reads from environment.
        base_url: Optional API base URL for OpenAI-compatible providers
            (e.g., "https://api.deepseek.com/v1").
        docker_image: Docker image to use for code execution (ignored when use_local=True).
        use_local: If True, use LocalCodeExecToolProvider instead of Docker.
        visualize: If True, attach a WebLogger and stream events to the web UI.
        max_tokens: Max output tokens per LLM call (per-request output limit, not context window).
            Use this to cap output size for models like deepseek-chat that have internal limits.

    Returns:
        Structured result dict with keys: task_id, sector, occupation, success,
        reason, output_files, token_usage, error.
    """
    task_id: str = task.get("task_id", "unknown")
    sector: str = task.get("sector", "")
    occupation: str = task.get("occupation", "")
    prompt: str = task.get("prompt", "")

    task_output_dir = output_base_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = task_output_dir / "reference"
    ref_files: list[Path] = []

    try:
        ref_files = await download_reference_files(task, ref_dir)
        logger.info("Task %s: downloaded %d reference files", task_id, len(ref_files))
    except Exception as exc:
        logger.warning("Task %s: failed to download reference files: %s", task_id, exc)

    # DeepSeek and some other providers require tool message content as plain string
    flatten = base_url is not None and "deepseek" in base_url.lower()
    client_kwargs: dict = dict(model=model, api_key=api_key, base_url=base_url, flatten_tool_content=flatten)
    if max_tokens is not None:
        client_kwargs["max_output_tokens"] = max_tokens  # semantic: per-call output limit

    # For known DeepSeek endpoints, inject context_window for correct overflow detection and
    # summarization threshold calculation
    if base_url and "deepseek" in base_url.lower():
        client_kwargs.setdefault("context_window", 64_000)

    client = EvalClient(**client_kwargs)

    if use_local:
        exec_provider = LocalCodeExecToolProvider()
    else:
        try:
            from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider
        except ImportError as e:
            raise ImportError(
                "Requires the docker extra. Install with: `uv pip install -e '.[docker]'`"
            ) from e
        exec_provider = DockerCodeExecToolProvider.from_image(docker_image)

    logger_for_agent = None
    if visualize:
        try:
            from visualize import WebLogger

            logger_for_agent = WebLogger()
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Failed to import WebLogger for visualization: %s", exc)

    agent = Agent(
        client=client,
        name="gdpval-agent",
        tools=[exec_provider, WebToolProvider()],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=logger_for_agent,
    )

    try:
        input_files = [str(p) for p in ref_files] if ref_files else None
        async with agent.session(
            output_dir=task_output_dir,
            input_files=input_files,
            cache_on_interrupt=False,
        ) as session:
            finish_params, history, metadata = await session.run(prompt)

        # Save full conversation trajectory (history is list[list[ChatMessage]])
        trajectory_path = task_output_dir / "trajectory.jsonl"
        with trajectory_path.open("w") as f:
            for segment in history:
                for msg in segment:
                    f.write(msg.model_dump_json() + "\n")

        # Collect token usage
        token_usage_list: list[TokenUsage] = metadata.get("token_usage", [])
        total_input = sum(u.input for u in token_usage_list)
        total_answer = sum(u.answer for u in token_usage_list)
        total_reasoning = sum(u.reasoning for u in token_usage_list)

        # Collect output files saved to task_output_dir
        output_files = [
            str(p.relative_to(task_output_dir))
            for p in task_output_dir.rglob("*")
            if p.is_file() and not p.is_relative_to(ref_dir) and p.name != "trajectory.jsonl"
        ]

        reason = finish_params.reason if finish_params else "No finish tool called"
        success = finish_params is not None

        # Grading
        grade_result = None
        if grade:
            from .grader import grade_task as _grade_task
            from .loader import download_deliverable_files

            deliverable_dir = task_output_dir / "deliverables"
            try:
                await download_deliverable_files(task, deliverable_dir)
            except Exception as exc:
                logger.warning("Task %s: failed to download deliverable files: %s", task_id, exc)

            try:
                grade_result = await _grade_task(
                    task=task,
                    output_dir=task_output_dir,
                    model=grading_model or model,
                    api_key=grading_api_key or api_key,
                    base_url=grading_base_url or base_url,
                    deliverable_dir=deliverable_dir,
                )
            except Exception as exc:
                logger.exception("Task %s: grading failed: %s", task_id, exc)

        return {
            "task_id": task_id,
            "sector": sector,
            "occupation": occupation,
            "success": success,
            "reason": reason,
            "output_files": output_files,
            "token_usage": {
                "input": total_input,
                "answer": total_answer,
                "reasoning": total_reasoning,
            },
            "error": None,
            "score": grade_result.score if grade_result else None,
            "max_score": grade_result.max_score if grade_result else None,
            "rubric_results": [r.model_dump() for r in grade_result.rubric_results] if grade_result else None,
            "grading_token_usage": grade_result.grading_token_usage if grade_result else None,
        }

    except Exception as exc:
        logger.exception("Task %s failed with error: %s", task_id, exc)
        return {
            "task_id": task_id,
            "sector": sector,
            "occupation": occupation,
            "success": False,
            "reason": "Task failed with an exception.",
            "output_files": [],
            "token_usage": {"input": 0, "answer": 0, "reasoning": 0},
            "error": str(exc),
        }
