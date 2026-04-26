"""LLM-as-Judge grading module for GDPVal evaluation.

Evaluates agent-produced deliverables against rubric criteria using an LLM
to judge each rubric item. Supports both single-item and batch grading modes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel  

from .file_extract import extract_text

logger = logging.getLogger(__name__)

# --- Prompt templates ---

GRADING_PROMPT_TEMPLATE = """\
You are a strict evaluation system for rubric-based grading.

## Task Description
{task_prompt}

## Agent's Output Content
{agent_output_text}

{reference_section}

## Rubric Criterion
Criterion: {criterion}
Maximum Score: {max_score}

## Evaluation Rules
- Only judge based on the provided agent output above. Do NOT use external knowledge.
- Do NOT infer or assume missing information. If the output does not explicitly contain the required evidence, assign 0.
- Partial satisfaction = 0. The criterion must be fully and explicitly met.
- Check BOTH content correctness AND format/structure correctness (e.g., file type, sheet names, column structure, JSON fields).
- If the criterion specifies a particular format, structure, or naming convention, verify it literally.

Respond in this exact JSON format and nothing else:
{{"awarded_score": <0 or {max_score}>, "judgment": "<brief reasoning in 1-2 sentences>"}}"""

BATCH_GRADING_PROMPT_TEMPLATE = """\
You are a strict evaluation system for rubric-based grading.

## Task Description
{task_prompt}

## Agent's Output Content
{agent_output_text}

{reference_section}

## Rubric Items to Evaluate
{rubric_items_text}

## Evaluation Rules
- Only judge based on the provided agent output above. Do NOT use external knowledge.
- Do NOT infer or assume missing information. If the output does not explicitly contain the required evidence, assign 0.
- Partial satisfaction = 0. Each criterion must be fully and explicitly met.
- Check BOTH content correctness AND format/structure correctness (e.g., file type, sheet names, column structure, JSON fields).
- If a criterion specifies a particular format, structure, or naming convention, verify it literally.

Respond with a JSON array, one entry per rubric item, in the same order as listed above.
Each entry must have this exact format:
{{"rubric_item_id": "<id>", "awarded_score": <0 or max_score>, "judgment": "<brief reasoning>"}}

Respond with the JSON array and nothing else."""

REFERENCE_SECTION_TEMPLATE = """\
## Reference Deliverable (for context only)
The following reference is provided as additional context.
Do NOT require exact matching with the reference.
Only evaluate based on the rubric criterion above.

{reference_text}"""

# Maximum number of rubric items to evaluate in a single LLM call
BATCH_SIZE = 10


# --- Data models ---

class RubricItem(BaseModel):
    """A single rubric criterion from the dataset."""

    rubric_item_id: str
    criterion: str
    score: int
    tags: list[str] = []


class RubricResult(BaseModel):
    """Judgment result for a single rubric item."""

    rubric_item_id: str
    criterion: str
    max_score: int
    awarded_score: int
    judgment: str


class GradeResult(BaseModel):
    """Complete grading result for a task."""

    task_id: str
    score: int
    max_score: int
    rubric_results: list[RubricResult]
    grading_token_usage: dict[str, int] = {"input": 0, "output": 0}
    error: str | None = None


# --- Helpers ---

def parse_rubric(task: dict[str, Any]) -> list[RubricItem]:
    """Parse task['rubric_json'] into RubricItem objects."""
    raw = task.get("rubric_json")
    if raw is None:
        return []

    if isinstance(raw, str):
        items = json.loads(raw)
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    rubric_items = []
    for item in items:
        if isinstance(item, str):
            item = json.loads(item)
        rubric_items.append(
            RubricItem(
                rubric_item_id=item.get("rubric_item_id", ""),
                criterion=item.get("criterion", ""),
                score=item.get("score", 1),
                tags=item.get("tags") or [],
            )
        )
    return rubric_items


def _collect_output_text(output_dir: Path) -> str:
    """Collect and extract text from all agent output files in a directory."""
    exclude_dirs = {"reference", "deliverables"}
    exclude_files = {"trajectory.jsonl"}

    parts: list[str] = []
    for file in sorted(output_dir.rglob("*")):
        if not file.is_file():
            continue
        if file.name in exclude_files:
            continue
        rel = file.relative_to(output_dir)
        if rel.parts and rel.parts[0] in exclude_dirs:
            continue

        text = extract_text(file)
        if text.strip():
            parts.append(f"=== File: {rel} ===\n{text}")

    return "\n\n".join(parts) if parts else "[No output files found]"


def _build_reference_section(reference_text: str | None) -> str:
    """Build the reference section for the grading prompt."""
    if not reference_text:
        return ""
    return REFERENCE_SECTION_TEMPLATE.format(reference_text=reference_text)


def _parse_json_response(text: str) -> Any:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
        text = text.strip()
    return json.loads(text)


# --- Single-item grading ---

async def _judge_rubric_item(
    client: Any,
    rubric_item: RubricItem,
    agent_output_text: str,
    reference_section: str,
    task_prompt: str,
) -> tuple[RubricResult, dict[str, int]]:
    """Ask the LLM to judge a single rubric item."""
    prompt = GRADING_PROMPT_TEMPLATE.format(
        task_prompt=task_prompt,
        agent_output_text=agent_output_text,
        reference_section=reference_section,
        criterion=rubric_item.criterion,
        max_score=rubric_item.score,
    )

    token_usage = {"input": 0, "output": 0}

    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=client._model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
            )
            text = response.choices[0].message.content or ""

            if response.usage:
                token_usage["input"] += response.usage.prompt_tokens
                token_usage["output"] += response.usage.completion_tokens

            parsed = _parse_json_response(text)
            awarded = parsed.get("awarded_score", 0)
            judgment = parsed.get("judgment", "")

            if awarded not in (0, rubric_item.score):
                awarded = 0

            return RubricResult(
                rubric_item_id=rubric_item.rubric_item_id,
                criterion=rubric_item.criterion,
                max_score=rubric_item.score,
                awarded_score=awarded,
                judgment=judgment,
            ), token_usage

        except (json.JSONDecodeError, KeyError) as exc:
            if attempt == 0:
                logger.debug("Grading parse error (retrying): %s", exc)
                continue
            logger.warning("Grading parse error for rubric %s: %s", rubric_item.rubric_item_id, exc)
            return RubricResult(
                rubric_item_id=rubric_item.rubric_item_id,
                criterion=rubric_item.criterion,
                max_score=rubric_item.score,
                awarded_score=0,
                judgment=f"PARSE_ERROR: {exc}",
            ), token_usage

    return RubricResult(
        rubric_item_id=rubric_item.rubric_item_id,
        criterion=rubric_item.criterion,
        max_score=rubric_item.score,
        awarded_score=0,
        judgment="UNEXPECTED_ERROR",
    ), token_usage


# --- Batch grading ---

async def _judge_rubric_batch(
    client: Any,
    rubric_items: list[RubricItem],
    agent_output_text: str,
    reference_section: str,
    task_prompt: str,
) -> tuple[list[RubricResult], dict[str, int]]:
    """Ask the LLM to judge multiple rubric items in one call."""
    rubric_items_text = "\n".join(
        f"{i + 1}. [id={item.rubric_item_id}] (max {item.score} pts): {item.criterion}"
        for i, item in enumerate(rubric_items)
    )

    prompt = BATCH_GRADING_PROMPT_TEMPLATE.format(
        task_prompt=task_prompt,
        agent_output_text=agent_output_text,
        reference_section=reference_section,
        rubric_items_text=rubric_items_text,
    )

    token_usage = {"input": 0, "output": 0}
    # Build a lookup for quick access
    item_map = {item.rubric_item_id: item for item in rubric_items}

    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=client._model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256 * len(rubric_items),
            )
            text = response.choices[0].message.content or ""

            if response.usage:
                token_usage["input"] += response.usage.prompt_tokens
                token_usage["output"] += response.usage.completion_tokens

            parsed_list = _parse_json_response(text)
            if not isinstance(parsed_list, list):
                raise ValueError("Expected JSON array")

            # Map results back to rubric items
            results: list[RubricResult] = []
            parsed_by_id = {r.get("rubric_item_id"): r for r in parsed_list if isinstance(r, dict)}

            for item in rubric_items:
                entry = parsed_by_id.get(item.rubric_item_id)
                if entry:
                    awarded = entry.get("awarded_score", 0)
                    if awarded not in (0, item.score):
                        awarded = 0
                    results.append(RubricResult(
                        rubric_item_id=item.rubric_item_id,
                        criterion=item.criterion,
                        max_score=item.score,
                        awarded_score=awarded,
                        judgment=entry.get("judgment", ""),
                    ))
                else:
                    results.append(RubricResult(
                        rubric_item_id=item.rubric_item_id,
                        criterion=item.criterion,
                        max_score=item.score,
                        awarded_score=0,
                        judgment="NOT_FOUND_IN_BATCH_RESPONSE",
                    ))

            return results, token_usage

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            if attempt == 0:
                logger.debug("Batch grading parse error (retrying): %s", exc)
                continue
            logger.warning("Batch grading parse error, falling back to single-item: %s", exc)
            # Fallback: grade each item individually
            all_results: list[RubricResult] = []
            for item in rubric_items:
                result, usage = await _judge_rubric_item(
                    client=client,
                    rubric_item=item,
                    agent_output_text=agent_output_text,
                    reference_section=reference_section,
                    task_prompt=task_prompt,
                )
                all_results.append(result)
                token_usage["input"] += usage["input"]
                token_usage["output"] += usage["output"]
            return all_results, token_usage

    # Should not reach here
    return [
        RubricResult(
            rubric_item_id=item.rubric_item_id,
            criterion=item.criterion,
            max_score=item.score,
            awarded_score=0,
            judgment="UNEXPECTED_ERROR",
        )
        for item in rubric_items
    ], token_usage


# --- Main grading entry point ---

async def grade_task(
    task: dict[str, Any],
    output_dir: Path,
    *,
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    deliverable_dir: Path | None = None,
    batch: bool = True,
) -> GradeResult:
    """Grade a single task's output against its rubric using LLM-as-Judge.

    Args:
        task: Task dict from the GDPVal dataset (must have rubric_json).
        output_dir: Directory containing agent's output files for this task.
        model: LLM model identifier for grading.
        api_key: API key for the grading LLM.
        base_url: Optional API base URL for OpenAI-compatible providers.
        deliverable_dir: Directory with reference deliverable files.
            If provided, their content is included as context (not as ground truth).
        batch: If True, evaluate multiple rubric items per LLM call to reduce cost.

    Returns:
        GradeResult with per-item scores and total.
    """
    import openai

    task_id = task.get("task_id", "unknown")
    task_prompt = task.get("prompt", "")

    # Parse rubric
    rubric_items = parse_rubric(task)
    if not rubric_items:
        return GradeResult(
            task_id=task_id,
            score=0,
            max_score=0,
            rubric_results=[],
            error="No rubric items found",
        )

    # Extract agent output text
    agent_output_text = _collect_output_text(output_dir)

    # Extract reference deliverable text (context only, not ground truth)
    reference_text: str | None = None
    if deliverable_dir and deliverable_dir.exists():
        ref_parts = []
        for file in sorted(deliverable_dir.rglob("*")):
            if file.is_file():
                text = extract_text(file)
                if text.strip():
                    ref_parts.append(f"=== Reference: {file.name} ===\n{text}")
        if ref_parts:
            reference_text = "\n\n".join(ref_parts)

    reference_section = _build_reference_section(reference_text)

    # Create OpenAI client for grading
    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    oai_client = openai.AsyncOpenAI(**client_kwargs)
    oai_client._model_id = model  # type: ignore[attr-defined]

    total_usage = {"input": 0, "output": 0}
    rubric_results: list[RubricResult] = []

    if batch and len(rubric_items) > 1:
        # Batch mode: evaluate BATCH_SIZE items per LLM call
        for batch_start in range(0, len(rubric_items), BATCH_SIZE):
            batch_items = rubric_items[batch_start : batch_start + BATCH_SIZE]
            logger.debug(
                "Grading task %s batch %d-%d/%d",
                task_id, batch_start + 1, batch_start + len(batch_items), len(rubric_items),
            )
            try:
                results, usage = await _judge_rubric_batch(
                    client=oai_client,
                    rubric_items=batch_items,
                    agent_output_text=agent_output_text,
                    reference_section=reference_section,
                    task_prompt=task_prompt,
                )
                rubric_results.extend(results)
                total_usage["input"] += usage["input"]
                total_usage["output"] += usage["output"]
            except Exception as exc:
                logger.warning("Batch grading error for task %s: %s", task_id, exc)
                for item in batch_items:
                    rubric_results.append(RubricResult(
                        rubric_item_id=item.rubric_item_id,
                        criterion=item.criterion,
                        max_score=item.score,
                        awarded_score=0,
                        judgment=f"ERROR: {exc}",
                    ))
    else:
        # Single-item mode
        for i, item in enumerate(rubric_items):
            logger.debug("Grading task %s rubric %d/%d: %s", task_id, i + 1, len(rubric_items), item.criterion[:80])
            try:
                result, usage = await _judge_rubric_item(
                    client=oai_client,
                    rubric_item=item,
                    agent_output_text=agent_output_text,
                    reference_section=reference_section,
                    task_prompt=task_prompt,
                )
                rubric_results.append(result)
                total_usage["input"] += usage["input"]
                total_usage["output"] += usage["output"]
            except Exception as exc:
                logger.warning("Grading error for task %s rubric %s: %s", task_id, item.rubric_item_id, exc)
                rubric_results.append(RubricResult(
                    rubric_item_id=item.rubric_item_id,
                    criterion=item.criterion,
                    max_score=item.score,
                    awarded_score=0,
                    judgment=f"ERROR: {exc}",
                ))

    total_score = sum(r.awarded_score for r in rubric_results)
    max_score = sum(r.max_score for r in rubric_results)

    logger.info(
        "Task %s graded: %d/%d (%.1f%%)",
        task_id, total_score, max_score, 100 * total_score / max_score if max_score else 0,
    )

    return GradeResult(
        task_id=task_id,
        score=total_score,
        max_score=max_score,
        rubric_results=rubric_results,
        grading_token_usage=total_usage,
    )
