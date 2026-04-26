from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from stirrup.core.models import LLMClient, SystemMessage, UserMessage
from stirrup.core.semantic_state import PlanArtifact, SemanticStateManager
from stirrup.prompts import PLANNER_PROMPT_TEMPLATE


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class Planner:
    """Pre-run planner that turns the task description into a structured plan artifact."""

    def __init__(self, client: LLMClient, *, max_turns: int) -> None:
        self._client = client
        self._max_turns = max_turns

    async def plan(self, task_description: str, uploaded_files: list[str]) -> PlanArtifact:
        response = await self._client.generate(
            [
                SystemMessage(content=PLANNER_PROMPT_TEMPLATE.format(max_turns=self._max_turns)),
                UserMessage(
                    content=(
                        f"Task description:\n{task_description}\n\n"
                        f"Uploaded file names:\n{json.dumps(uploaded_files, ensure_ascii=True)}\n\n"
                        "Return only JSON."
                    )
                ),
            ],
            {},
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        payload = _extract_json_object(content)
        if payload is None:
            return self._fallback(task_description, uploaded_files)
        try:
            return PlanArtifact.model_validate(payload)
        except ValidationError:
            return self._fallback(task_description, uploaded_files)

    def _fallback(self, task_description: str, uploaded_files: list[str]) -> PlanArtifact:
        return SemanticStateManager.build_fallback_plan(
            task_description=task_description,
            uploaded_files=uploaded_files,
            max_turns=self._max_turns,
        ).plan
