"""Tests for semantic-state planned variable progress."""

import json

from stirrup.core.semantic_state import PlanArtifact, PlannedPhase, SemanticStateManager


def _manager() -> SemanticStateManager:
    return SemanticStateManager(
        PlanArtifact(
            task_understanding="Create a final report file.",
            phases=[
                PlannedPhase(
                    name="execution",
                    goal="Create the final deliverable.",
                    turn_budget=2,
                    variables_to_extract=["final_deliverable_path", "primary_constraint"],
                )
            ],
            total_turn_budget=2,
        )
    )


def _progress(manager: SemanticStateManager, name: str) -> dict:
    state = json.loads(manager.serialize_persistent_state())
    return state["planned_variables"]["execution"][name]


def test_tool_stdout_evidence_does_not_complete_planned_variable() -> None:
    manager = _manager()

    evidence_id = manager.record_evidence(
        turn=1,
        source="code_exec",
        kind="tool_result",
        content="Saved report to final_report.xlsx",
        success=True,
    )

    progress = _progress(manager, "final_deliverable_path")
    assert evidence_id in manager.serialize_recent_evidence()
    assert progress["status"] == "pending"
    assert progress["bound_state_variables"] == []


def test_alias_state_update_requires_runtime_path_verification() -> None:
    manager = _manager()
    evidence_id = manager.record_evidence(
        turn=1,
        source="code_exec",
        kind="tool_result",
        content="Saved report to final_report.xlsx",
        success=True,
    )
    manager.apply_updates(
        [
            {
                "name": "saved_report_path",
                "value": "final_report.xlsx",
                "category": "intermediate_result",
                "source": "extractor",
                "evidence_refs": [evidence_id],
            }
        ],
        written_turn=1,
        confidence="extractor",
    )

    progress = _progress(manager, "final_deliverable_path")
    assert progress["status"] == "candidate"
    assert progress["bound_state_variables"] == ["saved_report_path"]
    assert progress["evidence_refs"] == [evidence_id]

    manager.record_runtime_env_state(
        "env.generated_files",
        {"count": 1, "paths": ["final_report.xlsx"], "truncated": False},
        written_turn=1,
        source="runtime",
        reason="file scan",
    )
    progress = _progress(manager, "final_deliverable_path")
    assert progress["status"] == "candidate"
    assert "finish has not validated" in progress["reason"]

    manager.record_runtime_env_state(
        "env.finish.validated_paths",
        ["final_report.xlsx"],
        written_turn=2,
        source="finish_tool",
        reason="finish validation",
    )
    progress = _progress(manager, "final_deliverable_path")
    assert progress["status"] == "satisfied"
    assert "validated" in progress["reason"]


def test_state_only_planned_variable_is_satisfied_by_state_update() -> None:
    manager = _manager()
    manager.apply_updates(
        [
            {
                "name": "primary_constraint",
                "value": "Return a spreadsheet.",
                "category": "task_constraint",
                "source": "agent",
            }
        ],
        written_turn=1,
        confidence="agent",
    )

    progress = _progress(manager, "primary_constraint")
    assert progress["status"] == "satisfied"
    assert progress["bound_state_variables"] == ["primary_constraint"]
