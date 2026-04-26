from __future__ import annotations

from copy import deepcopy
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class RuleSpec(BaseModel):
    rule_id: str
    category: str
    text: str
    resolution_strategy: str = "auto_patch"


class DeliverableSpec(BaseModel):
    name: str
    format: str
    constraints: list[str] = Field(default_factory=list)


class PlannedPhase(BaseModel):
    name: str
    goal: str
    turn_budget: int
    variables_to_extract: list[str] = Field(default_factory=list)


class PlanArtifact(BaseModel):
    task_understanding: str
    deliverables: list[DeliverableSpec] = Field(default_factory=list)
    phases: list[PlannedPhase]
    total_turn_budget: int
    key_rules: list[RuleSpec] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


@dataclass
class StateEntry:
    name: str
    value: str
    category: str
    source: str
    confidence: str
    written_turn: int
    update_mode: str = "initial"
    reason: str | None = None
    stale: bool = False
    value_history: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)


class SemanticVariable(BaseModel):
    name: str
    value: str
    category: str
    source: str = "agent"
    update_mode: str = "initial"
    reason: str | None = None
    derived_from: list[str] = Field(default_factory=list)


class ExtractorArtifact(BaseModel):
    variables: list[SemanticVariable] = Field(default_factory=list)


class Violation(BaseModel):
    rule_id: str
    severity: str
    message: str
    resolution_strategy: str = "follow_rule"
    rollback_turn: int | None = None
    suggested_fix: list[SemanticVariable] = Field(default_factory=list)


class ViolationArtifact(BaseModel):
    violations: list[Violation] = Field(default_factory=list)


@dataclass
class StateSnapshot:
    state_dict: dict[str, StateEntry]
    planned_variables: dict[str, dict[str, bool]]
    discovered_variables: set[str]


class SemanticStateManager:
    """Track task facts and inject them back into the agent loop each turn."""

    def __init__(self, plan: PlanArtifact) -> None:
        self.plan = plan
        self.active_rules = list(plan.key_rules)
        self.phase_schedule = list(plan.phases)
        self.planned_variables: dict[str, dict[str, bool]] = {
            phase.name: {name: False for name in phase.variables_to_extract}
            for phase in plan.phases
        }
        self.discovered_variables: set[str] = set()
        self.state_dict: dict[str, StateEntry] = {}
        self.turn_snapshots: dict[int, StateSnapshot] = {}
        self.pending_notices: list[str] = []
        self.pending_auto_corrections: dict[str, int] = {}
        self.rule_retry_counter: dict[str, int] = {}
        self.unresolvable_rules: set[str] = set()
        self.phase_warning_emitted: set[tuple[str, int]] = set()

    @classmethod
    def build_fallback_plan(
        cls,
        task_description: str,
        uploaded_files: list[str],
        max_turns: int,
    ) -> SemanticStateManager:
        analysis_budget = max(1, min(3, max_turns // 3 or 1))
        execution_budget = max(1, max_turns - analysis_budget)
        key_rules = [
            RuleSpec(
                rule_id="rule_no_silent_contradiction",
                category="implicit_constraint",
                text="Do not silently contradict values already stored in <current_state>.",
                resolution_strategy="rollback",
            ),
            RuleSpec(
                rule_id="rule_emit_state_update",
                category="deliverable_implied",
                text="When a stable fact or intermediate result is learned, emit a <state_update> block.",
                resolution_strategy="auto_patch",
            ),
            RuleSpec(
                rule_id="rule_explicit_overwrite",
                category="implicit_constraint",
                text="If a stored value must change, overwrite it explicitly and provide a reason.",
                resolution_strategy="rollback",
            ),
        ]
        for index, path in enumerate(uploaded_files, start=1):
            key_rules.append(
                RuleSpec(
                    rule_id=f"rule_uploaded_file_{index}",
                    category="explicit_instruction",
                    text=f"Inspect uploaded file before citing it: {path}",
                    resolution_strategy="rollback",
                )
            )
        artifact = PlanArtifact(
            task_understanding=task_description,
            phases=[
                PlannedPhase(
                    name="task_analysis",
                    goal="Infer deliverables, constraints, and likely anchor variables before heavy tool use.",
                    turn_budget=analysis_budget,
                    variables_to_extract=["deliverable_format", "primary_constraint"],
                ),
                PlannedPhase(
                    name="execution",
                    goal="Execute the task while preserving extracted facts and derived values.",
                    turn_budget=execution_budget,
                    variables_to_extract=["final_deliverable_path"],
                ),
            ],
            total_turn_budget=max_turns,
            key_rules=key_rules,
        )
        return cls(artifact)

    def get_phase_for_turn(self, turn_index: int) -> tuple[PlannedPhase, int, int]:
        turn_cursor = 0
        for phase in self.phase_schedule:
            start = turn_cursor + 1
            end = turn_cursor + phase.turn_budget
            if start <= turn_index <= end:
                used = turn_index - start + 1
                remaining = end - turn_index
                return phase, used, remaining
            turn_cursor = end
        final_phase = self.phase_schedule[-1]
        return final_phase, final_phase.turn_budget, 0

    def serialize_phase_status(self, turn_index: int) -> str:
        phase, used, remaining = self.get_phase_for_turn(turn_index)
        lines = [
            "<phase_status>",
            f"phase={phase.name}",
            f"used_turns={used}",
            f"remaining_turns={remaining}",
        ]
        for variable_name, extracted in self.planned_variables.get(phase.name, {}).items():
            lines.append(f"{'✓' if extracted else '✗'} {variable_name}")
        lines.append("</phase_status>")
        return "\n".join(lines)

    def serialize_notices(self) -> str:
        if not self.pending_notices:
            return ""
        lines = ["<semantic_state_notices>", *self.pending_notices, "</semantic_state_notices>"]
        return "\n".join(lines)

    def serialize_current_state(self) -> str:
        lines = ["<current_state>"]
        stale_lines: list[str] = []
        for entry in self.state_dict.values():
            summary = (
                f"{entry.name} = {entry.value} | category={entry.category} | source={entry.source} "
                f"| confidence={entry.confidence} | written_turn={entry.written_turn}"
            )
            if entry.value_history:
                summary += f" | overwritten={len(entry.value_history)}"
            if entry.stale:
                stale_lines.append(f"STALE {summary}")
            else:
                lines.append(summary)
        lines.extend(stale_lines)
        lines.append("</current_state>")
        return "\n".join(lines)

    def serialize_active_rules(self) -> str:
        lines = ["<active_rules>"]
        for rule in self.active_rules:
            lines.append(
                f"{rule.rule_id} | category={rule.category} | strategy={rule.resolution_strategy} | text={rule.text}"
            )
        lines.append("</active_rules>")
        return "\n".join(lines)

    def build_turn_context(self, turn_index: int) -> str:
        blocks = [
            self.serialize_notices(),
            self.serialize_phase_status(turn_index),
            self.serialize_current_state(),
            self.serialize_active_rules(),
        ]
        return "\n\n".join(block for block in blocks if block)

    def clear_pending_notices(self) -> None:
        self.pending_notices.clear()

    def snapshot_turn(self, turn_index: int) -> None:
        self.turn_snapshots[turn_index] = StateSnapshot(
            state_dict=deepcopy(self.state_dict),
            planned_variables=deepcopy(self.planned_variables),
            discovered_variables=set(self.discovered_variables),
        )

    def parse_state_update(self, assistant_text: str) -> list[dict[str, Any]]:
        match = re.search(r"<state_update>(.*?)</state_update>", assistant_text, flags=re.DOTALL)
        if not match:
            return []

        try:
            root = ET.fromstring(f"<state_update>{match.group(1)}</state_update>")
        except ET.ParseError:
            return []

        updates: list[dict[str, Any]] = []
        for child in root.findall("variable"):
            payload = dict(child.attrib)
            payload["derived_from"] = [
                item.strip() for item in payload.get("derived_from", "").split(",") if item.strip()
            ]
            updates.append(payload)
        return updates

    def parse_disputes(self, assistant_text: str) -> list[dict[str, str]]:
        disputes: list[dict[str, str]] = []
        for match in re.finditer(r'<dispute\s+rule_id="([^"]+)"\s+reason="([^"]*)"\s*/?>', assistant_text):
            disputes.append({"rule_id": match.group(1), "reason": match.group(2)})
        return disputes

    def apply_updates(self, updates: list[dict[str, Any]], *, written_turn: int, confidence: str) -> None:
        for payload in updates:
            name = payload["name"]
            update_mode = payload.get("update_mode", "initial")
            derived_from = payload.get("derived_from", [])
            if name in self.state_dict:
                update_mode = "overwrite"

            if update_mode == "overwrite" and name in self.state_dict:
                entry = self.state_dict[name]
                entry.value_history.append(entry.value)
                entry.value = payload["value"]
                entry.category = payload["category"]
                entry.source = payload.get("source", entry.source)
                entry.confidence = confidence
                entry.written_turn = written_turn
                entry.update_mode = update_mode
                entry.reason = payload.get("reason")
                entry.stale = False
                entry.derived_from = derived_from
                self._mark_dependents_stale(name)
                if entry.category == "data_fact":
                    self.pending_notices.append(
                        f'<warn kind="data_fact_overwrite" variable="{name}" turn="{written_turn}"/>'
                    )
            else:
                entry = StateEntry(
                    name=name,
                    value=payload["value"],
                    category=payload["category"],
                    source=payload.get("source", "agent"),
                    confidence=confidence,
                    written_turn=written_turn,
                    update_mode=update_mode,
                    reason=payload.get("reason"),
                    derived_from=derived_from,
                )
                self.state_dict[name] = entry

            self._mark_variable_completed(name)
            if name not in self._all_planned_variable_names():
                self.discovered_variables.add(name)

    def restore_snapshot(self, turn_index: int) -> bool:
        snapshot = self.turn_snapshots.get(turn_index)
        if snapshot is None:
            return False
        self.state_dict = deepcopy(snapshot.state_dict)
        self.planned_variables = deepcopy(snapshot.planned_variables)
        self.discovered_variables = set(snapshot.discovered_variables)
        return True

    def register_extractor_artifact(self, artifact: ExtractorArtifact, *, written_turn: int) -> None:
        updates = [variable.model_dump() for variable in artifact.variables]
        if updates:
            self.apply_updates(updates, written_turn=written_turn, confidence="extractor")

    def handle_disputes(self, disputes: list[dict[str, str]], *, turn_index: int) -> list[dict[str, str]]:
        outcomes: list[dict[str, str]] = []
        for dispute in disputes:
            rule_id = dispute["rule_id"]
            correction_turn = self.pending_auto_corrections.get(rule_id)
            if correction_turn is None:
                continue
            restored = self.restore_snapshot(correction_turn)
            if restored:
                self.pending_notices.append(
                    f'<state_rollback rule_id="{rule_id}" rollback_turn="{correction_turn}" '
                    f'reason="{self._escape_attr(dispute["reason"])}"/>'
                )
            del self.pending_auto_corrections[rule_id]
            self.rule_retry_counter[rule_id] = self.rule_retry_counter.get(rule_id, 0) + 1
            outcomes.append({"rule_id": rule_id, "action": "dispute_rollback"})
        self.check_phase_boundary(turn_index)
        return outcomes

    def handle_violations(self, artifact: ViolationArtifact, *, turn_index: int) -> dict[str, Any]:
        if not artifact.violations:
            self.rule_retry_counter.clear()
            self.check_phase_boundary(turn_index)
            return {"action": "none", "violations": []}

        outcomes: list[dict[str, Any]] = []
        for violation in artifact.violations:
            if violation.rule_id in self.unresolvable_rules:
                continue

            configured_strategy = self._resolution_strategy_for_rule(violation.rule_id)
            action = configured_strategy if violation.resolution_strategy == "follow_rule" else violation.resolution_strategy
            retry_count = self.rule_retry_counter.get(violation.rule_id, 0)
            if action == "auto_patch" and retry_count >= 1:
                action = "rollback"

            if action == "auto_patch" and violation.suggested_fix:
                self.rule_retry_counter[violation.rule_id] = retry_count + 1
                self.apply_updates(
                    [item.model_dump() for item in violation.suggested_fix],
                    written_turn=turn_index,
                    confidence="overridden_by_enforcer",
                )
                self.pending_auto_corrections[violation.rule_id] = turn_index
                self.pending_notices.append(
                    f'<auto_correction rule_id="{violation.rule_id}" reason="{self._escape_attr(violation.message)}"/>'
                )
                outcomes.append({"rule_id": violation.rule_id, "action": "auto_patch"})
                continue

            rollback_count = retry_count + 1
            self.rule_retry_counter[violation.rule_id] = rollback_count
            if rollback_count >= 3:
                self.unresolvable_rules.add(violation.rule_id)
                self.pending_notices.append(
                    f'<circuit_breaker rule_id="{violation.rule_id}" status="unresolvable" '
                    f'reason="{self._escape_attr(violation.message)}"/>'
                )
                outcomes.append({"rule_id": violation.rule_id, "action": "circuit_breaker"})
                continue

            rollback_to = violation.rollback_turn if violation.rollback_turn is not None else turn_index
            restored = self.restore_snapshot(rollback_to)
            if restored:
                self.pending_notices.append(
                    f'<state_rollback rule_id="{violation.rule_id}" rollback_turn="{rollback_to}" '
                    f'reason="{self._escape_attr(violation.message)}"/>'
                )
            outcomes.append({"rule_id": violation.rule_id, "action": "rollback"})

        self.check_phase_boundary(turn_index)
        return {"action": "handled", "violations": outcomes}

    def _mark_dependents_stale(self, updated_name: str) -> None:
        queue = [updated_name]
        visited = {updated_name}
        while queue:
            current = queue.pop(0)
            for entry in self.state_dict.values():
                if current in entry.derived_from and entry.name not in visited:
                    entry.stale = True
                    visited.add(entry.name)
                    queue.append(entry.name)

    def _mark_variable_completed(self, variable_name: str) -> None:
        for variable_map in self.planned_variables.values():
            if variable_name in variable_map:
                variable_map[variable_name] = True
                return

    def _all_planned_variable_names(self) -> set[str]:
        names: set[str] = set()
        for variable_map in self.planned_variables.values():
            names.update(variable_map)
        return names

    def _resolution_strategy_for_rule(self, rule_id: str) -> str:
        for rule in self.active_rules:
            if rule.rule_id == rule_id:
                return rule.resolution_strategy
        return "auto_patch"

    def check_phase_boundary(self, turn_index: int) -> None:
        phase, _used, remaining = self.get_phase_for_turn(turn_index)
        if remaining != 0:
            return
        missing = [
            name for name, extracted in self.planned_variables.get(phase.name, {}).items()
            if not extracted
        ]
        if not missing:
            return
        marker = (phase.name, turn_index)
        if marker in self.phase_warning_emitted:
            return
        self.phase_warning_emitted.add(marker)
        self.pending_notices.append(
            f'<phase_warning phase="{phase.name}" missing="{", ".join(missing)}"/>'
        )

    def _escape_attr(self, value: str) -> str:
        return value.replace('"', "'")

    def serialize_persistent_state(self) -> str:
        payload = {
            "plan": self.plan.model_dump(),
            "planned_variables": self.planned_variables,
            "discovered_variables": sorted(self.discovered_variables),
            "active_rules": [rule.model_dump() for rule in self.active_rules],
            "state_dict": {name: asdict(entry) for name, entry in self.state_dict.items()},
            "rule_retry_counter": self.rule_retry_counter,
            "unresolvable_rules": sorted(self.unresolvable_rules),
            "pending_notices": self.pending_notices,
            "pending_auto_corrections": self.pending_auto_corrections,
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)
