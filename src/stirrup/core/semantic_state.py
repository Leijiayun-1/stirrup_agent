from __future__ import annotations

from copy import deepcopy
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field

ENV_STATE_SOURCE_PRIORITY = {
    "agent": 10,
    "extractor": 20,
    "tool_provider": 70,
    "tool_runtime": 80,
    "finish_tool": 90,
    "runtime": 100,
}


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
    evidence_refs: list[str] = field(default_factory=list)
    matched_planned_variable_id: str | None = None


@dataclass
class PlannedVariableSpec:
    id: str
    canonical_name: str
    description: str
    category: str = "intermediate_result"
    value_type: str = "text"
    required: bool = True
    aliases: list[str] = field(default_factory=list)
    completion_policy: str = "state_only"
    evidence_policy: str = "optional"


@dataclass
class PlannedVariableProgress:
    spec: PlannedVariableSpec
    status: str = "pending"
    bound_state_variables: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    reason: str | None = None
    updated_turn: int | None = None

    @property
    def satisfied(self) -> bool:
        return self.status == "satisfied"


@dataclass
class EvidenceRecord:
    evidence_id: str
    turn: int
    source: str
    kind: str
    content_excerpt: str
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SemanticVariable(BaseModel):
    name: str
    value: str
    category: str
    source: str = "agent"
    update_mode: str = "initial"
    reason: str | None = None
    derived_from: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    matched_planned_variable_id: str | None = None


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
    planned_variables: dict[str, dict[str, PlannedVariableProgress]]
    discovered_variables: set[str]
    evidence_ledger: list[EvidenceRecord]


class SemanticStateManager:
    """Track task facts and inject them back into the agent loop each turn."""

    def __init__(self, plan: PlanArtifact) -> None:
        self.plan = plan
        self.active_rules = list(plan.key_rules)
        self.phase_schedule = list(plan.phases)
        self.planned_variables: dict[str, dict[str, PlannedVariableProgress]] = self._build_planned_variables(plan)
        self.discovered_variables: set[str] = set()
        self.state_dict: dict[str, StateEntry] = {}
        self.evidence_ledger: list[EvidenceRecord] = []
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
        for progress in self.planned_variables.get(phase.name, {}).values():
            symbol = {
                "satisfied": "✓",
                "candidate": "?",
                "blocked": "!",
                "stale": "!",
            }.get(progress.status, "✗")
            line = f"{symbol} {progress.spec.canonical_name} | status={progress.status}"
            if progress.bound_state_variables:
                line += f" | bound={','.join(progress.bound_state_variables)}"
            if progress.reason:
                line += f" | reason={progress.reason}"
            lines.append(line)
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
        env_lines = ["<env_state>"]
        for entry in self.state_dict.values():
            summary = (
                f"{entry.name} = {entry.value} | category={entry.category} | source={entry.source} "
                f"| confidence={entry.confidence} | written_turn={entry.written_turn}"
            )
            if entry.value_history:
                summary += f" | overwritten={len(entry.value_history)}"
            if entry.evidence_refs:
                summary += f" | evidence_refs={','.join(entry.evidence_refs[:5])}"
            if entry.matched_planned_variable_id:
                summary += f" | matched_planned_variable_id={entry.matched_planned_variable_id}"
            if entry.stale:
                stale_lines.append(f"STALE {summary}")
            elif entry.category == "env_state":
                env_lines.append(summary)
            else:
                lines.append(summary)
        if len(env_lines) > 1:
            env_lines.append("</env_state>")
            lines.extend(env_lines)
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
            self.serialize_recent_evidence(),
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
            evidence_ledger=deepcopy(self.evidence_ledger),
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
            payload["evidence_refs"] = [
                item.strip() for item in payload.get("evidence_refs", "").split(",") if item.strip()
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
            evidence_refs = payload.get("evidence_refs", [])
            category = payload["category"]
            source = payload.get("source", "agent")
            matched_planned_variable_id = payload.get("matched_planned_variable_id")

            if category == "env_state" and name in self.state_dict:
                existing = self.state_dict[name]
                if existing.category == "env_state" and not self._env_source_can_overwrite(source, existing.source):
                    self.pending_notices.append(
                        f'<warn kind="env_state_low_priority_update_ignored" variable="{name}" '
                        f'source="{self._escape_attr(source)}" existing_source="{self._escape_attr(existing.source)}"/>'
                    )
                    continue

            if name in self.state_dict:
                update_mode = "overwrite"

            if update_mode == "overwrite" and name in self.state_dict:
                entry = self.state_dict[name]
                if entry.category != "env_state" and category != "env_state":
                    entry.value_history.append(entry.value)
                entry.value = payload["value"]
                entry.category = category
                entry.source = source
                entry.confidence = confidence
                entry.written_turn = written_turn
                entry.update_mode = update_mode
                entry.reason = payload.get("reason")
                entry.stale = False
                entry.derived_from = derived_from
                entry.evidence_refs = evidence_refs
                entry.matched_planned_variable_id = matched_planned_variable_id
                self._mark_dependents_stale(name)
                if entry.category == "data_fact":
                    self.pending_notices.append(
                        f'<warn kind="data_fact_overwrite" variable="{name}" turn="{written_turn}"/>'
                    )
            else:
                entry = StateEntry(
                    name=name,
                    value=payload["value"],
                    category=category,
                    source=source,
                    confidence=confidence,
                    written_turn=written_turn,
                    update_mode=update_mode,
                    reason=payload.get("reason"),
                    derived_from=derived_from,
                    evidence_refs=evidence_refs,
                    matched_planned_variable_id=matched_planned_variable_id,
                )
                self.state_dict[name] = entry

            matched = self._update_planned_progress_for_entry(entry, written_turn=written_turn)
            if not matched and name not in self._all_planned_variable_names():
                self.discovered_variables.add(name)

        self._refresh_runtime_planned_progress(written_turn=written_turn)

    def restore_snapshot(self, turn_index: int) -> bool:
        snapshot = self.turn_snapshots.get(turn_index)
        if snapshot is None:
            return False
        self.state_dict = deepcopy(snapshot.state_dict)
        self.planned_variables = deepcopy(snapshot.planned_variables)
        self.discovered_variables = set(snapshot.discovered_variables)
        self.evidence_ledger = deepcopy(snapshot.evidence_ledger)
        return True

    def register_extractor_artifact(self, artifact: ExtractorArtifact, *, written_turn: int) -> None:
        updates = [variable.model_dump() for variable in artifact.variables]
        if updates:
            self.apply_updates(updates, written_turn=written_turn, confidence="extractor")

    def record_runtime_env_state(
        self,
        name: str,
        value: Any,
        *,
        written_turn: int,
        source: str = "runtime",
        reason: str | None = None,
        notice: str | None = None,
    ) -> None:
        """Write a runtime-verified env_state value with source priority protection."""
        if isinstance(value, str):
            serialized_value = value
        else:
            serialized_value = json.dumps(value, ensure_ascii=True, sort_keys=True)

        self.apply_updates(
            [
                {
                    "name": name,
                    "value": serialized_value,
                    "category": "env_state",
                    "source": source,
                    "update_mode": "overwrite" if name in self.state_dict else "initial",
                    "reason": reason,
                    "derived_from": [],
                }
            ],
            written_turn=written_turn,
            confidence="runtime_verified",
        )
        if notice:
            self.pending_notices.append(notice)

    def record_runtime_env_states(
        self,
        items: dict[str, Any],
        *,
        written_turn: int,
        source: str = "runtime",
        reason: str | None = None,
    ) -> None:
        for name, value in items.items():
            self.record_runtime_env_state(
                name,
                value,
                written_turn=written_turn,
                source=source,
                reason=reason,
            )

    def get_env_state_json(self, name: str, default: Any = None) -> Any:
        entry = self.state_dict.get(name)
        if entry is None or entry.category != "env_state":
            return default
        try:
            return json.loads(entry.value)
        except json.JSONDecodeError:
            return default

    def record_evidence(
        self,
        *,
        turn: int,
        source: str,
        kind: str,
        content: str,
        success: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        evidence_id = f"turn_{turn}.evidence_{len(self.evidence_ledger) + 1}"
        self.evidence_ledger.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                turn=turn,
                source=source,
                kind=kind,
                content_excerpt=self._truncate_evidence(content),
                success=success,
                metadata=metadata or {},
            )
        )
        return evidence_id

    def serialize_recent_evidence(self, *, limit: int = 8) -> str:
        if not self.evidence_ledger:
            return ""
        lines = ["<recent_evidence>"]
        for record in self.evidence_ledger[-limit:]:
            success = "" if record.success is None else f" | success={record.success}"
            lines.append(
                f"{record.evidence_id} | turn={record.turn} | source={record.source} "
                f"| kind={record.kind}{success} | content={record.content_excerpt}"
            )
        lines.append("</recent_evidence>")
        return "\n".join(lines)

    def serialize_planned_variables_for_extractor(self) -> str:
        payload: dict[str, Any] = {}
        for phase_name, variable_map in self.planned_variables.items():
            payload[phase_name] = []
            for progress in variable_map.values():
                payload[phase_name].append(
                    {
                        "id": progress.spec.id,
                        "canonical_name": progress.spec.canonical_name,
                        "description": progress.spec.description,
                        "category": progress.spec.category,
                        "value_type": progress.spec.value_type,
                        "aliases": progress.spec.aliases,
                        "completion_policy": progress.spec.completion_policy,
                        "evidence_policy": progress.spec.evidence_policy,
                        "status": progress.status,
                        "bound_state_variables": progress.bound_state_variables,
                    }
                )
        return json.dumps(payload, ensure_ascii=True)

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
                    self._mark_bound_progress_stale(entry.name)

    def _build_planned_variables(
        self,
        plan: PlanArtifact,
    ) -> dict[str, dict[str, PlannedVariableProgress]]:
        planned: dict[str, dict[str, PlannedVariableProgress]] = {}
        for phase in plan.phases:
            planned[phase.name] = {}
            for variable_name in phase.variables_to_extract:
                spec = self._make_planned_variable_spec(phase.name, variable_name)
                planned[phase.name][spec.canonical_name] = PlannedVariableProgress(spec=spec)
        return planned

    def _make_planned_variable_spec(self, phase_name: str, variable_name: str) -> PlannedVariableSpec:
        normalized = self._normalize_identifier(variable_name)
        value_type = "path" if any(token in normalized for token in ("path", "file", "output", "deliverable")) else "text"
        completion_policy = "runtime_verified" if value_type == "path" else "state_only"
        evidence_policy = "runtime_or_tool_verified" if value_type == "path" else "optional"
        aliases = self._default_aliases_for_variable(variable_name, value_type=value_type)
        return PlannedVariableSpec(
            id=f"{phase_name}.{variable_name}",
            canonical_name=variable_name,
            description=f"Planned variable `{variable_name}` for phase `{phase_name}`.",
            category="intermediate_result",
            value_type=value_type,
            aliases=aliases,
            completion_policy=completion_policy,
            evidence_policy=evidence_policy,
        )

    def _default_aliases_for_variable(self, variable_name: str, *, value_type: str) -> list[str]:
        aliases: set[str] = set()
        normalized = self._normalize_identifier(variable_name)
        if value_type == "path":
            aliases.update(
                {
                    "deliverable_path",
                    "final_output_path",
                    "final_file_path",
                    "output_file",
                    "output_path",
                    "report_path",
                    "saved_file_path",
                    "saved_report_path",
                }
            )
        if "format" in normalized:
            aliases.update({"output_format", "deliverable_type", "file_format"})
        if "constraint" in normalized:
            aliases.update({"main_constraint", "required_constraint", "task_requirement"})
        aliases.discard(variable_name)
        return sorted(aliases)

    def _normalize_identifier(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")

    def _all_planned_progress(self) -> list[PlannedVariableProgress]:
        return [progress for variable_map in self.planned_variables.values() for progress in variable_map.values()]

    def _find_planned_progress_for_payload(
        self,
        entry: StateEntry,
    ) -> PlannedVariableProgress | None:
        requested_id = entry.matched_planned_variable_id
        entry_name = self._normalize_identifier(entry.name)
        for progress in self._all_planned_progress():
            spec = progress.spec
            if requested_id and requested_id in {spec.id, spec.canonical_name}:
                return progress
            if entry_name == self._normalize_identifier(spec.canonical_name):
                return progress
            if entry_name in {self._normalize_identifier(alias) for alias in spec.aliases}:
                return progress
        return None

    def _update_planned_progress_for_entry(self, entry: StateEntry, *, written_turn: int) -> bool:
        progress = self._find_planned_progress_for_payload(entry)
        if progress is None:
            return False

        if entry.name not in progress.bound_state_variables:
            progress.bound_state_variables.append(entry.name)
        for evidence_ref in entry.evidence_refs:
            if evidence_ref not in progress.evidence_refs:
                progress.evidence_refs.append(evidence_ref)

        status, reason = self._verify_entry_against_planned_variable(entry, progress)
        self._set_progress_status(progress, status=status, reason=reason, written_turn=written_turn)
        return True

    def _verify_entry_against_planned_variable(
        self,
        entry: StateEntry,
        progress: PlannedVariableProgress,
    ) -> tuple[str, str]:
        if entry.stale:
            return "stale", f"{entry.name} is stale"
        if not str(entry.value).strip():
            return "candidate", f"{entry.name} has no value"

        policy = progress.spec.completion_policy
        if policy == "state_only":
            return "satisfied", f"{entry.name} written to semantic state"

        if policy == "state_with_evidence":
            if entry.evidence_refs:
                return "satisfied", f"{entry.name} written with evidence"
            return "candidate", f"{entry.name} lacks evidence_refs"

        if policy == "runtime_verified":
            if self._path_value_is_finish_validated(entry.value):
                return "satisfied", f"{entry.name} is validated by finish/runtime file state"
            if self._path_value_is_generated(entry.value):
                return "candidate", f"{entry.name} exists in generated files but finish has not validated it"
            return "candidate", f"{entry.name} has not been runtime-verified"

        return "candidate", f"unknown completion policy {policy}"

    def _set_progress_status(
        self,
        progress: PlannedVariableProgress,
        *,
        status: str,
        reason: str,
        written_turn: int,
    ) -> None:
        status_rank = {"pending": 0, "candidate": 1, "blocked": 1, "stale": 1, "satisfied": 2}
        current_rank = status_rank.get(progress.status, 0)
        new_rank = status_rank.get(status, 0)
        if new_rank < current_rank and progress.status == "satisfied":
            return
        progress.status = status
        progress.reason = reason
        progress.updated_turn = written_turn

    def _mark_bound_progress_stale(self, variable_name: str) -> None:
        for progress in self._all_planned_progress():
            if variable_name in progress.bound_state_variables:
                progress.status = "stale"
                progress.reason = f"{variable_name} became stale"

    def _refresh_runtime_planned_progress(self, *, written_turn: int) -> None:
        generated_paths = self._env_path_list("env.generated_files")
        finish_validated_paths = self._env_path_list("env.finish.validated_paths")
        for progress in self._all_planned_progress():
            if progress.spec.completion_policy != "runtime_verified":
                continue

            bound_entries = [
                self.state_dict[name]
                for name in progress.bound_state_variables
                if name in self.state_dict and self.state_dict[name].category != "env_state"
            ]
            if bound_entries:
                best_status = progress.status
                best_reason = progress.reason or ""
                for entry in bound_entries:
                    status, reason = self._verify_entry_against_planned_variable(entry, progress)
                    if status == "satisfied":
                        best_status, best_reason = status, reason
                        break
                    if status == "candidate":
                        best_status, best_reason = status, reason
                self._set_progress_status(progress, status=best_status, reason=best_reason, written_turn=written_turn)
                continue

            if finish_validated_paths:
                if "env.finish.validated_paths" not in progress.bound_state_variables:
                    progress.bound_state_variables.append("env.finish.validated_paths")
                self._set_progress_status(
                    progress,
                    status="satisfied",
                    reason="finish tool validated output path(s)",
                    written_turn=written_turn,
                )
            elif generated_paths:
                if "env.generated_files" not in progress.bound_state_variables:
                    progress.bound_state_variables.append("env.generated_files")
                self._set_progress_status(
                    progress,
                    status="candidate",
                    reason="runtime found generated file(s), but finish has not validated final path",
                    written_turn=written_turn,
                )

    def _path_value_is_finish_validated(self, value: str) -> bool:
        return self._path_value_in_env_list(value, "env.finish.validated_paths")

    def _path_value_is_generated(self, value: str) -> bool:
        return self._path_value_in_env_list(value, "env.generated_files")

    def _path_value_in_env_list(self, value: str, env_name: str) -> bool:
        candidates = self._extract_path_candidates(value)
        if not candidates:
            return False
        env_paths = {self._normalize_path(path) for path in self._env_path_list(env_name)}
        return any(self._normalize_path(candidate) in env_paths for candidate in candidates)

    def _env_path_list(self, env_name: str) -> list[str]:
        value = self.get_env_state_json(env_name, [])
        if isinstance(value, dict) and isinstance(value.get("paths"), list):
            return [str(path) for path in value["paths"]]
        if isinstance(value, list):
            return [str(path) for path in value]
        return []

    def _extract_path_candidates(self, value: str) -> list[str]:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, dict):
            paths = parsed.get("paths") or parsed.get("path") or parsed.get("validated_paths")
            if isinstance(paths, list):
                return [str(item) for item in paths]
            if isinstance(paths, str):
                return [paths]
        return [str(value)]

    def _normalize_path(self, path: str) -> str:
        return path.strip().replace("\\", "/").lstrip("./")

    def _all_planned_variable_names(self) -> set[str]:
        names: set[str] = set()
        for progress in self._all_planned_progress():
            names.add(progress.spec.canonical_name)
            names.update(progress.spec.aliases)
        return names

    def _resolution_strategy_for_rule(self, rule_id: str) -> str:
        for rule in self.active_rules:
            if rule.rule_id == rule_id:
                return rule.resolution_strategy
        return "auto_patch"

    def _env_source_can_overwrite(self, source: str, existing_source: str) -> bool:
        return ENV_STATE_SOURCE_PRIORITY.get(source, 0) >= ENV_STATE_SOURCE_PRIORITY.get(existing_source, 0)

    def check_phase_boundary(self, turn_index: int) -> None:
        phase, _used, remaining = self.get_phase_for_turn(turn_index)
        if remaining != 0:
            return
        missing = [
            progress.spec.canonical_name
            for progress in self.planned_variables.get(phase.name, {}).values()
            if progress.status != "satisfied"
        ]
        if not missing:
            return
        marker = (phase.name, turn_index)
        if marker in self.phase_warning_emitted:
            return
        self.phase_warning_emitted.add(marker)
        details = [
            f"{progress.spec.canonical_name}:{progress.status}:{progress.reason or 'no verified state'}"
            for progress in self.planned_variables.get(phase.name, {}).values()
            if progress.status != "satisfied"
        ]
        self.pending_notices.append(
            f'<phase_warning phase="{phase.name}" missing="{self._escape_attr(", ".join(missing))}" '
            f'details="{self._escape_attr("; ".join(details))}"/>'
        )

    def _escape_attr(self, value: str) -> str:
        return value.replace('"', "'")

    def _truncate_evidence(self, value: str, *, limit: int = 1200) -> str:
        normalized = re.sub(r"\s+", " ", value).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    def serialize_persistent_state(self) -> str:
        payload = {
            "plan": self.plan.model_dump(),
            "planned_variables": {
                phase_name: {
                    variable_name: asdict(progress)
                    for variable_name, progress in variable_map.items()
                }
                for phase_name, variable_map in self.planned_variables.items()
            },
            "discovered_variables": sorted(self.discovered_variables),
            "active_rules": [rule.model_dump() for rule in self.active_rules],
            "state_dict": {name: asdict(entry) for name, entry in self.state_dict.items()},
            "evidence_ledger": [asdict(record) for record in self.evidence_ledger],
            "rule_retry_counter": self.rule_retry_counter,
            "unresolvable_rules": sorted(self.unresolvable_rules),
            "pending_notices": self.pending_notices,
            "pending_auto_corrections": self.pending_auto_corrections,
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)
