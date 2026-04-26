# Stirrup Autoresearch Adaptation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold Stirrup so autoresearch can baseline and iterate on a framework-level semantic-state tracking experiment for GDPVal runs.

**Architecture:** Keep GDPVal evaluation logic unchanged and add a thin autoresearch wrapper around it. Implement the semantic-state feature inside `src/stirrup/core/semantic_state.py`, wire it into `Agent.run()` and the base system prompt, and persist autoresearch control artifacts under `autoresearch/` so bootstrap and loop stages can run without rereading planning context.

**Tech Stack:** Python 3.12, `uv`, Pydantic, Stirrup agent runtime, GDPVal evaluation entrypoint (`python -m evals.gdpval`).

---

## File Map

- Create `autoresearch/profile.yaml`: frozen runtime profile and experiment matrix for this spec.
- Create `autoresearch/state.yaml`: planning-stage state handoff for bootstrap.
- Create `autoresearch/results.tsv`: tabular run ledger with the exact columns frozen by the spec.
- Create `autoresearch/ledger.jsonl`: append-only event ledger for bootstrap/loop bookkeeping.
- Create `scripts/autoresearch_run.py`: thin adapter that runs GDPVal, writes a run log, and emits `metric=<integer>` on stdout.
- Create `evals/gdpval/task_lists/fast_subset.txt`: concrete fast subset task IDs for baseline and smoke runs.
- Create `src/stirrup/core/semantic_state.py`: planner artifact models, semantic-state manager, XML parsing, stale propagation, and turn-context serialization.
- Modify `src/stirrup/core/agent.py`: pre-run planner call, per-turn state injection, post-turn state application, summarization-safe persistence.
- Modify `src/stirrup/prompts/base_system_prompt.txt`: instruct the model to emit `<state_update>` blocks and respect injected state/rules.

## Scope Notes

- Thin adapter is applicable because the approved spec is `v1-bootstrap-fit` and GDPVal is the experiment entrypoint.
- No changes to `evals/` logic beyond adding `fast_subset.txt`; the spec explicitly forbids modifying GDPVal task loading, grading, or result formatting.
- No test-file edits are planned because `tests/` is outside the allowed edit scope. Verification is command-based.
- Log extraction is from adapter stdout, not from a JSONL artifact. The adapter must print exactly one `metric=<integer>` line.

### Task 1: Profile Generation

**Files:**
- Create: `autoresearch/profile.yaml`
- Verify: `autoresearch/profile.yaml`

- [ ] **Step 1: Write or update the artifact**

Write `autoresearch/profile.yaml` with this exact content:

```yaml
spec_path: docs/autoresearch/specs/2026-04-16-semantic-state-tracking-design.md
plan_path: docs/autoresearch/plans/2026-04-26-semantic-state-tracking-plan.md
compatibility_label: v1-bootstrap-fit

runtime:
  manager: uv
  env_prep_command: uv pip install -e '.[all,gdpval]'
  entry_command: python scripts/autoresearch_run.py --task-ids-file evals/gdpval/task_lists/fast_subset.txt
  timeout_seconds: 1200

experiment:
  time_budget_seconds: 12000
  metric_name: aggregate_score
  metric_direction: higher
  variables:
    state_trigger:
      - every_turn
      - after_file_parse
      - after_tool_success
    extractor_freq:
      - every_turn
      - every_3_turns
      - only_when_agent_silent
    inject_position:
      - system_message
      - user_message_prefix
      - after_tool_result
    state_granularity:
      - data_fact+task_constraint
      - data_fact+task_constraint+env_state

edit_scope:
  allowed_paths:
    - src/stirrup/prompts/base_system_prompt.txt
    - src/stirrup/core/semantic_state.py
    - src/stirrup/core/agent.py
  readonly_paths:
    - evals/
    - src/stirrup/clients/
    - src/stirrup/tools/
    - tests/
    - pyproject.toml
  primary_edit_target: src/stirrup/prompts/base_system_prompt.txt

baseline:
  must_run_first: true
  protocol: Run current codebase (no semantic state changes) on fast_subset tasks with grading enabled. Record aggregate_score as reference.
  baseline_description: Current Stirrup agent with unmodified base_system_prompt.txt and no semantic state management.

git_policy:
  branch_prefix: autoresearch/semantic-state
  commit_before_run: true
  keep_commit_strategy: keep-current-commit
  discard_strategy: hard-reset-to-pre-run-commit
  crash_strategy: keep-crash-commit-for-inspection

logging:
  run_log_path: logs/autoresearch/<branch>/<timestamp>/run.log
  metric_source: stdout
  summary_extract_command: python3 -c "import re, sys
for line in sys.stdin:
    m = re.search(r'metric=(\\d+)', line)
    if m:
        print(m.group(1))"
  results_columns:
    - run_id
    - commit_sha
    - aggregate_score
    - delta_vs_baseline
    - state_trigger
    - extractor_freq
    - inject_position
    - state_granularity
    - runtime_seconds
    - notes
```

- [ ] **Step 2: Run verification command**

Run: `sed -n '1,240p' autoresearch/profile.yaml`
Expected: the file contains `compatibility_label: v1-bootstrap-fit`, `entry_command: python scripts/autoresearch_run.py --task-ids-file evals/gdpval/task_lists/fast_subset.txt`, and the four experiment variable lists exactly as shown above.

- [ ] **Step 3: Record any additional readiness check needed for this artifact**

Run: `python3 - <<'PY'\nimport yaml\nfrom pathlib import Path\npath = Path('autoresearch/profile.yaml')\ndata = yaml.safe_load(path.read_text())\nassert data['runtime']['manager'] == 'uv'\nassert data['experiment']['metric_name'] == 'aggregate_score'\nassert data['logging']['metric_source'] == 'stdout'\nprint('profile-ok')\nPY`
Expected: `profile-ok`

- [ ] **Step 4: Commit**

```bash
git add autoresearch/profile.yaml
git commit -m "chore: add autoresearch profile"
```

### Task 2: State, Results, and Ledger Scaffolding

**Files:**
- Create: `autoresearch/state.yaml`
- Create: `autoresearch/results.tsv`
- Create: `autoresearch/ledger.jsonl`
- Verify: `autoresearch/state.yaml`
- Verify: `autoresearch/results.tsv`
- Verify: `autoresearch/ledger.jsonl`

- [ ] **Step 1: Write or update the artifact**

Write `autoresearch/state.yaml` with this exact content:

```yaml
current_stage: bootstrap
stage_status: pending
active_spec_path: docs/autoresearch/specs/2026-04-16-semantic-state-tracking-design.md
active_plan_path: docs/autoresearch/plans/2026-04-26-semantic-state-tracking-plan.md
next_allowed_skills:
  - autoresearch-bootstrap
rollback_target: null
blocker_reason: null
baseline_ref: null
best_ref: null
profile_status: pending
```

Write `autoresearch/results.tsv` with this exact content:

```tsv
run_id	commit_sha	aggregate_score	delta_vs_baseline	state_trigger	extractor_freq	inject_position	state_granularity	runtime_seconds	notes
```

Write `autoresearch/ledger.jsonl` with this exact content:

```jsonl
{"event":"planning_completed","spec_path":"docs/autoresearch/specs/2026-04-16-semantic-state-tracking-design.md","plan_path":"docs/autoresearch/plans/2026-04-26-semantic-state-tracking-plan.md","stage":"bootstrap"}
```

- [ ] **Step 2: Run verification command**

Run: `sed -n '1,120p' autoresearch/state.yaml && sed -n '1,5p' autoresearch/results.tsv && sed -n '1,5p' autoresearch/ledger.jsonl`
Expected: `current_stage: bootstrap` appears in `state.yaml`, the TSV header has exactly 10 tab-separated columns, and the ledger contains one JSON object with `"event":"planning_completed"`.

- [ ] **Step 3: Record any additional readiness check needed for this artifact**

Run: `python3 - <<'PY'\nimport json\nfrom pathlib import Path\nledger_line = Path('autoresearch/ledger.jsonl').read_text().strip()\nrecord = json.loads(ledger_line)\nassert record['event'] == 'planning_completed'\nassert Path('autoresearch/results.tsv').read_text().count('\\t') == 9\nprint('state-artifacts-ok')\nPY`
Expected: `state-artifacts-ok`

- [ ] **Step 4: Commit**

```bash
git add autoresearch/state.yaml autoresearch/results.tsv autoresearch/ledger.jsonl
git commit -m "chore: scaffold autoresearch state artifacts"
```

### Task 3: Fast Subset and Thin Adapter

**Files:**
- Create: `evals/gdpval/task_lists/fast_subset.txt`
- Create: `scripts/autoresearch_run.py`
- Verify: `evals/gdpval/task_lists/fast_subset.txt`
- Verify: `scripts/autoresearch_run.py`

- [ ] **Step 1: Write or update the artifact**

Write `evals/gdpval/task_lists/fast_subset.txt` with this exact content:

```text
05389f78-589a-473c-a4ae-67c61050bfca
4c18ebae-dfaa-4b76-b10c-61fcdf26734c
a45bc83b-22f9-4def-8d89-9c5661b2b86f
```

Write `scripts/autoresearch_run.py` with this exact content:

```python
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
    print(f"run_dir={run_dir}")
    print(f"log_path={log_path}")
    print(f"metric={aggregate_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run verification command**

Run: `python3 -m py_compile scripts/autoresearch_run.py && sed -n '1,20p' evals/gdpval/task_lists/fast_subset.txt`
Expected: no output from `py_compile`; the subset file prints exactly three task IDs, one per line.

- [ ] **Step 3: Record any additional readiness check needed for this artifact**

Run: `python3 scripts/autoresearch_run.py --help | sed -n '1,40p'`
Expected: help output includes `--task-ids-file`, `--output-root`, `--grading-model`, and the description `Thin autoresearch wrapper around python -m evals.gdpval`.

- [ ] **Step 4: Commit**

```bash
git add evals/gdpval/task_lists/fast_subset.txt scripts/autoresearch_run.py
git commit -m "chore: add autoresearch runner adapter"
```

### Task 4: Semantic State Manager Implementation

**Files:**
- Create: `src/stirrup/core/semantic_state.py`
- Verify: `src/stirrup/core/semantic_state.py`

- [ ] **Step 1: Write or update the artifact**

Write `src/stirrup/core/semantic_state.py` with this exact content:

```python
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class PlannedPhase(BaseModel):
    name: str
    goal: str
    turn_budget: int
    variables_to_extract: list[str] = Field(default_factory=list)


class PlanArtifact(BaseModel):
    task_understanding: str
    deliverables: list[dict[str, Any]] = Field(default_factory=list)
    phases: list[PlannedPhase]
    total_turn_budget: int
    key_rules: list[str] = Field(default_factory=list)
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


class SemanticStateManager:
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

    @classmethod
    def build_fallback_plan(cls, task_description: str, uploaded_files: list[str], max_turns: int) -> "SemanticStateManager":
        file_rules = [f"Inspect uploaded file: {path}" for path in uploaded_files]
        phase_a = PlannedPhase(
            name="task_analysis",
            goal="Infer deliverables and extract key constraints before tool-heavy work.",
            turn_budget=max(1, min(3, max_turns // 3)),
            variables_to_extract=["deliverable_format", "primary_constraint"],
        )
        phase_b = PlannedPhase(
            name="execution",
            goal="Execute the task while preserving extracted facts and constraints.",
            turn_budget=max_turns - phase_a.turn_budget,
            variables_to_extract=["final_deliverable_path"],
        )
        artifact = PlanArtifact(
            task_understanding=task_description,
            deliverables=[],
            phases=[phase_a, phase_b],
            total_turn_budget=max_turns,
            key_rules=[
                "Do not contradict values already stored in current_state without an overwrite reason.",
                "Emit <state_update> whenever a stable task fact, constraint, or intermediate result is learned.",
                *file_rules,
            ],
            risk_flags=[],
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
        lines = [f"<phase_status>", f"phase={phase.name}", f"used_turns={used}", f"remaining_turns={remaining}"]
        for variable_name, is_done in self.planned_variables.get(phase.name, {}).items():
            mark = "✓" if is_done else "✗"
            lines.append(f"{mark} {variable_name}")
        lines.append("</phase_status>")
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
        lines.extend(self.active_rules)
        lines.append("</active_rules>")
        return "\n".join(lines)

    def build_turn_context(self, turn_index: int) -> str:
        return "\n\n".join(
            [
                self.serialize_phase_status(turn_index),
                self.serialize_current_state(),
                self.serialize_active_rules(),
            ]
        )

    def parse_state_update(self, assistant_text: str) -> list[dict[str, Any]]:
        match = re.search(r"<state_update>(.*?)</state_update>", assistant_text, flags=re.DOTALL)
        if not match:
            return []
        wrapped = f"<state_update>{match.group(1)}</state_update>"
        root = ET.fromstring(wrapped)
        updates: list[dict[str, Any]] = []
        for child in root.findall("variable"):
            payload = dict(child.attrib)
            payload["derived_from"] = [
                item.strip() for item in payload.get("derived_from", "").split(",") if item.strip()
            ]
            updates.append(payload)
        return updates

    def apply_updates(self, updates: list[dict[str, Any]], *, written_turn: int, confidence: str) -> None:
        for payload in updates:
            name = payload["name"]
            derived_from = payload.get("derived_from", [])
            update_mode = payload.get("update_mode", "initial")
            reason = payload.get("reason")
            if name in self.state_dict:
                update_mode = "overwrite"
            if update_mode == "overwrite" and name in self.state_dict:
                existing = self.state_dict[name]
                existing.value_history.append(existing.value)
                existing.value = payload["value"]
                existing.category = payload["category"]
                existing.source = payload.get("source", existing.source)
                existing.confidence = confidence
                existing.written_turn = written_turn
                existing.reason = reason
                existing.stale = False
                existing.derived_from = derived_from
                self._mark_dependents_stale(name)
                entry = existing
            else:
                entry = StateEntry(
                    name=name,
                    value=payload["value"],
                    category=payload["category"],
                    source=payload.get("source", "agent"),
                    confidence=confidence,
                    written_turn=written_turn,
                    update_mode=update_mode,
                    reason=reason,
                    derived_from=derived_from,
                )
                self.state_dict[name] = entry
            self._mark_variable_completed(name)
            if name not in self._all_planned_variable_names():
                self.discovered_variables.add(name)

    def _mark_dependents_stale(self, updated_name: str) -> None:
        for entry in self.state_dict.values():
            if updated_name in entry.derived_from:
                entry.stale = True

    def _mark_variable_completed(self, variable_name: str) -> None:
        for phase_name, variable_map in self.planned_variables.items():
            if variable_name in variable_map:
                variable_map[variable_name] = True
                return

    def _all_planned_variable_names(self) -> set[str]:
        names: set[str] = set()
        for variable_map in self.planned_variables.values():
            names.update(variable_map)
        return names

    def serialize_persistent_state(self) -> str:
        serializable = {
            "plan": self.plan.model_dump(),
            "planned_variables": self.planned_variables,
            "discovered_variables": sorted(self.discovered_variables),
            "active_rules": self.active_rules,
            "state_dict": {
                key: {
                    "name": value.name,
                    "value": value.value,
                    "category": value.category,
                    "source": value.source,
                    "confidence": value.confidence,
                    "written_turn": value.written_turn,
                    "update_mode": value.update_mode,
                    "reason": value.reason,
                    "stale": value.stale,
                    "value_history": value.value_history,
                    "derived_from": value.derived_from,
                }
                for key, value in self.state_dict.items()
            },
        }
        return json.dumps(serializable, ensure_ascii=True, indent=2)
```

- [ ] **Step 2: Run verification command**

Run: `python3 -m py_compile src/stirrup/core/semantic_state.py`
Expected: no output and exit code `0`.

- [ ] **Step 3: Record any additional readiness check needed for this artifact**

Run: `python3 - <<'PY'\nfrom stirrup.core.semantic_state import SemanticStateManager\nmanager = SemanticStateManager.build_fallback_plan('compare vendors', ['quotes_1.docx'], 8)\nupdates = manager.parse_state_update('<state_update><variable name=\"unit_price\" value=\"USD 285\" category=\"data_fact\" source=\"quotes_1.docx:table1\"/></state_update>')\nmanager.apply_updates(updates, written_turn=1, confidence='agent')\nprint('unit_price' in manager.serialize_current_state())\nPY`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add src/stirrup/core/semantic_state.py
git commit -m "feat: add semantic state manager"
```

### Task 5: Agent Loop and Prompt Integration

**Files:**
- Modify: `src/stirrup/core/agent.py`
- Modify: `src/stirrup/prompts/base_system_prompt.txt`
- Verify: `src/stirrup/core/agent.py`
- Verify: `src/stirrup/prompts/base_system_prompt.txt`

- [ ] **Step 1: Write or update the artifact**

Update `src/stirrup/core/agent.py` with these exact edits:

1. Extend the imports near the top of the file with:

```python
from stirrup.core.semantic_state import PlanArtifact, SemanticStateManager
```

2. In `Agent.__init__`, immediately after `self._current_run_state = None`, add:

```python
        self._semantic_state_manager: SemanticStateManager | None = None
```

3. In `run()`, immediately after `msgs.append(SystemMessage(content=full_system_prompt))`, add:

```python
            uploaded_files = []
            state = _SESSION_STATE.get(None)
            if state and state.uploaded_file_paths:
                uploaded_files = list(state.uploaded_file_paths)
            task_description = init_msgs if isinstance(init_msgs, str) else str(init_msgs[-1].content)
            self._semantic_state_manager = SemanticStateManager.build_fallback_plan(
                task_description=task_description,
                uploaded_files=uploaded_files,
                max_turns=self._max_turns,
            )
            msgs.append(UserMessage(content=self._semantic_state_manager.build_turn_context(1)))
```

4. In the `for i in range(start_turn, self._max_turns):` loop, immediately before the `assistant_message, tool_messages, finish_params = await self.step(` line, add:

```python
            if self._semantic_state_manager is not None and i > start_turn:
                msgs.append(UserMessage(content=self._semantic_state_manager.build_turn_context(i + 1)))
```

5. In the same loop, immediately after `msgs.extend([assistant_message, *tool_messages, *user_messages])`, add:

```python
            if self._semantic_state_manager is not None:
                updates = self._semantic_state_manager.parse_state_update(str(assistant_message.content))
                if updates:
                    self._semantic_state_manager.apply_updates(
                        updates,
                        written_turn=i + 1,
                        confidence="agent",
                    )
                    run_metadata.setdefault("semantic_state", []).append(
                        self._semantic_state_manager.serialize_persistent_state()
                    )
```

Update `src/stirrup/prompts/base_system_prompt.txt` by appending this exact block to the end of the file:

```text

Semantic state protocol:
- When you learn a stable fact, task constraint, environment fact, or intermediate computed result, append a <state_update> block to your assistant message.
- Use XML of the form:
  <state_update>
    <variable name="example_name" value="example value" category="data_fact" source="document.ext:section"/>
  </state_update>
- Allowed categories are `data_fact`, `task_constraint`, `intermediate_result`, and `env_state`.
- If you revise a previously declared variable, include `update_mode="overwrite"` and `reason="..."`.
- Never contradict the injected <current_state> block silently. Either reuse the stored value or overwrite it explicitly with a reason.
```

- [ ] **Step 2: Run verification command**

Run: `python3 -m py_compile src/stirrup/core/agent.py && tail -n 12 src/stirrup/prompts/base_system_prompt.txt`
Expected: `py_compile` exits `0`; the prompt tail contains the `Semantic state protocol:` block and the four allowed categories.

- [ ] **Step 3: Record any additional readiness check needed for this artifact**

Run: `rg -n "SemanticStateManager|semantic_state|Semantic state protocol" src/stirrup/core/agent.py src/stirrup/prompts/base_system_prompt.txt`
Expected: one import line, one `_semantic_state_manager` field line, one `build_turn_context` injection before turn execution, one `parse_state_update` application block, and one prompt protocol section.

- [ ] **Step 4: Commit**

```bash
git add src/stirrup/core/agent.py src/stirrup/prompts/base_system_prompt.txt
git commit -m "feat: wire semantic state into agent loop"
```

### Task 6: Log Extraction and Loop Readiness Checks

**Files:**
- Verify: `scripts/autoresearch_run.py`
- Verify: `autoresearch/profile.yaml`
- Verify: `logs/autoresearch/`

- [ ] **Step 1: Write or update the artifact**

No new artifact is created in this task. Use the exact verification commands below to prove the adapter and profile satisfy the readiness contract.

- [ ] **Step 2: Run verification command**

Run: `printf 'noise\\nmetric=17\\n' | python3 -c "import re, sys\nfor line in sys.stdin:\n    m = re.search(r'metric=(\\d+)', line)\n    if m:\n        print(m.group(1))"`
Expected: `17`

- [ ] **Step 3: Record any additional readiness check needed for this artifact**

Run: `OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY}" uv run python scripts/autoresearch_run.py --task-ids-file evals/gdpval/task_lists/fast_subset.txt | tee /tmp/autoresearch-smoke.out`
Expected: stdout includes exactly one line matching `metric=<integer>`, the command exits `0` within the 1200-second timeout, and a run log exists under `logs/autoresearch/<branch>/<timestamp>/run.log`.

- [ ] **Step 4: Commit**

```bash
git add autoresearch/profile.yaml autoresearch/state.yaml autoresearch/results.tsv autoresearch/ledger.jsonl \
  scripts/autoresearch_run.py evals/gdpval/task_lists/fast_subset.txt \
  src/stirrup/core/semantic_state.py src/stirrup/core/agent.py src/stirrup/prompts/base_system_prompt.txt
git commit -m "chore: verify autoresearch loop readiness"
```

## Self-Review

- Spec coverage: `autoresearch/profile.yaml` covers frozen fields and experiment variables; `autoresearch/state.yaml`, `results.tsv`, and `ledger.jsonl` cover planning handoff; `scripts/autoresearch_run.py` and `fast_subset.txt` cover the thin adapter and fast subset; `src/stirrup/core/semantic_state.py`, `src/stirrup/core/agent.py`, and `src/stirrup/prompts/base_system_prompt.txt` cover the runtime semantic-state flow.
- Placeholder scan: no `TBD`, `TODO`, or deferred placeholders are used; every write step contains concrete content and every verification step contains an exact command.
- Artifact and identifier consistency: all paths consistently use `autoresearch/profile.yaml`, `scripts/autoresearch_run.py`, `evals/gdpval/task_lists/fast_subset.txt`, and `docs/autoresearch/plans/2026-04-26-semantic-state-tracking-plan.md`.
- Verification coverage: every created or modified artifact has at least one concrete verification command, and the final task verifies the metric extraction pipeline against both synthetic log text and a real GDPVal adapter invocation.
