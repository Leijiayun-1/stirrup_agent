"""Local AgentIF-OneDay dataset loader and attachment resolver."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("/home/shangguanyike/AgentIF-OneDay/agentif_oneday_data")

SUITE_DEFAULTS: dict[str, tuple[str, str]] = {
    "all": ("data.jsonl", "Attachments/Questions"),
    "data": ("data.jsonl", "Attachments/Questions"),
    "excel": ("ifoneday_excel/excel.jsonl", "ifoneday_excel/Questions/Questions"),
    "pdf": ("ifoneday_pdf/pdf.jsonl", "ifoneday_pdf/Questions"),
    "ppt": ("ifoneday_ppt/ppt.jsonl", "ifoneday_ppt/Questions"),
    "word": ("ifoneday_word/word.jsonl", "ifoneday_word/Questions"),
}


@dataclass(frozen=True)
class AgentIFTask:
    """One AgentIF-OneDay task record."""

    question_id: str
    title: str
    description: str
    attachment_filenames: list[str]
    score_criteria: list[dict[str, Any]]
    reference_answer_description: str = ""
    reference_answer_attachment_filenames: list[str] | None = None

    @classmethod
    def from_json(cls, item: dict[str, Any], *, lineno: int) -> "AgentIFTask":
        qid = str(item.get("question_id", "")).strip()
        if not qid:
            raise ValueError(f"Missing question_id at line {lineno}")

        attachments = [str(x) for x in item.get("attachment_filenames") or [] if str(x).strip()]
        score_criteria = [x for x in item.get("score_criteria") or [] if isinstance(x, dict)]
        ref_attachments = [
            str(x) for x in item.get("reference_answer_attachment_filenames") or [] if str(x).strip()
        ]

        return cls(
            question_id=qid,
            title=str(item.get("title", "")).strip(),
            description=str(item.get("description", "")).strip(),
            attachment_filenames=attachments,
            score_criteria=score_criteria,
            reference_answer_description=str(item.get("reference_answer_description", "")).strip(),
            reference_answer_attachment_filenames=ref_attachments,
        )


def resolve_suite_paths(
    suite: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    task_jsonl: Path | None = None,
    attachment_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve task JSONL, primary attachment dir, and recursive search root."""
    suite_key = suite.lower().strip()
    if suite_key not in SUITE_DEFAULTS:
        known = ", ".join(sorted(SUITE_DEFAULTS))
        raise ValueError(f"Unknown AgentIF suite '{suite}'. Expected one of: {known}")

    default_jsonl, default_attachments = SUITE_DEFAULTS[suite_key]
    resolved_data_root = data_root.resolve()
    resolved_task_jsonl = (task_jsonl or (resolved_data_root / default_jsonl)).resolve()
    resolved_attachment_dir = (attachment_dir or (resolved_data_root / default_attachments)).resolve()
    return resolved_task_jsonl, resolved_attachment_dir, resolved_data_root


def load_tasks(
    path: Path,
    question_ids: list[str] | None = None,
    max_tasks: int | None = None,
) -> list[AgentIFTask]:
    """Load AgentIF-OneDay tasks from a local JSONL manifest."""
    tasks: list[AgentIFTask] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {lineno}: {exc}") from exc
            tasks.append(AgentIFTask.from_json(item, lineno=lineno))

    if question_ids:
        wanted = set(question_ids)
        by_id = {task.question_id: task for task in tasks}
        missing = [qid for qid in question_ids if qid not in by_id]
        if missing:
            raise ValueError(f"Unknown question_id(s): {', '.join(missing)}")
        tasks = [by_id[qid] for qid in question_ids if qid in wanted]

    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    return tasks


def parse_question_ids(raw: str | None) -> list[str]:
    """Parse comma/newline/space separated question ids."""
    if not raw:
        return []
    normalized = raw.replace(",", "\n").replace(" ", "\n")
    return [token.strip() for token in normalized.splitlines() if token.strip()]


def load_question_ids_file(path: Path) -> list[str]:
    """Load question ids from a text file, ignoring blank lines and comments."""
    ids: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            ids.extend(parse_question_ids(line))
    return ids


def build_attachment_index(search_root: Path) -> dict[str, list[Path]]:
    """Index attachment files by filename under search_root."""
    index: dict[str, list[Path]] = {}
    if not search_root.exists():
        return index
    for path in search_root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, []).append(path.resolve())
    return index


def resolve_input_files(
    task: AgentIFTask,
    attachment_dir: Path,
    attachment_index: dict[str, list[Path]],
) -> list[Path]:
    """Resolve a task's attachment filenames to concrete local paths."""
    files: list[Path] = []
    missing: list[str] = []

    for name in task.attachment_filenames:
        direct = (attachment_dir / name).resolve()
        if direct.exists():
            files.append(direct)
            continue

        candidates = attachment_index.get(name, [])
        if len(candidates) == 1:
            files.append(candidates[0])
            continue

        if len(candidates) > 1:
            preferred = [candidate for candidate in candidates if task.question_id in candidate.name]
            files.append((preferred or candidates)[0])
            continue

        missing.append(name)

    if missing:
        raise FileNotFoundError(f"Missing attachments for {task.question_id}: {', '.join(missing)}")

    return files
