"""GDPVal dataset loader and reference file downloader."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HF_ORIGIN = "https://huggingface.co"


def _apply_hf_mirror(url: str) -> str:
    """Replace the HuggingFace origin with HF_ENDPOINT mirror if set."""
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    if endpoint and url.startswith(_HF_ORIGIN):
        return endpoint + url[len(_HF_ORIGIN):]
    return url


def load_gdpval_tasks(
    split: str = "train",
    task_ids: list[str] | None = None,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """Load GDPVal tasks from HuggingFace.

    Args:
        split: Dataset split to load (e.g., "train").
        task_ids: Optional list of task IDs to filter by.
        max_tasks: Optional maximum number of tasks to return (for debugging).

    Returns:
        List of task dicts from the dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Requires installation of the gdpval extra. "
            "Install with: `uv pip install -e '.[gdpval]'`"
        ) from e

    logger.info("Loading GDPVal dataset (split=%s)...", split)
    dataset = load_dataset("openai/gdpval", split=split)

    tasks: list[dict[str, Any]] = list(dataset)

    if task_ids:
        task_id_set = set(task_ids)
        tasks = [t for t in tasks if t.get("task_id") in task_id_set]
        logger.info("Filtered to %d tasks by task_ids", len(tasks))

    if max_tasks is not None:
        tasks = tasks[:max_tasks]
        logger.info("Limited to %d tasks", len(tasks))

    logger.info("Loaded %d tasks", len(tasks))
    return tasks


async def _download_files(
    urls: list[str],
    inline_files: list[Any],
    dest_dir: Path,
) -> list[Path]:
    """Shared download logic for reference and deliverable files.

    Downloads files from URLs and writes inline content to dest_dir.
    Skips files that already exist (resume-friendly).
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError("httpx is required for downloading files.") from e

    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    if urls:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            for url in urls:
                mirrored_url = _apply_hf_mirror(url)
                filename = url.split("/")[-1].split("?")[0] or "file"
                try:
                    from urllib.parse import unquote
                    filename = unquote(filename)
                except Exception:
                    pass
                dest_path = dest_dir / filename

                if dest_path.exists():
                    logger.debug("Skipping already-downloaded file: %s", dest_path)
                    downloaded.append(dest_path)
                    continue

                logger.info("Downloading %s -> %s", mirrored_url, dest_path)
                try:
                    response = await client.get(mirrored_url)
                    response.raise_for_status()
                    dest_path.write_bytes(response.content)
                    downloaded.append(dest_path)
                except Exception as exc:
                    logger.warning("Failed to download %s: %s", mirrored_url, exc)

    for item in inline_files:
        if isinstance(item, str):
            continue

        filename: str = item.get("filename", "file")
        content: str | bytes | None = item.get("content")

        if not content:
            continue

        dest_path = dest_dir / filename
        if dest_path.exists():
            logger.debug("Skipping already-written file: %s", dest_path)
            downloaded.append(dest_path)
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            dest_path.write_text(content, encoding="utf-8")
        else:
            dest_path.write_bytes(content)
        downloaded.append(dest_path)

    return downloaded


async def download_reference_files(task: dict[str, Any], dest_dir: Path) -> list[Path]:
    """Download reference files for a task.

    Downloads files from task["reference_file_urls"] (URLs) and writes
    task["reference_files"] (inline content) to dest_dir.
    Skips files that already exist (resume-friendly).
    """
    return await _download_files(
        urls=task.get("reference_file_urls") or [],
        inline_files=task.get("reference_files") or [],
        dest_dir=dest_dir,
    )


async def download_deliverable_files(task: dict[str, Any], dest_dir: Path) -> list[Path]:
    """Download deliverable (reference answer) files for a task.

    Downloads files from task["deliverable_file_urls"] (URLs) and writes
    task["deliverable_files"] (inline content) to dest_dir.
    Skips files that already exist (resume-friendly).
    """
    return await _download_files(
        urls=task.get("deliverable_file_urls") or [],
        inline_files=task.get("deliverable_files") or [],
        dest_dir=dest_dir,
    )
