"""AgentIF-OneDay single-task runner."""

from __future__ import annotations

import asyncio
import contextlib
import glob
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.core.models import ImageContentBlock, Tool, ToolResult
from stirrup.tools.browser_use import BrowserUseToolProvider
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.tools.view_image import ViewImageToolProvider
from stirrup.tools.web import WebToolProvider

from .loader import AgentIFTask


DEFAULT_SYSTEM_PROMPT = """
Execution policy:
- Do not fabricate, simulate, or placeholder any result.
- Never rename plain text files to binary extensions like .xlsx/.png/.jpg/.pdf/.doc.
- If a required website blocks simple HTTP requests, use browser tools first, then retry.
- For anti-bot stability, collect all required data from one site in as few navigations as possible before switching sites.
- If a page shows Cloudflare/security verification, wait and retry; do not keep scrolling/chasing text on the challenge page.
- The browser is only one possible way to obtain information. If website access becomes difficult or time-consuming, switch strategy and complete the deliverables using other available sources.
- Always call finish before max turns and include all produced output file paths.
- If you still cannot access required sources after retries, report failure clearly and do not claim completion.
- Produce real files with valid binary formats for requested outputs.
""".strip()


def build_prompt(task: AgentIFTask, include_score_criteria: bool = False) -> str:
    """Build the user prompt sent to Stirrup."""
    lines = [
        f"Task ID: {task.question_id}",
        f"Title: {task.title}",
        "",
        "Instruction:",
        task.description,
        "",
        "Requirements:",
        "- Complete the task end-to-end.",
        "- Use provided attachments when available.",
        "- Generate all required deliverables with exact filenames requested by the task.",
    ]

    if include_score_criteria and task.score_criteria:
        lines.extend(["", "Scoring hints:"])
        for idx, criterion in enumerate(task.score_criteria, start=1):
            content = str(criterion.get("content", "")).strip()
            score = criterion.get("score")
            lines.append(f"{idx}. [{score}] {content}")

    return "\n".join(lines).strip()


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return _to_jsonable(method())
            except Exception:
                pass
    return str(value)


def parse_bool(raw: str | None, default: bool) -> bool:
    """Parse a permissive env/CLI boolean string."""
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _is_executable_file(path: str | Path) -> bool:
    candidate = Path(path).expanduser()
    return candidate.exists() and candidate.is_file() and os.access(candidate, os.X_OK)


def _playwright_cache_root() -> Path:
    raw = os.getenv("PLAYWRIGHT_BROWSERS_PATH")
    if raw and raw.strip() and raw.strip() != "0":
        return Path(raw).expanduser()

    system = platform.system()
    if system == "Darwin":
        return Path("~/Library/Caches/ms-playwright").expanduser()
    if system == "Windows":
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "ms-playwright"
    return Path("~/.cache/ms-playwright").expanduser()


def _latest_globbed_executable(pattern: str) -> str | None:
    matches = sorted(glob.glob(str(Path(pattern).expanduser())))
    for match in reversed(matches):
        if _is_executable_file(match):
            return str(Path(match).resolve())
    return None


def _find_playwright_cached_browser(*, include_headless_shell: bool) -> str | None:
    cache_root = _playwright_cache_root()
    system = platform.system()
    if system == "Darwin":
        patterns = [
            cache_root / "chromium-*" / "chrome-mac" / "Chromium.app" / "Contents" / "MacOS" / "Chromium",
        ]
        if include_headless_shell:
            patterns.append(
                cache_root
                / "chromium_headless_shell-*"
                / "chrome-mac"
                / "Chromium.app"
                / "Contents"
                / "MacOS"
                / "Chromium"
            )
    elif system == "Windows":
        patterns = [
            cache_root / "chromium-*" / "chrome-win" / "chrome.exe",
        ]
        if include_headless_shell:
            patterns.append(cache_root / "chromium_headless_shell-*" / "chrome-win" / "chrome.exe")
    else:
        patterns = [
            cache_root / "chromium-*" / "chrome-linux*" / "chrome",
        ]
        if include_headless_shell:
            patterns.append(cache_root / "chromium_headless_shell-*" / "chrome-linux*" / "chrome")

    for pattern in patterns:
        if match := _latest_globbed_executable(str(pattern)):
            return match
    return None


def _find_system_browser() -> str | None:
    names = [
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
        "google-chrome-beta",
        "google-chrome-dev",
        "brave-browser",
        "microsoft-edge",
    ]
    for name in names:
        path = shutil.which(name)
        if path and _is_executable_file(path):
            return str(Path(path).resolve())

    system_paths = [
        "/usr/bin/google-chrome-stable",
        "/usr/bin/google-chrome",
        "/usr/local/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/local/bin/chromium",
        "/snap/bin/chromium",
        "/usr/bin/google-chrome-beta",
        "/usr/bin/google-chrome-dev",
        "/usr/bin/brave-browser",
    ]
    if platform.system() == "Darwin":
        system_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ]
    elif platform.system() == "Windows":
        system_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Chromium\Application\chrome.exe",
            r"C:\Program Files (x86)\Chromium\Application\chrome.exe",
            r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]

    for path in system_paths:
        expanded = os.path.expandvars(path)
        if _is_executable_file(expanded):
            return str(Path(expanded).resolve())
    return None


def _install_playwright_chromium() -> None:
    try:
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while installing Playwright Chromium") from exc
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"Failed to install Playwright Chromium: {message}") from exc


def resolve_browser_executable_path(
    *,
    explicit_path: str | None,
    cdp_url: str | None,
    headless: bool,
    install_if_missing: bool = True,
) -> str | None:
    """Resolve browser priority: explicit path/CDP, Playwright cache, system path, Playwright install."""
    if cdp_url:
        return explicit_path

    if explicit_path:
        return explicit_path

    if cached := _find_playwright_cached_browser(include_headless_shell=headless):
        return cached

    if system_browser := _find_system_browser():
        return system_browser

    if not install_if_missing:
        return None

    _install_playwright_chromium()
    return _find_playwright_cached_browser(include_headless_shell=headless)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ensure_localhost_no_proxy() -> None:
    required = {"127.0.0.1", "localhost", "::1"}
    for key in ("NO_PROXY", "no_proxy"):
        existing = {
            item.strip()
            for item in os.getenv(key, "").split(",")
            if item.strip()
        }
        combined = sorted(existing | required)
        os.environ[key] = ",".join(combined)


def _wait_for_cdp(port: int, *, timeout_seconds: float = 45.0) -> str:
    url = f"http://127.0.0.1:{port}/json/version"
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open(url, timeout=1) as response:
                if response.status == 200:
                    return f"http://127.0.0.1:{port}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for prelaunched Chromium CDP on port {port}: {last_error}")


def _terminate_browser_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


def _prelaunch_chromium_cdp(
    *,
    executable_path: str,
    profile_dir: Path,
    headless: bool,
    browser_user_agent: str | None,
    browser_timezone: str | None,
) -> tuple[str, subprocess.Popen[Any]]:
    """Launch Chromium directly and return a CDP URL for browser-use to attach to."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    port = _find_free_port()
    args = [
        executable_path,
        "--remote-debugging-address=127.0.0.1",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-gpu-sandbox",
        "--disable-setuid-sandbox",
        "--disable-blink-features=AutomationControlled",
        "--window-size=1920,1080",
        "--lang=en-US,en",
    ]
    if headless:
        args.append("--headless=new")
    if browser_user_agent:
        args.append(f"--user-agent={browser_user_agent}")
    if browser_timezone:
        args.append(f"--timezone={browser_timezone}")

    proxy = (os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY") or "").strip()
    if proxy:
        args.append(f"--proxy-server={proxy}")

    stderr_log_path = profile_dir / "chromium_stderr.log"
    stderr_log = stderr_log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(  # noqa: S603
            args,
            stdout=subprocess.DEVNULL,
            stderr=stderr_log,
            text=True,
        )
    finally:
        stderr_log.close()
    try:
        cdp_url = _wait_for_cdp(port)
    except Exception as exc:
        _terminate_browser_process(process)
        stderr_tail = ""
        with contextlib.suppress(Exception):
            stderr_tail = stderr_log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Failed to prelaunch Chromium from {executable_path}: {exc}\n{stderr_tail}") from exc
    return cdp_url, process


def infer_required_outputs(task: AgentIFTask) -> list[str]:
    """Infer explicitly requested output filenames from task text/rubrics."""
    primary_text = "\n".join([task.title, task.description])
    rubric_text = " ".join(str(c.get("content", "")) for c in task.score_criteria)

    def _extract(text: str) -> list[str]:
        names: list[str] = []
        for match in re.finditer(r"\[([^\]\n]+\.[A-Za-z0-9]{2,5})\]", text):
            names.append(match.group(1).strip())

        for pattern in [
            r"save(?:d)?(?:\s+(?:the\s+\w+|it))?\s+as\s+([A-Za-z0-9 _().\-]+\.[A-Za-z0-9]{2,5})",
            r"name(?:d)?\s+(?:the\s+file\s+)?as\s+([A-Za-z0-9 _().\-]+\.[A-Za-z0-9]{2,5})",
            r"format\s+([A-Za-z0-9 _().\-]+\.[A-Za-z0-9]{2,5})",
        ]:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                names.append(match.group(1).strip(" .,\"'"))
        return names

    names = _extract(primary_text) or _extract(rubric_text)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(name)
    return deduped


def _build_browser_extra_args(
    *,
    profile_dir: Path | None,
    browser_user_agent: str | None,
    browser_timezone: str | None,
) -> list[str]:
    args = [
        "--disable-blink-features=AutomationControlled",
        "--window-size=1920,1080",
        "--lang=en-US,en",
    ]
    if profile_dir is not None:
        args.append(f"--user-data-dir={profile_dir}")
    if browser_user_agent:
        args.append(f"--user-agent={browser_user_agent}")
    if browser_timezone:
        args.append(f"--timezone={browser_timezone}")

    proxy = (os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY") or "").strip()
    if proxy:
        args.append(f"--proxy-server={proxy}")
    return args


class PersistingBrowserUseToolProvider(BrowserUseToolProvider):
    """Persist browser screenshots into the code execution environment."""

    def __init__(
        self,
        *,
        code_provider: LocalCodeExecToolProvider,
        drop_search_tool: bool = True,
        cf_retry_attempts: int = 2,
        cf_retry_wait_seconds: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._code_provider = code_provider
        self._drop_search_tool = drop_search_tool
        self._cf_retry_attempts = max(cf_retry_attempts, 0)
        self._cf_retry_wait_seconds = max(cf_retry_wait_seconds, 1)
        self._screenshot_idx = 0
        self.cf_challenge_detected = False
        self.cf_challenge_unresolved = False

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.lower()
        if isinstance(content, list):
            return "\n".join(str(x) for x in content).lower()
        return str(content).lower()

    @classmethod
    def _looks_like_cloudflare(cls, content: Any) -> bool:
        text = cls._content_to_text(content)
        markers = (
            "performing security verification",
            "cloudflare security challenge",
            "cf-chl-widget",
            "ray id",
            "checking your browser",
        )
        return any(marker in text for marker in markers)

    async def _maybe_recover_cloudflare(
        self,
        *,
        current_result: ToolResult[Any],
        snapshot_tool: Tool[Any, Any] | None,
        wait_tool: Tool[Any, Any] | None,
        get_url_tool: Tool[Any, Any] | None,
        navigate_tool: Tool[Any, Any] | None,
    ) -> ToolResult[Any]:
        if not self._looks_like_cloudflare(current_result.content):
            return current_result

        self.cf_challenge_detected = True
        result = current_result

        for _ in range(self._cf_retry_attempts):
            if wait_tool is not None:
                wait_ret = wait_tool.executor(wait_tool.parameters(seconds=self._cf_retry_wait_seconds))
                if asyncio.iscoroutine(wait_ret):
                    await wait_ret

            url: str | None = None
            if get_url_tool is not None:
                url_ret = get_url_tool.executor(get_url_tool.parameters())
                if asyncio.iscoroutine(url_ret):
                    url_ret = await url_ret
                match = re.search(r"current url:\s*(\S+)", self._content_to_text(url_ret.content), re.IGNORECASE)
                if match:
                    url = match.group(1)

            if url and navigate_tool is not None and url != "about:blank":
                nav_ret = navigate_tool.executor(navigate_tool.parameters(url=url, new_tab=False))
                if asyncio.iscoroutine(nav_ret):
                    await nav_ret

            if snapshot_tool is not None:
                snap_ret = snapshot_tool.executor(snapshot_tool.parameters())
                if asyncio.iscoroutine(snap_ret):
                    snap_ret = await snap_ret
                result = snap_ret
                if not self._looks_like_cloudflare(result.content):
                    self.cf_challenge_unresolved = False
                    return result

        self.cf_challenge_unresolved = True
        return ToolResult(
            content=(
                f"{current_result.content}\n[cloudflare] challenge still active after "
                f"{self._cf_retry_attempts} auto-retries; continue with other sources or finish with failure."
            ),
            success=current_result.success,
            metadata=current_result.metadata,
        )

    async def __aenter__(self) -> list[Tool[Any, Any]]:
        tools = await super().__aenter__()
        screenshot_tool_name = self._tool_name("screenshot")
        search_tool_name = self._tool_name("search")
        snapshot_tool_name = self._tool_name("snapshot")
        wait_tool_name = self._tool_name("wait")
        get_url_tool_name = self._tool_name("get_url")
        navigate_tool_name = self._tool_name("navigate")
        tool_map = {tool.name: tool for tool in tools}
        wrapped_tools: list[Tool[Any, Any]] = []

        for tool in tools:
            if self._drop_search_tool and tool.name == search_tool_name:
                continue

            if tool.name == snapshot_tool_name:

                async def snapshot_with_cf_recover(
                    params: Any,
                    _tool: Tool[Any, Any] = tool,
                ) -> ToolResult[Any]:
                    result = _tool.executor(params)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return await self._maybe_recover_cloudflare(
                        current_result=result,
                        snapshot_tool=_tool,
                        wait_tool=tool_map.get(wait_tool_name),
                        get_url_tool=tool_map.get(get_url_tool_name),
                        navigate_tool=tool_map.get(navigate_tool_name),
                    )

                wrapped_tools.append(
                    Tool(
                        name=tool.name,
                        description=tool.description + " Includes Cloudflare auto-recovery retries.",
                        parameters=tool.parameters,
                        executor=snapshot_with_cf_recover,
                    )
                )
                continue

            if tool.name != screenshot_tool_name:
                wrapped_tools.append(tool)
                continue

            async def screenshot_with_persist(
                params: Any,
                _tool: Tool[Any, Any] = tool,
            ) -> ToolResult[Any]:
                result = _tool.executor(params)
                if asyncio.iscoroutine(result):
                    result = await result
                image_bytes: bytes | None = None
                if isinstance(result.content, list):
                    for block in result.content:
                        if isinstance(block, ImageContentBlock):
                            image_bytes = block.data
                            break
                if image_bytes is None:
                    return result

                self._screenshot_idx += 1
                filename = f"browser_screenshot_{self._screenshot_idx:03d}.png"
                await self._code_provider.write_file_bytes(filename, image_bytes)
                return ToolResult(
                    content=(
                        f"Screenshot captured and saved as {filename} in the code execution environment. "
                        "Use this exact path with view_image or code_exec."
                    ),
                    success=result.success,
                    metadata=result.metadata,
                )

            wrapped_tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description
                    + " Screenshot bytes are also persisted to a PNG file in the code execution environment.",
                    parameters=tool.parameters,
                    executor=screenshot_with_persist,
                )
            )

        return wrapped_tools


def _extract_finish_paths(response_payload: dict[str, Any]) -> list[str]:
    finish = response_payload.get("finish")
    if not isinstance(finish, dict):
        return []
    paths = finish.get("paths")
    if not isinstance(paths, list):
        return []
    return [str(x) for x in paths]


def _list_generated_files(task_out: Path) -> list[str]:
    ignored = {"stirrup_payload.json", "stirrup_response.json", "trajectory.jsonl"}
    files: list[str] = []
    for path in task_out.rglob("*"):
        if path.is_file() and path.name not in ignored:
            files.append(str(path.relative_to(task_out)))
    files.sort()
    return files


def _is_valid_binary_file(path: Path) -> tuple[bool, str]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".docx", ".pptx"}:
        if not zipfile.is_zipfile(path):
            return False, f"{path.name} is not a valid ZIP-based Office file"
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
        except Exception as exc:
            return False, f"{path.name} invalid office archive: {exc}"
        if "[Content_Types].xml" not in names:
            return False, f"{path.name} missing [Content_Types].xml"
        required_member = {
            ".xlsx": "xl/workbook.xml",
            ".docx": "word/document.xml",
            ".pptx": "ppt/presentation.xml",
        }[suffix]
        if required_member not in names:
            return False, f"{path.name} missing {required_member}"
        return True, ""

    try:
        raw = path.read_bytes()
    except Exception as exc:
        return False, f"failed to read {path.name}: {exc}"

    if suffix == ".png" and not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return False, f"{path.name} is not a valid PNG"
    if suffix in {".jpg", ".jpeg"} and not raw.startswith(b"\xff\xd8"):
        return False, f"{path.name} is not a valid JPEG"
    if suffix == ".pdf" and not raw.startswith(b"%PDF"):
        return False, f"{path.name} is not a valid PDF"
    if suffix == ".doc" and not raw.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        return False, f"{path.name} is not a valid binary .doc file"
    return True, ""


def _looks_like_simulation(response_payload: dict[str, Any]) -> bool:
    finish = response_payload.get("finish")
    if not isinstance(finish, dict):
        return False
    reason = str(finish.get("reason", "")).lower()
    return any(flag in reason for flag in ("simulat", "placeholder", "dummy", "cannot access", "could not access"))


def _validate_outputs(
    task: AgentIFTask,
    task_out: Path,
    finish_paths: list[str],
    generated_files: list[str],
    response_payload: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    required = infer_required_outputs(task)
    generated_lower = {path.lower() for path in generated_files}

    for required_name in required:
        if required_name.lower() not in generated_lower:
            errors.append(f"missing required output: {required_name}")

    for rel_path in generated_files:
        path = task_out / rel_path
        if not path.exists():
            errors.append(f"reported output missing on disk: {rel_path}")
            continue
        ok, message = _is_valid_binary_file(path)
        if not ok:
            errors.append(message)

    if _looks_like_simulation(response_payload):
        errors.append("finish reason indicates simulated/placeholder completion")

    for finish_path in finish_paths:
        resolved = Path(finish_path) if Path(finish_path).is_absolute() else (task_out / finish_path)
        if not resolved.exists():
            errors.append(f"finish path not found: {finish_path}")

    return errors


async def run_task(
    *,
    task: AgentIFTask,
    input_files: list[Path],
    output_base_dir: Path,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    max_turns: int = 30,
    client_timeout_seconds: int = 1800,
    web_timeout_seconds: int = 180,
    brave_api_key: str | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    include_score_criteria: bool = False,
    browser_headless: bool = False,
    browser_executable_path: str | None = None,
    browser_cdp_url: str | None = None,
    browser_profile_dir: Path | None = None,
    browser_user_agent: str | None = None,
    browser_timezone: str | None = None,
    cf_retry_attempts: int = 2,
    cf_retry_wait_seconds: int = 8,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one AgentIF-OneDay task with the local Stirrup agent."""
    task_out = output_base_dir / task.question_id
    if task_out.exists() and overwrite:
        shutil.rmtree(task_out)
    task_out.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(task, include_score_criteria=include_score_criteria)
    payload = {
        "question_id": task.question_id,
        "title": task.title,
        "model": model,
        "base_url": base_url or "",
        "prompt": prompt,
        "input_files": [str(path) for path in input_files],
        "output_dir": str(task_out),
        "max_turns": max_turns,
        "client_timeout_seconds": client_timeout_seconds,
        "web_timeout_seconds": web_timeout_seconds,
        "browser_headless": browser_headless,
        "browser_executable_path_requested": browser_executable_path or "",
        "browser_cdp_url": browser_cdp_url or "",
        "browser_profile_dir": str(browser_profile_dir) if browser_profile_dir is not None else "",
    }
    (task_out / "stirrup_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if dry_run:
        return {
            "question_id": task.question_id,
            "title": task.title,
            "status": "dry_run",
            "output_dir": str(task_out),
            "input_files": payload["input_files"],
            "error": None,
        }

    _ensure_localhost_no_proxy()
    code_provider = LocalCodeExecToolProvider()
    if browser_profile_dir is not None:
        browser_profile_dir.mkdir(parents=True, exist_ok=True)

    resolved_browser_executable_path = resolve_browser_executable_path(
        explicit_path=browser_executable_path,
        cdp_url=browser_cdp_url,
        headless=browser_headless,
    )
    prelaunched_browser_process: subprocess.Popen[Any] | None = None
    resolved_browser_cdp_url = browser_cdp_url
    if resolved_browser_cdp_url is None and resolved_browser_executable_path:
        profile_root = browser_profile_dir or (task_out / ".browser_profile")
        runtime_profile_dir = profile_root / "runtime_profiles" / (
            f"{task.question_id}-{os.getpid()}-{int(time.time() * 1000)}"
        )
        resolved_browser_cdp_url, prelaunched_browser_process = _prelaunch_chromium_cdp(
            executable_path=resolved_browser_executable_path,
            profile_dir=runtime_profile_dir,
            headless=browser_headless,
            browser_user_agent=browser_user_agent,
            browser_timezone=browser_timezone,
        )
    payload["browser_executable_path_resolved"] = resolved_browser_executable_path or ""
    payload["browser_cdp_url_resolved"] = resolved_browser_cdp_url or ""
    payload["browser_prelaunched"] = bool(prelaunched_browser_process)
    (task_out / "stirrup_payload.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    browser_provider = PersistingBrowserUseToolProvider(
        code_provider=code_provider,
        headless=browser_headless,
        executable_path=None if resolved_browser_cdp_url else resolved_browser_executable_path,
        cdp_url=resolved_browser_cdp_url,
        extra_args=_build_browser_extra_args(
            profile_dir=None if resolved_browser_cdp_url else browser_profile_dir,
            browser_user_agent=browser_user_agent,
            browser_timezone=browser_timezone,
        ),
        drop_search_tool=True,
        cf_retry_attempts=cf_retry_attempts,
        cf_retry_wait_seconds=cf_retry_wait_seconds,
    )
    tools = [
        browser_provider,
        WebToolProvider(timeout=float(max(web_timeout_seconds, 1)), brave_api_key=brave_api_key),
        code_provider,
        ViewImageToolProvider(exec_env=code_provider),
    ]

    flatten = base_url is not None and "deepseek" in base_url.lower()
    client = ChatCompletionsClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=float(max(client_timeout_seconds, 1)),
        flatten_tool_content=flatten,
    )
    agent = Agent(
        name="agentif-oneday-stirrup",
        client=client,
        max_turns=max(max_turns, 1),
        tools=tools,
        system_prompt=system_prompt,
    )

    try:
        async with agent.session(output_dir=task_out, input_files=input_files) as session:
            finish_params, full_msg_history, run_metadata = await session.run(prompt)

        with (task_out / "trajectory.jsonl").open("w", encoding="utf-8") as f:
            for segment in full_msg_history:
                for message in segment:
                    f.write(message.model_dump_json() + "\n")

        response_payload = {
            "finish": _to_jsonable(finish_params),
            "run_metadata": _to_jsonable(run_metadata),
            "message_group_count": len(full_msg_history),
            "last_group_message_count": len(full_msg_history[-1]) if full_msg_history else 0,
            "cloudflare_challenge_detected": browser_provider.cf_challenge_detected,
            "cloudflare_challenge_unresolved": browser_provider.cf_challenge_unresolved,
        }
        (task_out / "stirrup_response.json").write_text(
            json.dumps(response_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        finish_paths = _extract_finish_paths(response_payload)
        generated_files = _list_generated_files(task_out)
        validation_errors = _validate_outputs(
            task=task,
            task_out=task_out,
            finish_paths=finish_paths,
            generated_files=generated_files,
            response_payload=response_payload,
        )
        if response_payload.get("cloudflare_challenge_unresolved"):
            validation_errors.append("cloudflare challenge unresolved during run")

        required = infer_required_outputs(task)
        generated_lower = {path.lower() for path in generated_files}
        required_ready = bool(required) and all(required_name.lower() in generated_lower for required_name in required)
        status = "completed" if (finish_paths or required_ready) else "unfinished"
        if validation_errors:
            status = "error"

        return {
            "question_id": task.question_id,
            "title": task.title,
            "status": status,
            "success": status == "completed",
            "error": "; ".join(validation_errors) if validation_errors else None,
            "finish_paths": finish_paths,
            "generated_files": generated_files,
            "output_dir": str(task_out),
            "run_metadata": _to_jsonable(run_metadata),
        }
    except Exception as exc:
        return {
            "question_id": task.question_id,
            "title": task.title,
            "status": "error",
            "success": False,
            "error": str(exc),
            "finish_paths": [],
            "generated_files": _list_generated_files(task_out),
            "output_dir": str(task_out),
        }
    finally:
        _terminate_browser_process(prelaunched_browser_process)
