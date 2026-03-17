#!/usr/bin/env python3
"""
Stirrup Web Visualizer

Starts a local web server on http://localhost:8765 and opens a browser
to show real-time visualization of agent runs: LLM messages, tool calls,
and tool results.

Usage:
    python visualize.py [task description]

Example:
    python visualize.py "What is the capital of France?"
    python visualize.py  # Uses a default demo task
"""

import asyncio
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from queue import Empty, Queue
from socketserver import ThreadingMixIn
from typing import Any

from pydantic import BaseModel

from stirrup.core.models import AssistantMessage, ToolMessage, UserMessage
from stirrup.utils.logging import AgentLoggerBase

# ---------------------------------------------------------------------------
# SSE broadcast system
# ---------------------------------------------------------------------------

_clients: list[Queue[dict[str, Any]]] = []
_clients_lock = threading.Lock()

_event_history: list[dict[str, Any]] = []
_history_lock = threading.Lock()
_current_task_id: int = 0


def _broadcast(event: dict[str, Any]) -> None:
    """Send an event to all connected SSE clients, and persist to history."""
    global _current_task_id
    with _history_lock:
        if event.get("type") == "session_start" and event.get("depth", 1) == 0:
            _current_task_id += 1
        event = {**event, "task_id": _current_task_id}
        _event_history.append(event)
    with _clients_lock:
        for q in list(_clients):
            q.put_nowait(event)


def _content_to_str(content: Any) -> str:
    """Flatten Content (str | list[ContentBlock]) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            else:
                parts.append(f"[{type(block).__name__}]")
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Web logger — implements AgentLoggerBase
# ---------------------------------------------------------------------------


class WebLogger(AgentLoggerBase):
    """AgentLoggerBase implementation that broadcasts events to the web UI via SSE."""

    def __init__(self) -> None:
        self.name: str = "agent"
        self.model: str | None = None
        self.max_turns: int | None = None
        self.depth: int = 0
        self.finish_params: BaseModel | None = None
        self.run_metadata: dict[str, list[Any]] | None = None
        self.output_dir: str | None = None

    def __enter__(self) -> "WebLogger":
        _broadcast(
            {
                "type": "session_start",
                "name": self.name,
                "model": self.model,
                "max_turns": self.max_turns,
                "depth": self.depth,
            }
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        success = exc_type is None and self.finish_params is not None
        _broadcast(
            {
                "type": "session_end",
                "name": self.name,
                "success": success,
                "error": str(exc_val) if exc_val else None,
                "depth": self.depth,
            }
        )

    def on_step(
        self,
        step: int,
        tool_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        _broadcast(
            {
                "type": "step",
                "step": step,
                "tool_calls": tool_calls,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def assistant_message(
        self,
        turn: int,
        max_turns: int,
        assistant_message: AssistantMessage,
    ) -> None:
        tool_calls: list[dict[str, Any]] = []
        for tc in assistant_message.tool_calls:
            try:
                args: Any = json.loads(tc.arguments) if tc.arguments else {}
            except json.JSONDecodeError:
                args = tc.arguments
            tool_calls.append({"name": tc.name, "arguments": args})

        _broadcast(
            {
                "type": "assistant",
                "turn": turn,
                "max_turns": max_turns,
                "content": _content_to_str(assistant_message.content),
                "tool_calls": tool_calls,
                "input_tokens": assistant_message.token_usage.input,
                "output_tokens": assistant_message.token_usage.answer,
                "agent": self.name,
                "depth": self.depth,
            }
        )

    def user_message(self, user_message: UserMessage) -> None:
        _broadcast(
            {
                "type": "user",
                "content": _content_to_str(user_message.content),
                "agent": self.name,
                "depth": self.depth,
            }
        )

    def task_message(self, task: str | list[Any]) -> None:
        task_str = "\n".join(str(b) for b in task) if isinstance(task, list) else task
        _broadcast(
            {
                "type": "task",
                "content": task_str,
                "agent": self.name,
                "depth": self.depth,
            }
        )

    def tool_result(self, tool_message: ToolMessage) -> None:
        duration: float | None = None
        if tool_message.tool_end_time and tool_message.tool_start_time:
            duration = round(tool_message.tool_end_time - tool_message.tool_start_time, 2)

        _broadcast(
            {
                "type": "tool_result",
                "name": tool_message.name or "unknown",
                "content": _content_to_str(tool_message.content),
                "success": tool_message.success,
                "args_was_valid": tool_message.args_was_valid,
                "agent": self.name,
                "depth": self.depth,
                "duration": duration,
            }
        )

    def context_summarization_start(self, pct_used: float, cutoff: float) -> None:
        _broadcast(
            {
                "type": "context_summary_start",
                "pct_used": pct_used,
                "cutoff": cutoff,
            }
        )

    def context_summarization_complete(self, summary: str, bridge: str) -> None:
        _broadcast(
            {
                "type": "context_summary_complete",
                "summary": summary[:500] + "..." if len(summary) > 500 else summary,
            }
        )

    def debug(self, message: str, *args: object) -> None:
        pass

    def info(self, message: str, *args: object) -> None:
        pass

    def warning(self, message: str, *args: object) -> None:
        _broadcast({"type": "log", "level": "warning", "message": message % args if args else message})

    def error(self, message: str, *args: object) -> None:
        _broadcast({"type": "log", "level": "error", "message": message % args if args else message})


# ---------------------------------------------------------------------------
# HTTP server — serves visualize_index.html + SSE /events endpoint
# ---------------------------------------------------------------------------

_PORT = 8765
_INDEX_PATH = Path(__file__).with_name("visualize_index.html")


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            try:
                body = _INDEX_PATH.read_bytes()
            except FileNotFoundError:
                self.send_error(500, "visualize_index.html not found next to visualize.py")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()

        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q: Queue[dict[str, Any]] = Queue()
            # Register client BEFORE snapshotting history to avoid missing events
            with _clients_lock:
                _clients.append(q)
            # Snapshot history; some queued events may overlap (sent to queue after we registered)
            with _history_lock:
                history_snapshot = list(_event_history)
            history_ids = {id(ev) for ev in history_snapshot}

            try:
                # Replay history to the new client
                for ev in history_snapshot:
                    data = json.dumps(ev, ensure_ascii=False)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                # Drain any events queued during replay, skipping duplicates
                while True:
                    try:
                        ev = q.get_nowait()
                        if id(ev) not in history_ids:
                            data = json.dumps(ev, ensure_ascii=False)
                            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    except Empty:
                        break
                # Signal that the replay phase is complete
                self.wfile.write(b'data: {"type":"__replay_end__"}\n\n')
                self.wfile.flush()

                while True:
                    try:
                        event = q.get(timeout=15.0)
                        data = json.dumps(event, ensure_ascii=False)
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except Empty:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _clients_lock:
                    if q in _clients:
                        _clients.remove(q)

        elif self.path == "/reset":
            global _current_task_id  # noqa: PLW0603
            with _history_lock:
                _event_history.clear()
                _current_task_id = 0
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        else:
            self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # suppress HTTP access logs


def _start_server(port: int = _PORT) -> _ThreadingHTTPServer:
    server = _ThreadingHTTPServer(("localhost", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def start_visualizer_server(
    *,
    open_browser: bool = True,
    port: int = _PORT,
) -> _ThreadingHTTPServer:
    """Start the web visualizer server and optionally open the browser.

    This is a reusable entry-point for other modules (e.g. GDPVal runner)
    that want to enable real-time visualization.
    """
    server = _start_server(port)
    url = f"http://localhost:{port}"
    print(f"\n🌐 可视化界面: {url}")
    if open_browser:
        print("📡 正在打开浏览器...")
        try:
            webbrowser.open(url)
        except Exception as exc:  # pragma: no cover - best-effort UX
            print(f"⚠ 无法自动打开浏览器: {exc}")
    return server


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    from stirrup import Agent
    from stirrup.clients.chat_completions_client import ChatCompletionsClient
    from stirrup.tools import DEFAULT_TOOLS

    task = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "用 Python 写一个斐波那契数列函数，并计算前 20 项，最后输出结果。"
    )

    server = start_visualizer_server(open_browser=True, port=_PORT)

    # Give the browser a moment to connect before we start producing events
    await asyncio.sleep(2)

    print(f"🤖 任务: {task[:100]}{'...' if len(task) > 100 else ''}\n")

    client = ChatCompletionsClient(model="gpt-4o-mini")
    agent = Agent(
        client=client,
        name="stirrup-agent",
        tools=DEFAULT_TOOLS,
        logger=WebLogger(),
        max_turns=20,
    )

    async with agent.session() as session:
        await session.run(task)

    print("\n✅ Agent 运行完成，界面保持开放，按 Ctrl+C 退出。")
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 已退出")
