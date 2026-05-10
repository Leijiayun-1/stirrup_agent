# Context var for passing parent depth to sub-agent executors
import ast
import contextvars
import glob as glob_module
import inspect
import json
import logging
import re
import signal
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from itertools import chain, takewhile
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Annotated, Any, Self

import anyio
from pydantic import BaseModel, Field, ValidationError

from stirrup.constants import (
    AGENT_MAX_TURNS,
    CONTEXT_SUMMARIZATION_CUTOFF,
    FINISH_TOOL_NAME,
    TURNS_REMAINING_WARNING_THRESHOLD,
)
from stirrup.core.cache import CacheManager, CacheState, compute_task_hash
from stirrup.core.planner import Planner
from stirrup.core.semantic_state import ExtractorArtifact, SemanticStateManager, ViolationArtifact
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    ImageContentBlock,
    LLMClient,
    SubAgentMetadata,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolProvider,
    ToolResult,
    UserMessage,
)
from stirrup.prompts import MESSAGE_SUMMARIZER, MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE
from stirrup.skills import SkillMetadata, format_skills_section, load_skills_metadata
from stirrup.tools import DEFAULT_TOOLS
from stirrup.tools.code_backends.base import CodeExecToolProvider
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.tools.finish import SIMPLE_FINISH_TOOL
from stirrup.utils.logging import AgentLogger, AgentLoggerBase
from stirrup.utils.text import truncate_msg

_PARENT_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("parent_depth", default=0)

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Per-session state for resource lifecycle management.

    Kept minimal - only contains resources that need async lifecycle management
    (exit_stack, exec_env) and session-specific configuration (output_dir).

    Tool availability is managed via Agent._active_tools (instance-scoped),
    and run results are stored on the agent instance temporarily.

    For subagent file transfer:
    - parent_exec_env: Reference to the parent's exec env (for cross-env transfers)
    - depth: Agent depth (0 = root, >0 = subagent)
    - output_dir: For root agent, this is a local filesystem path. For subagents,
      this is a path within the parent's exec env.
    - exec_env_owned: Whether this session owns the exec_env and should clean it up.
      When share_parent_exec_env=True, the subagent borrows the parent's exec_env
      and exec_env_owned=False to prevent cleanup on subagent exit.
    """

    exit_stack: AsyncExitStack
    exec_env: CodeExecToolProvider | None = None
    output_dir: str | None = None  # String path (contextual: local for root, in parent env for subagent)
    parent_exec_env: CodeExecToolProvider | None = None
    depth: int = 0
    exec_env_owned: bool = True  # Whether this session owns (and should cleanup) the exec_env
    uploaded_file_paths: list[str] = field(default_factory=list)  # Paths of files uploaded to exec_env
    uploaded_file_records: list[dict[str, Any]] = field(default_factory=list)
    generated_files_snapshot: list[str] = field(default_factory=list)
    skills_metadata: list[SkillMetadata] = field(default_factory=list)  # Loaded skills metadata
    logger: AgentLoggerBase | None = None  # Logger for pause/resume during user input


_SESSION_STATE: contextvars.ContextVar[SessionState] = contextvars.ContextVar("session_state")

__all__ = [
    "Agent",
    "SubAgentParams",
]

LOGGER = logging.getLogger(__name__)


def _num_turns_remaining_msg(number_of_turns_remaining: int) -> UserMessage:
    """Create a user message warning the agent about remaining turns before max_turns is reached."""
    if number_of_turns_remaining == 1:
        return UserMessage(content="This is the last turn. You must call the finish tool now.")
    if number_of_turns_remaining <= 3:
        return UserMessage(
            content=f"You have {number_of_turns_remaining} turns remaining. Wrap up now and call the finish tool on the next turn.",
        )
    return UserMessage(
        content=f"You have {number_of_turns_remaining} turns remaining. If the core deliverables are ready, call the finish tool. Do not create extra files.",
    )


def _handle_text_only_tool_responses(tool_messages: list[ToolMessage]) -> tuple[list[ToolMessage], list[UserMessage]]:
    """Extract image blocks from tool messages and convert them to user messages for text-only models."""
    user_messages: list[UserMessage] = []
    for tm in tool_messages:
        if isinstance(tm.content, list):
            for idx, block in enumerate(tm.content):
                if isinstance(block, ImageContentBlock):
                    user_messages.append(
                        UserMessage(content=[f"Here is the image for tool call {tm.tool_call_id}", block]),
                    )
                    tm.content[idx] = f"Done! The User will provide the image for tool call {tm.tool_call_id}"
                elif isinstance(block, str):
                    continue
                else:
                    raise NotImplementedError(f"Unsupported content block: {type(block)}")

    return tool_messages, user_messages


def _get_total_token_usage(messages: list[list[ChatMessage]]) -> list[TokenUsage]:
    """
    Returns a list of TokenUsage objects aggregated from all AssistantMessage
    instances across the provided grouped message history.

    Args:
        messages: A list where each item is a list of ChatMessage objects representing a segment
                  or turn group of the conversation history.

    Returns:
        List of TokenUsage corresponding to each AssistantMessage in the flattened conversation history.
    """
    return [msg.token_usage for msg in chain.from_iterable(messages) if isinstance(msg, AssistantMessage)]


def _get_tool_durations(messages: list[list[ChatMessage]]) -> dict[str, list[float]]:
    """Collect tool execution durations grouped by tool name from message history."""
    durations: dict[str, list[float]] = {}
    for msg in chain.from_iterable(messages):
        if isinstance(msg, ToolMessage) and msg.name and msg.tool_duration is not None:
            durations.setdefault(msg.name, []).append(msg.tool_duration)
    return durations


def _get_model_speed_stats(messages: list[list[ChatMessage]], model_slug: str) -> dict[str, float | int | str]:
    """Compute speed stats for this agent's model from AssistantMessages.

    Returns a flat dict with model_slug, num_calls, output_tokens, duration, e2e_otps.
    Returns empty dict if no timed messages found.
    """
    num_calls = 0
    output_tokens = 0
    duration = 0.0
    for msg in chain.from_iterable(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        if msg.request_start_time is None or msg.request_end_time is None:
            continue
        msg_duration = msg.request_end_time - msg.request_start_time
        if msg_duration <= 0:
            continue
        num_calls += 1
        output_tokens += msg.token_usage.output
        duration += msg_duration
    if num_calls == 0:
        return {}
    return {
        "model_slug": model_slug,
        "num_calls": num_calls,
        "output_tokens": output_tokens,
        "duration": duration,
        "e2e_otps": output_tokens / duration if duration > 0 else 0.0,
    }


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block for block in content if isinstance(block, str))
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _format_planner_summary(plan_artifact: Any) -> str:
    """Format a concise planner summary for the run log."""
    deliverables = ", ".join(
        deliverable.name for deliverable in plan_artifact.deliverables[:3] if getattr(deliverable, "name", None)
    ) or "none"
    phases = ", ".join(
        f"{phase.name}({phase.turn_budget})" for phase in plan_artifact.phases[:4] if getattr(phase, "name", None)
    ) or "none"
    rules = ", ".join(rule.rule_id for rule in plan_artifact.key_rules[:4] if getattr(rule, "rule_id", None)) or "none"
    risks = ", ".join(plan_artifact.risk_flags[:3]) or "none"
    understanding = plan_artifact.task_understanding.strip()
    if len(understanding) > 180:
        understanding = understanding[:177] + "..."
    return (
        "<planner_artifact>\n"
        f"task_understanding={understanding}\n"
        f"deliverables={deliverables}\n"
        f"phases={phases}\n"
        f"key_rules={rules}\n"
        f"risk_flags={risks}\n"
        "</planner_artifact>"
    )


class SubAgentParams(BaseModel):
    """Parameters for sub-agent tool invocation."""

    task: Annotated[str, Field(description="The task/prompt for the sub-agent to complete")]
    input_files: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="List of file paths to upload to the sub-agent's execution environment. "
            "Use paths from output_dir (e.g., files saved by previous sub-agents).",
        ),
    ]


DEFAULT_SUB_AGENT_DESCRIPTION = "A sub agent that can be used to handle a contained, specific task."

# Agent name validation pattern: alphanumeric, underscores, hyphens, 1-128 chars
AGENT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class Agent[FinishParams: BaseModel, FinishMeta]:
    """Agent that executes tool-using loops with automatic context management.

    Runs up to max_turns iterations of: LLM generation → tool execution → message accumulation.
    When conversation history exceeds context window limits, older messages are automatically
    condensed into a summary to preserve working memory.

    The Agent can be used as an async context manager via .session() for automatic tool
    lifecycle management, logging, and file saving:

        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        # Create client and agent
        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(client=client, name="assistant")

        async with agent.session(output_dir="./output") as session:
            finish_params, history, metadata = await session.run("Your task here")
    """

    def __init__(
        self,
        client: LLMClient,
        name: str,
        *,
        max_turns: int = AGENT_MAX_TURNS,
        system_prompt: str | None = None,
        tools: list[Tool | ToolProvider] | None = None,
        finish_tool: Tool[FinishParams, FinishMeta] | None = None,
        # Agent options
        context_summarization_cutoff: float = CONTEXT_SUMMARIZATION_CUTOFF,
        turns_remaining_warning_threshold: int = TURNS_REMAINING_WARNING_THRESHOLD,
        run_sync_in_thread: bool = True,
        text_only_tool_responses: bool = True,
        block_successive_assistant_messages: bool = True,
        # Subagent options
        share_parent_exec_env: bool = False,
        # Logging
        logger: AgentLoggerBase | None = None,
    ) -> None:
        """Initialize the agent with an LLM client and configuration.

        Args:
            client: LLM client for generating responses. Use ChatCompletionsClient for
                    OpenAI/OpenAI-compatible APIs, or LiteLLMClient for other providers.
            name: Name of the agent (used for logging purposes)
            max_turns: Maximum number of turns before stopping
            system_prompt: System prompt to prepend to all runs (when using string prompts)
            tools: List of Tools and/or ToolProviders available to the agent.
                   If None, uses DEFAULT_TOOLS. ToolProviders are automatically
                   set up and torn down by Agent.session().
                   Use [*DEFAULT_TOOLS, extra_tool] to extend defaults.
            finish_tool: Tool used to signal task completion. Defaults to SIMPLE_FINISH_TOOL.
            context_summarization_cutoff: Fraction of context window (0-1) at which to trigger summarization
            run_sync_in_thread: Execute synchronous tool executors in a separate thread
            text_only_tool_responses: Extract images from tool responses as separate user messages
            block_successive_assistant_messages: If True (default), automatically inject a continue
                                               message when assistant responds without tool calls to
                                               prevent successive assistant messages.
            share_parent_exec_env: When True and used as a subagent, share the parent's code
                                   execution environment instead of creating a new one. This
                                   provides better performance (no file copying) and allows
                                   the subagent to see all files in the parent's environment.
                                   Only effective when the agent is used as a subagent via to_tool().
            logger: Optional logger instance. If None, creates AgentLogger() internally.

        """
        # Validate agent name
        if not AGENT_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid agent name '{name}'. "
                "Agent names must match pattern '^[a-zA-Z0-9_-]{1,128}$' "
                "(alphanumeric, underscores, hyphens only, 1-128 characters)."
            )

        self._client: LLMClient = client
        self._name = name
        self._max_turns = max_turns
        self._system_prompt = system_prompt
        self._tools = tools if tools is not None else DEFAULT_TOOLS
        self._finish_tool: Tool = finish_tool if finish_tool is not None else SIMPLE_FINISH_TOOL
        self._context_summarization_cutoff = context_summarization_cutoff
        self._turns_remaining_warning_threshold = turns_remaining_warning_threshold
        self._run_sync_in_thread = run_sync_in_thread
        self._text_only_tool_responses = text_only_tool_responses
        self._block_successive_assistant_messages = block_successive_assistant_messages
        self._share_parent_exec_env = share_parent_exec_env

        # Logger (can be passed in or created here)
        self._logger: AgentLoggerBase = logger if logger is not None else AgentLogger()

        # Session configuration (set during session(), used in __aenter__)
        self._pending_output_dir: Path | None = None
        self._pending_input_files: str | Path | list[str | Path] | None = None
        self._pending_skills_dir: Path | None = None
        self._resume: bool = False
        self._clear_cache_on_success: bool = True
        self._cache_on_interrupt: bool = True

        # Instance-scoped state (populated during __aenter__, isolated per agent instance)
        self._active_tools: dict[str, Tool] = {}
        self._last_finish_params: Any = None  # FinishParams type parameter
        self._last_run_metadata: dict[str, list[Any]] = {}
        self._transferred_paths: list[str] = []  # Paths transferred to parent (for subagents)

        # Cache state for resumption (set during run(), used in __aexit__ for caching on interrupt)
        self._current_task_hash: str | None = None
        self._current_run_state: CacheState | None = None
        self._semantic_state_manager: SemanticStateManager | None = None

    @property
    def name(self) -> str:
        """The name of this agent."""
        return self._name

    @property
    def client(self) -> LLMClient:
        """The LLM client used by this agent."""
        return self._client

    @property
    def tools(self) -> dict[str, Tool]:
        """Currently active tools (available after entering session context)."""
        return self._active_tools

    @property
    def finish_tool(self) -> Tool:
        """The finish tool used to signal task completion."""
        return self._finish_tool

    @property
    def logger(self) -> AgentLoggerBase:
        """The logger instance used by this agent."""
        return self._logger

    def session(
        self,
        output_dir: Path | str | None = None,
        input_files: str | Path | list[str | Path] | None = None,
        skills_dir: Path | str | None = None,
        resume: bool = False,
        clear_cache_on_success: bool = True,
        cache_on_interrupt: bool = True,
    ) -> Self:
        """Configure a session and return self for use as async context manager.

        Args:
            output_dir: Directory to save output files from finish_params.paths
            input_files: Files to upload to the execution environment at session start.
                        Accepts a single path or list of paths. Supports:
                        - File paths (str or Path)
                        - Directory paths (uploaded recursively)
                        - Glob patterns (e.g., "data/*.csv", "**/*.py")
                        Raises ValueError if no CodeExecToolProvider is configured
                        or if a glob pattern matches no files.
            skills_dir: Directory containing skill definitions to load and make available
                       to the agent. Skills are uploaded to the execution environment
                       and their metadata is included in the system prompt.
            resume: If True, attempt to resume from cached state if available.
                   The cache is identified by hashing the init_msgs passed to run().
                   Cached state includes message history, current turn, and execution
                   environment files from a previous interrupted run.
            clear_cache_on_success: If True (default), automatically clear the cache
                                   when the agent completes successfully. Set to False
                                   to preserve caches for inspection or debugging.
            cache_on_interrupt: If True (default), set up a SIGINT handler to cache
                               state on Ctrl+C. Set to False when running agents in
                               threads or subprocesses where signal handlers cannot
                               be registered from non-main threads.

        Returns:
            Self, for use with `async with agent.session(...) as session:`

        Example:
            async with agent.session(output_dir="./output", input_files="data/*.csv") as session:
                result = await session.run("Analyze the CSV files")

        Note:
            Multiple concurrent sessions from the same Agent instance are supported.
            Each session maintains isolated state via ContextVar.

        """
        self._pending_output_dir = Path(output_dir) if output_dir else None
        self._pending_input_files = input_files
        self._pending_skills_dir = Path(skills_dir) if skills_dir else None
        self._resume = resume
        self._clear_cache_on_success = clear_cache_on_success
        self._cache_on_interrupt = cache_on_interrupt
        return self

    def _handle_interrupt(self, _signum: int, _frame: object) -> None:
        """Handle SIGINT to ensure caching before exit.

        Converts the signal to a KeyboardInterrupt exception so that __aexit__
        is properly called and can cache the state before cleanup.
        """
        raise KeyboardInterrupt("Agent interrupted - state will be cached")

    def _resolve_input_files(self, input_files: str | Path | list[str | Path]) -> list[Path]:
        """Resolve input file paths, expanding globs and normalizing to Path objects.

        Args:
            input_files: Single path or list of paths (strings, Paths, or glob patterns)

        Returns:
            List of resolved Path objects

        Raises:
            ValueError: If a glob pattern matches no files

        """
        # Normalize to list
        paths = [input_files] if isinstance(input_files, str | Path) else list(input_files)

        resolved: list[Path] = []
        for path in paths:
            path_str = str(path)

            # Check if it looks like a glob pattern
            if any(c in path_str for c in ("*", "?", "[")):
                # Expand glob pattern
                matches = glob_module.glob(path_str, recursive=True)
                if not matches:
                    raise ValueError(f"Glob pattern '{path_str}' matched no files")
                resolved.extend(Path(m) for m in matches)
            else:
                # Regular path - add as-is (upload_files will handle non-existent)
                resolved.append(Path(path))

        return resolved

    def _collect_all_tools(self) -> list[Tool | ToolProvider]:
        """Collect all tools from this agent and any sub-agents recursively."""
        all_tools: list[Tool | ToolProvider] = list(self._tools)

        for tool in self._tools:
            # Check if this tool wraps a sub-agent (created via to_tool())
            if isinstance(tool, Tool) and hasattr(tool, "executor"):
                # Check if the executor is a closure that captured an Agent
                closure = getattr(tool.executor, "__closure__", None)
                if closure:
                    for cell in closure:
                        try:
                            cell_contents = cell.cell_contents
                            if isinstance(cell_contents, Agent):
                                # Recursively collect from sub-agent
                                all_tools.extend(cell_contents._collect_all_tools())  # noqa: SLF001
                        except ValueError:
                            # cell_contents can raise ValueError if empty
                            pass

        return all_tools

    def _collect_warnings(self) -> list[str]:
        """Collect warnings about agent configuration."""
        warnings = []

        # Collect all tools including from sub-agents
        all_tools = self._collect_all_tools()

        # Check for LocalCodeExecToolProvider (security risk) - only in top-level agent
        for tool in self._tools:
            if isinstance(tool, LocalCodeExecToolProvider):
                warnings.append(
                    "LocalCodeExecToolProvider can access your local filesystem. "
                    "Consider using DockerCodeExecToolProvider or E2BCodeExecToolProvider for sandboxed execution.",
                )
                break

        # Check for missing default tools (across entire agent tree)
        for default_tool in DEFAULT_TOOLS:
            default_type = type(default_tool)

            # Special case: For code exec providers, check if ANY CodeExecToolProvider is present
            if isinstance(default_tool, CodeExecToolProvider):
                found = any(isinstance(t, CodeExecToolProvider) for t in all_tools)
            else:
                found = any(isinstance(t, default_type) for t in all_tools)

            if not found:
                warnings.append(f"Missing default tool: {default_type.__name__}")

        # Check for code execution tool per-agent (including sub-agents)
        agents_without_code_exec = self._collect_agents_without_code_exec()
        warnings.extend(
            f"Agent '{agent_name}' has no code execution tool. It will not be able to save files to the output directory."
            for agent_name in agents_without_code_exec
        )

        # Check for code execution without output directory
        state = _SESSION_STATE.get(None)
        if state and state.exec_env and not state.output_dir:
            warnings.append(
                "Code execution environment is configured but no output_dir is set. "
                "Files created by the agent will be lost when the session ends.",
            )

        return warnings

    def _build_system_prompt(self) -> str:
        """Build the complete system prompt: base + input files + user instructions.

        Returns:
            Complete system prompt string combining base prompt, input file listing,
            and user's custom system_prompt (if provided).
        """
        from stirrup.prompts import BASE_SYSTEM_PROMPT_TEMPLATE

        parts: list[str] = []

        # Base prompt with max_turns
        parts.append(BASE_SYSTEM_PROMPT_TEMPLATE.format(max_turns=self._max_turns))

        # User interaction guidance based on whether user_input tool is available
        if "user_input" in self._active_tools:
            parts.append(
                " You have access to the user_input tool which allows you to ask the user "
                "questions when you need clarification or are uncertain about something."
            )
        else:
            parts.append(" You are not able to interact with the user during the task.")

        # Input files section (if any were uploaded)
        state = _SESSION_STATE.get(None)
        if state and state.uploaded_file_paths:
            files_section = "\n\nThe following input files have been provided for this task:"
            for file_path in state.uploaded_file_paths:
                files_section += f"\n- {file_path}"
            parts.append(files_section)

        # Skills section (if skills were loaded)
        if state and state.skills_metadata:
            skills_section = format_skills_section(state.skills_metadata)
            if skills_section:
                parts.append(f"\n\n{skills_section}")

        # User's custom system prompt (if provided)
        if self._system_prompt:
            parts.append(f"\n\nFollow these instructions from the User:\n{self._system_prompt}")

        return "".join(parts)

    def _collect_agents_without_code_exec(self) -> list[str]:
        """Collect names of agents (including self and sub-agents) that lack a code execution tool."""
        agents_missing: list[str] = []

        # Check if this agent has a code execution tool
        has_code_exec = any(isinstance(t, CodeExecToolProvider) for t in self._tools)
        if not has_code_exec:
            agents_missing.append(self._name)

        # Recursively check sub-agents
        for tool in self._tools:
            if isinstance(tool, Tool) and hasattr(tool, "executor"):
                closure = getattr(tool.executor, "__closure__", None)
                if closure:
                    for cell in closure:
                        try:
                            cell_contents = cell.cell_contents
                            if isinstance(cell_contents, Agent):
                                agents_missing.extend(cell_contents._collect_agents_without_code_exec())  # noqa: SLF001
                        except ValueError:
                            pass

        return agents_missing

    def _validate_subagent_code_exec_requirements(self) -> None:
        """Validate that if any subagent has code exec, the parent must also have code exec.

        This validation ensures proper file transfer chain - subagent files transfer to
        parent's exec env, so parent must have one to receive them.

        Raises:
            ValueError: If a subagent has code exec but this parent doesn't.

        """
        parent_has_code_exec = any(isinstance(t, CodeExecToolProvider) for t in self._tools)

        for tool in self._tools:
            if isinstance(tool, Tool) and hasattr(tool, "executor"):
                closure = getattr(tool.executor, "__closure__", None)
                if closure:
                    for cell in closure:
                        try:
                            cell_contents = cell.cell_contents
                            if isinstance(cell_contents, Agent):
                                subagent = cell_contents
                                subagent_has_code_exec = any(
                                    isinstance(t, CodeExecToolProvider)
                                    for t in subagent._tools  # noqa: SLF001
                                )

                                if subagent_has_code_exec and not parent_has_code_exec:
                                    raise ValueError(
                                        f"Subagent '{subagent._name}' has a code execution tool, "  # noqa: SLF001
                                        f"but parent agent '{self._name}' does not. "
                                        f"Parent must have a code execution tool to receive files from subagent."
                                    )

                                # Recursively validate nested subagents
                                subagent._validate_subagent_code_exec_requirements()  # noqa: SLF001
                        except ValueError as e:
                            if "code execution tool" in str(e):
                                raise
                            # cell_contents can raise ValueError if empty - ignore

    async def __aenter__(self) -> Self:
        """Enter session context: set up tools, logging, and resources.

        Creates a new SessionState and stores it in the _SESSION_STATE ContextVar,
        allowing concurrent sessions from the same Agent instance.
        """
        exit_stack = AsyncExitStack()
        await exit_stack.__aenter__()

        # Get parent state if exists (for subagent file transfer)
        parent_state = _SESSION_STATE.get(None)

        current_depth = _PARENT_DEPTH.get()

        # Create session state and store in ContextVar
        state = SessionState(
            exit_stack=exit_stack,
            output_dir=str(self._pending_output_dir) if self._pending_output_dir else None,
            parent_exec_env=parent_state.exec_env if parent_state else None,
            depth=current_depth,
            logger=self._logger,
        )
        _SESSION_STATE.set(state)

        try:
            # === TWO-PASS TOOL INITIALIZATION ===
            # First pass initializes CodeExecToolProvider so that dependent tools
            # (like ViewImageToolProvider) can access state.exec_env in second pass.
            active_tools: list[Tool] = []

            # Check if we should share parent's exec_env (subagent with share_parent_exec_env=True)
            should_share_exec_env = (
                self._share_parent_exec_env
                and current_depth > 0
                and parent_state is not None
                and parent_state.exec_env is not None
            )

            if should_share_exec_env:
                # SHARED EXEC ENV: Use parent's exec_env directly, don't create new one
                state.exec_env = parent_state.exec_env  # type: ignore[union-attr]
                state.exec_env_owned = False
                logger.debug(
                    "[%s __aenter__] Sharing parent's exec_env: %s (temp_dir=%s)",
                    self._name,
                    type(state.exec_env).__name__,
                    getattr(state.exec_env, "_temp_dir", "N/A"),
                )
                # Skip CodeExecToolProvider initialization but still need to add code exec tool
                # Create the tool from the shared exec_env using get_code_exec_tool()
                # (the exec_env is already entered by parent, so we just create the tool wrapper)
                if state.exec_env is None:
                    raise RuntimeError("Expected shared exec_env to be set, but it is None")
                code_exec_tool = state.exec_env.get_code_exec_tool()
                active_tools.append(code_exec_tool)
            else:
                # OWNED EXEC ENV: Initialize our own CodeExecToolProvider (at most one allowed)
                code_exec_providers = [t for t in self._tools if isinstance(t, CodeExecToolProvider)]
                if len(code_exec_providers) > 1:
                    raise ValueError(
                        f"Agent can only have one CodeExecToolProvider, found {len(code_exec_providers)}: "
                        f"{[type(p).__name__ for p in code_exec_providers]}"
                    )

                if code_exec_providers:
                    provider = code_exec_providers[0]
                    result = await exit_stack.enter_async_context(provider)
                    if isinstance(result, list):
                        active_tools.extend(result)
                    else:
                        active_tools.append(result)
                    state.exec_env = provider
                    state.exec_env_owned = True

            # Second pass: Initialize remaining ToolProviders and static Tools
            for tool in self._tools:
                if isinstance(tool, CodeExecToolProvider):
                    continue  # Already processed in first pass

                if isinstance(tool, ToolProvider):
                    # ToolProvider: enter context and get returned tool(s)
                    result = await exit_stack.enter_async_context(tool)
                    # Handle both single Tool and list[Tool] returns (e.g., MCPToolProvider)
                    if isinstance(result, list):
                        active_tools.extend(result)
                    else:
                        active_tools.append(result)
                else:
                    # Static Tool, use directly
                    active_tools.append(tool)

            # Build active tools dict with finish tool (stored on instance, not session)
            self._active_tools = {FINISH_TOOL_NAME: self._finish_tool}
            self._active_tools.update({t.name: t for t in active_tools})

            # Validate subagent code exec requirements (only at root level)
            if current_depth == 0:
                self._validate_subagent_code_exec_requirements()

            # Upload input files to exec_env if specified
            if self._pending_input_files:
                if not state.exec_env:
                    raise ValueError("input_files specified but no CodeExecToolProvider configured")

                logger.debug(
                    "[%s __aenter__] Uploading input files: %s, depth=%d, parent_exec_env=%s, parent_exec_env._temp_dir=%s, exec_env_owned=%s",
                    self._name,
                    self._pending_input_files,
                    state.depth,
                    type(state.parent_exec_env).__name__ if state.parent_exec_env else None,
                    getattr(state.parent_exec_env, "_temp_dir", "N/A") if state.parent_exec_env else None,
                    state.exec_env_owned,
                )

                if state.depth > 0 and state.parent_exec_env:
                    if not state.exec_env_owned:
                        # SHARED EXEC ENV: Files already accessible - no transfer needed
                        # Just record the paths as "uploaded" for system prompt
                        if isinstance(self._pending_input_files, (str, Path)):
                            state.uploaded_file_paths = [str(self._pending_input_files)]
                        else:
                            state.uploaded_file_paths = [str(p) for p in self._pending_input_files]
                        state.uploaded_file_records = [
                            {
                                "source_path": path,
                                "dest_path": path,
                                "size": None,
                                "status": "shared_parent_exec_env",
                            }
                            for path in state.uploaded_file_paths
                        ]
                        logger.debug(
                            "[%s __aenter__] Shared exec_env - files already accessible: %s",
                            self._name,
                            state.uploaded_file_paths,
                        )
                    else:
                        # SEPARATE EXEC ENV: Read files from parent's exec env, write to subagent's exec env
                        # input_files are paths within the parent's environment
                        result = await state.exec_env.upload_files(
                            *self._pending_input_files,
                            source_env=state.parent_exec_env,
                        )
                        logger.debug(
                            "[%s __aenter__] Upload result: uploaded=%s, failed=%s",
                            self._name,
                            result.uploaded,
                            result.failed,
                        )
                        state.uploaded_file_paths = [uf.dest_path for uf in result.uploaded]
                        state.uploaded_file_records = [
                            {
                                "source_path": str(uf.source_path),
                                "dest_path": uf.dest_path,
                                "size": uf.size,
                                "status": "uploaded",
                            }
                            for uf in result.uploaded
                        ]
                        if result.failed:
                            raise RuntimeError(f"Failed to upload files: {result.failed}")
                else:
                    # ROOT AGENT: Read files from local filesystem
                    resolved = self._resolve_input_files(self._pending_input_files)
                    result = await state.exec_env.upload_files(*resolved)
                    logger.debug(
                        "[%s __aenter__] Upload result: uploaded=%s, failed=%s",
                        self._name,
                        result.uploaded,
                        result.failed,
                    )
                    state.uploaded_file_paths = [uf.dest_path for uf in result.uploaded]
                    state.uploaded_file_records = [
                        {
                            "source_path": str(uf.source_path),
                            "dest_path": uf.dest_path,
                            "size": uf.size,
                            "status": "uploaded",
                        }
                        for uf in result.uploaded
                    ]
                    if result.failed:
                        raise RuntimeError(f"Failed to upload files: {result.failed}")
            self._pending_input_files = None  # Clear pending state

            # Upload skills directory if it exists and load metadata
            if self._pending_skills_dir:
                skills_path = self._pending_skills_dir
                if skills_path.exists() and skills_path.is_dir():
                    if state.exec_env:
                        logger.debug("[%s __aenter__] Uploading skills directory: %s", self._name, skills_path)
                        await state.exec_env.upload_files(skills_path, dest_dir="skills")
                    # Load skills metadata (even if no exec_env, for system prompt)
                    state.skills_metadata = load_skills_metadata(skills_path)
                    logger.debug("[%s __aenter__] Loaded %d skills", self._name, len(state.skills_metadata))
                self._pending_skills_dir = None  # Clear pending state
            elif parent_state and parent_state.skills_metadata:
                # Sub-agent: inherit skills from parent
                state.skills_metadata = parent_state.skills_metadata
                logger.debug("[%s __aenter__] Inherited %d skills from parent", self._name, len(state.skills_metadata))
                # Transfer skills directory from parent's exec_env to sub-agent's exec_env
                # (only if we have a separate exec_env)
                if state.exec_env and parent_state.exec_env and state.exec_env_owned:
                    await state.exec_env.upload_files("skills", source_env=parent_state.exec_env)

            # Configure and enter logger context
            self._logger.name = self._name
            self._logger.model = self._client.model_slug
            self._logger.max_turns = self._max_turns
            # depth is already set (0 for main agent, passed in for sub-agents)
            self._logger.__enter__()

            # Set up signal handler for graceful caching on interrupt (root agent only)
            if current_depth == 0 and self._cache_on_interrupt:
                self._original_sigint = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, self._handle_interrupt)

            return self

        except Exception:
            await exit_stack.__aexit__(None, None, None)
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit session context: save files, cleanup resources.

        File handling is depth-aware:
        - Root agent (depth=0): Saves files to local filesystem output_dir
        - Subagent (depth>0): Transfers files to parent's exec env at output_dir path
        """
        state = _SESSION_STATE.get()

        try:
            # Cache state on non-success exit (only at root level)
            should_cache = (
                state.depth == 0
                and (exc_type is not None or self._last_finish_params is None)
                and self._current_task_hash is not None
                and self._current_run_state is not None
            )

            logger.debug(
                "[%s __aexit__] Cache decision: should_cache=%s, depth=%d, exc_type=%s, "
                "finish_params=%s, task_hash=%s, run_state=%s",
                self._name,
                should_cache,
                state.depth,
                exc_type,
                self._last_finish_params is not None,
                self._current_task_hash,
                self._current_run_state is not None,
            )

            if should_cache:
                cache_manager = CacheManager(clear_on_success=self._clear_cache_on_success)

                exec_env_dir = state.exec_env.temp_dir if state.exec_env else None

                # Explicit checks to keep type checker happy - should_cache condition guarantees these
                if self._current_task_hash is None or self._current_run_state is None:
                    raise ValueError("Cache state is unexpectedly None after should_cache check")

                # Temporarily block SIGINT during cache save to prevent interruption
                original_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    cache_manager.save_state(
                        self._current_task_hash,
                        self._current_run_state,
                        exec_env_dir,
                    )
                finally:
                    signal.signal(signal.SIGINT, original_handler)
                self._logger.info(f"Cached state for task {self._current_task_hash}")
            # Save files from finish_params.paths based on depth
            if state.output_dir and self._last_finish_params and state.exec_env:
                paths = getattr(self._last_finish_params, "paths", None)
                if paths:
                    if state.depth == 0:
                        # ROOT AGENT: Save to local filesystem
                        output_path = Path(state.output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        logger.debug(
                            "[%s] ROOT AGENT (depth=0): Saving %d file(s) to local filesystem: %s -> %s",
                            self._name,
                            len(paths),
                            paths,
                            output_path,
                        )
                        result = await state.exec_env.save_output_files(paths, output_path, dest_env=None)
                        logger.debug(
                            "[%s] ROOT AGENT: Saved %d file(s), failed %d",
                            self._name,
                            len(result.saved),
                            len(result.failed),
                        )
                    else:
                        # SUBAGENT: Handle file transfer based on exec_env ownership
                        if not state.exec_env_owned:
                            # SHARED EXEC ENV: Files already in parent's env - no transfer needed
                            # Just record the paths for reporting to parent
                            self._transferred_paths = list(paths)
                            logger.debug(
                                "[%s] SUBAGENT (depth=%d, shared_exec_env): Files already in parent env: %s",
                                self._name,
                                state.depth,
                                self._transferred_paths,
                            )
                        elif state.parent_exec_env:
                            # SEPARATE EXEC ENV: Transfer to parent's exec env
                            logger.debug(
                                "[%s] SUBAGENT (depth=%d): Transferring %d file(s) to parent exec env: %s -> %s",
                                self._name,
                                state.depth,
                                len(paths),
                                paths,
                                state.output_dir,
                            )
                            result = await state.exec_env.save_output_files(
                                paths, state.output_dir, dest_env=state.parent_exec_env
                            )
                            # Store transferred paths for returning to parent
                            self._transferred_paths = [str(sf.output_path) for sf in result.saved]
                            logger.debug(
                                "[%s] SUBAGENT: Transferred %d file(s) to parent, failed %d. Paths: %s",
                                self._name,
                                len(result.saved),
                                len(result.failed),
                                self._transferred_paths,
                            )
                            if result.failed:
                                logger.warning("Failed to transfer some files to parent env: %s", result.failed)
                        else:
                            logger.warning(
                                "Subagent at depth %d has exec_env but no parent_exec_env. "
                                "Files will not be transferred.",
                                state.depth,
                            )
        finally:
            # Restore original signal handler (root agent only)
            if hasattr(self, "_original_sigint"):
                signal.signal(signal.SIGINT, self._original_sigint)
                del self._original_sigint

            # Exit logger context
            self._logger.finish_params = self._last_finish_params
            self._logger.run_metadata = self._last_run_metadata
            self._logger.output_dir = str(state.output_dir) if state.output_dir else None
            self._logger.__exit__(exc_type, exc_val, exc_tb)

            # Cleanup all async resources
            await state.exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def run_tool(self, tool_call: ToolCall, run_metadata: dict[str, list[Any]]) -> ToolMessage:
        """Execute a single tool call with error handling for invalid JSON/arguments.

        Returns a ToolMessage containing either the tool output or an error description.
        Metadata from the tool result is stored in the provided run_metadata dict.
        """
        tool = self._active_tools.get(tool_call.name)
        result: ToolResult
        args_valid = True

        # Ensure tool is tracked in metadata dict (even if no metadata returned)
        if tool_call.name not in run_metadata:
            run_metadata[tool_call.name] = []

        tool_start_time = perf_counter()

        if tool:
            try:
                # Normalize empty arguments to valid empty JSON object
                args = tool_call.arguments if tool_call.arguments and tool_call.arguments.strip() else "{}"
                params = tool.parameters.model_validate_json(args)

                # Set parent depth for sub-agent tools to read
                prev_depth = _PARENT_DEPTH.set(self._logger.depth)
                try:
                    if inspect.iscoroutinefunction(tool.executor):
                        result = await tool.executor(params)  # ty: ignore[invalid-await]
                    elif self._run_sync_in_thread:
                        # ty: ignore - type checker doesn't understand iscoroutinefunction narrowing
                        result = await anyio.to_thread.run_sync(tool.executor, params)  # ty: ignore[unresolved-attribute]
                    else:
                        # ty: ignore - iscoroutinefunction check above ensures this is sync
                        result = tool.executor(params)  # ty: ignore[invalid-assignment]
                finally:
                    _PARENT_DEPTH.reset(prev_depth)

                # Store metadata if present
                if result.metadata is not None:
                    run_metadata[tool_call.name].append(result.metadata)
            except ValidationError:
                LOGGER.debug(
                    "LLMClient tried to use the tool %s but the tool arguments are not valid: %r",
                    tool_call.name,
                    tool_call.arguments,
                )
                result = ToolResult(content="Tool arguments are not valid", success=False)
                args_valid = False
        else:
            LOGGER.debug(f"LLMClient tried to use the tool {tool_call.name} which is not in the tools list")
            result = ToolResult(content=f"{tool_call.name} is not a valid tool", success=False)

        tool_end_time = perf_counter()

        return ToolMessage(
            content=result.content,
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            args_was_valid=args_valid,
            success=result.success,
            tool_start_time=tool_start_time,
            tool_end_time=tool_end_time,
        )

    async def step(
        self,
        messages: list[ChatMessage],
        run_metadata: dict[str, list[Any]],
        turn: int = 0,
        max_turns: int = 0,
    ) -> tuple[AssistantMessage, list[ToolMessage], FinishParams | None, list[dict[str, Any]]]:
        """Execute one agent step: generate assistant message and run any requested tool calls.

        Args:
            messages: Current conversation messages
            run_metadata: Metadata storage for tool results
            turn: Current turn number (1-indexed) for logging
            max_turns: Maximum turns for logging

        Returns the assistant message, tool execution results, and finish tool call (if present).

        """
        assistant_message = await self._client.generate(messages, self._active_tools)

        # Log assistant message immediately
        if turn > 0:
            self._logger.assistant_message(turn, max_turns, assistant_message)

        finish_params: FinishParams | None = None
        finish_attempts: list[dict[str, Any]] = []
        tool_messages: list[ToolMessage] = []
        if assistant_message.tool_calls:
            tool_messages = []
            for tool_call in assistant_message.tool_calls:
                tool_message = await self.run_tool(tool_call, run_metadata)
                tool_messages.append(tool_message)

                if tool_message.name == FINISH_TOOL_NAME:
                    reported_paths: list[str] = []
                    try:
                        params = self._finish_tool.parameters.model_validate_json(tool_call.arguments)
                        reported_paths = list(params.paths)
                        if tool_message.success:
                            finish_params = params
                    except ValidationError:
                        params = None
                    missing_paths: list[str] = []
                    missing_match = re.search(r"Files do not exist: (\[.*?\])", _content_to_text(tool_message.content))
                    if missing_match:
                        try:
                            parsed_missing = ast.literal_eval(missing_match.group(1))
                            if isinstance(parsed_missing, list):
                                missing_paths = [str(item) for item in parsed_missing]
                        except Exception:
                            missing_paths = []
                    finish_attempts.append(
                        {
                            "turn": turn,
                            "status": "validated" if tool_message.success else "failed",
                            "reported_paths": reported_paths,
                            "validated_paths": reported_paths if tool_message.success else [],
                            "missing_paths": missing_paths,
                            "message": truncate_msg(_content_to_text(tool_message.content), 1000),
                        }
                    )

                # Log tool result immediately
                self._logger.tool_result(tool_message)

        return assistant_message, tool_messages, finish_params, finish_attempts

    async def summarize_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Condense message history using LLM to stay within context window."""
        task_context: list[ChatMessage] = list(takewhile(lambda m: not isinstance(m, AssistantMessage), messages))

        summary_prompt = [*messages, UserMessage(content=MESSAGE_SUMMARIZER)]

        # We need to pass the tools to the client so that it has context of tools used in the conversation
        summary = await self._client.generate(summary_prompt, self._active_tools)

        summary_bridge_prompt = MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE.format(summary=summary.content)
        summary_bridge = UserMessage(content=summary_bridge_prompt)
        acknowledgement_msg = UserMessage(content="Got it, thanks!")

        # Log the completed summary
        summary_content = summary.content if isinstance(summary.content, str) else str(summary.content)
        self._logger.context_summarization_complete(summary_content, summary_bridge_prompt)

        return [*task_context, summary_bridge, acknowledgement_msg]

    def _get_browser_runtime_config(self) -> dict[str, Any] | None:
        for tool in self._tools:
            if type(tool).__name__ != "BrowserUseToolProvider" and not hasattr(tool, "_tool_prefix"):
                continue
            tool_prefix = getattr(tool, "_tool_prefix", "")
            if tool_prefix and not any(name.startswith(f"{tool_prefix}_") for name in self._active_tools):
                continue
            return {
                "provider": type(tool).__name__,
                "headless": getattr(tool, "_headless", None),
                "executable_path": getattr(tool, "_executable_path", None),
                "cdp_url": getattr(tool, "_cdp_url", None),
                "use_cloud": getattr(tool, "_use_cloud", None),
            }
        return None

    def _summarize_file_list(self, paths: list[str], *, limit: int = 200) -> dict[str, Any]:
        paths = sorted(dict.fromkeys(paths))
        return {
            "count": len(paths),
            "paths": paths[:limit],
            "truncated": len(paths) > limit,
        }

    def _escape_notice_attr(self, value: str) -> str:
        return value.replace('"', "'")

    async def _record_initial_env_state(self, *, written_turn: int) -> None:
        if self._semantic_state_manager is None:
            return
        state = _SESSION_STATE.get(None)
        if state is None:
            return

        active_tool_names = sorted(self._active_tools)
        browser_config = self._get_browser_runtime_config()
        facts: dict[str, Any] = {
            "env.output_dir": state.output_dir or "",
            "env.exec_environment": {
                "available": state.exec_env is not None,
                "type": type(state.exec_env).__name__ if state.exec_env else None,
                "owned": state.exec_env_owned,
                "depth": state.depth,
                "temp_dir": str(state.exec_env.temp_dir) if state.exec_env and state.exec_env.temp_dir else None,
                "path_policy": "Use relative paths inside the execution environment.",
            },
            "env.exec_cwd": {
                "display": ".",
                "temp_dir": str(state.exec_env.temp_dir) if state.exec_env and state.exec_env.temp_dir else None,
                "path_policy": "Use relative paths from the execution environment root.",
            },
            "env.uploaded_files": state.uploaded_file_records,
            "env.tool_availability": {
                "active_tools": active_tool_names,
                "code_exec": state.exec_env is not None and "code_exec" in self._active_tools,
                "browser": any(name.startswith("browser_") for name in self._active_tools),
                "web_fetch": "fetch_web_page" in self._active_tools,
                "web_search": "web_search" in self._active_tools,
                "view_image": "view_image" in self._active_tools,
                "finish": FINISH_TOOL_NAME in self._active_tools,
            },
            "env.browser": {"available": False} if browser_config is None else {"available": True, **browser_config},
        }
        self._semantic_state_manager.record_runtime_env_states(
            facts,
            written_turn=written_turn,
            source="runtime",
            reason="session initialized",
        )
        await self._record_generated_files_env_state(written_turn=written_turn)

    async def _record_generated_files_env_state(self, *, written_turn: int) -> None:
        if self._semantic_state_manager is None:
            return
        state = _SESSION_STATE.get(None)
        if state is None or state.exec_env is None:
            return
        try:
            workspace_files = await state.exec_env.list_files(".")
        except Exception as exc:
            self._semantic_state_manager.record_runtime_env_state(
                "env.generated_files_scan_error",
                {"error": str(exc), "turn": written_turn},
                written_turn=written_turn,
                source="runtime",
                reason="failed to scan execution environment files",
            )
            return

        uploaded = set(state.uploaded_file_paths)
        generated = [
            path
            for path in workspace_files
            if path not in uploaded and not path.startswith("skills/")
        ]
        state.generated_files_snapshot = generated
        self._semantic_state_manager.record_runtime_env_states(
            {
                "env.workspace_files": self._summarize_file_list(workspace_files),
                "env.generated_files": self._summarize_file_list(generated),
            },
            written_turn=written_turn,
            source="runtime",
            reason="execution environment file scan",
        )

    def _extract_tool_error(self, tool_message: ToolMessage, *, turn_index: int) -> dict[str, Any] | None:
        text = _content_to_text(tool_message.content)
        tool_name = tool_message.name or "unknown"
        error: dict[str, Any] | None = None

        if not tool_message.args_was_valid:
            error = {
                "tool": tool_name,
                "turn": turn_index,
                "error_type": "invalid_tool_arguments",
                "message": "Tool arguments failed schema validation.",
            }
        elif not tool_message.success:
            error = {
                "tool": tool_name,
                "turn": turn_index,
                "error_type": "tool_failed",
                "message": truncate_msg(text, 1000),
            }

        if tool_name == "code_exec":
            error_kind_match = re.search(r"<error_kind>(.*?)</error_kind>", text, flags=re.DOTALL)
            exit_code_match = re.search(r"<exit_code>(-?\d+)</exit_code>", text)
            stderr_match = re.search(r"<stderr>(.*?)</stderr>", text, flags=re.DOTALL)
            details_match = re.search(r"<details>(.*?)</details>", text, flags=re.DOTALL)
            stderr = (stderr_match or details_match).group(1).strip() if (stderr_match or details_match) else ""
            exit_code = int(exit_code_match.group(1)) if exit_code_match else None
            if error_kind_match or (exit_code is not None and exit_code != 0):
                error = {
                    "tool": tool_name,
                    "turn": turn_index,
                    "error_type": error_kind_match.group(1).strip() if error_kind_match else "command_failed",
                    "exit_code": exit_code,
                    "message": truncate_msg(stderr or text, 1000),
                }

        if error is None:
            return None

        missing_module_match = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
        if missing_module_match:
            error["missing_dependency"] = missing_module_match.group(1)
            error["error_type"] = "missing_dependency"
        elif "ModuleNotFoundError" in text:
            error["error_type"] = "missing_dependency"
        elif "FileNotFoundError" in text or "No such file or directory" in text or "not found" in text.lower():
            error["error_type"] = "path_or_file_not_found"
        elif "PermissionError" in text or "permission denied" in text.lower():
            error["error_type"] = "permission_error"
        elif "timed out" in text.lower() or error.get("error_type") == "timeout":
            error["error_type"] = "timeout"
        elif tool_name in {"fetch_web_page", "web_search"}:
            error["error_type"] = "network_or_web_error"

        return error

    async def _record_tool_env_state(
        self,
        tool_messages: list[ToolMessage],
        *,
        turn_index: int,
        finish_attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._semantic_state_manager is None:
            return

        errors = [
            error
            for tool_message in tool_messages
            if (error := self._extract_tool_error(tool_message, turn_index=turn_index)) is not None
        ]
        if errors:
            latest = errors[-1]
            self._semantic_state_manager.record_runtime_env_state(
                "env.last_tool_error",
                latest,
                written_turn=turn_index,
                source="tool_runtime",
                reason="tool failure detected by runtime",
                notice=(
                    f'<auto_correction rule_id="env_last_tool_error" '
                    f'reason="{self._escape_notice_attr(str(latest.get("message", latest)))}"/>'
                ),
            )

            missing_dependencies = self._semantic_state_manager.get_env_state_json("env.missing_dependencies", [])
            if not isinstance(missing_dependencies, list):
                missing_dependencies = []
            known = {
                item.get("module") for item in missing_dependencies
                if isinstance(item, dict)
            }
            changed = False
            for error in errors:
                module = error.get("missing_dependency")
                if module and module not in known:
                    missing_dependencies.append(
                        {
                            "module": module,
                            "tool": error.get("tool"),
                            "turn": turn_index,
                            "status": "missing",
                        }
                    )
                    known.add(module)
                    changed = True
            if changed:
                self._semantic_state_manager.record_runtime_env_state(
                    "env.missing_dependencies",
                    missing_dependencies,
                    written_turn=turn_index,
                    source="tool_runtime",
                    reason="ModuleNotFoundError detected in code execution",
                    notice=(
                        '<auto_correction rule_id="env_missing_dependency" '
                        'reason="A required Python dependency is missing; install it in code_exec or use an available alternative."/>'
                    ),
                )

        if finish_attempts:
            finish_status = finish_attempts[-1]
            self._semantic_state_manager.record_runtime_env_state(
                "env.finish_status",
                finish_status,
                written_turn=turn_index,
                source="finish_tool",
                reason="finish tool result",
                notice=(
                    '<auto_correction rule_id="env_finish_failed" '
                    'reason="Finish failed; verify reported paths exist and call finish again."/>'
                    if finish_status.get("status") != "validated" else None
                ),
            )
            self._semantic_state_manager.record_runtime_env_states(
                {
                    "env.finish.reported_paths": finish_status.get("reported_paths", []),
                    "env.finish.validated_paths": finish_status.get("validated_paths", []),
                    "env.finish.missing_paths": finish_status.get("missing_paths", []),
                },
                written_turn=turn_index,
                source="finish_tool",
                reason="finish tool path validation",
            )

        await self._record_generated_files_env_state(written_turn=turn_index)

    async def _semantic_state_json_call(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        response = await self._client.generate(
            [
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            {},
        )
        return _extract_json_object(_content_to_text(response.content))

    async def _run_semantic_state_extractor(
        self,
        assistant_message: AssistantMessage,
        tool_messages: list[ToolMessage],
        turn_index: int,
    ) -> ExtractorArtifact:
        if self._semantic_state_manager is None:
            return ExtractorArtifact()

        system_prompt = (
            "You are a semantic-state extractor. Return only JSON with the schema "
            '{"variables":[{"name":"...","value":"...","category":"...","source":"...",'
            '"update_mode":"initial","reason":null,"derived_from":["..."]}]}. '
            "Extract stable variables present in tool results or the assistant reply that are missing from current state. "
            "Do not repeat already-known variables unless a correction is clearly required."
        )
        user_prompt = (
            f"Turn: {turn_index}\n\n"
            f"Known variables:\n{self._semantic_state_manager.serialize_current_state()}\n\n"
            f"Planned variables:\n{json.dumps(self._semantic_state_manager.planned_variables, ensure_ascii=True)}\n\n"
            f"Assistant message:\n{_content_to_text(assistant_message.content)}\n\n"
            f"Tool results:\n{json.dumps([_content_to_text(msg.content) for msg in tool_messages], ensure_ascii=True)}"
        )
        payload = await self._semantic_state_json_call(system_prompt, user_prompt)
        if payload is None:
            return ExtractorArtifact()
        try:
            return ExtractorArtifact.model_validate(payload)
        except ValidationError:
            return ExtractorArtifact()

    async def _run_semantic_state_auditor(
        self,
        assistant_message: AssistantMessage,
        tool_messages: list[ToolMessage],
        turn_index: int,
    ) -> ViolationArtifact:
        if self._semantic_state_manager is None:
            return ViolationArtifact()

        system_prompt = (
            "You are a semantic-state rule auditor. Return only JSON with the schema "
            '{"violations":[{"rule_id":"...","severity":"soft|hard","message":"...",'
            '"resolution_strategy":"follow_rule|auto_patch|rollback","rollback_turn":1,'
            '"suggested_fix":[{"name":"...","value":"...","category":"...","source":"...",'
            '"update_mode":"overwrite","reason":"...","derived_from":["..."]}]}]}. '
            "Check the assistant reply and current state against the active rules. "
            "Use follow_rule unless you have a strong reason to override the rule's configured strategy."
        )
        user_prompt = (
            f"Turn: {turn_index}\n\n"
            f"Active rules:\n{self._semantic_state_manager.serialize_active_rules()}\n\n"
            f"Current state:\n{self._semantic_state_manager.serialize_current_state()}\n\n"
            f"Assistant message:\n{_content_to_text(assistant_message.content)}\n\n"
            f"Tool results:\n{json.dumps([_content_to_text(msg.content) for msg in tool_messages], ensure_ascii=True)}"
        )
        payload = await self._semantic_state_json_call(system_prompt, user_prompt)
        if payload is None:
            return ViolationArtifact()
        try:
            return ViolationArtifact.model_validate(payload)
        except ValidationError:
            return ViolationArtifact()

    async def run(
        self,
        init_msgs: str | list[ChatMessage],
        *,
        depth: int | None = None,
    ) -> tuple[FinishParams | None, list[list[ChatMessage]], dict[str, Any]]:
        """Execute the agent loop until finish tool is called or max_turns reached.

        A base system prompt is automatically prepended to all runs, including:
        - Agent purpose and max_turns info
        - List of input files (if provided via session())
        - User's custom system_prompt (if configured in __init__)

        Args:
            init_msgs: Either a string prompt (converted to UserMessage) or a list of
                      ChatMessage to extend the conversation after the system prompt.
            depth: Logging depth for sub-agent runs. If provided, updates logger.depth for this run.

        Returns:
            Tuple of (finish params, message history, run metadata).
            finish params is None if max_turns reached.
            run metadata maps tool/agent names to lists of metadata returned by each call.

        Example:
            # Simple string prompt
            await agent.run("Analyze this data and create a report")

            # Multiple messages
            await agent.run([
                UserMessage(content="First, read the data"),
                AssistantMessage(content="I've read the data file..."),
                UserMessage(content="Now analyze it"),
            ])

        """

        # Compute task hash for caching/resume
        task_hash = compute_task_hash(init_msgs)
        self._current_task_hash = task_hash

        # Initialize cache manager
        cache_manager = CacheManager(clear_on_success=self._clear_cache_on_success)
        start_turn = 0
        resumed = False

        # Try to resume from cache if requested
        if self._resume:
            state = _SESSION_STATE.get()
            cached = cache_manager.load_state(task_hash)
            if cached:
                # Restore files to exec env
                if state.exec_env and state.exec_env.temp_dir:
                    cache_manager.restore_files(task_hash, state.exec_env.temp_dir)

                # Restore state
                msgs = cached.msgs
                full_msg_history = cached.full_msg_history
                run_metadata = cached.run_metadata
                start_turn = cached.turn
                resumed = True
                self._logger.info(f"Resuming from cached state at turn {start_turn}")
            else:
                self._logger.info(f"No cache found for task {task_hash}, starting fresh")

        if not resumed:
            msgs: list[ChatMessage] = []
            run_metadata: dict[str, list[Any]] = {}

            uploaded_files: list[str] = []
            state = _SESSION_STATE.get(None)
            if state and state.uploaded_file_paths:
                uploaded_files = list(state.uploaded_file_paths)
            task_description = init_msgs if isinstance(init_msgs, str) else str(init_msgs[-1].content)
            self._logger.info("Running planner for %s uploaded file(s)", len(uploaded_files))
            planner = Planner(self._client, max_turns=self._max_turns)
            plan_artifact = await planner.plan(task_description=task_description, uploaded_files=uploaded_files)
            self._semantic_state_manager = SemanticStateManager(plan_artifact)
            run_metadata.setdefault("_planner", []).append(plan_artifact.model_dump())
            await self._record_initial_env_state(written_turn=0)

            # Build the complete system prompt (base + input files + user instructions)
            full_system_prompt = self._build_system_prompt()
            msgs.append(SystemMessage(content=full_system_prompt))

            if isinstance(init_msgs, str):
                msgs.append(UserMessage(content=init_msgs))
            else:
                msgs.extend(init_msgs)
            msgs.append(UserMessage(content=self._semantic_state_manager.build_turn_context(1)))

            full_msg_history: list[list[ChatMessage]] = []

        # Set logger depth if provided (for sub-agent runs)
        if depth is not None:
            self._logger.depth = depth

        # Log the task at run start (only if not resuming)
        if not resumed:
            self._logger.task_message(task_description)
            self._logger.user_message(UserMessage(content=_format_planner_summary(plan_artifact)))

        # Show warnings (top-level only, if logger supports it)
        if self._logger.depth == 0 and isinstance(self._logger, AgentLogger):
            run_warnings = self._collect_warnings()
            if run_warnings:
                self._logger.warnings_message(run_warnings)

        # Use logger callback if available and not overridden
        step_callback = self._logger.on_step

        full_msg_history: list[list[ChatMessage]] = []

        # Cumulative stats for spinner
        total_tool_calls = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for i in range(start_turn, self._max_turns):
            if self._semantic_state_manager is not None:
                self._semantic_state_manager.snapshot_turn(i + 1)
            # Capture current state for potential caching (before any async work)
            self._current_run_state = CacheState(
                msgs=list(msgs),
                full_msg_history=[list(group) for group in full_msg_history],
                turn=i,
                run_metadata=dict(run_metadata),
                task_hash=task_hash,
                agent_name=self._name,
            )
            if self._max_turns - i <= self._turns_remaining_warning_threshold and i != 0:
                num_turns_remaining_msg = _num_turns_remaining_msg(self._max_turns - i)
                msgs.append(num_turns_remaining_msg)
                self._logger.user_message(num_turns_remaining_msg)

            if self._semantic_state_manager is not None and i > start_turn:
                semantic_state_msg = UserMessage(content=self._semantic_state_manager.build_turn_context(i + 1))
                msgs.append(semantic_state_msg)
                self._logger.user_message(semantic_state_msg)
                self._semantic_state_manager.clear_pending_notices()

            # Pass turn info to step() for real-time logging
            assistant_message, tool_messages, finish_params, finish_attempts = await self.step(
                msgs,
                run_metadata,
                turn=i + 1,
                max_turns=self._max_turns,
            )

            # Update cumulative stats
            total_tool_calls += len(tool_messages)
            total_input_tokens += assistant_message.token_usage.input
            total_output_tokens += assistant_message.token_usage.output

            # Call progress callback after step completes
            if step_callback:
                step_callback(i + 1, total_tool_calls, total_input_tokens, total_output_tokens)

            user_messages: list[UserMessage] = []
            if self._text_only_tool_responses:
                tool_messages, user_messages = _handle_text_only_tool_responses(tool_messages)

            # Log user messages (e.g., image content extracted from tool responses)
            for user_msg in user_messages:
                self._logger.user_message(user_msg)

            msgs.extend([assistant_message, *tool_messages, *user_messages])

            if self._semantic_state_manager is not None:
                await self._record_tool_env_state(
                    tool_messages,
                    turn_index=i + 1,
                    finish_attempts=finish_attempts,
                )
                disputes = self._semantic_state_manager.parse_disputes(str(assistant_message.content))
                if disputes:
                    dispute_outcomes = self._semantic_state_manager.handle_disputes(
                        disputes,
                        turn_index=i + 1,
                    )
                    run_metadata.setdefault("_semantic_state_events", []).append(
                        {
                            "turn": i + 1,
                            "disputes": disputes,
                            "outcome": dispute_outcomes,
                        }
                    )
                updates = self._semantic_state_manager.parse_state_update(str(assistant_message.content))
                if updates:
                    self._semantic_state_manager.apply_updates(
                        updates,
                        written_turn=i + 1,
                        confidence="agent",
                    )
                extractor_artifact = await self._run_semantic_state_extractor(
                    assistant_message,
                    tool_messages,
                    i + 1,
                )
                self._semantic_state_manager.register_extractor_artifact(
                    extractor_artifact,
                    written_turn=i + 1,
                )
                violation_artifact = await self._run_semantic_state_auditor(
                    assistant_message,
                    tool_messages,
                    i + 1,
                )
                violation_outcome = self._semantic_state_manager.handle_violations(
                    violation_artifact,
                    turn_index=i + 1,
                )
                run_metadata.setdefault("_semantic_state_events", []).append(
                    {
                        "turn": i + 1,
                        "extractor": extractor_artifact.model_dump(),
                        "auditor": violation_artifact.model_dump(),
                        "outcome": violation_outcome,
                    }
                )
                run_metadata.setdefault("_semantic_state", []).append(
                    self._semantic_state_manager.serialize_persistent_state()
                )

            if finish_params:
                break

            pct_context_used = assistant_message.token_usage.total / self._client.max_tokens
            if pct_context_used >= self._context_summarization_cutoff and i + 1 != self._max_turns:
                self._logger.context_summarization_start(pct_context_used, self._context_summarization_cutoff)
                full_msg_history.append(msgs)
                msgs = await self.summarize_messages(msgs)

            # Avoid successive assistant messages (only if next turn won't show turns remaining)
            next_turn_will_show_warning = self._max_turns - (i + 1) <= self._turns_remaining_warning_threshold
            if (
                self._block_successive_assistant_messages
                and not tool_messages
                and not user_messages
                and not next_turn_will_show_warning
            ):
                msgs.extend([UserMessage(content="Please continue the task")])
        else:
            LOGGER.error(
                f"Maximum number of turns reached: {self._max_turns}. The agent was not able to finish the task. Consider increasing the max_turns parameter.",
            )

        full_msg_history.append(msgs)

        # Add agent's own token usage, tool durations, and model speed to run_metadata
        run_metadata["token_usage"] = _get_total_token_usage(full_msg_history)
        run_metadata["_tool_durations"] = _get_tool_durations(full_msg_history)  # type: ignore[assignment]
        run_metadata["_model_speed"] = _get_model_speed_stats(full_msg_history, self._client.model_slug)  # type: ignore[assignment]
        # Store for __aexit__ to access (on instance for this agent)
        self._last_finish_params = finish_params
        self._last_run_metadata = run_metadata

        # Clear cache on successful completion (finish_params is set)
        if finish_params is not None and cache_manager.clear_on_success:
            cache_manager.clear_cache(task_hash)
            self._current_task_hash = None
            self._current_run_state = None

        return finish_params, full_msg_history, run_metadata

    def to_tool(
        self,
        *,
        description: str = DEFAULT_SUB_AGENT_DESCRIPTION,
        system_prompt: str | None = None,
    ) -> Tool[SubAgentParams, SubAgentMetadata]:
        """Convert this Agent to a Tool for use as a sub-agent.

        Args:
            description: Tool description shown to the parent agent
            system_prompt: Optional system prompt to prepend when running

        Returns:
            Tool that executes this agent when called, returning SubAgentMetadata
            containing token usage, message history, and any metadata from tools
            the sub-agent used.

        """
        agent = self  # Capture self for closure

        async def sub_agent_executor(params: SubAgentParams) -> ToolResult[SubAgentMetadata]:
            """Execute the sub-agent with the given task.

            Sub-agents enter their own full session to ensure:
            1. Tool isolation - each agent only sees its own tools (fixes recursive sub-agent bug)
            2. Proper ToolProvider lifecycle - sub-agent's ToolProviders are initialized
            3. Correct logging - logger context is entered for proper output formatting
            """
            # Get parent's depth and calculate subagent depth
            parent_depth = _PARENT_DEPTH.get()
            sub_agent_depth = parent_depth + 1

            # Save parent's session state so we can restore it after subagent completes
            # This ensures sibling subagents see the parent's state, not a previous sibling's stale state
            parent_session_state = _SESSION_STATE.get(None)
            logger.debug(
                "[%s] PRE-SESSION: _SESSION_STATE=%s, exec_env=%s, exec_env._temp_dir=%s",
                agent.name,
                id(parent_session_state) if parent_session_state else None,
                type(parent_session_state.exec_env).__name__
                if parent_session_state and parent_session_state.exec_env
                else None,
                getattr(parent_session_state.exec_env, "_temp_dir", "N/A")
                if parent_session_state and parent_session_state.exec_env
                else None,
            )

            # Set _PARENT_DEPTH to subagent's depth BEFORE entering session
            # so that __aenter__ reads the correct depth for SessionState.depth
            prev_depth = _PARENT_DEPTH.set(sub_agent_depth)
            try:
                init_msgs: list[ChatMessage] = []
                if system_prompt:
                    init_msgs.append(SystemMessage(content=system_prompt))
                init_msgs.append(UserMessage(content=params.task))

                # Sub-agent enters its own full session for tool isolation and proper lifecycle
                # output_dir is a path within the parent's exec env (not local filesystem)
                # Files are transferred to parent's env at __aexit__ via save_output_files(dest_env=parent)
                async with agent.session(
                    output_dir=".",  # Path in parent's exec env
                    input_files=list(params.input_files) if params.input_files else None,  # ty: ignore[invalid-argument-type]
                ) as agent_session:
                    # Override logger depth for proper indentation in console output
                    agent_session._logger.depth = sub_agent_depth  # noqa: SLF001

                    finish_params, msg_history, run_metadata = await agent_session.run(init_msgs)

                    # Extract the last assistant message with actual content (not just tool calls)
                    last_assistant_msg: AssistantMessage | None = None
                    for msg_group in reversed(msg_history):
                        for msg in reversed(msg_group):
                            if isinstance(msg, AssistantMessage) and msg.content:
                                last_assistant_msg = msg
                                break
                        if last_assistant_msg:
                            break

                    # Build content from the assistant message and/or finish params
                    content_parts: list[str] = []

                    if last_assistant_msg and last_assistant_msg.content:
                        content = last_assistant_msg.content
                        if isinstance(content, list):
                            content = "\n".join(str(block) for block in content)
                        content_parts.append(content)

                    # Include finish params if available (they often contain the actual result)
                    if finish_params is not None:
                        finish_dict = finish_params.model_dump()
                        if finish_dict:
                            content_parts.append(f"Finish params: {finish_dict}")

                    # Report files transferred to parent's exec env (set in __aexit__)
                    transferred_paths = agent_session._transferred_paths  # noqa: SLF001
                    if transferred_paths:
                        content_parts.append(f"Files available in your environment: {transferred_paths}")

                    if not content_parts:
                        result_content = "<sub_agent_result>\n<error>No assistant message or finish params found</error>\n</sub_agent_result>"
                    else:
                        content = "\n".join(content_parts)
                        result_content = (
                            f"<sub_agent_result>"
                            f"\n<response>{content}</response>"
                            f"\n<finished>{finish_params is not None}</finished>"
                            f"\n</sub_agent_result>"
                        )

                    # Create subagent metadata with token usage, message history, and run metadata
                    sub_metadata = SubAgentMetadata(
                        message_history=msg_history,
                        run_metadata=run_metadata,
                    )

                    return ToolResult(content=result_content, metadata=sub_metadata)

            except Exception as e:
                # On error, return empty metadata
                error_metadata = SubAgentMetadata(
                    message_history=[],
                    run_metadata={},
                )
                return ToolResult(
                    content=f"<sub_agent_result>\n<error>{e!s}</error>\n</sub_agent_result>",
                    success=False,
                    metadata=error_metadata,
                )
            finally:
                # DEBUG: Log SESSION_STATE after subagent session
                post_session_state = _SESSION_STATE.get(None)
                logger.debug(
                    "[%s] POST-SESSION: _SESSION_STATE=%s, exec_env=%s, exec_env._temp_dir=%s",
                    agent.name,
                    id(post_session_state) if post_session_state else None,
                    type(post_session_state.exec_env).__name__
                    if post_session_state and post_session_state.exec_env
                    else None,
                    getattr(post_session_state.exec_env, "_temp_dir", "N/A")
                    if post_session_state and post_session_state.exec_env
                    else None,
                )

                # Restore parent's depth
                _PARENT_DEPTH.reset(prev_depth)
                # Restore parent's session state so next sibling subagent sees it
                if parent_session_state is not None:
                    _SESSION_STATE.set(parent_session_state)

        return Tool[SubAgentParams, SubAgentMetadata](
            name=self._name,
            description=description,
            parameters=SubAgentParams,
            executor=sub_agent_executor,  # ty: ignore[invalid-argument-type]
        )
