"""Eval-specific LLM client with corrected overflow detection."""
from __future__ import annotations

import logging
from typing import Any

from stirrup.clients.chat_completions_client import ChatCompletionsClient

LOGGER = logging.getLogger(__name__)


class EvalClient(ChatCompletionsClient):
    """ChatCompletionsClient，修复了 finish_reason='length' 误报为上下文溢出的问题。

    适用于 DeepSeek 等自带内部输出上限的模型：这些模型在达到自身输出上限时
    返回 finish_reason='length'，而框架会将其误判为输入上下文溢出。

    修复原理：
    - 将内部 _max_tokens 设为 1，使溢出检测条件 output_tokens >= _max_tokens 始终成立，
      finish_reason='length' 始终被视为输出截断（警告），而非上下文溢出（异常）。
    - 真实的 API max_completion_tokens 通过 kwargs 传入（覆盖 request_kwargs 中的先前赋值）。
    - max_tokens 属性覆盖后返回 context_window，agent.py 的摘要阈值计算不受影响。
    """

    def __init__(
        self,
        model: str,
        context_window: int = 64_000,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: LLM model identifier.
            context_window: 模型的真实上下文窗口大小（token 数）。
                用于 agent.py 的上下文摘要阈值。DeepSeek-chat 为 64000。
            max_output_tokens: 每次 API 调用的最大输出 token 数。
                若为 None，默认使用 context_window。
            **kwargs: 透传给 ChatCompletionsClient（api_key、base_url 等）。
        """
        # Extract nested kwargs dict to avoid duplicate key passing
        extra_api_kwargs = kwargs.pop("kwargs", {})
        # Inject real output limit, overrides max_completion_tokens in request_kwargs
        extra_api_kwargs.setdefault("max_completion_tokens", max_output_tokens or context_window)

        # max_tokens=1: makes overflow detection condition always True, disabling false positives
        super().__init__(model=model, max_tokens=1, kwargs=extra_api_kwargs, **kwargs)
        self._context_window_size = context_window

    @property
    def max_tokens(self) -> int:
        """返回真实上下文窗口大小，供 agent.py 计算摘要阈值。"""
        return self._context_window_size
