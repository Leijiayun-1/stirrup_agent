# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Stirrup 是一个轻量级的 Python 框架，用于构建 AI 智能体。它的设计理念是与模型协作而非对抗，不强加僵化的工作流程，融合了 Claude Code 等领先智能体的最佳实践。

## 开发命令

### 环境配置
```bash
# 以可编辑模式安装所有依赖
pip install -e '.[all]'
# 或使用 uv:
uv venv && uv pip install -e '.[all]'
```

### 代码质量
```bash
# 格式化代码
uv run ruff format

# 代码检查
uv run ruff check

# 类型检查
uv run ty check
```

### 测试
```bash
# 运行所有测试
uv run pytest tests

# 运行单个测试文件
uv run pytest tests/test_agent.py

# 跳过可选依赖测试（docker、e2b、browser）
uv run pytest -m "not docker and not e2b and not browser"
```

### 文档
```bash
# 本地启动文档服务
uv run mkdocs serve
```

## 架构设计

### 核心组件

**Agent (`src/stirrup/core/agent.py`)**
- 主要编排器，运行智能体循环
- 管理对话历史和上下文摘要
- 处理工具执行和会话生命周期
- 关键概念：
  - `Agent.session()`: 资源生命周期管理的上下文管理器（工具、文件、清理）
  - `session.run()`: 执行智能体循环，直到调用 finish 工具或达到 max_turns
  - 子智能体：智能体可以生成子智能体，支持共享或隔离的执行环境

**Models (`src/stirrup/core/models.py`)**
- 核心数据结构：`ChatMessage`、`Tool`、`ToolProvider`、`ToolResult`
- `LLMClient` 协议：LLM 提供商的抽象接口
- 内容块：`ImageContentBlock`、`VideoContentBlock`、`AudioContentBlock`，支持自动格式转换
- `ToolProvider`：管理需要生命周期的工具（连接、临时目录等）
- `Tool`：简单的无状态可调用对象，包含名称、描述、参数、执行器

**Clients (`src/stirrup/clients/`)**
- `ChatCompletionsClient`：默认客户端，使用 OpenAI SDK，支持任何 OpenAI 兼容 API
- `OpenResponsesClient`：使用 OpenAI Responses API 格式
- `LiteLLMClient`：多提供商支持（需要 `stirrup[litellm]`）

### 工具系统

**Tool vs ToolProvider**
- `Tool`：用于简单操作的无状态可调用对象
- `ToolProvider`：通过异步上下文管理器管理资源（连接、临时目录等）

**默认工具 (`DEFAULT_TOOLS`)**
- `LocalCodeExecToolProvider`：在隔离的临时目录中执行 shell 命令
- `WebToolProvider`：网页抓取和搜索（搜索需要 `BRAVE_API_KEY`）

**可选工具**（需要显式导入和额外依赖）：
- `DockerCodeExecToolProvider`：在 Docker 中执行代码（`stirrup[docker]`）
- `E2BCodeExecToolProvider`：在 E2B 沙箱中执行代码（`stirrup[e2b]`）
- `MCPToolProvider`：MCP 服务器集成（`stirrup[mcp]`）
- `BrowserUseToolProvider`：浏览器自动化（`stirrup[browser]`）

### 技能系统

技能是 `skills/` 目录中的模块化指令包。每个技能包含：
- `SKILL.md`：YAML 前置元数据（名称、描述）+ 详细指令
- 可选的参考文档和资源
- 通过 `Agent(skills=["skills/data_analysis"])` 加载

### 上下文管理

- 接近上下文限制时自动进行对话摘要
- 截止阈值：`src/stirrup/constants.py` 中的 `CONTEXT_SUMMARIZATION_CUTOFF`
- 摘要器使用较小的模型压缩历史记录，同时保留关键信息

### 会话状态

`SessionState` 管理每个会话的资源：
- `exec_env`：代码执行环境（本地、Docker、E2B）
- `output_dir`：输出目录（根智能体为本地路径，子智能体为父环境中的路径）
- `depth`：智能体嵌套层级（0 = 根，>0 = 子智能体）
- `exec_env_owned`：会话是否拥有 exec_env（用于清理）
- 子智能体可以共享父级的 exec_env 或使用隔离环境

## 代码风格

- 需要 Python 3.12+
- 行长度：120 字符
- 使用 Ruff 进行格式化和检查（配置在 `pyproject.toml`）
- 需要类型提示（由 `ty` 类型检查器强制执行）
- 所有 I/O 操作使用 async/await

## 测试

- 测试使用 `pytest`，`anyio_mode = "auto"` 支持异步
- 单元测试使用模拟 LLM 客户端（参见 `tests/test_agent.py`）
- 可选依赖的标记：`@pytest.mark.docker`、`@pytest.mark.e2b`、`@pytest.mark.browser`
- 测试结构与源代码对应：`tests/test_*.py` 对应 `src/stirrup/*.py`
