# Semantic State Tracking via Framework-Level Interception

## Problem Statement

Stirrup agent 在 gdpval 长程工作流评测中，语义定义变量（从文档中提取的数据、任务约束、中间计算结果）未被显式追踪。这些变量仅以隐式知识存在于对话上下文中，在长对话中发生漂移、丢失或前后不一致，导致交付物质量下降。

**目标**: 在 Stirrup 框架层引入 SemanticStateManager，通过 agent 自声明 + 小模型校验的双通道机制，显式追踪语义变量，并在每轮注入回 context，验证其对 gdpval 评分的提升效果。

## Compatibility Label

`v1-bootstrap-fit`

Stirrup 不是传统训练 repo，但 eval 系统（`evals/gdpval/`）可作为实验入口。需要一个薄适配器脚本将 eval 运行 + 分数提取封装为标准命令。适配器只负责运行和提取分数，不修改 eval 逻辑。

## Chosen Approach and Rationale

**方案 1: 自声明为主 + 提取器校验**

1. System prompt 指导 agent 在每轮输出 `<state_update>` XML 块声明变量更新
2. 框架用规则解析器（正则/XML）提取声明的变量
3. 每轮工具执行完毕后，一个小模型（haiku）对照 tool result 校验并补全遗漏变量
4. 验证后的状态字典在每轮开头以 `<current_state>` 块注入 context
5. 状态在上下文压缩（summarize）时独立保留

**选择理由**:
- 自声明让 agent 保持主体意识，格式遵守度本身可作为诊断信号
- 提取器兜底防止遗漏，成本可控（haiku 调用）
- 规则解析速度快、确定性强，符合"规则解码"设计理念

**排除的方案**:
- 方案 2（提取器主导）: agent 与状态解耦，每轮多一次完整提取调用，成本高
- 方案 3（专用工具声明）: 增加工具调用开销，agent 容易忽略调用

## Key Decisions and Trade-offs

| 决策 | 选择 | 理由 |
|------|------|------|
| 提取机制 | A+B 混合（自声明 + 提取器校验） | 自声明低成本，提取器补漏 |
| 状态格式 | XML 块（`<state_update>`, `<current_state>`） | 与 LLM 输出自然融合，正则可解析 |
| 状态注入位置 | 实验变量（system/user/tool_result 后均测） | 无先验最优，需实验验证 |
| 提取器模型 | haiku 级别 | 成本低、速度快、结构化提取足够 |
| 上下文压缩时处理 | 状态字典独立保留，不参与压缩 | 核心价值：防止压缩导致变量丢失 |

## Adapter Boundary

**需要的适配器**:

1. `scripts/autoresearch_run.py` — 薄包装脚本
   - 清空上一轮 results.jsonl
   - 调用 `python -m evals.gdpval --local --task-ids-file <subset> --grade --output-dir <run_dir>`
   - 从 results.jsonl 提取 aggregate_score，输出 `metric=<score>` 到 stdout
   - **不修改** eval 逻辑本身

2. `evals/gdpval/task_lists/fast_subset.txt` — 3 个代表性任务 ID（1 低分、1 中分、1 高分）

**适配器不可触碰的边界**:
- eval 的 task loading、grading、result 格式
- LLM client 实现
- 工具实现

## Open Questions

None — 所有设计决策已在交互中确认。

---

## Frozen Profile Fields

```
runtime.manager: uv
runtime.env_prep_command: uv pip install -e '.[all]'
runtime.entry_command: python scripts/autoresearch_run.py --task-ids-file evals/gdpval/task_lists/fast_subset.txt
runtime.timeout_seconds: 1200

experiment.time_budget_seconds: 12000
experiment.metric_name: aggregate_score
experiment.metric_direction: higher

edit_scope.allowed_paths:
  - src/stirrup/prompts/base_system_prompt.txt
  - src/stirrup/core/semantic_state.py
  - src/stirrup/core/agent.py
edit_scope.readonly_paths:
  - evals/
  - src/stirrup/clients/
  - src/stirrup/tools/
  - tests/
  - pyproject.toml
edit_scope.primary_edit_target: src/stirrup/prompts/base_system_prompt.txt

baseline.must_run_first: true
baseline.protocol: Run current codebase (no semantic state changes) on fast_subset tasks with grading enabled. Record aggregate_score as reference.
baseline.baseline_description: Current Stirrup agent with unmodified base_system_prompt.txt, no semantic state management.

git_policy.branch_prefix: autoresearch/semantic-state
git_policy.commit_before_run: true
git_policy.keep_commit_strategy: keep-current-commit
git_policy.discard_strategy: hard-reset-to-pre-run-commit
git_policy.crash_strategy: keep-crash-commit-for-inspection

logging.run_log_path: logs/autoresearch/<branch>/<timestamp>/run.log
logging.summary_extract_command: "python3 -c \"import re, sys\nfor line in sys.stdin:\n    m = re.search(r'metric=(\\d+)', line)\n    if m: print(m.group(1))\""
logging.results_columns:
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

## Experiment Variables

| 变量 | 候选值 |
|------|--------|
| state_trigger | every_turn / after_file_parse / after_tool_success |
| extractor_freq | every_turn / every_3_turns / only_when_agent_silent |
| inject_position | system_message / user_message_prefix / after_tool_result |
| state_granularity | data_fact+task_constraint / data_fact+task_constraint+env_state |
