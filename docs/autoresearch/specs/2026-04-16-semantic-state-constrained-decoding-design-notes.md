# 语义状态追踪与规划器设计：约束解码思想的框架层实现

## 背景与问题定义

在 gdpval 长程工作流评测中，Stirrup agent 表现出以下系统性问题：

- **语义变量漂移**：同一个变量（如供应商单价）在不同 turn 中被赋予不同的值，前后不一致
- **推断重复**：已经从文档中提取过的数据，在后续 turn 中被重新推断，结论可能不同
- **约束遗忘**：任务要求（如"2页PDF"、"30张幻灯片"）在执行过程中被遗忘，导致交付物不达标
- **变量在 context 压缩中丢失**：对话历史被 summarize 后，隐式的推断结论随之消失
- **turn 预算失控**：没有阶段性规划，前期松散后期紧张，临近 turn 上限才匆忙提交

**核心洞察**：这些问题的共同根源是"语义定义变量未显式化"——重要的推断结论和任务约束只存在于模型的隐式上下文中，没有被框架层显式追踪和强制执行。

---

## 核心思想：约束解码的宏观借鉴

### 原始约束解码（token 级别）

约束解码（Constrained Decoding）是 LLM 推理时的一种技术：在每一步 token 生成时，不从整个词表中自由采样，而是只允许模型从"合法的下一个 token"集合中选择。这个合法集合由外部约束逻辑（FSM/正则/JSON Schema）实时计算得出。

实现上通过维护一个有限状态机（FSM）追踪当前输出状态，在每步推理前计算"token mask"，将不合法的 token 的 logit 设为负无穷，使其概率归零。Outlines、llama.cpp grammar、vLLM guided decoding 都是这类实现。

### 宏观借鉴：turn 级别的约束

我们不访问模型 logits（黑盒 API），而是借鉴约束解码的**思想**：

> **已推断过的结论不再重新推断，当前输出必须与已声明的规则和已有状态保持一致。**

把 FSM 的"合法 token 集合"放大到 turn 级别的"合法行为集合"：

| 约束解码概念 | 框架层对应物 |
|---|---|
| FSM（有限状态机） | SemanticStateManager（跟踪已推断的事实） |
| 产生式规则 / 文法 | Planner 输出的 `key_rules` + `phases` |
| token mask | 每轮注入的 `<current_state>` + 活跃规则 |
| "合法下一个 token" | "与当前状态和规则一致的下一步行为" |
| parse state 向前推进 | 语义状态随每轮工具执行更新 |
| 已决策的 token 不可逆 | 已存入状态的值是终态，不可矛盾引用 |

**Planner 是编译期**（解析任务描述，生成文法），**SemanticStateManager 是运行期**（维护 FSM 状态，计算每步的合法约束集）。

---

## 系统架构

```
task description
      │
      ▼
┌─────────────┐   structured JSON output（JSON Schema 强制）
│   Planner   │──────────────────────────────────────────┐
│  (pre-run,  │                                           │
│  1 LLM call)│                                           │
└─────────────┘                                           ▼
                                             ┌──────────────────────────┐
                                             │  Plan Artifact            │
                                             │  - phases + turn_budgets  │
                                             │  - key_rules              │
                                             │  - variables_to_extract   │
                                             │  - deliverables           │
                                             └──────────────┬───────────┘
                                                            │ 注入 system prompt
                                                            │ 初始化 SemanticStateManager
                                                            ▼
                                           ┌──────────────────────────────┐
                                           │       agent.run() loop        │
                                           │  每个 turn：                  │
                                           │  1. 注入 <phase_status>       │
                                           │     + <current_state>         │
                                           │     + <active_rules>          │
                                           │  2. agent 生成输出            │
                                           │     + <state_update>（自声明）│
                                           │  3. haiku 校验 + 补全         │
                                           │  4. 一致性 + 规则检查         │
                                           │  5. state_dict 更新           │
                                           └──────────────────────────────┘
```

---

## 组件一：Planner（前置规划器）

### 定位

Planner 在 `session.run()` 之前执行，是一次独立的 LLM 调用。当前 Stirrup 没有规划器，agent 在 turn 1 既要理解任务又要开始执行，等于边开车边看地图。Planner 的作用是**在执行开始前把任务结构显式化**。

### 输入与输出

- **输入**：用户任务描述 + 已上传的文件名列表
- **输出**：JSON Schema 强制的结构化计划（via OpenAI Structured Outputs API 或 prompt + 验证重试）

### Plan Artifact Schema

```json
{
  "task_understanding": "string",
  "deliverables": [
    {
      "name": "string",
      "format": "string",
      "constraints": ["string"]
    }
  ],
  "phases": [
    {
      "name": "string",
      "goal": "string",
      "turn_budget": "int",
      "variables_to_extract": ["string"]
    }
  ],
  "total_turn_budget": "int",
  "key_rules": ["string"],
  "risk_flags": ["string"]
}
```

### key_rules 的提取逻辑

Planner prompt 引导 LLM 对任务描述做三类扫描：

**扫描类型 1：显式约束（用户明确说的）**
```
"终止邮件发给 Juvoxa CEO"  → key_rule: "终止邮件收件人为 Juvoxa Optics CEO"
"财务分析用 INR"           → key_rule: "所有货币金额以 INR 表示"
"CPO 报告 2-3 页"          → key_rule: "CPO 报告长度 2-3 页"
```

**扫描类型 2：隐式领域约束（任务类型决定的）**
```
供应商对比报告  → key_rule: "Autonexis 与 Vendrax 必须在相同基准上比较"
```

**扫描类型 3：交付物完整性约束（从 deliverables 字段反推）**
```
要求"财务影响分析"  → key_rule: "报告必须包含：单价、工装成本、年度总成本对比"
```

**重要限制**：Planner 在执行前运行，尚未读过输入文件内容，只知道文件名。因此 key_rules 只能从任务描述推断，主要是**任务层面的约束**，而非数据层面的约束。

### variables_to_extract 的提取逻辑

Planner 问自己："要完成这个任务，我必须知道哪些具体数值？"然后按 phase 分配。

以任务 05389f78（汽车供应商分析）为例：

```
任务描述提到 quotes_1.docx → 文档解析 phase 需要：
  - autonexis_unit_price（报价单标准字段）
  - vendrax_unit_price
  - autonexis_tooling
  - vendrax_tooling
  - autonexis_lead_time
  - vendrax_lead_time
  - annual_volume（任务要求"商业影响"，需要这个才能算总成本）

任务要求"财务影响分析" → 分析 phase 需要计算并存储：
  - total_cost_autonexis_INR
  - total_cost_vendrax_INR
  - recovery_amount_INR（合同违约追偿金额）
```

**重要说明**：这个列表是基于任务描述的**最小预期集合**，不是穷举。Agent 在执行中可能发现额外变量，SemanticStateManager 允许动态添加。

### phases 与 turn_budget 的作用

解决轨迹分析中发现的"turn 预算失控"问题。Agent 在 turn 1 就知道各阶段边界：

```json
"phases": [
  {"name": "文档解析",  "turn_budget": 6,  "variables_to_extract": ["autonexis_unit_price", "..."]},
  {"name": "财务分析",  "turn_budget": 8,  "variables_to_extract": ["total_cost_autonexis_INR", "..."]},
  {"name": "文档生成",  "turn_budget": 6,  "variables_to_extract": []}
]
```

框架实时追踪当前 phase，在 context 中提示剩余 turns，防止在一个阶段过度停留。

---

## 组件二：SemanticStateManager（语义状态管理器）

### 双通道提取机制

**通道 A：Agent 自声明**

System prompt 要求 agent 在完成关键信息提取后，在输出末尾写 `<state_update>` XML 块：

```xml
<state_update>
  <variable name="autonexis_unit_price" value="USD 285"
            category="data_fact" source="quotes_1.docx:table1"/>
  <variable name="annual_volume"        value="12000 units"
            category="data_fact" source="quotes_1.docx:header"/>
</state_update>
```

框架用规则解析器（正则/XML parser）提取，写入 state_dict。

Variable 类别定义：

| category | 含义 | 示例 |
|---|---|---|
| `data_fact` | 从输入文档中提取的数据值 | 供应商单价、合同日期 |
| `task_constraint` | 来自用户任务描述的约束 | 页数限制、文件格式 |
| `intermediate_result` | 计算得出的中间值 | 年度总成本、汇率换算结果 |
| `env_state` | 执行环境状态 | 已安装的库、已创建的文件 |

**通道 B：Haiku 提取器校验**

每轮工具执行完毕后，haiku 对照 tool result 校验并补全遗漏变量：

```
prompt: "以下是工具执行结果。当前已记录的变量是：[autonexis_unit_price, annual_volume]。
         还有哪些变量应该被提取但未被 agent 声明？"
```

返回结构化 JSON，框架将补全的变量合并进 state_dict，标记 `confidence: extractor`（区别于 agent 自声明的 `confidence: agent`）。

### 每轮 Context 注入

在 agent 看到每一个新 turn 之前，框架注入三块信息：

```xml
<!-- 注入 1：当前 phase 状态 -->
<phase_status>
  当前阶段：文档解析 (turn 3/6)
  待提取变量：autonexis_unit_price ✗, vendrax_unit_price ✗,
              autonexis_tooling ✗, vendrax_tooling ✗,
              autonexis_lead_time ✗, annual_volume ✗
</phase_status>

<!-- 注入 2：已确认的语义状态 -->
<current_state>
  autonexis_unit_price = USD 285    (turn 4, source: quotes_1.docx, confidence: agent)
  annual_volume        = 12000 units (turn 4, source: quotes_1.docx, confidence: agent)
  vendrax_unit_price   = USD 310    (turn 4, source: quotes_1.docx, confidence: extractor)
</current_state>

<!-- 注入 3：全程活跃规则 -->
<active_rules>
  1. 所有货币金额以 INR 表示
  2. Autonexis 与 Vendrax 必须在相同基准上比较
  3. 终止邮件收件人为 Juvoxa Optics CEO
  4. CPO 报告长度 2-3 页
</active_rules>
```

### Phase 边界检查

每个 phase 的最后一个 turn 结束时，框架对比 `expected_variables` vs `extracted_variables`：

```
预期：autonexis_unit_price ✓  vendrax_unit_price ✓
      autonexis_tooling ✓      vendrax_tooling ✓
      autonexis_lead_time ✗    ← 缺失！
      annual_volume ✓
```

注入警告：

```xml
<phase_warning level="high">
  文档解析阶段结束，以下变量未提取：autonexis_lead_time
  建议：在进入分析阶段前补充提取，否则后续交货周期对比将无法完成。
</phase_warning>
```

### 一致性检查：解决语义变量漂移（核心机制）

这是最核心的机制，直接对应原始轨迹中发现的"数字漂移"问题。

**案例还原（任务 05389f78，turn 28）：**

Turn 12，agent 计算完成，状态字典写入：
```
total_cost_autonexis_INR = ₹28,500,000
（来源：USD 285 × 12000 × 汇率 8.33）
```

Turn 28，agent 在生成 CPO 报告时写了 `₹24,500,000`（与 turn 12 的结论不同）。

Haiku 一致性检查捕获到矛盾，框架注入：

```xml
<consistency_violation>
  检测到变量值与已存储结论冲突：
  你输出的 Autonexis 总成本 ₹24,500,000 与 turn 12 确认的
  total_cost_autonexis_INR = ₹28,500,000 不符。
  请使用已确认的值，不要重新计算。
</consistency_violation>
```

**这就是约束解码"已决策的 token 不可逆"在 turn 级别的对应**：已推断并存入状态的值是终态，后续引用必须与之一致。

### 规则违规检测

Haiku 在每轮检查 agent 输出是否违反 key_rules：

```xml
<rule_violation>
  检测到规则违反：active_rule[1] "所有货币金额以 INR 表示"
  你的输出中使用了 USD 2,850，请转换为 INR。
</rule_violation>
```

**为什么选择 warn 而非 hard block？**

在真正的约束解码中，非法 token 概率被设为 0，模型没有选择。但 hard block 在 agent 框架中有两个风险：
1. **卡死风险**：如果规则制定得不够好，hard block 可能让 agent 永远无法前进
2. **规则可能有例外**：比如 key_rule 说"价格用 INR"，但某一步 agent 需要先用美元查询再换算，这是合理的中间步骤

因此选择 **warn 模式**：注入 warning message，让 agent 在下一轮自行纠正。这是对约束解码思想的务实适配。

### 上下文压缩时的特殊处理

当对话历史触发 `summarize_messages()` 时，原始消息中的推断过程会被压缩丢弃。但 state_dict **独立保存，不参与压缩**，在压缩后重新注入为新的 `<current_state>` 块。

这直接解决了"长程任务中变量漂移/丢失"问题的根本原因——语义状态与对话历史解耦。

---

## 整体数据流

```
Planner（session.run() 之前）
  ├─ 扫描任务描述 → key_rules（全程约束）
  ├─ 推断需要的数值 → variables_to_extract（按 phase 分配）
  └─ 分配 turn 预算 → phases
           │
           ▼
SemanticStateManager 初始化
  ├─ expected_variables = Planner 的 variables_to_extract
  ├─ active_rules       = Planner 的 key_rules
  ├─ phase_schedule     = Planner 的 phases + turn_budgets
  └─ state_dict         = {}（空）

每个 turn：
  ① 框架注入：<phase_status> + <current_state> + <active_rules>
  ② Agent 生成输出 + <state_update>（自声明，机制 A）
  ③ 规则解析器提取 state_update → state_dict
  ④ Haiku 对照 tool_result 补全遗漏变量 → state_dict（机制 B）
  ⑤ Haiku 检查：
       a. key_rules 是否被违反？→ inject <rule_violation> warn
       b. 已有变量是否被矛盾引用？→ inject <consistency_violation> warn
  ⑥ Phase 边界检查：expected vs extracted → inject <phase_warning> if missing
  ⑦ state_dict 独立持久化（不参与 context 压缩）
```

---

## 解决的具体轨迹问题对照

| 原始轨迹问题 | 发生任务 | 对应机制 |
|---|---|---|
| turn 28 数字漂移（₹24.5M vs ₹28.5M） | 05389f78 | 一致性检查：state_dict 值是终态 |
| 生成5页而非2页PDF | 11e1b169 | key_rules 包含页数约束，每轮注入 |
| 14张而非30张幻灯片 | 9e8607e7 | deliverables.constraints 被监控 |
| 变量被 context 压缩丢失 | 多个任务 | state_dict 独立保存不参与压缩 |
| 调试循环中反复计算同一个值 | 40a8c4b1 | variables_to_extract 提前声明，提取后即引用 |
| turn 预算失控，最后紧急收尾 | 40a8c4b1, 9e8607e7 | phases + turn_budget 阶段性管控 |
| 文档解析失败后数据漂移 | 05389f78 | haiku 补全确保关键变量不漏提取 |

---

## 实验设计

### 兼容性标签

`v1-bootstrap-fit`

### 入口命令

```bash
python scripts/autoresearch_run.py \
  --task-ids-file evals/gdpval/task_lists/fast_subset.txt
```

### Metric

`aggregate_score`：fast_subset 3 个代表性任务的 LLM-as-Judge 评分之和（higher is better）

### Edit Scope

**可编辑**：
- `src/stirrup/prompts/base_system_prompt.txt`（自声明格式指令）
- `src/stirrup/prompts/planner_prompt.txt`（新建，规划器 system prompt）
- `src/stirrup/core/planner.py`（新建，pre-run LLM call + JSON Schema 验证）
- `src/stirrup/core/semantic_state.py`（新建，SemanticStateManager 实现）
- `src/stirrup/core/agent.py`（集成 planner 调用 + state manager hook）

**只读**：
- `evals/`（eval 逻辑、grader、数据加载）
- `src/stirrup/clients/`（LLM 客户端）
- `src/stirrup/tools/`（工具实现）
- `tests/`、`pyproject.toml`

### 实验变量矩阵

| 变量 | 候选值 |
|---|---|
| `plan_enforcement` | `structured_outputs_api` / `prompt_retry` |
| `state_trigger` | `every_turn` / `after_file_parse` / `after_tool_success` |
| `extractor_freq` | `every_turn` / `every_3_turns` / `only_when_agent_silent` |
| `inject_position` | `system_message` / `user_message_prefix` / `after_tool_result` |
| `state_granularity` | `data_fact+task_constraint` / `+intermediate_result` / `+env_state` |
| `rule_check_freq` | `every_turn` / `phase_boundary_only` |
| `planner_model` | `same_as_agent` / `haiku`（更快更便宜） |

### 实施顺序

1. **阶段一**：先实现并测试 Planner（不带 SemanticStateManager）→ 看规划质量和 turn 利用率
2. **阶段二**：加入 SemanticStateManager（不带 Planner）→ 看单独的状态追踪效果
3. **阶段三**：两者结合 → 看完整系统的协同效果

### Baseline

当前 Stirrup agent，无规划器，无语义状态管理，`base_system_prompt.txt` 未修改。必须先跑 baseline 建立 reference score。

---

## 开放问题

1. **Planner 的调用时机**：是在 `session.__aenter__()` 中调用，还是在 `session.run()` 的最开始？后者更自然，但前者可以在 session 设置阶段就完成规划。

2. **Plan 注入位置**：plan artifact 是作为 SystemMessage 的一部分注入，还是作为第一条 UserMessage 之前的专用消息块？SystemMessage 更稳定（不参与普通压缩），但某些 API 对 SystemMessage 的更新有限制。

3. **Variables_to_extract 的动态扩展**：当 agent 在执行中发现 Planner 未预见的重要变量时，是直接加入 state_dict，还是需要经过一个"注册"流程？动态添加灵活但可能引入噪声。

4. **Haiku 提取器的成本**：每轮一次 haiku 调用，在 60-turn 任务上会增加显著成本。是否需要一个"跳过条件"（如当前 turn 没有工具调用时跳过）？
