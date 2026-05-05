# 使用 Stirrup 跑 AgentIF-OneDay

本文说明如何用当前仓库里的本地修改版 Stirrup 跑 `/home/shangguanyike/AgentIF-OneDay` 数据。不要重新下载 Stirrup。

## 1. 环境

推荐使用已有的 conda 环境或新建环境：

```bash
conda activate agentif-oneday-stirrup
```

如果该环境不存在，可以用已有 AgentIF/Stirrup 环境克隆：

```bash
conda create -n agentif-oneday-stirrup --clone agentif-stirrup-word
conda activate agentif-oneday-stirrup
```

然后在本地 Stirrup 仓库中安装当前代码和 AgentIF 依赖：

```bash
cd /home/shangguanyike/stirrup_update/stirrup_agent
python -m pip install -e '.[agentif-oneday]'
python -m pip install playwright
```

当前机器已有 `/usr/bin/google-chrome`，`.env` 默认会让 browser-use 使用系统 Chrome；只有没有系统 Chrome 时才需要额外运行 `python -m playwright install chromium`。

如果不想安装 editable 包，也可以在运行前设置：

```bash
export PYTHONPATH=/home/shangguanyike/stirrup_update/stirrup_agent/src:/home/shangguanyike/stirrup_update/stirrup_agent
```

## 2. 配置 `.env`

仓库根目录已创建 `.env`，请填写模型配置：

```bash
STIRRUP_MODEL=your-model-name
STIRRUP_API_KEY=your-api-key
STIRRUP_BASE_URL=https://your-openai-compatible-endpoint/v1

BRAVE_API_KEY=optional-brave-search-key
AGENTIF_ONEDAY_DATA_ROOT=/home/shangguanyike/AgentIF-OneDay/agentif_oneday_data
STIRRUP_BROWSER_HEADLESS=false
STIRRUP_BROWSER_EXECUTABLE_PATH=/usr/bin/google-chrome
```

`STIRRUP_API_KEY` 为空时，CLI 会继续尝试 `MODEL_API_KEY`、`OPENAI_API_KEY`、`OPENROUTER_API_KEY`。

## 3. 先做 dry-run 检查映射

```bash
cd /home/shangguanyike/stirrup_update/stirrup_agent
python -m evals.agentif_oneday \
  --suite excel \
  --question-ids taskif_88 \
  --output-dir ./agentif_oneday_outputs_excel88 \
  --dry-run
```

dry-run 不调用模型，只会生成：

- `run_manifest.json`
- `results.jsonl`
- `taskif_xxx/stirrup_payload.json`

用它确认 task、附件和输出目录映射是否正确。

## 4. 正式运行

运行单题：

```bash
python -m evals.agentif_oneday \
  --suite excel \
  --question-ids taskif_88 \
  --output-dir ./agentif_oneday_outputs_excel88 \
  --max-turns 30 \
  --include-score-criteria
```

运行某个子集的前 N 题：

```bash
python -m evals.agentif_oneday \
  --suite pdf \
  --max-tasks 3 \
  --output-dir ./agentif_oneday_outputs_pdf_smoke
```

运行完整 `data.jsonl`：

```bash
python -m evals.agentif_oneday \
  --suite all \
  --output-dir ./agentif_oneday_outputs_all \
  --concurrency 1
```

支持的 `--suite`：

- `all` / `data`: `agentif_oneday_data/data.jsonl`
- `excel`: `ifoneday_excel/excel.jsonl`
- `pdf`: `ifoneday_pdf/pdf.jsonl`
- `ppt`: `ifoneday_ppt/ppt.jsonl`
- `word`: `ifoneday_word/word.jsonl`

## 5. 输出

每次运行的输出目录包含：

- `run_manifest.json`: 本次运行配置
- `results.jsonl`: 每题状态汇总
- `taskif_xxx/stirrup_payload.json`: 发送给 Stirrup 的任务映射
- `taskif_xxx/stirrup_response.json`: finish 和运行元信息
- `taskif_xxx/trajectory.jsonl`: 完整对话轨迹
- `taskif_xxx/*`: 智能体生成的交付文件

默认会跳过 `results.jsonl` 中已经 `completed` 的题。需要强制重跑时加：

```bash
--overwrite
```

## 6. 常见调整

使用已有浏览器：

```bash
python -m evals.agentif_oneday \
  --suite all \
  --question-ids taskif_33 \
  --browser-cdp-url http://127.0.0.1:9222
```

覆盖数据路径：

```bash
python -m evals.agentif_oneday \
  --task-jsonl /path/to/tasks.jsonl \
  --attachment-dir /path/to/Questions \
  --attachment-search-root /path/to/agentif_oneday_data
```

如果网页任务不稳定，优先保持 `STIRRUP_BROWSER_HEADLESS=false`，并复用 `STIRRUP_BROWSER_PROFILE_DIR` 中的浏览器状态。当前机器已有 `/usr/bin/google-chrome`，默认通过 `STIRRUP_BROWSER_EXECUTABLE_PATH` 使用系统 Chrome。
