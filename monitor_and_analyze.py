#!/usr/bin/env python3
"""监控任务完成并自动分析所有轨迹"""
import json
import time
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("test_output")
RESULTS_FILE = OUTPUT_DIR / "results.jsonl"
TOTAL_TASKS = 30

def count_completed():
    if not RESULTS_FILE.exists():
        return 0
    with open(RESULTS_FILE) as f:
        return sum(1 for _ in f)

def load_all_results():
    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            results.append(json.loads(line))
    return results

def analyze_trajectory(task_id):
    """分析单个任务的轨迹"""
    traj_file = OUTPUT_DIR / task_id / "trajectory.jsonl"
    if not traj_file.exists():
        return None

    with open(traj_file) as f:
        msgs = [json.loads(line) for line in f]

    analysis = {
        'task_id': task_id,
        'total_turns': len(msgs),
        'tool_calls': defaultdict(int),
        'errors': [],
        'file_creations': [],
        'finish_turn': None,
        'py_file_pattern': []  # 检测写.py然后执行的模式
    }

    for i, msg in enumerate(msgs):
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                analysis['tool_calls'][tc['name']] += 1
                if tc['name'] == 'finish':
                    analysis['finish_turn'] = i

        if msg.get('role') == 'tool':
            content = str(msg.get('content', ''))

            # 检测错误
            if 'exit_code>1' in content or 'exit_code>100' in content:
                stderr = ''
                if '<stderr>' in content:
                    start = content.find('<stderr>') + 8
                    end = content.find('</stderr>')
                    stderr = content[start:end][:200]
                analysis['errors'].append({
                    'turn': i,
                    'tool': msg.get('name'),
                    'stderr': stderr
                })

            # 检测文件创建
            stdout = ''
            if '<stdout>' in content:
                start = content.find('<stdout>') + 8
                end = content.find('</stdout>')
                stdout = content[start:end]

            for line in stdout.split('\n'):
                if any(keyword in line.lower() for keyword in ['created:', 'saved', 'writing']):
                    analysis['file_creations'].append(line.strip()[:100])

        # 检测写.py文件然后执行的模式
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                if tc['name'] == 'code_exec':
                    try:
                        args = json.loads(tc.get('arguments', '{}'))
                        cmd = args.get('cmd', '')
                    except json.JSONDecodeError:
                        # 跳过无法解析的JSON
                        continue

                    # 检测 cat > *.py
                    if 'cat >' in cmd and '.py' in cmd:
                        analysis['py_file_pattern'].append(('write', i))
                    # 检测 python3 *.py
                    elif 'python3 ' in cmd and '.py' in cmd and 'EOF' not in cmd:
                        # 检查前一个是否是写.py
                        if analysis['py_file_pattern'] and analysis['py_file_pattern'][-1][0] == 'write':
                            prev_turn = analysis['py_file_pattern'][-1][1]
                            if i - prev_turn <= 4:  # 在4轮内
                                analysis['py_file_pattern'].append(('exec', i, prev_turn))

    return analysis

def generate_report(all_results, all_analyses):
    """生成最终的问题汇总报告"""
    report = []
    report.append("# GDPVal 评估完整分析报告\n")
    report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 总览
    total = len(all_results)
    success_count = sum(1 for r in all_results if r['success'])
    fail_count = total - success_count

    report.append("## 总览\n")
    report.append(f"- 总任务数: {total}\n")
    report.append(f"- 成功: {success_count} ({success_count*100//total}%)\n")
    report.append(f"- 失败: {fail_count} ({fail_count*100//total}%)\n")

    total_input = sum(r['token_usage']['input'] for r in all_results)
    total_output = sum(r['token_usage']['answer'] for r in all_results)
    report.append(f"- 总token消耗: {total_input:,} input + {total_output:,} output = {total_input+total_output:,}\n\n")

    # 问题统计
    report.append("## 问题模式统计\n\n")

    # 1. 写.py和执行分两步
    py_pattern_tasks = []
    for task_id, analysis in all_analyses.items():
        if analysis and analysis['py_file_pattern']:
            exec_count = sum(1 for p in analysis['py_file_pattern'] if p[0] == 'exec')
            if exec_count > 0:
                py_pattern_tasks.append((task_id[:8], exec_count))

    report.append(f"### 1. 写.py文件和执行分两步 (浪费轮次)\n")
    report.append(f"出现次数: {len(py_pattern_tasks)} 个任务\n")
    if py_pattern_tasks:
        report.append("涉及任务:\n")
        for tid, count in py_pattern_tasks[:10]:
            report.append(f"  - {tid}: {count}次\n")
    report.append("\n")

    # 2. finish拖延
    finish_delay_tasks = []
    for task_id, analysis in all_analyses.items():
        if analysis and analysis['finish_turn']:
            ratio = analysis['finish_turn'] / analysis['total_turns']
            if ratio > 0.8:  # 超过80%才finish
                finish_delay_tasks.append((task_id[:8], analysis['finish_turn'], analysis['total_turns']))

    report.append(f"### 2. finish调用拖延 (超过80%轮次才调用)\n")
    report.append(f"出现次数: {len(finish_delay_tasks)} 个任务\n")
    if finish_delay_tasks:
        report.append("涉及任务:\n")
        for tid, fturn, total in finish_delay_tasks[:10]:
            report.append(f"  - {tid}: 第{fturn}轮/{total}轮 ({fturn*100//total}%)\n")
    report.append("\n")

    # 3. 重复错误
    repeated_error_tasks = []
    for task_id, analysis in all_analyses.items():
        if analysis and len(analysis['errors']) >= 2:
            repeated_error_tasks.append((task_id[:8], len(analysis['errors'])))

    report.append(f"### 3. 重复错误 (同一任务出现2+次错误)\n")
    report.append(f"出现次数: {len(repeated_error_tasks)} 个任务\n")
    if repeated_error_tasks:
        report.append("涉及任务:\n")
        for tid, count in repeated_error_tasks[:10]:
            report.append(f"  - {tid}: {count}次错误\n")
    report.append("\n")

    # 4. 文件过度创建
    excessive_files_tasks = []
    for task_id, analysis in all_analyses.items():
        if analysis and len(analysis['file_creations']) > 10:
            excessive_files_tasks.append((task_id[:8], len(analysis['file_creations'])))

    report.append(f"### 4. 文件过度创建 (创建10+个文件)\n")
    report.append(f"出现次数: {len(excessive_files_tasks)} 个任务\n")
    if excessive_files_tasks:
        report.append("涉及任务:\n")
        for tid, count in excessive_files_tasks[:10]:
            report.append(f"  - {tid}: {count}个文件\n")
    report.append("\n")

    # 失败任务详情
    report.append("## 失败任务详情\n\n")
    for result in all_results:
        if not result['success']:
            tid = result['task_id'][:8]
            report.append(f"### Task {tid}\n")
            report.append(f"- Sector: {result['sector']}\n")
            report.append(f"- Occupation: {result['occupation']}\n")
            report.append(f"- Reason: {result['reason']}\n")
            if result.get('error'):
                report.append(f"- Error: {result['error']}\n")

            analysis = all_analyses.get(result['task_id'])
            if analysis:
                report.append(f"- 总轮次: {analysis['total_turns']}\n")
                report.append(f"- 错误数: {len(analysis['errors'])}\n")
            report.append("\n")

    # 成功但低效的任务
    report.append("## 成功但低效的任务 (存在多个问题)\n\n")
    inefficient_tasks = []
    for result in all_results:
        if result['success']:
            tid = result['task_id']
            analysis = all_analyses.get(tid)
            if not analysis:
                continue

            issues = []
            # 检查各种问题
            if any(p[0] == 'exec' for p in analysis['py_file_pattern']):
                issues.append("写.py分两步")
            if analysis['finish_turn'] and analysis['finish_turn'] / analysis['total_turns'] > 0.8:
                issues.append("finish拖延")
            if len(analysis['errors']) >= 2:
                issues.append("重复错误")
            if len(analysis['file_creations']) > 10:
                issues.append("文件过度创建")

            if len(issues) >= 2:
                inefficient_tasks.append((tid[:8], issues, analysis['total_turns']))

    for tid, issues, turns in inefficient_tasks[:15]:
        report.append(f"### Task {tid}\n")
        report.append(f"- 总轮次: {turns}\n")
        report.append(f"- 问题: {', '.join(issues)}\n\n")

    # 改进建议
    report.append("## 改进建议\n\n")
    report.append("### 1. System Prompt优化\n")
    report.append("- 强化\"使用heredoc直接执行Python，不要分步写.py文件\"\n")
    report.append("- 明确\"只创建任务明确要求的文件，不要创建中间文件、摘要、README等\"\n")
    report.append("- 强调\"完成核心交付物后立即调用finish，不要继续优化或创建额外文档\"\n\n")

    report.append("### 2. 警告机制调整\n")
    report.append("- 当前\"You have N turns remaining\"的措辞暗示需要填满所有轮次\n")
    report.append("- 建议改为\"If core deliverables are ready, call finish now\"\n\n")

    report.append("### 3. 错误处理改进\n")
    report.append("- 检测到相同错误重复时，应该改变策略而不是重试相同方法\n")
    report.append("- 对于常见错误（如python3 -c语法问题），应该在system prompt中预先说明\n\n")

    return ''.join(report)

def main():
    print("开始监控任务完成情况...")

    # 等待所有任务完成
    while True:
        completed = count_completed()
        print(f"[{time.strftime('%H:%M:%S')}] 进度: {completed}/{TOTAL_TASKS}")

        if completed >= TOTAL_TASKS:
            print("✓ 所有任务已完成，开始分析...")
            break

        time.sleep(300)  # 每5分钟检查一次

    # 加载所有结果
    all_results = load_all_results()
    print(f"加载了 {len(all_results)} 条结果")

    # 分析所有轨迹
    all_analyses = {}
    unique_tasks = set(r['task_id'] for r in all_results)
    print(f"开始分析 {len(unique_tasks)} 个唯一任务的轨迹...")

    for i, task_id in enumerate(unique_tasks, 1):
        print(f"  [{i}/{len(unique_tasks)}] 分析 {task_id[:8]}...")
        analysis = analyze_trajectory(task_id)
        if analysis:
            all_analyses[task_id] = analysis

    # 生成报告
    print("生成最终报告...")
    report = generate_report(all_results, all_analyses)

    # 保存报告
    report_file = OUTPUT_DIR / "EVALUATION_FINDINGS.md"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"✓ 报告已保存至: {report_file}")
    print(f"  总任务数: {len(all_results)}")
    print(f"  成功: {sum(1 for r in all_results if r['success'])}")
    print(f"  失败: {sum(1 for r in all_results if not r['success'])}")

if __name__ == '__main__':
    main()
