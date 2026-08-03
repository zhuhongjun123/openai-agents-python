# 运行日志

## 2026-08-03 Baseline 冒烟测试 ✅

- **环境**：Windows 11 / conda env `agent-project`（Python 3.12.13）/ openai-agents 0.19.2（editable）
- **模型**：DeepSeek `deepseek-chat`，走 OpenAI 兼容 Chat Completions 接口（`OpenAIChatCompletionsModel` + 自定义 client + 关闭 tracing）
- **命令**：直接调用 env python 运行 `examples/rs_paper_agent/baseline_smoke.py`（见下方坑）
- **耗时**：约 5 秒；**输出**：`我是中文助手，专注于简洁准确地解答问题；Agent 运行成功，已就绪。`
- **坑记录**：`conda run` 在 Windows 上捕获输出时用 GBK 解码导致 `UnicodeEncodeError`；改用 `conda run --no-capture-output` 或直接调用 `envs/agent-project/python.exe` + `PYTHONIOENCODING=utf-8` 解决。

## 2026-08-03 遥感论文分析多 Agent（Ownership Build 第一轮）✅

- **架构**：Planner（拆解任务）→ Researcher（`search_corpus` 检索）→ Writer（结构化报告），**顺序编排**，显式传递 final_output。
- **踩坑与修复**：
  1. SDK 自动 handoffs 在 DeepSeek 上不收敛（MaxTurnsExceeded）→ 改顺序编排，面试可讲"为什么不用 handoffs"；
  2. Researcher 反复调用工具循环（每次并行 2 个 query、不断换英文重查）→ 给工具加**去重守卫**（相似 query 命中 60% 直接提示停止）+ 限制单子任务检索次数；
  3. 中文 query 在英文全文语料上命中差，量化句排不进 top-2 → 新增 `knowledge_base/key_findings.md` 结构化关键发现库，并对知识库命中**加权 5 倍**；
  4. 句子级摘要：按句切分、优先含数字的句子，单文件最多 3 句、350 字符。
- **评估**：`eval_harness.py` 5 问全部通过（耗时约 80-110s），量化数据准确（-190%/24%/96 FLUXNET、31,248 采样点/1982–2019、+143→+51 W/m²、0.01/0.05–0.10、R² 0.055→0.0036）。
- **成本**：DeepSeek deepseek-chat，5 问约 15-20 次 LLM 调用，成本 < 0.5 元。
- **下一步**：Interview Pack（简历 4-5 行 / Q&A / 代码讲解 / PPT）→ GitHub 发布 → 网站接入。
