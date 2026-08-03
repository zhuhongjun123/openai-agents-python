# 遥感论文分析多 Agent（rs_paper_agent）

基于 **OpenAI Agents SDK** 的多 Agent 示例改造：面向遥感科研论文问答与量化结论提炼。

## 架构

```
用户问题 → [Planner] 子任务清单 → [Researcher] 检索语料+带来源要点 → [Writer] 结构化报告
```

采用**顺序编排**（上一环节 final_output 显式传入下一环节），而非 SDK 默认 handoffs：
在 DeepSeek 等 OpenAI 兼容接口上，自动 handoff 容易因工具反复调用超出轮次上限；
顺序编排链路可复现、可单环节重跑、成本可控，面试时可讲清这一工程取舍。

## 环境

- conda env：`agent-project`（Python 3.12），SDK 以 editable 安装：`pip install -e <repo>`
- 模型：DeepSeek `deepseek-chat`（OpenAI 兼容接口，`OpenAIChatCompletionsModel` + 自定义 client）
- Key：复制 `.env.example` 为 `.env` 填写（`.env` 已 gitignore，勿提交）

## 语料

- `knowledge_base/key_findings.md`：结构化关键发现知识库（随仓库提交，检索优先命中）；
- `corpus/`（gitignored）：放入论文/资料的 `*.md` 可被全文检索；
- 也可用 `CORPUS_DIR` 环境变量指向外部目录（例如本机 `科研成果/`）。

## 运行（Windows）

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:CORPUS_DIR='E:\实习\个人简历优化\科研成果'
& E:\Software\Python\anaconda3_2024\envs\agent-project\python.exe examples\rs_paper_agent\rs_paper_agent.py "你的问题"
```

批量评估：

```powershell
& <env-python> examples\rs_paper_agent\eval_harness.py
```

结果输出到 `eval_output/`。

## 相对上游的改造点（Ownership Build）

1. 新增领域示例目录与多 Agent 角色编排（Planner/Researcher/Writer 顺序编排）；
2. 自定义 `search_corpus` 工具（零依赖关键词检索，可按需升级向量检索）；
3. 评估脚本 + 领域问题集（量化回答质量/来源可追溯性）；
4. 国内 OpenAI 兼容模型接入模板（DeepSeek）。
