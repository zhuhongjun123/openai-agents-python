"""遥感论文分析多 Agent 系统（基于 OpenAI Agents SDK）。

角色：
- Planner：把科研问题拆解为检索/分析子任务，交接给 Researcher；
- Researcher：调用 search_corpus 工具检索论文语料，输出带来源的要点，交接给 Writer；
- Writer：综合要点写成结构化中文报告（含量化数据与来源标注）。

编排方式：确定性顺序编排（sequential orchestration）——三个 Agent 各司其职，
上一环节的 final_output 显式作为下一环节输入。相比 SDK 的 handoffs 自动交接，
顺序编排在 DeepSeek 等兼容接口上更稳定、可复现、易讲清（取舍见 README）。

环境变量（.env 已 gitignore）：
  LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
  CORPUS_DIR  ：语料目录（*.md），默认取本目录 corpus/（gitignored）

用法（Windows，避免 conda run 的 GBK 坑）：
  $env:PYTHONIOENCODING='utf-8'
  & <env-python> examples\rs_paper_agent\rs_paper_agent.py "你的科研问题"
"""

import asyncio
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.decorators import tool

load_dotenv(Path(__file__).resolve().parent / ".env")

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
API_KEY = os.getenv("LLM_API_KEY", "").strip()
MODEL = os.getenv("LLM_MODEL", "deepseek-chat").strip()
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", Path(__file__).resolve().parent / "corpus"))
CORPUS_DIR.mkdir(exist_ok=True)
KB_DIR = Path(__file__).resolve().parent / "knowledge_base"

if not API_KEY:
    raise SystemExit("请先在 examples/rs_paper_agent/.env 中配置 LLM_API_KEY")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_tracing_disabled(disabled=True)  # 无 OpenAI 平台 key，关闭 tracing


def _model() -> OpenAIChatCompletionsModel:
    return OpenAIChatCompletionsModel(model=MODEL, openai_client=client)


_SEEN_QUERIES: list[set[str]] = []


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；\n])", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


@tool
def search_corpus(query: str, top_k: int = 2) -> str:
    """在遥感论文语料库（markdown 文件）中检索与 query 相关的句子，返回带文件来源的要点。
    同一主题只需检索一次；若返回"已检索过相似内容"，说明数据已获取，直接作答。"""
    files = sorted(KB_DIR.glob("*.md")) + sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        return "知识库/语料库为空：请确认 knowledge_base/ 与 CORPUS_DIR 存在。"
    terms = {t for t in re.split(r"[\s，。、；：（）()/]+", query.lower()) if len(t) >= 2}
    if not terms:
        return "请提供更具体的关键词。"
    # 去重守卫：与历史检索高度重叠时提示停止，避免模型反复检索不收敛
    for seen in _SEEN_QUERIES:
        if len(terms & seen) / max(1, len(seen)) >= 0.6:
            return "已检索过相似内容，直接基于已有检索结果作答，不要重复调用 search_corpus。"
    _SEEN_QUERIES.append(terms)

    scored: list[tuple[int, Path, list[str]]] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        lower = text.lower()
        is_kb = f.parent == KB_DIR
        # 知识库（结构化关键发现）加权，确保高信号句子优先命中
        hit = (5 if is_kb else 1) * sum(lower.count(t) for t in terms)
        sentences = [s for s in _split_sentences(text) if any(t in s.lower() for t in terms)]
        # 优先含数字的句子（量化数据），最多 3 句
        sentences.sort(key=lambda s: (0 if any(c.isdigit() for c in s) else 1, len(s)))
        scored.append((hit, f, sentences[:3]))
    scored.sort(key=lambda item: -item[0])
    out: list[str] = []
    for hit, f, sentences in scored[:top_k]:
        if not sentences:
            continue
        joined = " ".join(sentences)[:350]
        out.append(f"【来源 {f.name}】（命中 {hit} 次）\n{joined}")
    return "\n\n".join(out)


researcher = Agent(
    name="Researcher",
    instructions=(
        "你是遥感科研助理。针对给出的每个子任务，调用 search_corpus 工具检索论文语料，"
        "提取与任务相关的方法、数据、结果与结论，并标注来源文件名。"
        "每个子任务最多调用 search_corpus 一次；若返回'已检索过相似内容'，"
        "说明数据已获取，直接整理输出，不要重复检索。"
        "不得编造语料中不存在的数据；检索不到就明确说明。"
        "最终输出格式：按子任务逐条给出【结论】【依据（含数字/单位/来源）】。"
    ),
    tools=[search_corpus],
    model=_model(),
)

writer = Agent(
    name="Writer",
    instructions=(
        "你是遥感论文分析报告撰写者。基于 Researcher 提供的带来源要点，写结构化中文报告："
        "① 直接结论；② 依据与量化数据（保留数字、单位与范围）；③ 来源标注。"
        "报告不超过 400 字，数据必须来自要点，不得虚构。"
    ),
    model=_model(),
)

planner = Agent(
    name="Planner",
    instructions=(
        "你是遥感论文分析任务的规划者。把用户的科研问题拆解为 1-3 个具体检索子任务，"
        "只输出编号任务清单（每行一个），不要回答问题本身。"
    ),
    model=_model(),
)


async def run_analysis(question: str) -> str:
    # 环节 1：Planner 拆解任务
    plan = await Runner.run(planner, question, max_turns=5)
    # 环节 2：Researcher 检索语料并产出带来源要点
    findings = await Runner.run(
        researcher,
        f"原始问题：{question}\n\n子任务清单：\n{plan.final_output}",
        max_turns=12,
    )
    # 环节 3：Writer 综合成稿
    report = await Runner.run(
        writer,
        f"原始问题：{question}\n\nResearcher 要点：\n{findings.final_output}",
        max_turns=5,
    )
    return report.final_output


async def main(question: str) -> None:
    print("=== 最终报告 ===")
    print(await run_analysis(question))


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        q = "GLASS AVHRR 与 MODIS 的发射率差异对蒸散发反演的影响有多大？请给出量化结论与验证范围。"
    asyncio.run(main(q))
