"""Baseline smoke test：OpenAI Agents SDK + 国内 OpenAI 兼容接口。

运行前设置环境变量（也可复制 .env.example 为 .env 后由 python-dotenv 读取）：
  LLM_BASE_URL=https://api.deepseek.com/v1                  # DeepSeek
  LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # 通义千问
  LLM_API_KEY=<你的 API Key>
  LLM_MODEL=deepseek-chat / qwen-plus

用法：
  conda run -n agent-project python examples/rs_paper_agent/baseline_smoke.py
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

load_dotenv(Path(__file__).resolve().parent / ".env")


def _env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"请先设置环境变量 {name}")
    return value


async def main() -> None:
    base_url = _env("LLM_BASE_URL")
    api_key = _env("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL", "deepseek-chat").strip()

    # 国内 OpenAI 兼容接口：自建 client + Chat Completions 模型
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    set_tracing_disabled(disabled=True)  # 无 OpenAI 平台 key 时关闭 tracing

    agent = Agent(
        name="Baseline",
        instructions="你是中文助手，回答简洁准确。",
        model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
    )

    result = await Runner.run(agent, "用一句话介绍你自己，并说明 Agent 运行成功。")
    print("=== final_output ===")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
