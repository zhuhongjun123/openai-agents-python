"""评估脚本：对一组遥感论文分析问题批量跑多 Agent，输出保存到 eval_output/。

用法：
  $env:PYTHONIOENCODING='utf-8'
  & <env-python> examples/rs_paper_agent/eval_harness.py
"""

import asyncio
import json
from pathlib import Path

from rs_paper_agent import run_analysis

QUESTIONS = [
    "GLASS AVHRR 与 MODIS 的发射率差异对蒸散发反演的影响有多大？给出量化结论与验证范围。",
    "论文中生成全球 BBE 长时序产品用了多少采样点、覆盖什么时间范围？",
    "复杂地形下蒸散发反演引入 RH 反馈后 Bias 改善多少？",
    "地形各向异性对地表温度反演误差的影响有哪些量化结果？",
    "短波蓝空反照率校准后高海拔地区差异降低多少？",
]


async def main() -> None:
    out_dir = Path(__file__).resolve().parent / "eval_output"
    out_dir.mkdir(exist_ok=True)
    results: list[dict] = []
    for i, q in enumerate(QUESTIONS, start=1):
        print(f"[{i}/{len(QUESTIONS)}] {q[:30]}...")
        try:
            answer = await run_analysis(q)
        except Exception as exc:  # noqa: BLE001
            answer = f"[错误] {type(exc).__name__}: {exc}"
        (out_dir / f"q{i}.md").write_text(f"## 问题\n{q}\n\n## 回答\n{answer}\n", encoding="utf-8")
        results.append({"question": q, "answer": answer})
    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("完成，结果保存到", out_dir)


if __name__ == "__main__":
    asyncio.run(main())
