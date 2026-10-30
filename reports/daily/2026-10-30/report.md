# 隔夜研究报告 2026-10-30

状态：实时快照不可用，停止选股
生成时间：2026-10-30T18:52:04.702270+08:00

运行说明：使用腾讯原生接口，随时可运行；盘中获取实时，盘后获取收盘。

## 候选（分数不代表胜率）

无可用候选。看状态和错误明细。

## 数据与学习

{
  "state": "实时快照不可用，停止选股",
  "strategy_id": "manual-v6-tx-27d32c79d642",
  "errors": [
    {
      "task": "spot",
      "error": "ValueError: 未能获取到腾讯行情数据，可能是网络限制或接口变更\nTraceback (most recent call last):\n  File \"/home/runner/work/a-/a-/engine.py\", line 321, in _worker\n    raise ValueError(\"未能获取到腾讯行情数据，可能是网络限制或接口变更\")\nValueError: 未能获取到腾讯行情数据，可能是网络限制或接口变更\n"
    },
    {
      "error": "腾讯全市场快照返回为空"
    }
  ],
  "version": "manual-v6-tx",
  "finished_at": "2026-10-30T18:52:04.701712+08:00"
}
