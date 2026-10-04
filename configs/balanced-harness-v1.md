# 均衡 Harness 示范 v1

训练数据入口为 `runs/issue28-balanced-harness/frozen-v1/data/manifest.json`。
`balanced-harness-v1.lock.json` 固定数据、来源、审阅、家族及配置身份。
详细结果见 [采集报告](../runs/issue28-balanced-harness/REPORT.md)。

数据采用 bash-block-v4、原始 Qwen3.5-4B 的学生 tokenizer 和关闭 thinking
的编码。覆盖原有 60 个训练家族，三类各 20 个。共接纳 62 条静态示范，
输入 58,924 token，监督 11,581 token；13 条含真实失败后的修复，占 20.97%。
两个家族各有第二条经审阅的互补示范。

来源为 Codex 参考辅助构造。原始文本、生成端参考上下文和作者单独保存。
生成端上下文仅含任务指令、初始文件及参考实现，不含隐藏评分材料。
学生上下文只含正常任务指令、行动和真实观察。每个行动保留完整 token；
系统、任务和观察的训练权重为零。回放及静态示范没有采样 logprob，
也不用于 on-policy 更新或教师评分。

正式计划固定每家族最多三次尝试、最多两条接纳示范，总尝试不超过 180 次，
候选不超过 120 条，采集墙钟上限 3600 秒。新增付费和 GPU 使用上限均为零。
来源耗尽、次数或时间上限、基础设施中断均保留停止理由及覆盖缺口。
该数据不承诺 SFT 能力收益；新训练方案属于任务 29。

## 复跑

保留原始冻结目录。以下新目录须尚不存在：

```bash
uv run --locked bashlet collect-demonstrations configs/balanced-harness-v1.json \
  -c configs/balanced-harness-v1.yaml -c configs/bash-block-v4.yaml \
  --output-dir runs/NEW-balanced-collection
```

逐行动审阅新执行结果。审阅使用 schema 2，按顺序覆盖每个 `attempt_id`。
每个行动填写 `supervise`、`reason`、`author` 与 `action_type`。
第二条示范填写 `complementary_reason`；工作区题填写 `workspace_use` 的
读取步、实现步、路径和实际信息利用理由。额外公开检查的恢复证据填写
`recovery` 的失败步、修改步、重新验证步及理由。

```bash
uv run --locked bashlet freeze-demonstrations runs/NEW-balanced-collection/collection.json \
  --review runs/NEW-balanced-review.json \
  -c configs/balanced-harness-v1.yaml -c configs/bash-block-v4.yaml \
  --output-dir runs/NEW-balanced-frozen
```

冻结入口执行精确回放、独立评分及实际训练加载。观察不一致、输入或反馈
被修改、无提交、评分失败及无意义重复均留下拒绝理由。旧 schema 1 计划
和旧训练数据继续使用原有路径，不改写历史字节。

仅重新加载既有数据，不启动训练：

```bash
uv run --locked python - <<'PY'
from pathlib import Path
from bashlet.training import load_demonstrations
manifest, datums = load_demonstrations(
    Path('runs/issue28-balanced-harness/frozen-v1/data/manifest.json')
)
print(manifest['counts'], len(datums))
PY
```
