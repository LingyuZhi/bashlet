# 冻结的 Harness 示范 v1

训练入口使用 `runs/issue24-curated-harness/frozen-v1/data/manifest.json`。
`curated-harness-v1.lock.json` 固定本轮数据与证据的 SHA-256。
原始采样、人工审阅和回放观察分别保存，训练仍经过已有
`prepare_sft` / `load_demonstrations` 路径。

## 收集条件

`curated-harness-v1.json` 在采样前固定 60 个训练家族、逐任务内容哈希、
候选清单哈希、过滤规则与停止条件。每个家族只尝试一次，不重试或换题。
三类初始各 20 个；最多接纳 60 条，最多补充 3 条 Codex 修正。
开发与最终测试任务没有进入收集、修正或筛选。

采样模型为本地原始 Qwen3.5-4B，revision 为
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，vLLM 0.17.1，BF16，
thinking 关闭。实际权重、tokenizer、服务配置和原始 token/logprob
随轨迹保存。服务探针先核验身份，再允许采样。
`curated-harness-sampling-v1.yaml` 固定 temperature=1、top_p=1、seed=4，
每次生成最多 1024 token，每个回合最多 10 步、6144 生成 token、300 秒，
连续 3 次格式错误终止。没有新增付费 API 请求或训练更新。

## 使用方式

先按已有本地推理流程启动原始模型服务，再在计算节点运行：

```bash
uv run --locked bashlet collect-demonstrations configs/curated-harness-v1.json \
  -c configs/local-base.yaml -c configs/curated-harness-sampling-v1.yaml \
  --set model.base_url=http://127.0.0.1:8000 \
  --output-dir runs/my-harness-collection
```

输出目录必须不存在。`plan.json` 和已解析配置先落盘；每次尝试在执行前
登记，结束后保存运行哈希。基础设施错误保留不完整清单并停止，不补采。
清单与每个 Task 的来源、split、家族和内容均使用已有准入规则校验。

审阅文件使用 schema 1。`collection_sha256` 绑定完整 `collection.json`。
`attempts` 按尝试顺序覆盖全部记录，每项包含 `attempt`、`accept`、
`reason` 和逐步对应的 `actions`。每个行动写 `supervise`、`reason`、
`author`（model 或 codex）及人工判定的主要 `action_type`。
失败轨迹的行动也登记，但不进入训练。

定点修正使用 `correction_reason` 说明实际失败与覆盖缺口，
`corrections` 中每项为 `reply` 和 `supervision`；后者采用相同行动字段。
具体例子见本轮 `quality-review.json`。新的采样必须重新审阅，旧审阅
不能套用到另一份收集清单。

```bash
uv run --locked bashlet freeze-demonstrations \
  runs/issue24-curated-harness/collection/collection.json \
  --review runs/issue24-curated-harness/quality-review.json \
  -c configs/curated-harness-export-v1.yaml \
  --output-dir runs/my-frozen-demonstrations
```

导出配置的 14 步、8192 生成 token、连续 4 次格式错误上限只用于回放和
追加修正；原始采样条件不变。完整历史上限为 8192 token，不截掉失败前缀。
训练配置的 `dataset` 指向导出的 `data/manifest.json` 即可；本票不选择
训练超参数或启动下一票的小试。

## 实际交付

| 项目 | 短程序 | 代码修改 | 工作区 | 总计 |
| --- | ---: | ---: | ---: | ---: |
| 原始基模尝试 | 20 | 20 | 20 | 60 |
| 接纳轨迹／任务／家族 | 6 | 1 | 1 | 8 |
| 其中纯基模轨迹 | 6 | 0 | 0 | 6 |
| 其中 Codex 修正轨迹 | 0 | 1 | 1 | 2 |
| 距初始目标的缺口 | 14 | 19 | 19 | 52 |
| 训练输入 token | 5376 | 1231 | 1231 | 7838 |
| 受监督 token | 852 | 323 | 157 | 1332 |

8 条示范有 7846 个完整历史 token，手动自回归移位后输入为 7838。
每条输入长度 min/median/max 为 630/951/1262；监督长度为 76/156/323。
518 个监督 token 来自有效写入或修复行动，含完整代码正文；没有将写文件
行动统称为转义开销。其他监督 token 为检查 439、提交 359、读取 16。
11 个格式错误、2 个错误实现与 3 个无用重复读取共 576 个生成 token 被屏蔽。
系统提示、任务、环境观察和其他上下文也不接受监督。

60 次原始尝试中，10 次提交通过独立评分，2 次提交被评分拒绝，
其余 48 次未成功提交。人工选择 10 条成功轨迹和 2 条待修正前缀。
其中 4 条成功轨迹的 `ls -la` 含时间戳，精确回放无法复现，记录拒绝。
没有删掉这些行动或伪造原观察。基模自己完成的两条失败恢复示范分别修复
返回类型理解错误和不存在的 heapq API。Codex 仅为 `last-record` 与
`stock-value` 编写 8 个行动，经过真实执行与隐藏评分。

首轮导出与最终冻结各执行 12 次回放，共 24 次；每轮接纳相同的 8 个家族。
后一次还检查审阅与收集清单的哈希绑定。两轮回放均有单独目录；没有新增
基模尝试。重复回放不增加独立轨迹或家族数量，也不是 on-policy 样本。

这是一份很小且不均衡的静态 SFT 数据。代码修改与工作区各只有一条 Codex
修正，没有可接纳的纯基模样本；不能据此声称三类行为已充分覆盖，也不能
把收集成功比例当独立能力评测。后续小试只能检查链路与初步学习信号，
不足以证明泛化提升。未为凑满数量放宽质量或追加采样。

## 证据位置

- `runs/issue24-curated-harness/collection/`：采样前计划、实际服务身份、
  全部 60 次原始尝试及完整 token/logprob、提交与评分。
- `runs/issue24-curated-harness/quality-review.json`：逐行动决定、理由、
  人工行动类型与两条修正的覆盖依据。
- `runs/issue24-curated-harness/frozen-v1/`：可加载训练输入、单独回放轨迹、
  分布报告、拒绝清单与全部冻结文件哈希。
- `runs/issue24-curated-harness/acceptance.json`：原始 token 与完整历史抽核、
  加载验证、两轮回放计数与冻结哈希复核。

`frozen/` 保留首轮导出证据，后续训练使用 `frozen-v1/`。
生成文件遵循仓库约定保存在 `runs/`；本轮服务结束后恢复原 SFT 服务。
