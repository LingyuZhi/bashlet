# 已执行轨迹的 SFT 输入

`prepare-sft` 保留原有 `reference-implementation-v1` 三步参考示范模式。
加上 `--trajectories` 后，导出 `executed-trajectory-v1`，使用 schema 2。
新模式不要求参考程序、固定步数、反馈命令或每一步成功。

```bash
uv run --locked bashlet prepare-sft runs/harness-tasks-v1/manifest.json \
  --trajectories runs/my-review.json -c configs/executed-trajectories.yaml \
  --max-context-tokens 32768 --output-dir runs/my-sft-data
uv run --locked bashlet train-sft configs/my-sft-training.yaml \
  --output-dir runs/my-sft-training
```

训练配置的 `dataset` 指向导出的 `manifest.json`。沿用现有 `train-sft`、
交叉熵损失和手动自回归移位；不新增训练后端。

审阅文件示例：

```json
{
  "schema_version": 1,
  "trajectories": [{
    "run_dir": "sampled/task-run",
    "run_sha256": "完整 run.json 的 SHA-256",
    "actions": [
      {"supervise": false, "reason": "协议错误", "author": "model"},
      {"supervise": true, "reason": "失败测试揭示当前 bug", "author": "model"}
    ],
    "corrections": [{
      "reply": "```bash\n修正行动的 bash 命令\n```",
      "supervision": {
        "supervise": true,
        "reason": "修复实际反馈中发现的问题",
        "author": "codex"
      }
    }]
  }]
}
```

`run_dir` 相对于审阅文件所在目录解析，也可写绝对路径。
`actions` 与原轨迹逐步对应，包括不监督的行动。每步必须给出决定和理由。
运行测试发现 bug 可以接受监督，即使退出码非零。协议错误、不可执行的
bash 语法和连续无新状态或信息的重复行动不能接受监督。

离线导出先验证冻结的训练 Task、来源和编码，再从初始工作区重放原始
token，逐步比对真实观察和工作区内容。不会 decode 后重新 tokenize
已采样历史。Codex 修正只使用学生可用的 bash 操作，接在完整失败前缀后，
每一步都实际执行；已提交的原轨迹不能追加修正。
需要扩大原失败回合的步数或无效行动预算时，通过版本化配置或 `--set`
显式设置，新的预算指纹写入导出数据。

包含时间戳、随机输出等无法逐字复现观察的已有运行会被拒绝，原因和重放
证据保留。在线调用者也可在 `run_experiment` 返回后审阅实际轨迹，再调用
公开的 `export_demonstration(..., supervision=decisions)` 和自己的
`SFTAdapter`；这条路径保留刚执行的真实观察，不需要重新执行已知轨迹。
Codex 独立回放的模型身份须使用 `codex/` 前缀，不能把旧参考示范改标为 Codex。

有效行动监督完整 assistant 生成内容，包括代码、思考内容和生成的停止
token。系统提示、任务输入、真实观察、模板预填 token 和不监督的行动
权重为零。限制按完整历史 token 数计算；超过限制的样本明确拒绝，
不会裁掉前缀。未提交、评分失败或缺少修正的未完成来源也会记录拒绝原因。

产物记录原始模型身份、原始 token 与 logprob、作者、每步监督决定、
原始执行记录哈希、任务身份、完整提交快照和独立评分。训练加载器再次
检查因果历史、监督范围、移位和提交文件内容。所有这些静态 SFT 样本
均标记 `on_policy_eligible: false`。重放产生的 Generation 也保持
`source: replayed`，不能当作当前学生的 OPSD rollout。
