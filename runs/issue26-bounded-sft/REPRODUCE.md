# Harness SFT 小试复跑

使用仓库根目录与锁定的 uv 环境。原始训练数据、模型权重和完整轨迹继续放在 `runs/`，本文件不复制它们。先校验 [preflight.json](preflight.json) 的数据、配置及冻结方案身份。数据为任务 24 的 8 条示范；开发规则为 `configs/sft-acceptance-v1.json`，冻结 SHA-256 为 `4265077a30b820c4deb97bfaf1d3c1022fdeb9437d96befa732945b7c29367ec`。

当前记录均已完成。不要覆盖它们，也不要把同 seed 重跑结果叠加成新证据。以下给出实际使用的入口；新实验必须使用新的输出目录，并将客户端与服务配置中的快照路径对应改为新目录。模型 identity 必须从新 `snapshot.json` 读取，不能沿用下方旧 identity。

## 训练与恢复

本次入口如下。只有检查前一个快照的生成和冻结评测后，才执行下一条。已有完整运行再次执行会拒绝额外更新。

```bash
uv run --env-file configs/bounded-harness-training.env --locked bashlet train-sft \
  configs/bounded-harness-sft-v1.yaml --output-dir runs/issue26-bounded-sft/training --stop-after 1
uv run --env-file configs/bounded-harness-training.env --locked bashlet train-sft \
  configs/bounded-harness-sft-v1.yaml --output-dir runs/issue26-bounded-sft/training --resume --stop-after 4
uv run --env-file configs/bounded-harness-training.env --locked bashlet train-sft \
  configs/bounded-harness-sft-v1.yaml --output-dir runs/issue26-bounded-sft/training --resume
```

配置明确 `max_epochs: 1`、batch=1、steps=8、learning rate=1e-5；sampler 权重只用于推理，恢复由完整 `training_state` 执行。继续使用原 `runs/spend.sqlite3` 和 `configs/spend.yaml`。不能新建账本、放宽额度或删除未知费用来复跑。

## 导出与本地服务

对 1、4、8 三个检查点分别使用对应目录。导出会把当时的训练 journal 保存为 `training/training-records/<sha256>.json`，使早期快照在训练恢复后仍可验证。以 step 8 为例：

```bash
uv run --locked bashlet export-snapshot runs/issue26-bounded-sft/training \
  --step 8 --api-key-env pytrio_api_key --output-dir runs/issue26-bounded-sft/step8-export
uv run --env-file configs/bounded-harness-local.env --locked bashlet prepare-local-snapshot \
  runs/issue21-local-inference/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --export runs/issue26-bounded-sft/step8-export/export.json --merge \
  --output-dir runs/issue26-bounded-sft/local-step8
```

在 `n30126` 先运行 `module avail`，本轮实际使用 `module load cuda/12.8`。核对当前 API server 的完整命令后停止，等待进程退出与显存释放，再启动目标服务：

```bash
bash scripts/serve-local-vllm.sh configs/local-vllm/serve-bounded-sft-step8.yaml
```

复用已有 18025 隧道；没有监听时才建立：

```bash
ssh -N -L 127.0.0.1:18025:127.0.0.1:8000 n30126
```

客户端层叠顺序为 `local-base.yaml`、对应 `bounded-harness-stepN.yaml`、`acceptance-local-v1.yaml`。协议、环境、预算、编码、BF16、服务配置、HTTP 超时与重试保持一致。实际服务 identity 和 token/logprob 探针由既有模型入口核验。

## 行为与开发评测

本次训练任务行为探针使用 `mbpp/623`、`harness/last-record`、`harness/stock-value`，均不计入开发成绩。step 4/8 的薄入口为：

```bash
uv run --locked python runs/issue26-bounded-sft/probe.py 8
```

原始输出目录已存在时，不要复跑该脚本覆盖索引；复制薄入口并改为新的诊断输出路径。核心执行和评分仍调用 `run_experiment`。

已有 step 8 在目标服务下进行独立复核时，使用新的输出目录，且标记为复跑，不纳入原始选优分母：

```bash
uv run --locked bashlet evaluate-acceptance configs/sft-acceptance-v1.json \
  --snapshot vllm/Qwen/Qwen3.5-4B@trio://run_19taybw3zq00/sampler_weights/sampler-0008:0df7c78acf25c64a9c8ce8587f0a93d2a3706837dc8052291f3dac2c2424a7e6 \
  -c configs/local-base.yaml -c configs/bounded-harness-step8.yaml -c configs/acceptance-local-v1.yaml \
  --monitor-config configs/bounded-harness-monitoring.yaml \
  --training-run runs/issue26-bounded-sft/training --output-dir runs/issue26-replay-step8
```

基线引用 `runs/issue25-acceptance/base-v1/acceptance.json`，其原 SHA-256 为 `108f341b6dfff119aba21f00203376b0eafcde3bac686b9bc575962296e70761`。完成 seed 4 后，有希望候选才用 `--stage confirmation --resume <acceptance.json>` 为候选及基线扩展 seed 17/29；完成结果复用，不重复计作新题。本轮没有有希望候选，因此没有执行确认。

## 选优和核验

```bash
uv run --locked bashlet select-sft runs/issue26-bounded-sft/training \
  --baseline runs/issue25-acceptance/base-v1/acceptance.json \
  --evaluation runs/issue26-bounded-sft/step1-screening/acceptance.json \
  --evaluation runs/issue26-bounded-sft/step4-screening/acceptance.json \
  --evaluation runs/issue26-bounded-sft/step8-screening/acceptance.json \
  --monitor-config configs/bounded-harness-monitoring.yaml --output-dir runs/NEW-selection
uv run --locked python runs/issue26-bounded-sft/build-report.py
```

检查 `student_snapshot`、`frozen_teacher_snapshot`、`sft_training_state` 均为 null。此结果不能初始化 OPSD。权威选择为 `selection-verified/selection.json`；`selection/selection.json` 保留修复前的身份误判。两次选择使用相同评测，没有增加采样分母。选择器从当前 journal 或对应内容哈希归档核验训练身份，不能用异源训练记录代替。完整训练状态及原始轨迹保持不变。

监控断网恢复使用既有入口，云端仅接收允许字段：

```bash
uv run --locked bashlet monitor sync runs/issue26-bounded-sft/step8-screening/harness/seed-4
uv run --locked bashlet monitor sync runs/issue26-bounded-sft/step8-screening/coding/seed-4
uv run --locked bashlet monitor sync runs/issue26-bounded-sft/step8-screening/variant/seed-4
uv run --locked bashlet monitor sync runs/issue26-bounded-sft/selection-verified
uv run --locked python runs/issue26-bounded-sft/verify-monitoring.py
```

实现验证使用 `uv run --locked mypy`、`uv run --locked ruff check`、`uv run --locked ruff format --check src tests`、`uv run --locked pytest -q tests/test_training.py` 和最终全量 `uv run --locked pytest -q`。原运行记录中的 code revision 是实施前父提交 `b3d6c2b`；实际实现差异保存在 `implementation.patch`，其哈希见 `validation.json`，并随当前分支提交。该补丁为无上下文格式，恢复时使用 `git apply --unidiff-zero`。step 1/4 的历史训练记录由完成日志重建，字节哈希与对应导出当时捕获的哈希完全相同，保存在 `training/step1-record.json` 与 `training/step4-record.json`。原始模型路径和数据依赖来自任务 21、24、25 的已保留产物。
