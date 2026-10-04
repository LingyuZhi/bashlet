# 任务 27 复跑

从仓库根目录操作，使用 uv 锁定环境。原始 Task、清单和模型快照须与
`preflight.json` 的 SHA-256 一致。已有完成目录不可覆盖；新尝试不是本轮
18 次中的替换记录。以下命令中 `runs/NEW-pilot-*` 必须改为新的输出目录。
本轮固定产物目录仅用于阅读和重新加载。

## 原始服务

在 `n30126` 先检查模块、显存、当前进程及端口。保留旧 8000 服务，
新原始模型监听 8001。若资源或端口已变化，先准备并记录新条件，不能
将两协议放在不同窗口或推理条件下。

```bash
ssh n30126
module avail
module load cuda/12.8
cd /data/home/zdhs0148/opd/bashlet
nvidia-smi
timeout --signal=TERM --kill-after=45s 6900s \
  bash scripts/serve-local-vllm.sh configs/local-vllm/serve-collection-pilot-v1.yaml
```

本轮实际后台启动薄入口为 `start-service.sh`，PID、日志和 GPU 监控另存。
整个阶段限两小时；启动命令的 6900 秒时限预留准备与收尾时间。复跑须
单独记录准备、能力探针、两组采样、回放和收尾，不能只统计生成时长。

在客户端建立独立 loopback 隧道：

```bash
ssh -N -L 127.0.0.1:18027:127.0.0.1:8001 -o ExitOnForwardFailure=yes n30126
```

`collect-demonstrations` 会在允许尝试前重新检查实际 snapshot 身份、权重、
tokenizer、编码、token 与 logprob 能力。本轮另保留了单独的能力探针，
三次一 token 探针的用量已分别计入记录。所有模型调用均为本地服务。

## 配对采样

两次使用同一个冻结计划，先 v3 后 v4；sampling、seed、初始状态、预算、
环境、服务窗口、编码、HTTP 条件全部相同。

```bash
uv run --locked bashlet collect-demonstrations configs/collection-pilot-v1.json \
  -c configs/local-base.yaml -c configs/collection-pilot-v1.yaml \
  --output-dir runs/NEW-pilot-v3
uv run --locked bashlet collect-demonstrations configs/collection-pilot-v1.json \
  -c configs/local-base.yaml -c configs/collection-pilot-v1.yaml \
  -c configs/bash-block-v4.yaml --output-dir runs/NEW-pilot-v4
```

不能重试失败或换题来改变既定分母。未完成时保留 collection 与 episode
记录，报告分母仍为 18。本轮采样全量 18 次已经完成，不应再次启动来
重复核验文档或报告。采样结束后核对新服务进程完整命令，再停止新服务；
保留旧服务，并用 `curl --noproxy '*'` 或 `httpx(trust_env=False)` 核验旧 API。

## 审阅、精确回放、冻结与加载

新的采样必须新写两份完整审阅，不能套用本轮审阅。格式沿用既有
`freeze-demonstrations`：schema=1，collection_sha256 绑定对应清单，
attempts 按顺序覆盖全部尝试，每个 actions 项有 supervise、reason、author
和 action_type。配对报告还要求每条回复的 observation_review，字段为
invented 布尔值、人工理由和原 model_call.text 的 UTF-8 SHA-256。
示例为本轮 `v3-quality-review.json`、`v4-quality-review.json`。

```bash
uv run --locked bashlet freeze-demonstrations runs/NEW-pilot-v3/collection.json \
  --review runs/NEW-v3-review.json -c configs/collection-pilot-export-v1.yaml \
  --output-dir runs/NEW-v3-frozen
uv run --locked bashlet freeze-demonstrations runs/NEW-pilot-v4/collection.json \
  --review runs/NEW-v4-review.json -c configs/collection-pilot-export-v1.yaml \
  -c configs/bash-block-v4.yaml --output-dir runs/NEW-v4-frozen
uv run --locked bashlet report-collection-pilot configs/collection-pilot-v1.json \
  --collection runs/NEW-pilot-v3/collection.json --review runs/NEW-v3-review.json \
  --collection runs/NEW-pilot-v4/collection.json --review runs/NEW-v4-review.json \
  --output-dir runs/NEW-comparison
```

输出目录必须不存在。冻结入口复用真实 bash 与独立评分，接纳的训练输入
经过实际加载器检查。回放观察保存到新 episode，不修改采样历史。
时间戳差异会保留为拒绝原因，不删除这一步或替换原始输出。

仅重新汇总本轮证据时，可使用原两个 collection 和质量审阅路径，并将
报告输出改为新目录。此命令不采样、回放或调用远端模型。

## 独立辅助链路

本轮工作区模型候选为零，辅助示范仅验证已有参考来源的执行与加载链路。
下列命令从冻结 Task 取参考实现，所有观察仍来自真实环境：

```bash
uv run --locked bashlet prepare-sft \
  runs/issue27-collection-pilot/tasks/manifest.json --task-id harness/stock-value \
  --max-context-tokens 16384 -c configs/collection-pilot-export-v1.yaml \
  -c configs/bash-block-v4.yaml --output-dir runs/NEW-auxiliary-reference
```

辅助记录不计入模型配对分母或原始模型合格率，token 为 replayed。
质量审阅、内容哈希冻结和加载证据在本轮 `auxiliary-reference/` 中单独保存。

无需训练或 API 请求即可重查三份数据的实际训练加载：

```bash
uv run --locked python - <<'PY'
from pathlib import Path
from bashlet.training import load_demonstrations
root = Path('runs/issue27-collection-pilot')
for relative in ('v3-frozen/data/manifest.json', 'v4-frozen/data/manifest.json',
                 'auxiliary-reference/manifest.json'):
    manifest, datums = load_demonstrations(root / relative)
    print(relative, manifest['counts'], len(datums))
PY
```

## 实现验证

```bash
uv run --locked mypy
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
uv run --locked pytest -q tests/test_collection_pilot.py
uv run --locked pytest -q tests/test_curation.py
uv run --locked pytest -q
```

实际工具执行使用 `UV_CACHE_DIR=/tmp/bashlet-uv-cache` 和 `--no-sync`，因为
仓库的锁定环境已安装，受限执行器不能写默认 uv 缓存。真实沙箱测试及
采样在允许 bwrap 与 loopback 网络的宿主执行环境中完成。首次受限环境
测试失败与后续宿主测试结果分开记录；没有改变沙箱探针来绕过失败。
