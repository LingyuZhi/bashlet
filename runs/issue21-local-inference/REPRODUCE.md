# 任务 21 的输入与复跑

本轮只使用任务 07 已有 step 12 sampler，不训练新模型。付费请求使用既有 `runs/spend.sqlite3` 和 `configs/spend.yaml` 的累计 `local-snapshot-inference: 5` 子额度。开始时历史保守占用为 ¥8.40、在途 ¥0、已入金 ¥10，因此实际可用额度最多 ¥1.60；历史实扣仍未知，不释放或分摊预留。本轮新增付费模型计算请求为零。

基模为 `Qwen/Qwen3.5-4B`，HF revision 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。`local-base/snapshot.json` 保存完整基模、tokenizer、聊天模板文件哈希。SFT 来源为 `trio://run_x2uvy7hltg83/sampler_weights/sampler-0012`，checkpoint 为 `ckpt_wftn2nfe3pnl`；`sft-step12/export.json` 保存 SDK、训练来源、下载 archive 与 adapter 哈希。云端模型 revision 未公开，保存为 null。

## 准备已有输入

在仓库根目录操作。新输出目录不能已有完成的 export／snapshot manifest。下面是本轮使用的命令；复跑时改成新目录，并更新运行 YAML 的产物路径。大体积权重不进入 Git，下载仍需原账户授权和 checkpoint 保留。

```bash
uv run --locked bashlet spend status --policy configs/spend.yaml
uv run --locked bashlet export-snapshot runs/issue07-sft-v1 \
  --step 12 --api-key-env pytrio_api_key \
  --output-dir runs/issue21-local-inference/sft-step12
```

下载基模时，在 `n30126` 上先检查模块，再使用锁定的独立环境。优先使用本地缓存；本轮通过 HF-Mirror 下载指定 revision 的 JSON、Jinja 和 safetensors 文件。

```bash
ssh n30126
module avail
module load cuda/12.8
cd /data/home/zdhs0148/opd/bashlet
uv sync --project configs/local-vllm --locked
HF_ENDPOINT=https://hf-mirror.com uv run --project configs/local-vllm --locked python -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', cache_dir='runs/issue21-local-inference/hub', allow_patterns=['*.json', '*.jinja', '*.safetensors']))"
```

对下载目录创建基模和 SFT manifest。设置线程数只控制 CPU 合并工作，不改变推理采样参数。

```bash
LOCAL_BASE=runs/issue21-local-inference/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
uv run --locked bashlet prepare-local-snapshot "$LOCAL_BASE" \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --output-dir runs/issue21-local-inference/local-base
OPENBLAS_NUM_THREADS=8 uv run --locked bashlet prepare-local-snapshot "$LOCAL_BASE" \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --export runs/issue21-local-inference/sft-step12/export.json --merge \
  --output-dir runs/issue21-local-inference/local-sft-step12-merged-v2
```

转换逐项校验 249 个原始 LoRA 目标。24 组 GDN split Q/K/V 使用拼接 A、分块 B 和零填充，统一 rank 64、alpha 128，保持原始 alpha/r=32/16；重排本身不改变 BF16 数值。合并应用 201 个目标矩阵，并对 head-only LoRA 解除输出 head 绑定，保持原始输入 embedding。`merge-verification.json` 用独立 PyTorch FP32 运算抽核每个原始目标的三个输出行，容差为一个 BF16 ULP，并全量检查 input embedding 相等；这不是云端权重一致性证明。

[PyTRIO 下载说明](https://docs.pytrio.cn/docs/content/guide/download/content.md) 使用 `RestClient.download_checkpoint` 导出已有 LoRA。锁定 SDK 0.2.9 在本轮返回 TAR，文档示例为 ZIP，导出入口支持两者。直接 LoRA 服务实际拒绝 `language_model.lm_head`，完整错误在 `server-lora-failed.log`；可复跑的验收路径为显式 `--merge`，没有移除或忽略这个目标。

## 服务切换与恢复

启动前核对当前服务 PID 的命令行，向实际 API server PID 发送 TERM，等待退出及显存释放，再启动目标服务。不要仅凭旧 PID 文件停止未知进程。任务 20 被替换的 API server 为 1309835，完整参数仍保存在 `runs/issue20-vllm/start-service.sh`。

```bash
ps -p <当前 API server PID> -o pid,ppid,args
kill -TERM <已核对的 API server PID>
bash scripts/serve-local-vllm.sh configs/local-vllm/serve.yaml
# 同一端口切换到 SFT：先停止上面的已核对进程，再运行
bash scripts/serve-local-vllm.sh configs/local-vllm/serve-sft.yaml
```

本轮后台启动命令保存在 `start-service.sh`，分别写入 `server-base.log`、`server-sft.log` 和 `service.pid`；持续运行的 SFT 日志另留固定验收副本 `server-sft-acceptance.log`。新服务同样仅监听 `127.0.0.1:8000`，Python 3.12、vLLM 0.17.1、Torch 2.10.0、CUDA 12.8、BF16、8192 上下文、单序列、eager、显存利用率 0.6、`generation-config=vllm`、language-model-only。实际配置与系统版本另存 `api-base.json`、`api-sft.json`，运行时验证证据保留在每个 episode。

Git 中固定服务日志去除终端 CR 和行尾空白，原始与规范化 SHA-256 保存于 `log-format.json`；episode JSON、token/logprob 和仍在写入的服务日志未改写。

恢复任务 20：停止已核对的新 API server，执行 `bash -l runs/issue20-vllm/start-service.sh`，恢复其 vLLM 0.11.0 与 Qwen3-4B-Instruct-2507 配置。恢复原始 Qwen3.5 则使用 `serve.yaml`。恢复服务不改变旧记录的开发身份。

## 真实回合

客户端保留 loopback 隧道：`ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 n30126`。基模和 SFT 必须与当前服务匹配；SFT 客户端叠加两个 YAML。

```bash
uv run --locked bashlet task runs/issue21-local-inference/smoke-task.json \
  -c configs/local-base.yaml --output-dir runs/issue21-local-inference/base-smoke
uv run --locked bashlet task runs/issue21-local-inference/smoke-task.json \
  -c configs/local-base.yaml -c configs/local-sft.yaml \
  --output-dir runs/issue21-local-inference/sft-smoke
LOCAL_SNAPSHOT_ID=$(uv run --locked python -c \
  "import json; print(json.load(open('runs/issue21-local-inference/local-sft-step12-merged-v2/snapshot.json'))['identity'])")
uv run --locked bashlet evaluate configs/mbpp-local-smoke-v1.json \
  --snapshot "$LOCAL_SNAPSHOT_ID" -c configs/local-base.yaml -c configs/local-sft.yaml \
  --monitor-config configs/local-monitoring.yaml --training-run runs/issue07-sft-v1 \
  --output-dir runs/issue21-local-inference/sft-evaluation
```

基模评测使用相同 plan 和 `local-base.yaml`，身份为 `vllm/Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。固定 smoke Task 明确要求在公开反馈通过后提交，只用于加载、采样、执行与独立评分链路验证；不能作为自然任务成功率或 SFT 有效性的证据。MBPP 单题的失败和首次实现评分全部保留，不能用 smoke 成功覆盖失败结果。

## 审查后的准入与验收

实验模式要求明确的 `auto`／`safetensors` 加载器及 `lazy`／`eager` 策略，不能有额外加载配置或过滤权重的 ignore patterns；`dummy` 和缺失加载配置会失败。服务应使用 `generation-config=vllm`，采样覆盖完整进入比较条件，自定义 logits processor 和外部生成／模型配置不受支持。[vLLM 0.17.1 原生模型配置](https://github.com/vllm-project/vllm/blob/v0.17.1/vllm/config/model.py) 会应用服务采样覆盖，因此这些设置必须可追溯。

`*-review/` 是修正探针后使用相同 Task／plan、参数和权重重新运行的证据，`api-*-review.json` 保存当时完整加载与推理配置。`acceptance.json` 指定最终采用的目录和比较指纹，原始四个回合继续保留。[validation.json](validation.json) 区分审查前全量 304 项通过与修正后 64 项回归通过；[review.json](review.json) 保存两轴初审与复审结果。

最终活动服务日志为 `server-sft-review.log`，固定验收副本为 `server-sft-review-acceptance.log`；`migration.json` 记录复跑后的实际 PID 和恢复方式。
