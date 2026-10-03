# 任务 25：双轨验收

本地真实筛选使用 seed 4。原始模型为 Harness 4/30、直接代码生成 27/37；已有 step 12 SFT 为 3/30、20/37。两者代码修改及工作区类别均为 0/10，许可证政策变体也未成功。候选没有满足扩展规则，未增加确认 seed。`selection/selection.json` 保留证据不足及缺失的确认 seed，学生、教师和训练状态起点均为 null。该结果不宣称确认的能力回退或提升。

`acceptance.json` 保存结果、原始评测路径／SHA-256、每类固定分母、典型轨迹和有效条件比较。完整原始 token、logprob、观察、工作区与评分在 `base-v1/` 和 `sft-step12/`，不作为 W&B 上传内容。10 个 MBPP 家族重叠，两轨分母分别报告。环境变体仍属于同一个 Harness 家族。

## 复跑

仓库根目录执行。先按 AGENTS.md 进入 `n30126`，检查模块并加载本次已核实的 `cuda/12.8`。检查当前 API server PID 的完整命令后再停止，等待退出和显存释放。使用既有锁定 Python 3.12 / vLLM 0.17.1 / BF16 服务配置启动目标模型：

```bash
bash scripts/serve-local-vllm.sh configs/local-vllm/serve.yaml
# 切换到已有 SFT 时，先停止已核对的基模 API server，再执行：
bash scripts/serve-local-vllm.sh configs/local-vllm/serve-sft.yaml
```

客户端建立 loopback 隧道：

```bash
ssh -N -L 127.0.0.1:18025:127.0.0.1:8000 n30126
```

基线命令：

```bash
uv run --locked bashlet evaluate-acceptance configs/sft-acceptance-v1.json \
  --snapshot vllm/Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  -c configs/local-base.yaml -c configs/acceptance-local-v1.yaml \
  --monitor-config configs/acceptance-monitoring.yaml --output-dir runs/NEW-base
```

候选叠加 `local-sft.yaml`，`--snapshot` 使用其 `snapshot.json` 的完整 identity，并指定 `--training-run runs/issue07-sft-v1`。新运行使用新输出目录。有希望时，在对应服务下对基线和候选各使用 `--stage confirmation --resume <acceptance.json>`；完整同 seed 结果复用。源文件／评测哈希、客户端 HTTP 策略、编码及实际服务条件不同会拒绝复用或资格判定。

本次选优：

```bash
uv run --locked bashlet select-sft runs/issue07-sft-v1 \
  --baseline runs/issue25-acceptance/base-v1/acceptance.json \
  --evaluation runs/issue25-acceptance/sft-step12/acceptance.json \
  --monitor-config configs/acceptance-monitoring.yaml \
  --output-dir runs/NEW-selection
```

真实正向结果没有出现，未生成工作区收益审阅来强行接纳候选。正向、格式改善但任务变差、单类退化、独立编程回退、全淘汰及基础设施缺口等规则由可控模型／真实沙箱 CLI 测试验证，测试替身不能产生生产起点。

## 保存和限制

没有新增训练或付费模型请求。复用任务 21 的基模与 merged step 12 权重；云端原始模型 revision 仍未知，不宣称与本地逐位相同。结束后保留 SFT 服务，实际 API server PID 为 1748544；再次使用前仍需核对进程。两次服务启动日志位于本目录。

早期 `base/` 在离线监控被错误要求在线验证时退出，尚无任务采样；`sft-step12-identity-rejected/` 因传入错误快照哈希被拒绝，也没有任务尝试。这些记录单独保留，不计为追加实验样本。

监控清单为 `monitoring-upload.json`，包含 7 个完成运行及其允许字段。直接编程监控使用 `coding-base-monitor/`、`coding-sft-monitor/` 从原评测导入，明确为历史导入，并排除未计算的首次实现／恢复指标。原始数据和哈希未改。自动审批两次拒绝在线上传，因此当前仅保存离线缓存；获明确授权后，可对清单中的各 `run_dir` 使用 `bashlet monitor sync` 同步到既有 `lingyuzhi-chinese-academy-of-sciences/bashlet`。

`validation.json` 与 `review.json` 记录验证范围及两轴审查。全量 382 项通过；最后的 HTTP 策略修正另跑受影响文件，不把此前全量结果冒充修正后的全量复跑。
