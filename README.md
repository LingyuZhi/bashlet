<h1 align="center">bashlet</h1>

<p align="center">
  <img src="assets/bashlet_logo.png" alt="bashlet logo" width="480">
</p>

<p align="center">
  <strong>A Tiny Coding Agent with On-Policy Self-Distillation.</strong>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.13%2B-3776AB?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="Python 3.13+"></a>
  <a href="configs/defaults.yaml"><img src="https://img.shields.io/badge/Harness-Bash-10B981?style=for-the-badge&amp;logo=gnubash&amp;logoColor=white" alt="Bash Harness"></a>
  <a href="#training"><img src="https://img.shields.io/badge/Training-OPSD-7C3AED?style=for-the-badge" alt="SFT &amp; OPSD"></a>
  <a href="environments/local-vllm/"><img src="https://img.shields.io/badge/Inference-vLLM-0EA5E9?style=for-the-badge" alt="vLLM"></a>
</p>

<p align="center">
  <a href="#results">📊 实验结果</a> ·
  <a href="#features">✨ 核心能力</a> ·
  <a href="#quickstart">🚀 快速开始</a> ·
  <a href="#models">🔌 接入真实模型</a> ·
  <a href="#training">🧬 训练与评测入口</a> ·
  <a href="#development">🛠️ 开发检查</a>
</p>

---

bashlet 是一个通过 bash 命令编写、运行和修复代码的极简编程智能体。

项目研究 Harness + On-Policy Self-Distillation（OPSD）：学生模型和教师模型都从 [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) 初始化得到；学生执行编程任务生成真实交互轨迹，教师模型保持冻结，并结合参考实现为学生生成的轨迹提供 token-level 的监督。

<a id="results"></a>

## 📊 实验结果

在相同的 500 个开发集评估回合上，对比 Baseline 和两个 OPSD checkpoint。每个模型覆盖 5 个评估 seed，每个 seed 100 回合。

![开发集总体与分类最终通过率：Baseline step-0 为 70.2%，OPSD step-268 为 77.4%，OPSD step-536 为 80.4%。](assets/experiment-results.svg)

| 评估对象 | 通过回合数 | 最终通过率 | 平均生成 tokens | 平均交互步数 |
| --- | ---: | ---: | ---: | ---: |
| Baseline step-0 | 351 / 500 | 70.2% | 1708.908 | 8.424 |
| OPSD step-268 | 387 / 500 | 77.4% | 1979.854 | 9.152 |
| OPSD step-536 | 402 / 500 | 80.4% | 1796.788 | 8.564 |

> 生成 tokens 包含 thinking。两项开销均为全部 500 回合的每回合均值。

<a id="features"></a>

## ✨ 核心能力

- 💻 **任务交互**：`agent` 接收自然语言指令和工作目录；`task` 运行带独立评分的编程任务。
- 🛡️ **隔离执行**：使用 [Bubblewrap](https://github.com/containers/bubblewrap) 沙箱，每步启动新 shell，工作区文件在步骤之间保留。
- 📝 **轨迹记录**：保存命令、真实观察、工作区变化，以及采样生成的原始 token 和 logprob。
- 🧠 **后训练与评测**：通过 [PyTRIO](https://pytrio.com) 进行 OPSD 训练；同时支持本地 vLLM 采样与评测。

<a id="quickstart"></a>

## 🚀 快速开始

### 🧰 环境要求

- **宿主环境**：Python 3.13+ 和 uv。
- **执行环境**：Linux，允许创建 user、mount、PID、network namespaces；提供 `/usr/bin/bwrap`、`/usr/bin/prlimit`、`/bin/bash`，以及 git、awk、sed、grep。
- **默认沙箱环境**：`/usr/bin/python3` 为 Python 3.10.12，独立虚拟环境使用锁定的 pytest 9.1.1。其他环境需调整[默认配置](configs/defaults.yaml)中的 `environment` 设置。

### 🧪 运行示例任务

在仓库根目录执行：

```bash
uv sync --locked
uv sync --project environments/sandbox-python --python /usr/bin/python3 --locked
uv run --locked bashlet task examples/sum.json
```

此示例使用 `ReplayModel` 的预写回复，在真实沙箱中运行代码、执行反馈测试并提交评分，无需 GPU 或 API Key。结束后会显示评分，产物保存在 `runs/<运行 ID>/`。

查看运行记录时，将 `runs/RUN_ID` 替换为 CLI 输出的运行目录：

```bash
uv run --locked bashlet show runs/RUN_ID
```

更多任务示例见 [examples/](examples/)。

### 🤖 直接运行 agent

对工作目录下达指令。以下示例仍使用预写回复：

```bash
mkdir -p runs/demo-input
uv run --locked bashlet agent "创建 result.txt，然后提交" \
  --workdir runs/demo-input \
  --set 'model.replies=["```bash\ntouch result.txt\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"]'
```

`agent` 操作输入目录的副本，结果保存在本次运行的 `workspace/` 中。通过 `-c` 选择模型配置；多个配置按顺序覆盖，`--set key=value` 覆盖单项设置。

<a id="models"></a>

## 🔌 接入真实模型

将 `PYTRIO_API_KEY=你的密钥` 写入仓库根目录的 `.env`。正式训练使用 [configs/opsd.yaml](configs/opsd.yaml) 中的 PyTRIO 设置，采样和训练共用 `runs/spend.sqlite3` 账本及其中的 `spend` 费用策略。

本地 vLLM 的依赖锁位于 [environments/local-vllm/](environments/local-vllm/)。准备模型权重后，调整 `opsd.yaml` 的 `evaluation.base_path`、服务地址和 `evaluation.server.options`，再启动服务：

```bash
uv sync --project environments/local-vllm --python 3.12 --locked
bash scripts/serve-local-vllm.sh
```

服务脚本将当前参数写入 `runs/local-vllm/serve.yaml`。在计算节点上运行前，先加载该节点可用的 CUDA module。

<a id="training"></a>

## 🧬 训练与评测入口

OPSD 从原始指令模型开始，学生更新，特权教师冻结。训练使用 PyTRIO，评测使用本地 vLLM。

`configs/` 只包含两份默认配置：[defaults.yaml](configs/defaults.yaml) 定义 Harness、协议和沙箱；[opsd.yaml](configs/opsd.yaml) 汇总训练、评测、数据质量检查与费用策略。配置中的路径相对声明它的 YAML 文件解析。本地 tokenizer 路径使用 `./` 或 `../` 前缀，Hub 模型名保持原样。先按[数据准备说明](docs/data-preparation.md)生成数据，再准备模型权重与服务环境，并按实际资源调整配置。

```bash
uv run --locked bashlet train-opsd configs/opsd.yaml
uv run --locked bashlet train-opsd configs/opsd.yaml --resume
uv run --locked bashlet evaluate-checkpoint configs/opsd.yaml --phase final
uv run --locked bashlet watch-opsd --config configs/opsd.yaml
```

`train-opsd` 直接运行正式训练。每轮评测共享训练的 Harness、预算和采样条件，模型权重由保存的 checkpoint 确定。更改新实验的轮数、种子或预算只需改配置；同一运行恢复时必须保持其冻结条件。`--phase test` 仅允许在完整训练结束后访问最终测试集。

| 用途 | CLI 命令 | 配置或说明 |
| --- | --- | --- |
| 准备 MBPP 任务 | `prepare-mbpp --source-lock data/sources.json` | [来源定义](data/sources.json) |
| 导入公开任务 | `import-public-tasks data/sources.json --catalog public-python`（或 `p3`） | [来源与导入计划](data/sources.json) |
| 验证和冻结混合任务集 | `prepare-harness --cohort data/cohort.json` | [任务划分](data/cohort.json)、[准备说明](docs/data-preparation.md) |
| 导出保存的权重 | `export-snapshot` | 使用 OPSD 运行中的 `training/` 目录 |
| 准备本地模型 | `prepare-local-snapshot` | 基模与导出的 adapter |
| 独立任务评测 | `evaluate` | 固定任务清单与已验证的本地模型 |

通过 `uv run --locked bashlet --help` 可以查看入口。

<a id="development"></a>

## 🛠️ 开发检查

```bash
uv run --locked mypy
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

测试需要可用的 Bubblewrap namespaces，以及测试所用的 Qwen tokenizer 本地缓存。

---

项目术语见 [CONTEXT.md](CONTEXT.md)，架构决策见 [docs/adr/](docs/adr/)。
