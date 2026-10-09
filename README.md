<h1 align="center">bashlet</h1>

<p align="center">
  <img src="assets/bashlet_logo.png" alt="bashlet logo" width="480">
</p>

<p align="center">
  <strong>A Tiny Coding Agent with On-Policy Self-Distillation.</strong>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.13%2B-3776AB?style=for-the-badge&amp;logo=python&amp;logoColor=white" alt="Python 3.13+"></a>
  <a href="configs/bash-block-v3.yaml"><img src="https://img.shields.io/badge/Harness-Bash-10B981?style=for-the-badge&amp;logo=gnubash&amp;logoColor=white" alt="Bash Harness"></a>
  <a href="#training"><img src="https://img.shields.io/badge/Training-SFT%20%26%20OPSD-7C3AED?style=for-the-badge" alt="SFT &amp; OPSD"></a>
  <a href="configs/local-vllm/"><img src="https://img.shields.io/badge/Inference-vLLM-0EA5E9?style=for-the-badge" alt="vLLM"></a>
</p>

<p align="center">
  <a href="#features">✨ 核心能力</a> ·
  <a href="#quickstart">🚀 快速开始</a> ·
  <a href="#models">🔌 接入真实模型</a> ·
  <a href="#training">🧬 训练与评测入口</a> ·
  <a href="#development">🛠️ 开发检查</a>
</p>

---

bashlet 是一个通过 bash 命令编写、运行和修复代码的极简编程智能体。项目为任务交互、后训练和评测提供共用的 Harness。

项目研究 On-Policy Self-Distillation（OPSD）：学生模型生成真实交互轨迹，同源的冻结教师利用训练任务的参考实现，为学生已采样的 token 提供概率监督。

<a id="features"></a>

## ✨ 核心能力

- 💻 **任务交互**：`agent` 接收自然语言指令和工作目录；`task` 运行带独立评分的编程任务。
- 🛡️ **隔离执行**：使用 Bubblewrap 沙箱，每步启动新 shell，工作区文件在步骤之间保留。
- 📝 **轨迹记录**：保存命令、真实观察、工作区变化，以及采样生成的原始 token 和 logprob。
- 🧠 **后训练与评测**：通过 PyTRIO 进行 SFT 和 OPSD；支持本地 vLLM 采样与评测。

<a id="quickstart"></a>

## 🚀 快速开始

### 🧰 环境要求

- **宿主环境**：Python 3.13+ 和 uv。
- **执行环境**：Linux，允许创建 user、mount、PID、network namespaces；提供 `/usr/bin/bwrap`、`/usr/bin/prlimit`、`/bin/bash`，以及 git、awk、sed、grep。
- **默认沙箱环境**：`/usr/bin/python3` 为 Python 3.10.12，独立虚拟环境使用锁定的 pytest 9.1.1。其他环境需调整[默认配置](configs/bash-block-v3.yaml)中的 `environment` 设置。

### 🧪 运行示例任务

在仓库根目录执行：

```bash
uv sync --locked
uv sync --project configs/sandbox-python --python /usr/bin/python3 --locked
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
uv run --locked bashlet agent "创建并运行 hello.py，然后提交" \
  --workdir runs/demo-input -c configs/replay-agent.yaml
```

`agent` 操作输入目录的副本，结果保存在本次运行的 `workspace/` 中。通过 `-c` 选择模型配置；多个配置按顺序覆盖，`--set key=value` 覆盖单项设置。

<a id="models"></a>

## 🔌 接入真实模型

将 `PYTRIO_API_KEY=你的密钥` 写入仓库根目录的 `.env`，使用 [PyTRIO 配置](configs/pytrio-smoke.yaml)运行云端 Qwen3.5-4B：

```bash
uv run --locked bashlet spend status
uv run --locked bashlet task examples/pytrio-smoke.json \
  -c configs/pytrio-smoke.yaml
```

远端采样与训练会产生 API 费用，统一使用 `runs/spend.sqlite3` 账本。预算设置见 [configs/spend.yaml](configs/spend.yaml)。

本地 vLLM 的环境见 [configs/local-vllm/](configs/local-vllm/)，运行配置见 [configs/local-base.yaml](configs/local-base.yaml)。使用前需准备模型权重，并调整服务地址、权重和快照清单路径。

<a id="training"></a>

## 🧬 训练与评测入口

OPSD 可以直接从指令模型开始；SFT 是可选起点。

| 🎯 用途 | ⌨️ CLI 命令 | 📄 配置或说明 |
| --- | --- | --- |
| 准备 MBPP 任务 | `prepare-mbpp` | [数据源配置](configs/mbpp-sanitized-v1.json) |
| 准备多类型任务 | `prepare-harness` | [v1 配置](configs/harness-tasks-v1.md)、[v2 数据角色](configs/harness-tasks-v2.md)、[836 家族冻结](configs/opsd-expanded-data-v8.md) |
| 导出 SFT 数据 | `prepare-sft` | [已执行轨迹的输入格式](configs/executed-trajectories.md) |
| SFT 训练 | `train-sft` | [SFT 配置](configs/sft-v1.yaml) |
| OPSD 训练 | `train-opsd` | [OPSD 小试配置](configs/opsd-pilot-v1.yaml) |
| 固定开发集评测 | `evaluate` | [开发集清单](configs/mbpp-dev-8-v1.json) |

通过 `uv run --locked bashlet --help` 查看命令，各子命令的 `--help` 提供详细参数。示例配置引用已准备的数据和模型产物，使用前需调整路径与凭证设置。

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
