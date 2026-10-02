# bashlet

A Tiny Coding Agent with On-Policy Self-Distillation。

当前实现任务 01、02、12–20：Harness 入口 `bashlet.run_episode` 负责指令、工作区和模型之间的交互；实验入口 `bashlet.run_experiment` 负责准备 Task、对提交工作区做快照、独立评分和事后候选／修复分析。CLI 可以实时观察回合并回放运行记录。每个交互回合都由模型驱动：可以使用根据真实观察采样行动的策略，也可以用 `ReplayModel` 编码预写回复；两者共用同一 token 上下文、Harness、预算与评分。开发用 vLLM 可在计算节点运行，正式实验的 PyTRIO 采样由任务 04 接入。

## 运行

宿主使用 Python 3.13.12 和 uv；依赖版本锁定在 `uv.lock`，默认使用清华 PyPI 镜像。
执行节点需提供 Linux user/mount/PID/network namespaces、Bubblewrap、`/usr/bin/prlimit`、`/bin/bash` 和 `/usr/bin/python3`。

```bash
uv sync --locked
uv sync --project configs/sandbox-python --python /usr/bin/python3 --locked
uv run bashlet task examples/sum.json
uv run bashlet task examples/package-repair.json
```

沙箱 Python 环境独立于宿主开发环境：`configs/sandbox-python/uv.lock` 锁定 pytest 9.1.1 及其依赖，解释器固定为本节点的 Python 3.10.12。第二条命令准备这个环境，运行时只读挂载。缺少环境或版本不符时，能力探针会停止回合。使用安装后的包或另一个环境位置时，可用 `--set environment.python.source=/绝对路径/venv` 指定已按该锁文件准备的环境；YAML 中的相对 source 路径以该 YAML 所在目录为基准。

也可以用 `uv run python -m bashlet task examples/sum.json`。`task` 经过实验层，结束时显示评分；退出码 0 表示评分通过，1 表示未通过或未提交，2 表示基础设施或配置错误，130 表示中断。原来的请求文件入口已合并进 `task`，不再直接用文件名作为 CLI 命令。`package-repair.json` 演示多模块预置 bug、公开 pytest 反馈、修复和隐藏测试评分。

不需要 Task 或评分器时，用一段指令和已有目录直接运行 Harness。下面的回放演示无需 GPU：

```bash
mkdir -p runs/demo-input
uv run bashlet agent "创建并运行 hello.py，然后提交" --workdir runs/demo-input -c configs/replay-agent.yaml
uv run bashlet show runs/<运行 ID>
```

`agent` 将工作目录复制到本次运行的 `workspace/`，保留文件内容、权限和符号链接，再让 agent 操作副本；源文件不受修改。若产物根目录位于源目录中，复制时排除该产物目录，避免递归复制运行记录。`agent` 退出码 0 表示显式提交，不做评分。两个执行命令都支持 `--output-dir <目录>`，每次建立独立的运行目录。

实时视图和 `show` 共用一个 rich 滚动渲染器，展示每步的预算、可选行动说明、命令、模型实际看到的观察与格式错误，以及最终状态和评分。观察使用轨迹中保存的 `feedback`，保留原有截断和字面文本。提交或中断时没有后续模型观察，单独展示命令结果。视图只提供观察，不接受人工干预；Ctrl-C 会停止沙箱进程，保存已完成步骤、当前行动、捕获到的输出和工作区，再以 `interrupted` 结束。`show` 只读取运行记录，不重新执行命令。行动说明不依赖 `THOUGHT:`；原生 thinking 默认折叠，`agent`、`task` 和 `show` 都可加 `--show-thinking` 显示完整思考。

目录复制或工作区快照期间也可以中断：复制尚未完成时记录 `Workspace copy interrupted`，不调用模型；命令已完成但快照被打断时，先保留该步的行动和观察，再完成工作区证据的保存。

`task` 接受可信的 Task JSON 对象（字段见下文），模型在 YAML 配置中选择。也支持原有的 `{"task": ..., "replies": [...]}` 请求文件，将 `replies` 作为所选模型的构造参数并写进合并配置；同一次运行只从请求或 `model.replies` 中选一个来源。协议、执行环境、交互预算和采样参数来自版本化配置，不写在请求里。`replies` 是完整回复文本的列表，行动正文只要求一个非空 bash 围栏块，块前说明可选。命令返回码为 0 且合并输出首行恰好是 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 时提交整个工作区，立即结束回合。命令失败、超时和超量输出保留真实观察，后续回复仍可继续执行。旧的 `actions` 请求，以及写在请求里的 `limits` 或 `budget`，会被视为配置错误。

## 实验配置

默认条件在 `configs/bash-block-v3.yaml`，打包时作为内置配置随包安装，分四块：

- `protocol`：agent 可见的文本、提交标记、观察首尾字符数，以及正常观察、超时、截断和格式错误的模板。字符串是 Jinja2 模板，未定义变量在运行开始前报错。
- `environment`：执行环境标识 `id`、执行器路径、只读挂载、符号链接、生成的最小 `/etc`、独立 Python 环境和版本、HOME、环境变量，以及单条命令的超时和资源限制。命令默认超时 30 秒，实验层通过配置按任务集统一设置。
- `budget`：步数、生成 token 数、墙钟时间和连续格式错误上限。
- `model`：`class` 选择后端类，`sampling` 由 Harness 在每步采样时传入，其余字段作为模型构造参数。回放可用 `tokenizer`、`tokenizer_revision` 和 `thinking` 选择真实 Qwen 编码及思考模式，默认关闭 thinking。

`class` 可写内置类名 `ReplayModel`，或者完整导入路径，例如 `bashlet.replay.ReplayModel` / `bashlet.replay:ReplayModel`。回放模型默认使用 `Utf8Encoding`；`replies` 为预写回复列表，`identity` 可选。自定义类须满足同一个 token 模型和编码接口，采样输出仍接受 Harness 的 token / logprob 校验；构造失败时在创建运行目录前报配置错误。Python 调用者仍可以直接传入 `model=` 对象。

```yaml
model:
  class: ReplayModel
  identity: replay/manual-demo
  replies:
    - |-
      ```bash
      printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n'
      ```
```

可以叠加多个文件，后面的覆盖前面的；`--set key=value` 在所有文件之后覆盖单项，值按 YAML 解析。未知键、类型错误和非正的限额都会在建立运行目录之前失败。

```bash
uv run bashlet task examples/sum.json -c tighter.yaml --set budget.max_steps=4
```

Python 调用使用同一套层叠：`run_experiment(..., config=[path], overrides=["budget.max_steps=4"])`。

每次 `run.json` 写入合并后的完整 `config`，以及 `fingerprints.protocol`、`fingerprints.environment`、`fingerprints.budget`。指纹是对应块的稳定 SHA-256。同一配置得到相同指纹；只改一块时，只有那一块的指纹变化。`model` 不计入这三个指纹，所以更换采样参数或后端名称不会被当成更换了 Harness 或交互预算。协议指纹包含文本、行动与提交规则和观察截断配置，不包含候选、验证或修复分析。修改 agent 可见的协议文本会改变指纹，需要新的协议名。

默认回放模型使用回放专用的 `Utf8Encoding`，以 UTF-8 字节作为 token，编码身份为 `replay/utf8-v1`。使用真实 Qwen tokenizer 时，YAML 可设置：

```yaml
model:
  class: ReplayModel
  tokenizer: Qwen/Qwen3-1.7B
  tokenizer_revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
  thinking: false
```

`tokenizer_revision` 必填，支持 Hub commit、tag 或 branch；运行记录保存实际解析到的 snapshot commit 和文件 SHA-256。`tokenizer` 也可指向本地 tokenizer 目录，此时 revision 由调用者指定，文件哈希仍参与身份。默认先读本地 Hugging Face 缓存，缺少资产时从 HF-Mirror 下载 tokenizer 文件，不下载模型权重；可用 `tokenizer_cache_dir` 指定缓存位置。依赖精确锁定为 transformers 4.57.1、tokenizers 0.22.1 和 huggingface-hub 0.36.0。

`QwenChatMLEncoding` 用官方 `apply_chat_template` 构造首个 prompt，thinking 开关对应 `enable_thinking`；后续只追加 ChatML 观察与同一生成提示。编码提供 `stop_tokens`、真实思考标记的 token ID，以及 `reply(tokens)` 返回的行动正文、思考内容和结束状态。原始 token 不改写；即使官方多轮模板会移除早先的思考，本项目也保留它。调用方式依据 [Qwen 官方说明](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)和 [Transformers 聊天模板文档](https://huggingface.co/docs/transformers/v4.57.1/chat_templating)。Qwen3.5-4B 的模板与边界校验由任务 04 完成（H-11）。

Python 调用者可传入 `ReplayModel(replies, encoding=QwenChatMLEncoding(name, revision=revision, thinking=True))`，用学生编码构造示范；每步仍标注 `replayed`。原生 thinking 需要真实 tokenizer 或调用者提供的编码。以下演示使用本地缓存的 Qwen3-1.7B，思考里的命令不会执行，第二步行动没有说明：

```bash
uv run bashlet agent "创建并运行 hello.py，然后提交" --workdir runs/demo-input -c configs/replay-qwen-thinking.yaml
uv run bashlet show runs/<运行 ID>
uv run bashlet show runs/<运行 ID> --show-thinking
```

## 开发用 vLLM

`VLLMModel`（也可写完整导入路径 `bashlet.vllm.VLLMModel`）只产生开发证据。每个运行的 `policy_details.development_only` 固定为 `true`，实时视图和 `show` 也会注明排除实验结果。该字段同样保留在探针失败、服务失败和中断的记录中；有评分的开发回合也不作为实验结果。

服务环境独立锁定在 `configs/dev-vllm/`，使用 Python 3.12、vLLM 0.11.0、PyTorch 2.8.0 和 CUDA 12.8 wheel；tokenizer 库与宿主版本一致。首次使用先核对 `/version`、`/v1/models`，再用一个 prompt token 采样一个 token。探针结果、实际服务版本、模型 revision、服务报告的权重路径和上下文上限写入 `policy_details`，tokenizer 的文件哈希仍写入 `encoding_details`。探针失败时没有任务生成或命令执行。

采样使用 [`/v1/completions` 的 token prompt、`return_token_ids=true` 和 `logprobs=0`](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/protocol.py)，按[该版本的返回实现](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/serving_completion.py)读取 `choices[0].token_ids` 与 `logprobs.token_logprobs`，并校验 logprob 标识对应的 token。服务返回的文本不参与生成解析；不对返回 token 重新编码，不补概率。缺失、错位或非法数值结束为 `model_error`。网络错误与 HTTP 408/429/500/502/503/504 默认最多重试两次，最终失败为 `infrastructure_error`；其他 HTTP 错误不重试。请求的生成量按剩余上下文裁剪，达到模型上限或服务报告上下文超限时结束为 `context_exhausted`，已采样的 token 与概率仍保留。

安装 vLLM、下载权重和使用计算节点前须取得使用者同意。获准后，在 `n30126` 上执行以下命令；先列模块，再加载实际存在的模块。模型与 tokenizer revision 固定为 `cdbee75f17c01a7cc42f958dc650907174af0554`，优先复用本地缓存，缺少的资产通过 HF-Mirror 下载：

```bash
ssh n30126
module avail
module load cuda/12.8
cd /data/home/zdhs0148/opd/bashlet
uv sync --project configs/dev-vllm --python 3.12 --locked
HF_ENDPOINT=https://hf-mirror.com uv run --project configs/dev-vllm --locked hf download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554
HF_ENDPOINT=https://hf-mirror.com uv run --project configs/dev-vllm --locked vllm serve Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 --served-model-name Qwen/Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.6 --max-num-seqs 1 --enforce-eager --generation-config vllm
```

服务只绑定节点的 loopback。另开终端建立只绑定本机的 SSH 隧道，再用示例配置启动 CLI：

```bash
ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 n30126
```

```bash
mkdir -p runs/dev-vllm-input
uv run bashlet agent "创建 hello.py，运行后提交" --workdir runs/dev-vllm-input -c configs/dev-vllm.yaml
uv run bashlet show runs/<运行 ID>
```

服务可以持续运行，供后续开发使用；本次任务验收后保留服务和 GPU 占用。使用带认证的服务时，通过 `BASHLET_VLLM_API_KEY` 环境变量提供 API key，或用 `model.api_key_env` 指定环境变量名；配置与运行记录只保存变量名。真实服务不加入自动测试；错误映射由本地假 HTTP 服务在实验入口和 CLI 子进程验证。

2026-10-02 已在 `n30126` 的 A100 40GB 上完成真实 CLI 冒烟回合：上述 Qwen3-4B 模型在 3 步内创建并运行 `hello.py`，输出 `5` 后显式提交；48 个采样 token 均保存原始 id 和逐 token logprob，能力与隔离探针通过。开发证据保存在 `runs/issue20-vllm/`：`api-probe.json` 保存真实 token 接口请求和响应，`acceptance.json` 汇总验收，完整运行记录位于 `episodes/a2a47dbc534c43178098b1addd3f609c/run.json`。

## 两层入口与 Task v2

`run_episode(instruction=..., workspace=..., model=..., config=...)` 在已有工作区中运行一个回合，不接收 Task 或评分器。`config` 是 `bashlet.config.load_config()` 返回的完整配置；默认加载内置配置。返回 `EpisodeResult`，包含轨迹、终止状态、最终工作区路径 `workspace`、预算用量和可选的 `submission_text`。此入口直接修改指定工作区，运行记录默认保存在唯一的 `runs/<ID>/` 中；调用者也可指定独立的 `run_dir`，通过 `on_update` 接收进度。每个回合的状态都属于调用实例。

`run_experiment` 接收 Task，将 `initial_files` 中的相对路径和文本复制进本次运行的工作区，再调用同一个 Harness。Task 还包含稳定的 `task_id`、指令 `prompt`、执行环境标识 `environment_id`、宿主持有的 `FunctionGrader` 或 `CommandGrader`，以及可选的参考实现、反馈命令和实现路径。默认环境标识为 `linux-bwrap-v1`，须与配置的 `environment.id` 一致。初始文件可以放在任意子目录；文件名、反馈测试和评分方式都由任务数据声明。

bash-block-v3 的系统提示只说明行动格式、shell 规则和提交方式，不含任务集的文件名或格式。任务指令原样成为第一条 user 消息；文件和反馈命令由 Task 声明。MBPP 类 Task 可以在 `initial_files` 中提供 `task.md` 和 `feedback_tests.py`，另一种任务可以使用完全不同的文件布局，协议指纹保持相同。

Python 调用者使用实验入口：

```python
from pathlib import Path
from bashlet import (
    FunctionGrader,
    GradingCase,
    ReplayModel,
    Task,
    Utf8Encoding,
    run_experiment,
)

result = run_experiment(
    task=Task(
        task_id="manual/double-v1",
        prompt="Write solution.py with solve(n), returning twice n.",
        initial_files={
            "feedback_tests.py": "from solution import solve\nassert solve(2) == 4\n",
        },
        grader=FunctionGrader(
            module_path="solution.py",
            function_name="solve",
            cases=(GradingCase(args=(-3,), expected=-6),),
        ),
        implementation_paths=("solution.py",),
        feedback_command="python3 feedback_tests.py",
    ),
    model=ReplayModel(
        [
            "Implement the function.\n```bash\nprintf 'def solve(n): return n * 2\\n' > solution.py\n```",
            "Check the implementation.\n```bash\npython3 feedback_tests.py\n```",
            "```bash\nprintf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```",
        ],
        encoding=Utf8Encoding(),
    ),
    output_dir=Path("runs"),
)
print(result.status, result.grade)
```

## 行动协议 bash-block-v3

协议完整文本定义在[默认配置](configs/bash-block-v3.yaml)的 `protocol` 块，由 `src/bashlet/protocol.py` 渲染和解析。SFT 示范、学生 rollout 和评测共用它。系统提示、观察模板、历史编码和提交规则连同指纹写入每次运行 `run.json` 的 `protocol` 字段。v3 文本已交用户审阅，指纹固定为 `90720d09056f51501e9c5761577733296ecaf1c0f6792d8a7f00f1e0f070cb13`；修改协议必须更换版本、更新测试中的指纹并重建示范。v2 的交付记录保留在任务 16，已有示例已迁移；仓库尚无需要重建的 SFT 示范数据。

- 行动正文恰好包含一个非空、闭合的 ```` ```bash ```` 围栏块，块前说明可选，不要求 `THOUGHT:`。命令在隔离工作区执行。每一步都启动新的 shell，`cd`、`export` 和后台进程不跨步保留，工作区文件保留。
- 开启 thinking 时，只解析最后一个思考结束 token 之后的正文；起始标记可以已经在 prompt 中，采样无需重复。思考里的块不参与行动块计数，原始思考 token 和 logprob 保留在轨迹与上下文中。关闭时，正文可直接给出 bash 块。
- 只有成功命令的合并输出首行恰好为 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 时才提交，其余行原样记为 `submission_text`。标记在行动说明或思考中、位于后续行或命令返回非零均不提交；硬截断造成的不完整首行也不能作为标记。提交步仍记录实际 bash 命令、观察和工作区变化。
- 观察以 `<returncode>` 和 `<output>` 标签作为下一条 user 消息返回，stdout/stderr 在同一管道中合并以保持写入顺序。输出缺少末尾换行时补一个，标签内容不转义。原始捕获输出另存于 `observation.output`。
- 超过 `protocol.observation_head_chars + protocol.observation_tail_chars` 时，观察保留首尾（默认各 2000 字符），注明省略的字符数并提示缩小输出范围。`environment.output_bytes` 是捕获字节硬上限，超出部分直接丢弃并标记 `output_truncated`；命令继续执行，回合也继续。超时有独立文案，并终止该步全部进程。
- 格式错误（没有块、多个块、空块、块未闭合、思考未结束、因长度上限被截断）不执行，返回 `<format_error>` 说明并计入步数；连续 `max_consecutive_invalid` 次后以 `format_error` 终止，合法命令重置连续计数。
- 历史是只追加的 token 序列：先 `encoding.prompt(system, task.prompt)`，之后每步追加原始生成 token 和 `encoding.continuation(tokens, feedback)`。采样和回放的 token 都不会 decode 后重新 tokenize；观察只在对应行动实际执行后追加。

模型通过 `Model` Adapter 接入（`identity`、`encoding`、`sample(prompt, sampling) -> Generation`）；编码还提供 `encode(text)`，供回放使用。每次 `Generation` 标注 `source`：默认 `sampled`，必须带等长、有限、非正的逐 token `logprobs`；`replayed` 的 `logprobs` 为 `None`。回放 token 不属于 on-policy 样本。返回内容违反约定或无法解码时以 `model_error` 结束，被拒输出保存在 `rejected_generation`，不进入上下文；模型调用本身抛错（服务、网络、Adapter）记为 `infrastructure_error`。测试用可控采样 Adapter 位于 `tests/adapters.py`。

Harness 回合结果互斥：`submitted`（显式提交）、`interrupted`、`format_error`、`model_error`、`context_exhausted`（模型上下文耗尽）、`budget_exhausted`（`InteractionBudget` 的步数、生成 token 数或墙钟秒数）和 `infrastructure_error`。实验层保留 `episode_status`，将已提交回合进一步判为 `completed`、`invalid_submission` 或评分阶段的 `infrastructure_error`；`result.passed` 只在可信评分通过时为真。未提交回合不评分，计为失败，最终工作区保留供诊断。回放回复用尽后返回空回复，按连续格式错误上限或交互预算结束。每次请求的 `max_tokens` 裁剪到剩余生成额度；回放也按此裁剪 token，并将被截断的回复标记为 `length`。`usage` 记录实际步数、生成量、无效行动数与耗时。墙钟预算从隔离探针通过后开始计时，在步与步之间检查；单条命令另受 `Limits.timeout_seconds` 约束，单次模型调用的超时由接入的 Adapter 负责。

每个交互步（包括格式错误、提交和执行器故障）之后记录工作区清单：每条相对路径包含文件类型，普通文件另含 SHA-256、字节数和权限；符号链接只记录目标，FIFO、socket 等特殊文件只记录类型，不读取或跟随。相同字节只在 `blobs/<sha256>` 保存一份，删除或覆盖文件后仍可按历史清单取回原内容。不可读的路径保留类型和读取错误，供后续行动修复；初始化时也保存一份清单。非 UTF-8 文件名和符号链接目标通过 JSON 转义保存，读取记录时可恢复原路径。

候选实现、验证与修复尝试在回合结束后，由轨迹和清单计算。`Task.implementation_paths` 声明接受分析的文件，`Task.feedback_command` 声明反馈命令；MBPP 类任务在任务数据中填写 `("solution.py",)` 和 `"python3 feedback_tests.py"`。未声明实现路径时不做候选或修复分析；只声明实现路径时仍记录候选，但不判定验证。

候选按全部声明文件的内容去重，回退时沿用原编号，第 0 个候选即首次实现。声明文件须都可读、为普通文件且合计不超过提交大小限制；缺失、特殊或过大的文件状态沿用最后一个候选，保持原有行为。单文件候选直接指向内容 blob；多文件候选的 `path` 指向路径到内容哈希的 JSON 清单，`files` 列出各文件哈希。只有整条命令与任务的反馈命令一致（忽略首尾空白），且当前反馈材料与初始清单相同，才算验证；成功与否取真实观察状态。修复尝试从失败验证开始，到验证一个不同候选时结束，重复验证同一候选不新开尝试。反馈材料来自 Task 的初始文件，排除声明的实现文件。验证分析使用 Task 的声明，不写入协议配置或指纹。

## 隔离与评分边界

每次实验先运行真实访问探针：验证 namespace 已分离、环境变量已清理、宿主私有文件和包含参考实现／评分材料的配置无法读取、任务目录和 HOME 可写，以及新增挂载只读、账户和 localhost 解析正常、Python/pytest 版本符合配置、awk/sed/grep/git/pytest 可用。探针使用独立的 `environment.probe_timeout_seconds`（默认 30 秒），避免任务的短命令超时影响环境检查；生效窗口写入运行记录。探针不通过时记录 `infrastructure_error`，不执行任何任务命令，也不退回宿主直接执行。缺少执行器可用 `--set environment.bwrap=/does-not-exist/bwrap` 重现。

沙箱只读挂载 `/usr/bin`、`/usr/lib`、`/usr/lib64`、`/etc/alternatives` 和独立 Python 环境，并挂载当前任务目录。`/etc` 的账户、hosts、名称解析与 git 配置由配置生成，不挂载宿主 `/etc`。每一步使用独立 `/proc`、`/dev`、`/tmp` 和可写 HOME，清理环境和继承的文件描述符，断开宿主网络，移除 capabilities 并建立新会话；HOME 与临时文件不会跨步保留。实现依据 [Bubblewrap 0.6.1 的说明](https://github.com/containers/bubblewrap/blob/v0.6.1/README.md)及[参数文档](https://github.com/containers/bubblewrap/blob/v0.6.1/bwrap.xml)。任务 Python/pytest 的实际版本与宿主 Python 分别记录。默认环境关闭 Python 字节码写入和 [pytest 缓存插件](https://docs.pytest.org/en/stable/how-to/cache.html)，避免工作区快照产生 `__pycache__` 和 `.pytest_cache`；`pytest` 和 `python3 -m pytest` 都使用只读 Python 环境。

`json-function-v1` 评分按 `Task.grader` 中的 `module_path` 和 `function_name` 加载模块、调用函数，接受 JSON 位置参数，返回值须可表示为 JSON。模块可导入工作区中的其他包或模块，并读取提交的数据文件。每个评分用例在新沙箱中执行快照的只读副本，宿主比较真实返回值与期望值，保留逐用例结果。参考实现只用于可信材料记录，不执行、不做源码相等比较；评分期望值始终留在宿主。学生的退出码 0、自报通过或被修改的反馈测试均不能代替评分。此切片尚未实现任意 MBPP 测试程序的适配。

`test-command-v1` 对应 `CommandGrader(command=..., hidden_files=...)`。实验层在显式提交后，把隐藏文件写入 `private/grading-workspace/` 中的快照副本，再在新沙箱中只读执行测试命令；隐藏文件不会写回工作区、提交快照或学生观察。隐藏文件覆盖同路径的学生文件；若学生文件或目录阻碍注入，该提交评分失败。命令退出码为 0 即通过，输出文本或截断不影响判断；超时为失败，执行器故障单独记录。Grade 中的 `total_cases=1` 表示执行了一条测试命令，命令内部可运行多个 pytest 用例。

测试命令评分的信任边界：**学生代码和测试在同一个进程里运行，恶意提交可以伪造通过**，例如导入时调用 `os._exit(0)`。这是与函数调用评分不同的已接受限制；OPSD 训练信号不使用评分结果，轨迹保留供审计。CLI 的可信 Task JSON 使用以下 grader 字段选择此评分器：

```json
"grader": {
  "command": "pytest -q tests/test_hidden.py",
  "hidden_files": {
    "tests/test_hidden.py": "from solution import solve\ndef test_solve(): assert solve(-2) == -4\n"
  }
}
```

提交快照包含整个工作区的普通文件和目录，包括空目录、二进制文件与文件的执行权限；符号链接、FIFO、socket、设备和不可读文件都会使提交无效。快照从提交步的清单与内容存档生成，不跟随链接、不复用工作区 inode，后续对工作区的修改不影响它。`environment.submission_bytes` 限制全部文件的总字节数，`environment.submission_files` 限制路径条目数（含目录）。

限制来自配置的 `environment` 块，覆盖墙钟超时、捕获输出字节、提交大小和条目数、每进程地址空间、CPU 时间、单文件大小及打开文件数。超时会终止整个沙箱及后代进程；输出超限只丢弃后续字节，继续排空管道，不阻止命令执行。函数评分单独捕获 stdout/stderr，以便宿主解析 JSON；评分输出被硬截断时记为未通过，不能把一个有效 JSON 前缀误当成完整返回值。生效值随运行记录保存；它们不是 cgroup 总量配额，不能视为完整的多租户资源隔离。安全边界依赖节点内核和系统运行库。

## 运行产物

`runs/<唯一运行 ID>/` 与源码分离，权限为当前用户私有。以下是实验运行的完整产物；`agent` 只保存 Harness 证据和工作区，没有 Task、候选分析或评分：

- `run.json`（schema version 10）：任务身份与材料摘要、执行环境标识、Harness 终止状态、提交文本、合并后的完整配置、protocol／environment／budget 三个指纹、协议、采样条件、模型与编码身份（`encoding_details` 保存 tokenizer 名称、实际 revision、文件和模板哈希、库版本、thinking 与停止标记）、预算与用量、隔离检查、实际轨迹（含每次生成的原始 token、来源、logprob、上下文位置、原文 `text`、分离后的 `reply`、学生实际看到的反馈、每步工作区清单和当时的预算用量）、初始工作区清单、完整 token 上下文、事后计算的候选与修复尝试、整个提交的内容哈希、清单和评分。回合进行时只保存证据，候选与修复结果在结束时写入。回放的 `logprobs` 保存为 `null`。`completed` 表示评分流程完成；任务成功由 `grade.passed` 决定。评分基础设施故障单独记录，未执行的用例不会计为通过。`show` 也支持没有逐步用量快照或分离后正文的历史记录，以已存的 token、耗时和原文恢复展示。
- `private/task.json`：完整可信任务配置，包括参考实现和评分材料。
- `workspace/`：Task 的初始文件及学生操作后的实际文件状态。
- `blobs/<sha256>`：初始状态及各步中普通文件内容的只读存档；跨路径、跨步骤去重。
- `candidates/NNN.json`：多文件候选的只读清单；单文件候选直接复用 blob。
- `submission/`：整个工作区的独立提交快照，包含路径、目录和原始文件内容；评分时只读挂载。
- `private/grading-workspace/`：评分阶段创建的快照副本；函数评分逐用例执行，命令评分仅在此副本注入隐藏文件。
- `probe/`：能力探针使用的临时工作区。

`run_experiment` 返回值及整个运行目录供实验操作者使用；模型上下文只包含系统提示、公开任务与各步 `feedback` 中的真实观察。评分结果、参考实现、隐藏输入及期望值均不回写本回合工作区。

## 验证

```bash
uv run mypy
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

测试均通过小型实验入口保留真实隔离、bash 和评分，不调用付费模型。需要在允许 Bubblewrap 创建 namespaces 的节点执行；受限外层沙箱中的隔离测试应失败，不能靠跳过测试宣称节点隔离已验证。
