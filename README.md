# bashlet

A Tiny Coding Agent with On-Policy Self-Distillation。

当前实现任务 01、02、12–15：Harness 入口 `bashlet.run_episode` 负责指令、工作区和模型之间的交互；实验入口 `bashlet.run_experiment` 负责准备 Task、对提交工作区做快照、独立评分和事后候选／修复分析。每个交互回合都由 `model=` 驱动：可以使用根据真实观察采样行动的策略，也可以用 `ReplayModel` 编码预写回复；两者共用同一 token 上下文、Harness、预算与评分。真实 PyTRIO 采样由任务 04 接入。

## 运行

宿主使用 Python 3.13.12 和 uv；依赖版本锁定在 `uv.lock`，默认使用清华 PyPI 镜像。
执行节点需提供 Linux user/mount/PID/network namespaces、Bubblewrap、`/usr/bin/prlimit`、`/bin/bash` 和 `/usr/bin/python3`。

```bash
uv sync --locked
uv run bashlet examples/sum.json
```

也可以用 `uv run python -m bashlet examples/sum.json`。终端返回状态和运行目录：退出码 0 表示提交通过，1 表示任务未通过或未提交，2 表示基础设施或配置错误。
用 `--output-dir <目录>` 指定产物根目录，每次都会建立新的运行目录。

JSON 请求是可信任务输入，只包含 `task` 和 `replies`。协议、执行环境、交互预算和采样参数来自版本化配置，不写在请求里。`replies` 是完整回复文本的列表，每条回复包含一个 bash 围栏块；块内容为 `submit` 时提交整个工作区并立即结束回合。命令失败、超时和超量输出保留真实观察，后续回复仍可继续执行。旧的 `actions` 请求，以及写在请求里的 `limits` 或 `budget`，会被视为配置错误。

## 实验配置

默认条件在 `src/bashlet/configs/bash-block-v1.yaml`，分四块：

- `protocol`：agent 可见的文本和协议规则。字符串是 Jinja2 模板，未定义变量在运行开始前报错。当前默认渲染结果就是 bash-block-v1，系统提示、观察和格式错误文本与原先逐字节相同。
- `environment`：执行环境标识 `id`、执行器路径、只读挂载、符号链接、`PATH`/`LANG`，以及单条命令的超时和资源限制。
- `budget`：步数、生成 token 数、墙钟时间和连续格式错误上限。
- `model`：采样参数、后端名称和 thinking 开关。本切片仍由调用方传入模型对象；thinking 只被记录，解析留到后续协议。

可以叠加多个文件，后面的覆盖前面的；`--set key=value` 在所有文件之后覆盖单项，值按 YAML 解析。未知键、类型错误和非正的限额都会在建立运行目录之前失败。

```bash
uv run bashlet examples/sum.json --config tighter.yaml --set budget.max_steps=4
```

Python 调用使用同一套层叠：`run_experiment(..., config=[path], overrides=["budget.max_steps=4"])`。

每次 `run.json` 写入合并后的完整 `config`，以及 `fingerprints.protocol`、`fingerprints.environment`、`fingerprints.budget`。指纹是对应块的稳定 SHA-256。同一配置得到相同指纹；只改一块时，只有那一块的指纹变化。`model` 不计入这三个指纹，所以更换采样参数或后端名称不会被当成更换了 Harness 或交互预算。协议指纹沿用 v1 的定义摘要，默认配置下仍是测试固定的那个值；修改任何 agent 可见的协议文本都会改变它，并需要新的协议名。

请求文件 CLI 使用回放专用的 `Utf8Encoding`，以 UTF-8 字节作为 token，编码身份为 `replay/utf8-v1`。Python 调用者可以把学生的真实编码传给 `ReplayModel`，用于构造示范轨迹。

## 两层入口与 Task v2

`run_episode(instruction=..., workspace=..., model=..., config=...)` 在已有工作区中运行一个回合，不接收 Task 或评分器。`config` 是 `bashlet.config.load_config()` 返回的完整配置；默认加载内置配置。返回 `EpisodeResult`，包含轨迹、终止状态、最终工作区路径 `workspace`、预算用量和可选的 `submission_text`。此入口直接修改指定工作区，运行记录默认保存在唯一的 `runs/<ID>/` 中；调用者也可指定独立的 `run_dir`，通过 `on_update` 接收进度。每个回合的状态都属于调用实例。

`run_experiment` 接收 Task，将 `initial_files` 中的相对路径和文本复制进本次运行的工作区，再调用同一个 Harness。Task 还包含稳定的 `task_id`、指令 `prompt`、执行环境标识 `environment_id`、宿主持有的 `FunctionGrader`，以及可选的参考实现、反馈命令和实现路径。默认环境标识为 `linux-bwrap-v1`，须与配置的 `environment.id` 一致。初始文件可以放在任意子目录；文件名、反馈测试和入口函数都由任务数据声明。

本切片继续使用冻结的 bash-block-v1 文本和指纹；系统提示仍提及原来的三个文件，通用协议文本由任务 16 升级。实验层已不再自动创建它们，MBPP 类 Task 在 `initial_files` 中显式提供 `task.md` 和 `feedback_tests.py`。

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
            "```bash\nprintf 'def solve(n): return n * 2\\n' > solution.py\n```",
            "```bash\npython3 feedback_tests.py\n```",
            "```bash\nsubmit\n```",
        ],
        encoding=Utf8Encoding(),
    ),
    output_dir=Path("runs"),
)
print(result.status, result.grade)
```

## 行动协议 bash-block-v1

协议文本定义在默认配置的 `protocol` 块，由 `src/bashlet/protocol.py` 渲染和解析。SFT 示范、学生 rollout 和评测共用它。完整系统提示、观察模板、历史编码和验证判定连同指纹写入每次运行 `run.json` 的 `protocol` 字段；测试固定该指纹，修改任何协议文本都必须更换协议名、更新指纹并重建示范。

- 每条回复恰好包含一个 ```` ```bash ```` 围栏块，块外文本允许。块内容为 `submit` 时提交；否则以 bash 在隔离工作区执行。
- 观察以 `<status>`、`<exit_code>`、`<stdout>`、`<stderr>` 标签作为下一条 user 消息返回；缺少末尾换行时补一个。标签内容不转义，原始输出另存于 `observation`。
- 无效行动（零个或多个块、空块、因长度上限被截断）不执行，返回 `<invalid_action>` 说明并计入步数；连续 `max_consecutive_invalid` 次后以 `invalid_action` 终止。
- 历史是只追加的 token 序列：先 `encoding.prompt(system, task.prompt)`，之后每步追加原始生成 token 和 `encoding.continuation(tokens, feedback)`。采样和回放的 token 都不会 decode 后重新 tokenize；观察只在对应行动实际执行后追加。

模型通过 `Model` Adapter 接入（`identity`、`encoding`、`sample(prompt, sampling) -> Generation`）；编码还提供 `encode(text)`，供回放使用。每次 `Generation` 标注 `source`：默认 `sampled`，必须带等长、有限、非正的逐 token `logprobs`；`replayed` 的 `logprobs` 为 `None`。回放 token 不属于 on-policy 样本。返回内容违反约定或无法解码时以 `model_error` 结束，被拒输出保存在 `rejected_generation`，不进入上下文；模型调用本身抛错（服务、网络、Adapter）记为 `infrastructure_error`。测试用可控采样 Adapter 位于 `tests/adapters.py`。

Harness 回合结果互斥：`submitted`（显式提交）、`interrupted`、`invalid_action`、`model_error`、`budget_exhausted`（`InteractionBudget` 的步数、生成 token 数或墙钟秒数）和 `infrastructure_error`。实验层保留 `episode_status`，将已提交回合进一步判为 `completed`、`invalid_submission` 或评分阶段的 `infrastructure_error`；`result.passed` 只在可信评分通过时为真。未提交回合不评分，计为失败，最终工作区保留供诊断。回放回复用尽后返回空回复，按连续格式错误上限或交互预算结束。每次请求的 `max_tokens` 裁剪到剩余生成额度；回放也按此裁剪 token，并将被截断的回复标记为 `length`。`usage` 记录实际步数、生成量、无效行动数与耗时。墙钟预算从隔离探针通过后开始计时，在步与步之间检查；单条命令另受 `Limits.timeout_seconds` 约束，单次模型调用的超时由接入的 Adapter 负责。

每个交互步（包括格式错误、提交和执行器故障）之后记录工作区清单：每条相对路径包含文件类型，普通文件另含 SHA-256 和字节数；符号链接只记录目标，FIFO、socket 等特殊文件只记录类型，不读取或跟随。相同字节只在 `blobs/<sha256>` 保存一份，删除或覆盖文件后仍可按历史清单取回原内容。不可读的路径保留类型和读取错误，供后续行动修复；初始化时也保存一份清单。非 UTF-8 文件名和符号链接目标通过 JSON 转义保存，读取记录时可恢复原路径。

候选实现、验证与修复尝试在回合结束后，由轨迹和清单计算。`Task.implementation_paths` 声明接受分析的文件，`Task.feedback_command` 声明反馈命令；MBPP 类任务在任务数据中填写 `("solution.py",)` 和 `"python3 feedback_tests.py"`。未声明实现路径时不做候选或修复分析；只声明实现路径时仍记录候选，但不判定验证。

候选按全部声明文件的内容去重，回退时沿用原编号，第 0 个候选即首次实现。声明文件须都可读、为普通文件且合计不超过提交大小限制；缺失、特殊或过大的文件状态沿用最后一个候选，保持 v1 行为。单文件候选直接指向内容 blob；多文件候选的 `path` 指向路径到内容哈希的 JSON 清单，`files` 列出各文件哈希。只有整条命令与任务的反馈命令一致（忽略首尾空白），且当前反馈材料与初始清单相同，才算验证；成功与否取真实观察状态。修复尝试从失败验证开始，到验证一个不同候选时结束，重复验证同一候选不新开尝试。反馈材料来自 Task 的初始文件，排除声明的实现文件；v1 配置中原有的验证描述保留用于兼容冻结指纹，实际分析使用 Task 的声明。

## 隔离与评分边界

每次实验先运行真实访问探针：验证 namespace 已分离、环境变量已清理、宿主私有文件和包含参考实现／评分材料的配置无法读取，以及任务目录可写。探针不通过时记录 `infrastructure_error`，不执行任何任务命令，也不退回宿主直接执行。缺少执行器可用 `--set environment.bwrap=/does-not-exist/bwrap` 重现。

沙箱只挂载只读的 `/usr/bin`、`/usr/lib`、`/usr/lib64` 和当前任务目录，使用独立 `/proc`、`/dev`、`/tmp`，清理环境和继承的文件描述符，断开宿主网络，移除 capabilities 并建立新会话。实现依据 [Bubblewrap 0.6.1 的说明](https://github.com/containers/bubblewrap/blob/v0.6.1/README.md)及[参数文档](https://github.com/containers/bubblewrap/blob/v0.6.1/bwrap.xml)。节点内实际任务 Python 来自 `/usr/bin/python3`，与宿主 Python 分别记录；当前节点为 Python 3.10.12。首版任务仅使用该解释器的标准库。

`json-function-v1` 评分按 `Task.grader` 中的 `module_path` 和 `function_name` 加载模块、调用函数，接受 JSON 位置参数，返回值须可表示为 JSON。模块可导入工作区中的其他包或模块，并读取提交的数据文件。每个评分用例在新沙箱中执行快照的只读副本，宿主比较真实返回值与期望值，保留逐用例结果。参考实现只用于可信材料记录，不执行、不做源码相等比较；评分期望值始终留在宿主。学生的退出码 0、自报通过或被修改的反馈测试均不能代替评分。此切片尚未实现任意 MBPP 测试程序的适配。

提交快照包含整个工作区的普通文件和目录，包括空目录与二进制文件；符号链接、FIFO、socket、设备和不可读文件都会使提交无效。快照从提交步的清单与内容存档生成，不跟随链接、不复用工作区 inode，后续对工作区的修改不影响它。`environment.submission_bytes` 限制全部文件的总字节数，`environment.submission_files` 限制路径条目数（含目录）。

限制来自配置的 `environment` 块，覆盖墙钟超时、stdout 与 stderr 合计字节、提交大小和条目数、每进程地址空间、CPU 时间、单文件大小及打开文件数。超时或输出超限会终止整个沙箱及后代进程。生效值随运行记录保存；它们不是 cgroup 总量配额，不能视为完整的多租户资源隔离。安全边界依赖节点内核和系统运行库。

## 运行产物

`runs/<唯一运行 ID>/` 与源码分离，权限为当前用户私有：

- `run.json`（schema version 6）：任务身份与材料摘要、执行环境标识、Harness 终止状态、合并后的完整配置、protocol／environment／budget 三个指纹、协议、采样条件、模型与编码身份、预算与用量、隔离检查、实际轨迹（含每次生成的原始 token、来源、logprob、上下文位置、学生实际看到的反馈和每步工作区清单）、初始工作区清单、完整 token 上下文、事后计算的候选与修复尝试、整个提交的内容哈希、清单和评分。回合进行时只保存证据，候选与修复结果在结束时写入。回放的 `logprobs` 保存为 `null`。`completed` 表示评分流程完成；任务成功由 `grade.passed` 决定。评分基础设施故障单独记录，未执行的用例不会计为通过。
- `private/task.json`：完整可信任务配置，包括参考实现和评分材料。
- `workspace/`：Task 的初始文件及学生操作后的实际文件状态。
- `blobs/<sha256>`：初始状态及各步中普通文件内容的只读存档；跨路径、跨步骤去重。
- `candidates/NNN.json`：多文件候选的只读清单；单文件候选直接复用 blob。
- `submission/`：整个工作区的独立提交快照，包含路径、目录和原始文件内容；评分时只读挂载。
- `private/grading-workspace/`：评分阶段创建的快照副本，逐用例运行在新沙箱中。
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
