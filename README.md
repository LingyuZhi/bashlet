# bashlet

A Tiny Coding Agent with On-Policy Self-Distillation。

当前实现任务 01–05、12–20：Harness 入口 `bashlet.run_episode` 负责指令、工作区和模型之间的交互；实验入口 `bashlet.run_experiment` 负责准备 Task、对提交工作区做快照、独立评分和事后候选／修复分析。CLI 可以实时观察回合、回放运行记录，并评测固定开发任务清单。每个交互回合都由模型驱动：可以使用根据真实观察采样行动的策略，也可以用 `ReplayModel` 编码预写回复；两者共用同一 token 上下文、Harness、预算与评分。正式实验使用 PyTRIO 的 Qwen/Qwen3.5-4B，在提交付费请求前通过共用费用账本检查额度。

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

`class` 可写内置类名 `ReplayModel`、`VLLMModel` 或 `PyTRIOModel`，或者完整导入路径，例如 `bashlet.replay.ReplayModel` / `bashlet.replay:ReplayModel`。回放模型默认使用 `Utf8Encoding`；`replies` 为预写回复列表，`identity` 可选。自定义类须满足同一个 token 模型和编码接口，采样输出仍接受 Harness 的 token / logprob 校验；构造失败时在创建运行目录前报配置错误。Python 调用者仍可以直接传入 `model=` 对象。

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

`tokenizer_revision` 必填，支持 Hub commit、tag 或 branch；运行记录保存实际解析到的 snapshot commit 和文件 SHA-256。`tokenizer` 也可指向本地 tokenizer 目录，此时 revision 由调用者指定，文件哈希仍参与身份。默认先读本地 Hugging Face 缓存，缺少资产时从 HF-Mirror 下载 tokenizer 文件，不下载模型权重；可用 `tokenizer_cache_dir` 指定缓存位置。依赖精确锁定为 transformers 4.57.3、tokenizers 0.22.1 和 huggingface-hub 0.36.0。

`QwenChatMLEncoding` 用官方 `apply_chat_template` 构造首个 prompt，thinking 开关对应 `enable_thinking`；后续只追加 ChatML 观察与同一生成提示。编码提供 `stop_tokens`、真实思考标记的 token ID，以及 `reply(tokens)` 返回的行动正文、思考内容和结束状态。原始 token 不改写；即使官方多轮模板会移除早先的思考，本项目也保留它。调用方式依据 [Qwen 官方说明](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)和 [Transformers 聊天模板文档](https://huggingface.co/docs/transformers/v4.57.3/chat_templating)。任务 04 已用 Qwen3.5-4B 的真实 tokenizer 验证两种 thinking 模式、官方模板、思考与行动边界及多轮原始 token 保留（H-11）。

Python 调用者可传入 `ReplayModel(replies, encoding=QwenChatMLEncoding(name, revision=revision, thinking=True))`，用学生编码构造示范；每步仍标注 `replayed`。原生 thinking 需要真实 tokenizer 或调用者提供的编码。以下演示使用本地缓存的 Qwen3-1.7B，思考里的命令不会执行，第二步行动没有说明：

```bash
uv run bashlet agent "创建并运行 hello.py，然后提交" --workdir runs/demo-input -c configs/replay-qwen-thinking.yaml
uv run bashlet show runs/<运行 ID>
uv run bashlet show runs/<运行 ID> --show-thinking
```

## PyTRIO 与费用账本

`PyTRIOModel` 使用官方 SDK 0.2.9，模型固定为 `Qwen/Qwen3.5-4B`。API Key 放在忽略的 `.env` 中，建议格式为 `PYTRIO_API_KEY=...`；也支持一行裸 `trio_...` 密钥，环境变量优先。密钥不会写入 YAML、模型上下文或运行记录。

```bash
uv run bashlet task examples/pytrio-smoke.json -c configs/pytrio-smoke.yaml --output-dir runs/issue04-pytrio
uv run bashlet spend status
```

配置固定 tokenizer 的 Hub commit 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，默认关闭 thinking；可用 `--set model.thinking=true` 启用。服务端基座模型只暴露别名，无法按 Hub commit 固定权重；`policy_details.server_model_revision` 明确为 `null`，参考 revision 与 tokenizer 文件哈希另行保存。

首次使用前核对实时公开价格、模型权限及上下文上限，再用完整 ChatML 前缀采样一个 token，并用 [`compute_logprobs`](https://docs.pytrio.com/docs/advanced/compute_logprobs) 重评分同一原始序列。返回概率逐位置对应 token，首个位置允许 `None`；新增 token 的两种 logprob 相差超过 0.05 时停止回合。此检查仅覆盖温度 1、top-p 1 与聊天前缀；冻结 SFT 教师及其他参数的训练兼容性留给任务 09。SDK 丢弃原始响应的 request id，因此这个锁定版本的适配器捕获两条固定传输事件，核对发送和返回身份，不采集日志正文。

`configs/spend.yaml` 保留常规 ¥150、预留 ¥50、项目 ¥200，同时用已充值的 ¥10 限制实际提交；`pytrio-smoke` 累计上限 ¥0.50，`opsd-pilot` 累计上限 ¥10。每个模型实例的能力探测另限 ¥0.20，并计入所属子额度。后续付费采样、教师评分、训练、示范和评测都须通过同一个 `SpendLedger.execute`；不同费用类别不会重置额度，上下文和观察全部计价。

账本默认为 `runs/spend.sqlite3`。SQLite 事务在提交前原子检查已发生费用、在途预留与新请求上界；并发进程共用此路径，禁止用新账本规避累计预算。重启检查账本策略与 workspace 身份。SDK 隐式重试关闭；明确重试需要新预留。额度不足、子额度未知或价格无法约束时结束为 `budget_exhausted`。已返回的失败与被丢弃结果保留全部费用上界；超时、中断和网关失败保留在途预留，拒绝重复提交相同未结算工作。

价格来自[官方目录](https://pytrio.cn/api/model-prices)，版本 `2026-08-21`：未缓存 prefill 为 ¥1.49/M token，采样与训练为 ¥4.54/M token。评分归类尚未确认，按最高费率估计，并计入评分接口内部额外生成的一个 token。每个阶段向上预留到 ¥0.01，再乘 2；`billing_basis=upper_bound` 表示保守费用占用，不能当作真实账单。余额查询、最小收费粒度、失败请求计费和缓存收费仍需账单确认。取得对应请求的完成与账单证据后，可人工核对：

```bash
uv run bashlet spend reconcile <账本请求ID> --actual-cny 0.01 --outcome completed --evidence runs/bill.json
```

`bill.json` 是操作者核实的非空 JSON 证据，须对应服务请求身份、终态及实际费用。费用未知时不能填零释放预留；实际费用高于原上界后，账本会阻止后续提交。

2026-10-02 的真实 smoke 在 3 步、56 个任务生成 token 内提交，独立函数评分 3/3 通过；聊天前缀的采样与重评分 logprob 差异为 0。完整记录在 `runs/issue04-pytrio/74500de4b2f244868b5cccf297774f66/run.json`，验收与费用假设在上层的 `acceptance.json`。此前裸单 token 探测相差约 4.20，失败证据也保留；原因尚未得到服务端确认。两次运行共占用 ¥0.28 上界、在途 ¥0；账单接口对当前 API Key 返回 401，实际扣费尚未核实。

## 开发用 vLLM

以下保留任务 20 的历史开发配置和证据。任务 04 起的实际调用统一使用上面的 Qwen3.5-4B 配置。

`VLLMModel`（也可写完整导入路径 `bashlet.vllm.VLLMModel`）只产生开发证据。每个运行的 `policy_details.development_only` 固定为 `true`，实时视图和 `show` 也会注明排除实验结果。该字段同样保留在探针失败、服务失败和中断的记录中；有评分的开发回合也不作为实验结果。

服务环境独立锁定在 `configs/dev-vllm/`，使用 Python 3.12、vLLM 0.11.0、PyTorch 2.8.0 和 CUDA 12.8 wheel。首次使用先核对 `/version`、`/v1/models`，再用一个 prompt token 采样一个 token。探针结果、实际服务版本、模型 revision、服务报告的权重路径和上下文上限写入 `policy_details`，tokenizer 的文件哈希仍写入 `encoding_details`。探针失败时没有任务生成或命令执行。

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

## 已验证的 MBPP 任务集

`prepare-mbpp` 将锁定的 MBPP sanitized 转为当前 Task 数据结构，使用现有 `json-function-v1` 函数评分和 bash-block-v3 行动协议。Task v2 是任务 15 引入的数据结构名称，与行动协议的版本独立。

默认数据选择记录在 [configs/mbpp-sanitized-v1.json](configs/mbpp-sanitized-v1.json)：Google Research 提交 `e49bbfe381c9c0e564b937f1c4e163a2273c65cc`，原文件 SHA-256 为 `ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9`。这是本次实施的例行数据配置。按[官方原始题号划分](https://github.com/google-research/google-research/blob/e49bbfe381c9c0e564b937f1c4e163a2273c65cc/mbpp/README.md)，sanitized 的 427 题包含训练 120、开发 43、最终测试 257 和提示示例 7 题；提示示例排除。原始题号 601–974 属于 train，511–600 属于 dev，11–510 属于 test，不重新随机划分。

先准备 [sandbox Python 环境](#运行)，然后运行：

```bash
uv run bashlet prepare-mbpp \
  --source-lock configs/mbpp-sanitized-v1.json \
  --output-dir runs/mbpp-sanitized-v1
```

下载遵循 `HTTP_PROXY`、`HTTPS_PROXY`、`NO_PROXY`，网络异常及 HTTP 408/429/500/502/503/504 最多重试两次。可用 `--source-file /path/to/sanitized-mbpp.json` 离线重建，包括复用此前产物的 `private/source.json`。从镜像取得的文件同样必须通过固定 SHA-256；不自动用另一个 HF 数据版本替换原文件。输出目录必须是新目录，避免覆盖已有清单和验证证据；中断后用缓存原文件重建到新目录。

转换规则 `mbpp-json-functions-v1`：第一条原始断言公开为 `feedback_tests.py`，其余不同输入用于隐藏评分。只支持单一命名函数的位置参数调用与字面量相等断言；输入必须保持 JSON 类型语义，tuple、set、非字符串键字典、复杂调用与不支持的断言会被排除并记录理由。输出 tuple 按既有 JSON 评分约定序列化为 list；不支持集合、复数和近似比较。重复反馈输入不算独立评分用例；没有独立用例的任务排除。该规则保留少量隐藏用例，**不是 MBPP+ 增强评测**；MBPP+ 的完整格式、类型恢复和增强用例导入不在这个转换器中。

每个候选任务先在私有验证工作区真实运行全部原始断言，再用学生所用的函数评分器逐用例验证参考实现。参考验证只使用固定回放命令，没有模型采样；错误参考解被排除，基础设施故障使准备过程失败，不据此筛掉题目。最终测试材料只进入可信数据验证与评分流程，这一步不作模型选优或探索性评测。

产物全部保存在权限为 `0700` 的输出目录下：

- `manifest.json`：源版本、内容哈希、转换规则、验证环境、所有原题的家族／split、可用状态、排除原因和验证记录路径；只有 `status: complete` 才能从清单选择任务。
- `private/source.json`：校验过的原始数据，供离线重建。
- `private/tasks/{train,dev,test}/<原题号>.json`：可直接传给既有实验入口的可信 Task；包含宿主持有的参考实现和评分用例，不能整份复制给学生。
- `private/reference-validation/`：参考实现的真实隔离执行、原始断言结果和逐用例评分证据，供操作者审计。

例如选择训练题 602，通过已有任务入口运行；模型配置仍由 `-c` 决定：

```bash
uv run bashlet task runs/mbpp-sanitized-v1/manifest.json \
  --task-id mbpp/602 --split train -c configs/dev-vllm.yaml
```

选择入口检查整份清单的原题号、家族和 split，校验所选 Task 的内容哈希和来源。即使增强来源把 `Mbpp/602` 标成 `test`，其家族仍为 `mbpp/602`，归属 train；错误的目标 split 会在模型启动前被拒绝。直接 Task JSON 和 `run_experiment` 同样校验来源中的家族与原始划分。`Task.origin`（`TaskOrigin`）保存数据来源和家族身份，目前支持 MBPP 来源；派生任务用 `dataclasses.replace(task, task_id=..., prompt=..., initial_files=...)` 保留它。每次实验的 `run.json.task_origin` 同样保留该身份，供示范和 rollout 继承；修改衍生内容不产生新的原题家族。

2026-10-02 已真实完成全部 427 题的数据验证：训练 95、开发 37、最终测试 197，共 329 项可用任务；其余 98 项为提示示例或不支持的转换格式，均记录理由，未发现可转换参考解验证失败。清单与全部参考验证证据位于 `runs/mbpp-sanitized-v1-verified/`。真实训练题 `mbpp/602` 通过清单选择、v3 回放、反馈测试和独立评分（2/2），使用的实现与原参考程序不同；验收摘要为 `runs/issue03-mbpp-smoke/acceptance.json`。这些是数据与入口验收，不是模型能力评测。

## 固定开发集评测

任务 05 的 `evaluate` 逐项调用同一个 `run_experiment`，在第一次付费探测前保存所有预定任务，完整结果写入 `runs/<目录>/<运行 ID>/evaluation.json`。固定清单见 [configs/mbpp-dev-8-v1.json](configs/mbpp-dev-8-v1.json)：将 37 道已验证开发题按原始题号排序，取位置 `floor(i * 36 / 7)`（i=0…7）的 8 道，选择不依据模型结果。此入口只接受 `dev`，不开展最终测试选参。

```bash
uv run --locked bashlet evaluate configs/mbpp-dev-8-v1.json \
  --snapshot pytrio/Qwen/Qwen3.5-4B@unversioned-base \
  -c configs/pytrio-smoke.yaml -c configs/development-baseline-v1.yaml \
  --output-dir runs/issue05-development-baseline
```

每题最多 6 步、4096 个生成 token、300 秒；单次生成最多 1024 token，temperature=1、top-p=1、seed=4，关闭 thinking。`development-baseline` 累计子额度为 ¥2，探测、失败和重跑也计入，并继续受同一个 `runs/spend.sqlite3` 的 ¥10 充值额度和 ¥200 项目上限限制。账本允许在原有全局限制、子额度和历史支出不变时追加子额度，记录政策变更；不能通过重启提高已有上限。

评测记录保存模型身份、参考 revision、服务端会话／采样身份、完整条件、数据来源与 Task 哈希、反馈命令、每题运行路径和文件哈希。`comparison_sha256` 覆盖任务顺序与内容、评分和分析规则、协议、环境、交互预算、采样参数、编码与实际上下文上限；模型快照另行保存，不混进比较条件。后续 SFT／OPSD 使用原清单和交互／采样条件，选择对应 `model.model_path`，并把 `--snapshot` 写为 `pytrio/Qwen/Qwen3.5-4B@<该路径>`；阶段评测子额度可另行分配，仍共用原账本及全局现金上限，不能重置已用额度。快照加载使用 [SDK 的 create_sampling_client(model_path=...)](https://docs.pytrio.com/docs/api/ServiceClient#create_sampling_client)，仍要在相应训练任务中验证真实保存／加载。原始基座只有服务端别名，`server_model_revision=null` 明示无法确认权重 commit，不把 tokenizer revision 当成权重证据。

首次实现采用 `first-recorded-candidate-v1`：某个已执行步骤结束后，所有 Task 声明的实现路径首次成为有内容存档、总大小未超实现限制的普通文件。空文件、语法错误以及任务原有但未修改的实现都算候选；无需先验证或提交。回合结束后，从该步清单和 blobs 恢复整个工作区，再用原评分器独立评分。结果保存在 `first_implementation` 和 `private/first-implementation/`，不会成为学生反馈。单题 API 可用 `assess_first_implementation=True` 启用，普通交互默认不增加这次评分。

最终通过率和首次实现通过率的分母均为全部预定任务。无候选计首次实现未通过；服务、现金额度或中断造成的缺口另列，不能缩小分母。失败恢复分母为首次实现已独立评分且失败的任务 ID 集合，分子要求最终显式提交并通过；没有失败候选时比例为 `null`。各模型的失败集合可能不同，不能把这些比例当作同一失败起点上的配对比较。

每题只运行一个回合，回合级不自动重试；后端请求仅按原配置有限重试并逐次记账。模型失败、模型输出证据错误、协议错误、交互／上下文预算耗尽、服务错误分别记录。服务故障、现金不足或中断后停止，剩余题保留为 `not_started`，整次运行标为 `incomplete`，CLI 返回 2；运行完整返回 0，通过率不影响此退出码。重跑创建全新记录并沿用原分母和账本，不能合并不同运行中的成功样本。条件变化产生新指纹，需另存配置／任务清单新版本并重跑受影响对照，最终测试前冻结条件。

日常验收用 `--evidence-kind test-double` 显式标识替身或回放；默认 `experimental` 只接受 PyTRIO 生产后端。真实评测与测试替身的输出目录、证据标签和比较指纹分开。费用摘要关联本次 sampler 的账本请求；`charged_upper_bound_cny` 与未结算预留都是保守占用，只有全部请求具有已核实账单时 `actual_cny` 才是数值。

2026-10-02 的真实基线完整执行 8 题，最终通过 5/8（62.5%），首次实现通过 6/8（75%），错误为协议错误 2 题、交互步数耗尽 1 题，服务缺口 0。失败恢复分母仅 `mbpp/581`，恢复成功 0/1。`mbpp/573` 的首次实现评分通过，但模型随后连续输出观察格式，未提交；`mbpp/593` 未形成候选；`mbpp/581` 修复后公开反馈通过，却在提交前用尽 6 步，所以仍为失败。这 8 题显示出协议适配和交互效率的改进空间，不能作为最终测试能力估计；本次没有据此修改协议或扩大预算重跑。

本次共 39 次任务采样、2907 个任务生成 token，加上 2 次探测共 41 个收费请求，账本占用上界 ¥1.64、在途 ¥0；全项目累计上界 ¥1.92，充值剩余额度 ¥8.08。账单接口仍返回 401，实际扣费未知。结果与逐项哈希验收在 `runs/issue05-development-baseline/1dd96303a0784be6a0988c6a22bc31dc/` 的 `evaluation.json`、`acceptance.json`、`billing-verification.json`；比较指纹为 `cc0efb2e89e26e7d5dfcc7a148623eeb60042cde20a5c207a4ea0b4f292c9da8`。对应 sampler 为 `sample_3sv7dco753ii`，每题的 `run` 路径可交给 `bashlet show` 回放。

后续人工费用证据另存 `billing-user-report.json`，保持原始运行记录不变。用户确认后台累计显示 `sample:2.97K ¥0.01`，另有 prefill ¥0.04，其中包含上一个任务；这两个分项合计 ¥0.05，不能精确分摊到本次评测。该累计金额与本次账本上界分别记录，尚未对 41 个请求擅自分摊费用或释放账本占用。替身验收记录单独归档在 `runs/issue05-test-double/`。实现通过 256 项全量测试；审查修复后评测／PyTRIO 两个文件的 38 项测试、mypy、ruff 均通过，两轴审查没有遗留代码问题。

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

`json-function-v1` 评分按 `Task.grader` 中的 `module_path` 和 `function_name` 加载模块、调用函数，接受 JSON 位置参数，返回值须可表示为 JSON。模块可导入工作区中的其他包或模块，并读取提交的数据文件。每个评分用例在新沙箱中执行快照的只读副本，宿主比较真实返回值与期望值，保留逐用例结果。学生实验不会执行参考实现或比较源码；MBPP 准备阶段在独立的可信验证运行中执行参考实现。评分期望值始终留在宿主。学生的退出码 0、自报通过或被修改的反馈测试均不能代替评分。MBPP 转换仅支持上文记录的子集。

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

编码测试使用预先缓存的真实 tokenizer：Qwen3-1.7B 的上述 revision 放在默认 Hugging Face 缓存，Qwen3.5-4B 的上述 revision 放在 `runs/tokenizers`。只需 tokenizer、模板和配置文件，不需模型权重。
