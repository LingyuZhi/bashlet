# bashlet：极简 bash agent 的 SFT → OPSD 实验规格

Status: ready-for-agent

Type: spec

Recorded: 2026-09-28

Revised: 2026-10-03（任务 21 允许经验证的本地采样与评测；PyTRIO 仍是唯一生产训练后端）

Project root: `/data/home/zdhs0148/opd/bashlet`

本规格综合已结束的 grill-session：用户在 A18 确认项目范围，并在随后确认本规格的主要测试 seam。`ready-for-agent` 表示可进入实施，不表示代码已存在、账户已可用或实验已达标；本次交付只完成领域文档、ADR 和规格。

2026-09-29 的 Harness grill（Q1–Q18）改变了 Harness 的定位与设计：Harness 与任务解耦，参考 mini-swe-agent 2.4.6，并增加 CLI 和可配置的模型后端（ADR-0004）。相关决定见 D-15–D-29 和 Testing Decisions 中的「Harness v2 的测试 seam」，由任务 12–20 实施，执行顺序见 plan.md。决定后括号里的 Q 编号指该次 grill 的问题。

2026-10-01：用户确认行动正文不再强制 `THOUGHT:` 前缀或非空说明，以避免在模型原生 thinking 之外重复要求思考文本。此次变更由尚未完成的任务 19 实施，协议升级为 bash-block-v3；任务 18 提前支持可选行动说明的展示，任务 16 的交付记录保留为 v2 历史。本规格记录的是最新目标，尚不表示 v3 已实现。

2026-10-02：用户确认 W&B（`wandb`）为训练与配套评测的统一监控工具，project 为 `bashlet`，entity 使用账号默认空间；上传指标、脱敏配置和 checkpoint 引用，完整轨迹与权重保留在原存储位置。运行中断网时继续本地记录，恢复后同步。用户已在 `.env` 配置凭证，实际认证、权限和上传链路仍需任务 07 验证；本次只更新现有待实施文档，不表示 W&B 已接入。

## Problem Statement

用户是硕士一年级学生，希望通过一个可在 Assistant Professor 的 Research Intern／RA 面试中介绍的 tiny project，建立真实的 Agentic 后训练经验和研究工程能力。目标约为研究判断力 55%、工程能力 45%，需要能解释任务、训练信号、对照和结果，不要求提出新算法或达到论文发表标准。

此前项目出现过两类具体失败：后训练效果低于 SFT，以及随着开发推进，冗余文件、中间产物和多套实现增加，核心逻辑无法继续维护。本项目必须同时产生真实训练证据和可理解的代码，避免只完成包装、只复述方法名称，或再次失去对项目结构的掌握。

用户每天约有 4 小时，时间目标见 D-01。本地实际可用一张 A100 40GB，用于开发推理及任务 21 起经验证的示范采样和评测；生产训练使用 PyTRIO，以减少多套训练基础设施排错；新增 API 实验支出总上限为 ¥200。用户要求独立评测上的实际提升，阴性结果不能作为项目成功的替代标准。

任务 02 完成后，用户发现首版 Harness（bash-block-v1）过度贴合 MBPP：系统提示点名固定的任务文件，提交只能是一个文件，验证绑定一条命令，沙箱缺少 awk、HOME 和 pytest 等常用环境，输出稍长就终止命令。用户要的是一个小而通用的 bash 编程 agent，像 mini-swe-agent 那样能面对任意编程任务，再对它做后训练。设计 Harness 时应当把它当作一个真正的 agent 来做，而不是因为要在 MBPP 上训练，就把 MBPP 的数据格式写进 Harness。训练任务集从 MBPP 起步，以后可能扩展到更难的任务。用户还希望能直接看到 agent 工作时的表现，并接入不同的模型；无论哪个模型，都必须能取得 token 级的采样概率，取不到就显式报错。

## Solution

构建一个由用户能够读懂和维护的极简 bash Harness，让 Qwen/Qwen3.5-4B 从已有指令能力出发，先进行任务 SFT，再优先验证并运行 OPSD。Agent 根据自然语言指令在任务工作区中编写或修改程序，通过 bash 执行真实反馈测试，在收到观察后继续修复，最后提交。首版任务集为 MBPP，以后可替换或扩展。

Harness 是一个小而通用的 bash 编程 agent：给它一段指令、一个工作目录和一个模型，它就按固定规则工作到提交或终止。任务数据、评分和候选实现分析属于其上的实验层，因此换任务集只换数据（ADR-0004）。用户可以在终端里实时观察 agent 的每一步，也可以回放已保存的运行。模型后端通过配置接入，任何采样后端都必须返回 token 和逐 token 的采样 logprob。协议、执行环境、交互预算和模型选择等实验条件写在版本化配置中。

SFT 的主要目标是在保留原有编程能力的前提下，让模型更可靠地通过 bashlet Harness 完成任务：按协议输出可执行的 bash 行动，利用真实环境信息或执行反馈继续交互，并正确提交。它不要求提升原有解题能力，也不与 MBPP 或某一种写文件模板绑定；只有对任务表现有益的行为才纳入示范，不强制每题都读取、测试或修复。

SFT 使用经过执行验证的示范轨迹，验收边界见 D-06。OPSD 从合格的 SFT 模型快照出发：学生继续学习，冻结教师使用同一 SFT 快照并额外看到训练题的参考实现，沿学生实际轨迹计算概率监督。教师使用采样 token 监督；本项目不宣称复现完整词表 JSD。

训练和评测共享同一 Harness。原始指令模型、SFT 模型和 OPSD 模型使用一致的任务与交互预算进行比较；核心验收是 OPSD 的最终测试通过率高于 SFT。研究分析关注改善发生在首次实现还是失败恢复，正式消融、规模扩展和多种训练算法对比留到有余量时。

先完成受限小试，再根据实际 token、账单与开发集表现安排训练规模。交付可重跑的代码、模型快照／adapter 与训练状态、结果表、典型交互轨迹和短实验报告。

SFT、OPSD 小试与主训练，以及原始模型基线、开发集和最终评测，统一使用 W&B 展示进度、学习诊断、任务执行指标和费用。W&B 记录关联本地实验运行与模型状态；完整本地产物和费用总账继续承担审计与恢复职责，具体约定见 D-30。

## User Stories

1. As a research learner, I want a complete SFT-to-OPSD experiment, so that I can explain a training process I have actually run.
2. As an RA applicant, I want an explicit question about task success and failure recovery, so that I can discuss evidence beyond reproducing a method name.
3. As the project owner, I want one small Harness that I can understand, so that I can maintain the project after the initial implementation.
4. As a researcher, I want the same Harness during training and evaluation, so that changes in task success are not caused by changing the interaction rules.
5. As an agent, I want a natural-language instruction and a Task Workspace, so that I can write or repair an implementation in that workspace.
6. As an agent, I want to execute bash Actions and receive real Observations, so that I can use test failures to guide later Actions.
7. As a researcher, I want an Episode to have an explicit submission or termination outcome, so that incomplete attempts are distinguishable from successful submissions.
8. As a researcher, I want Interaction Steps and Repair Attempts recorded separately, so that several shell commands are not misreported as several independent repairs.
9. As a data curator, I want stable Task identities across variants and Trajectories, so that related examples cannot silently cross dataset splits.
10. As a data curator, I want separate Training, Development and Held-out Test Splits, so that model selection does not consume final evaluation evidence.
11. As an evaluator, I want Feedback Tests separated from Grading Tests, so that observing or changing feedback material cannot determine the final grade.
12. As a researcher, I want Reference Implementations verified before use, so that teacher supervision is based on valid task behavior.
13. As a researcher, I want SFT to start from an instruction-capable model, so that the limited experiment focuses on the Harness and the task.
14. As a data curator, I want reference-derived Demonstrations executed and their origin recorded, so that constructed examples are not presented as genuine exploratory trajectories.
15. As a researcher, I want optional Demonstration Generators to contribute verified repair examples, so that existing large-model APIs can help without being mistaken for token-scoring Teachers.
16. As a researcher, I want SFT to supervise model-generated content while preserving Observations as context, so that the model learns the intended interaction behavior.
17. As a researcher, I want current Students to produce the OPSD Trajectories, so that the teacher supervises states the learner actually visits.
18. As a researcher, I want the Frozen Teacher to load the selected SFT Model Snapshot, so that it does not accidentally use the original instruction model or later student weights.
19. As a researcher, I want Privileged Information visible only to the Teacher during OPSD, so that the Student learns under its normal information constraints.
20. As a researcher, I want Teacher Scoring on the original student token sequence, so that a separately generated teacher answer cannot replace the on-policy signal.
21. As a researcher, I want token identities, policy provenance and Supervision Masks preserved, so that the update can be audited against the original Trajectory.
22. As a researcher, I want invalid probability alignment to prevent an update, so that corrupted supervision cannot silently consume training budget.
23. As a researcher, I want subsequent rollouts to use the updated Student while the Teacher remains frozen, so that the learning process stays on-policy.
24. As an experiment operator, I want both sampling snapshots and resumable Training States identified, so that evaluation, teacher loading and recovery use the right saved artifact.
25. As an evaluator, I want the original instruction, SFT and OPSD stages measured under the same protocol, so that the report shows both the training increment and the overall behavior.
26. As an evaluator, I want final test tasks excluded from checkpoint selection, so that the final comparison remains independent.
27. As a researcher, I want real execution pass rates distinguished from loss and KL diagnostics, so that optimization progress is not mistaken for capability improvement.
28. As a researcher, I want first-implementation and failure-recovery evidence interpreted with their denominators, so that changes in the failed-task population do not create a misleading conclusion.
29. As an experiment operator, I want a small bounded Pilot before the main run, so that I can check API behavior, billing, supervision and parameter updates early.
30. As the budget owner, I want student sampling, teacher scoring, training, data generation, evaluation and retries included in Experiment Spend, so that the ¥200 limit applies to the whole project.
31. As the budget owner, I want pending remote requests considered before new work is submitted, so that asynchronous execution cannot silently exceed the remaining budget.
32. As an experiment operator, I want runtime failures and billed work recorded, so that failed calls do not disappear from either the report or the budget.
33. As a maintainer, I want each Experiment Run to own its configuration and artifacts, so that intermediate products do not accumulate among the core source files.
34. As a researcher, I want commands, dataset versions and snapshot identities associated with results, so that a later reader can reproduce or audit the experiment.
35. As a maintainer, I want behavioral tests through the high-level experiment seam, with the CLI as the only second seam, so that refactoring the internal implementation does not require rewriting a suite of private-helper tests.
36. As a maintainer, I want controllable model-call Adapters for ordinary tests, so that local checks can exercise real task execution without spending API credit.
37. As an experiment operator, I want model-generated bash isolated from credentials, reference answers and grading material, so that task execution does not compromise the host or the evaluation.
38. As a researcher, I want a real PyTRIO smoke run as well as local tests, so that a passing test double is not mistaken for a verified remote training path.
39. As the project owner, I want confirmed decisions, proposed details and unverified hypotheses kept distinct, so that later agents preserve intent without inventing evidence.
40. As an RA applicant, I want checkpoints, results and concrete Trajectories in a short report, so that I can explain both what worked and what the experiment does not establish.
41. As the project owner, I want the Harness independent of any Task Set, so that moving beyond MBPP only changes task data, not the interaction rules.
42. As the project owner, I want a CLI that shows an Episode step by step while it runs and replays saved runs, so that I can see how the agent actually behaves.
43. As a researcher, I want to plug in different model backends through configuration, each required to return token IDs and sampled-token logprobs or fail loudly, so that every backend preserves trajectory fidelity.
44. As an agent, I want the system prompt to describe only the action format, the shell rules and how to submit, so that I can work on any Task Set without instructions tied to one benchmark.
45. As an agent, I want the task text delivered as my first user message, so that the same Harness framing applies whatever the Task asks.
46. As an agent, I want to provide exactly one bash block with an optional action explanation, so that I can act without a required THOUGHT prefix or duplicate reasoning.
47. As an agent, I want to be told that every step runs in a fresh shell, so that I include the directory changes or variables a command needs instead of assuming they persist.
48. As an agent, I want to submit by printing an agreed marker as the first output line of a successful command, so that submission needs no special syntax and works for any kind of Task.
49. As an agent, I want each Observation to show the return code and the combined output, so that I can read errors in the order they happened.
50. As an agent, I want long output shown as its head and tail with the number of omitted characters and a hint, so that I still see the key parts and know to narrow the command.
51. As an agent, I want a timeout reported with its own message, so that I can tell a slow command from a failed one.
52. As an agent, I want a format error explained without anything being executed, so that I can correct my reply before the Episode ends.
53. As an agent, I want common tools such as awk, sed, grep, git and pytest, plus a writable home directory, so that ordinary coding workflows work inside the sandbox.
54. As an agent, I want a Task Workspace that can contain several modules and existing buggy code, so that I can repair programs, not only write one new file.
55. As an agent in thinking mode, I want only the text after my reasoning to be parsed as the Action, so that commands I merely considered while thinking are never executed.
56. As the project owner, I want a Task to consist of an instruction, initial files and a correctness criterion, so that a new Task Set is only a new data conversion.
57. As the project owner, I want MBPP-specific formats confined to the MBPP conversion, so that replacing or extending MBPP never touches the Harness.
58. As the project owner, I want a multi-file Task with a pre-planted bug and a hidden test command to pass end to end, so that the decoupling is proven by a test rather than asserted.
59. As the project owner, I want the Harness runnable on any directory with a plain instruction and no Task, so that I can use and inspect it as an agent in its own right.
60. As the project owner, I want the Task and Harness interfaces not to preclude per-Task container environments, so that harder benchmarks later need a new environment rather than a new protocol.
61. As a researcher, I want protocol, execution environment and Interaction Budget kept in one versioned configuration with a fingerprint for each, so that I can tell exactly which Experimental Conditions changed between two runs.
62. As a researcher, I want two runs treated as comparable only when all three fingerprints match, so that "same Harness, same Interaction Budget" is checked rather than assumed.
63. As a researcher, I want changing the model backend or its sampling settings to leave the protocol fingerprint unchanged, so that debugging on a development model is not mistaken for a protocol change.
64. As a researcher, I want every model-generated segment labelled as sampled or replayed, so that only genuinely sampled tokens can become on-policy training signal.
65. As a data curator, I want Demonstrations produced by replaying pre-written replies through the same Harness path as student rollouts, so that SFT inputs share the exact template, whitespace and Observation rendering the Student will see.
66. As a data curator, I want repair examples from a Demonstration Generator encoded with the student tokenizer and marked as replayed, so that they can join SFT without being mistaken for student rollouts.
67. As a researcher, I want Candidate Implementations, Verifications and Repair Attempts derived afterwards from the recorded Trajectory and the Task's declared implementation paths and feedback check, so that changing an analysis definition never requires a new protocol version.
68. As a researcher, I want the Task Workspace state recorded after every step, so that analyses and the CLI can show exactly what each Action changed.
69. As a researcher, I want thinking mode switchable in configuration with raw thinking tokens kept in the context, so that thinking can be piloted without breaking token-level fidelity.
70. As a researcher, I want running out of context to end an Episode with its own outcome, so that context exhaustion is not confused with a model error or a budget limit.
71. As an evaluator, I want grading to run from the host against a copy of the submitted Task Workspace in a fresh sandbox, so that nothing the agent left behind can influence the grade.
72. As an evaluator, I want hidden grading files copied in only at grading time, so that they never appear in the Task Workspace, the Observations or any log the Student can reach.
73. As an evaluator, I want function-call grading compared on the host, so that MBPP-style Tasks keep a tamper-resistant grade with per-case results.
74. As an evaluator, I want the trust limits of test-command grading documented, so that nobody reads its exit code as stronger evidence than it is.
75. As an evaluator, I want an Episode that ends without a Submission counted as a failure while its final Task Workspace is kept, so that the denominator stays honest and failures can still be diagnosed.
76. As an experiment operator, I want the live view to show budget usage, any action explanation, the command and the Observation exactly as the model saw it, so that I can follow the agent step by step even when no explanation is provided.
77. As an experiment operator, I want thinking collapsed by default in the live view, so that long reasoning does not bury the Actions.
78. As an experiment operator, I want the live view to be watch-only, so that what I see is a pure policy rollout with no human edits in the context.
79. As an experiment operator, I want Ctrl-C to stop an Episode with an interrupted outcome and a complete record, so that stopping a run never loses or corrupts data.
80. As an experiment operator, I want a saved run replayed with the same renderer, so that I can review past Episodes exactly as they looked live.
81. As a developer, I want a free self-hosted inference backend on the local A100, so that debugging prompts, parsing and the CLI does not consume Experiment Spend.
82. As a researcher, I want runs on the development backend never reported as experiment results, so that a different model and serving stack cannot contaminate the evidence.
83. As a maintainer, I want each backend chosen by a class name or import path in the configuration, so that adding a backend needs no change to the Harness.
84. As a maintainer, I want every backend to pass a small capability probe before its first use, so that a backend that cannot return token IDs and logprobs is rejected before an Episode starts.
85. As a maintainer, I want the chat encoding bound to a recorded tokenizer identity and checked against the official chat template, so that the prompts the model sees match the format it was trained on.
86. As a maintainer, I want transient backend failures retried a bounded number of times and then recorded as infrastructure errors, so that service problems are neither hidden nor blamed on the model.
87. As an experiment operator, I want the new /etc, home directory and tool environment added without weakening isolation, so that usability never reopens access to credentials or grading material.
88. As an operator on a shared cluster, I want the development inference service reachable only from my own session, so that no unauthenticated model endpoint is exposed on the cluster network.
89. As a maintainer, I want Experimental Conditions kept in configuration files rather than code constants, so that changing a prompt or a limit is a reviewed, fingerprinted change instead of a code edit.
90. As an experiment operator, I want SFT, OPSD and their evaluations monitored in W&B, so that learning diagnostics, execution results and spend can be compared using traceable run and snapshot identities.
91. As an experiment operator, I want monitoring to keep local records during a network outage and sync them afterwards, so that interrupted uploads do not lose evidence or reset training progress.

## Implementation Decisions

### 已确认决定

- **D-01 — 范围与约束。** 首版以入门和可信实验为目标，四天目标、一周上限，每天约 4 小时；新增 API 实验支出不得超过 ¥200。不要求新算法、论文级结论或先做完整消融。A1–A7、A13、A18 已确认这些范围。2026-09-29：Harness v2（ADR-0004）扩大了范围，时间目标与上限各顺延两天，自 2026-09-28 起算，目标 2026-10-03、上限 2026-10-06；现金上限不变。
- **D-02 — 单一生产训练后端。** 使用 PyTRIO 上的 LoRA，首选 Qwen/Qwen3.5-4B 现成指令模型；本地承担数据、Harness、工具执行和实验组织，并在任务 21 起支持经身份、编码及推理条件验证的示范采样和评测。现有一张 A100 40GB 不转化为另一套生产训练后端，也不默认租卡；遵循 ADR-0001 的 2026-10-03 修订。任务 21 先导出任务 07 已有 sampler 并验证本地加载，再考虑新增训练；本轮新增付费请求受项目总账及累计 ¥5 子额度约束，实际可用资金不足时遵守更小的额度。
- **D-03 — 自写、共用、与任务无关的 Harness。** 参考 mini-swe-agent 2.4.6 的简单交互思路，重新实现一个小而通用的 bash 编程 agent。它只接收指令、任务工作区和模型，不包含任何任务集的格式；任务数据、评分和候选实现分析属于其上的实验层。Harness 与其行动规则同时服务 SFT 示范、OPSD rollout 和评测；遵循 ADR-0002 与 ADR-0004。
- **D-04 — 任务与数据。** 首版使用 MBPP 作为任务起点；同一题及其改写、错误实现、示范和 rollout 继承同一 split。开发集用于方法选择与选择模型快照，最终测试集留到方案固定后。使用 MBPP+ 时按原始题号恢复划分，不能把其整体名为 test 的发布形式误认为独立于 MBPP 训练题。
- **D-05 — 反馈与评分分离。** 学生可接收真实反馈测试结果；参考实现、增强评分测试及其期望输出由可信数据／评分流程掌握，不放入学生工作区或观察。最终评分针对学生提交的实现重新执行评分判据，不信任学生自行汇报的通过率。
- **D-06 — SFT 目标与验收。** 从现成指令模型出发，学习可靠使用 Harness 并保留原有编程能力。分别检查协议遵循、真实交互与提交可靠性，以及独立的编程表现；至少在一种需要真实环境信息或执行反馈的场景中取得可复核的任务收益。格式改善、更多交互或 loss 下降不能单独证明目标达成；确认存在编程能力回退的候选不合格，明显回退时停止继续更新并诊断，证据不足时不自动判为通过。没有合格候选时可以记录“本轮没有合格 SFT 快照”并暂缓 OPSD，不能仅在 SFT 候选中强行选优。示范须真实执行、保留来源与观察，不把构造或回放冒充模型探索；具体数据、训练和评测方案在实施任务中记录。
- **D-07 — 冻结 SFT 教师的 OPSD。** 学生和教师均从选定 SFT 快照出发；学生接受后续更新，教师保留固定参数，并在教师上下文中看到训练题参考实现。OPSD 必须使用学生实际采样和执行的轨迹，遵循 ADR-0003。
- **D-08 — 训练目标口径。** 采用 PyTRIO OPSD 示例的 sampled-token reverse KL 口径：对每个受监督的学生 token，以教师 logprob 减去 rollout 时的学生 logprob，再乘正的蒸馏系数，形成固定的逐 token advantage，通过 importance_sampling 更新学生。教师更认可该 token 时信号为正；教师参数不参与更新。它不是完整词表 JSD，也不默认叠加 GRPO 或终局奖励；任务评分用于独立评测，不能混称为教师概率监督。
- **D-09 — 信息权限分阶段说明。** SFT 可以学习由训练参考程序构造的示范；OPSD 中参考实现是教师的特权信息，不进入该阶段学生的任务输入。学生评测始终使用正常任务输入，不向任何被比较模型提供参考实现。
- **D-10 — 验收与研究切口。** 记录原始指令模型、SFT 和 OPSD 三阶段成绩。主验收是同一最终测试集、同一 Harness 和相同交互预算下，OPSD 最终通过率高于 SFT；首次实现和失败恢复分析用于解释收益。A7 已将 SFT 作为主要比较起点，不另加“必须打败其他算法”的要求。
- **D-11 — 费用与分阶段投入。** OPSD 小试支出上限 ¥10，包含在项目 ¥200 总上限内；常规支出计划控制在 ¥150 内，保留 ¥50 余量。所有数据生成、学生采样、教师评分、训练、评测、失败调用和重跑都进入账本，不能只计算训练费用，也不能把观察的 loss mask 当成免计费依据。
- **D-12 — 单一路线与备选。** 优先验证 OPSD，再集中训练选中的路线；9B OPD 仅为小试不成立后的备选，不同时铺开两套完整实验，也不因切换备选增加总预算。备选启用要保留原因、教师身份和实验条件，不能混入 OPSD 主线结果。
- **D-13 — 可维护性与交付。** 核心逻辑与运行产物分离，每次实验集中保存配置、数据身份、轨迹、模型状态、指标和费用。交付可重跑代码、模型快照／adapter、结果表、典型轨迹及短报告；AI Coding 是开发方式，不作为单独的产品功能或验收指标。
- **D-14 — 测试 seam。** 用户已确认以“一次小型实验：执行 bash 任务 → 记录轨迹 → 构造一次训练更新 → 独立评分与费用记录”为主要测试 seam。日常测试用可控 Adapter 替换付费模型调用，真实 bash 执行、数据隔离和评分仍运行；真实 PyTRIO 验证沿用已约定的小试额度。2026-09-30：用户确认 Harness v2 保留 `run_experiment` 为主要 seam，另以 CLI 子进程为第二个 seam，见 Testing Decisions。
- **D-15 — Harness 与实验层两层结构。** Harness 只接收指令、任务工作区、模型和配置，按协议运行一个交互回合，返回轨迹、终止结果、最终工作区和可选的提交文本；它不知道 Task、评分器或任何任务集。实验层依据 Task 准备工作区，调用 Harness，在提交后对工作区做快照，在新沙箱中评分，并在事后做分析。公开入口只有实验入口 `run_experiment` 和供 CLI 使用的 Harness 入口 `run_episode`。Harness 不持有模块级状态，以后可以由实验层并发运行多个回合。（Q2）
- **D-16 — 行动格式。** 任务 19 将行动协议升级为 bash-block-v3，任务 16 交付的 bash-block-v2 保留为历史记录。行动正文给出恰好一个非空 bash 围栏块，块前的自然语言说明可选，不要求 `THOUGHT:` 前缀或非空说明。没有块、多个块、空块、块未闭合，或回复因长度上限被截断，都算格式错误：不执行，返回说明并计入步数；连续次数达到配置上限（初始为 3）时终止。系统提示只描述行动格式、shell 规则和提交方式，不含任何任务集的文件名或格式；任务文本来自 Task 的指令，作为第一条 user 消息。开启 thinking 时只解析最后一个思考结束标记之后的正文，思考内容不计入行动块数量。（Q3、Q13；2026-10-01 用户修订）
- **D-17 — 提交。** 某一步命令输出的第一行是 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 且返回码为 0 时，回合以提交结束，其余输出行记为提交文本。提交交付的是此刻的整个任务工作区，其中哪些内容接受评分由 Task 的评分器决定。没有提交就终止的回合不评分，计为任务失败，最终工作区保留供诊断；诊断性评分不计入通过率。（Q4 及默认项）
- **D-18 — 观察格式。** 观察由返回码和合并后的输出组成，stderr 与 stdout 合并并保持先后顺序。输出超过阈值时保留首尾各一段（初始各 2000 字符），注明省略的字符数，并提示模型缩小输出范围；命令不会因输出过长被终止。沙箱另设字节硬上限，防止输出失控，超出部分被丢弃，不算模型错误，也不结束回合。超时使用单独的文案，该步的全部进程都会被终止。（Q6）
- **D-19 — 执行环境。** 每一步都在新的 bwrap 沙箱中启动新的 bash，只有工作区跨步保留。当前目录、环境变量、后台进程、临时目录和 HOME 每步重置，系统提示会说明这一点。首版只有一个执行环境，包含：
  - 只读的系统程序与库；
  - 生成的最小 /etc（账户、hosts、名称解析与 git 配置），外加只读的 alternatives 链接；
  - 每步新建的可写 HOME；
  - 一个版本锁定、只读挂载、含 pytest 的 Python 环境。字节码与 pytest 缓存的写入被关闭，避免弄脏工作区快照。

  单条命令默认超时 30 秒，由实验层按任务集统一设置。隔离强度不降低：分离全部 namespace、清空环境变量、断网、移除 capabilities、施加资源限制。能力探针覆盖新增的挂载，并检查常用工具可用。2026-09-29 的临时探测已确认：在本节点的 bwrap 0.6.1 下，生成的 /etc 加 alternatives 链接能让 awk、git、whoami 正常工作，HOME 可写。（Q5、Q7 及默认项）
- **D-20 — 交互预算与终止结果。** 交互预算包括步数、生成 token 数、上下文 token 数、墙钟时间和连续格式错误次数。Harness 的终止结果互斥且可区分：提交、格式错误达到上限、预算耗尽（注明是哪一项）、上下文耗尽、模型错误、基础设施错误、中断。实验层另外区分无效提交和评分阶段的基础设施错误。（Q12、Q14 及默认项）
- **D-21 — 轨迹与运行记录。** 每次生成保留原始 token、逐 token logprob、上下文位置、来源（采样或回放）、策略身份与采样条件。上下文只追加；观察只在对应行动执行之后追加；已生成的 token 不会被 decode 后重新编码。每一步记录工作区清单（路径到内容哈希），发生变化的文件按内容寻址保存，供分析层和 CLI 显示每步的改动。运行记录包含合并后的完整配置、三个指纹、编码与 tokenizer 身份、后端身份、隔离探针结果和执行环境描述。（默认项）
- **D-22 — 模型接口。** 从 mini-swe-agent 借用「按配置选择模型类，并把配置写进轨迹」的做法，但不采用它「消息进、文本出」的接口。
  - **编码**：按模型家族实现，绑定 tokenizer 身份，负责构造首个 prompt、每步的 continuation 和 decode，并掌握停止 token 与思考标记。首个实现是 Qwen ChatML，首个 prompt 必须与官方聊天模板一致。
  - **采样契约**：模型接收 token 形式的上下文和采样参数，返回生成的 token、逐 token logprob、停止原因和请求标识。采样后端必须返回 token ID 和每个采样 token 的 logprob。用户所说的 logits 按此理解，不要求 top-k 或完整词表分布。缺失、长度不一致或数值非有限时显式报错，不补值。
  - **能力探针与重试**：每个后端首次使用前先做一次极小的能力探针，不通过就不开始回合。瞬时错误由后端有限次重试，仍失败时记为基础设施错误。
  - **后端选择**：由配置中的类名或导入路径决定，其余字段作为构造参数。

  （Q12）
- **D-23 — 模型后端范围。** 首版实现三个来源：PyTRIO（唯一生产训练后端及远端采样／评分，任务 04）、自建 vLLM（任务 20 的开发模式；任务 21 起经验证的本地示范采样与评测）和回放模型。vLLM 运行在 AGENTS.md 约定的 A100 40GB 节点，只监听本机，通过 SSH 隧道访问或启用 API key。实验模式必须使用原始 Qwen/Qwen3.5-4B 或明确导出的 SFT 快照，绑定权重、tokenizer、模板、checkpoint 与服务条件，并取得实际加载与 token／逐 token logprob 证据；失败不能静默回退。比较条件包含实际推理版本、精度、上下文、缓存和并行等设定，条件变化需要匹配的基线。云端未公开的基模 revision 保持未知，不由名称或 tokenizer commit 推断权重逐位一致。任务 20 的 Qwen3-4B-Instruct-2507 与历史运行继续为开发证据，不能改标签追认为新基线。HF transformers 本地推理和只返回文本的聊天 API 不作为采样后端。（Q11、2026-09-29 资源说明及任务 21 的用户范围调整）
- **D-24 — 回放模型与样本来源。** 回放模型用真实编码把预先写好的回复文本转成 token 交给 Harness；之后的解析、执行、观察渲染和上下文追加，与学生 rollout 走同一条代码路径。它取代任务 02 的受控行动路径，用于测试、CLI 演示和 SFT 示范（任务 06）。回放 token 没有采样 logprob，每步都标注为回放来源，OPSD 更新拒绝回放来源的步骤。示范生成模型给出的文本也经回放模型、用学生 tokenizer 编码并标注来源。（Q9）
- **D-25 — thinking 模式。** 配置中的 thinking 开关支持开启和关闭两种模式，首次小试默认关闭。两种模式均不要求 `THOUGHT:` 或行动说明。关闭时，行动正文可以直接给出 bash 块；开启时，只解析思考结束标记之后的正文，不要求再生成思考说明。原始思考 token 留在上下文中。官方多轮模板会在后续轮次丢弃早先的思考，我们不这样做，这是保持严格 on-policy 的代价。思考中出现的 bash 块不会执行；思考未结束就触及长度上限时按格式错误处理。思考边界由真实 tokenizer 和生成提示确定，不能假设采样文本一定包含思考起始标记。Qwen3.5 模板的细节在实施时用真实 tokenizer 校验。（Q13；2026-10-01 用户修订）
- **D-26 — 配置与指纹。** 实验条件写在版本化的 YAML 配置中，不写成代码常量。
  - **结构**：配置分四块：protocol（所有 agent 可见的文本与规则）、environment（挂载、工具环境、环境变量、单条命令超时与资源限制）、budget，以及 model（后端类、采样参数、编码与 thinking 开关）。可以叠加多个文件，也可以用 `key=value` 覆盖单项，合并后的完整配置写入运行记录。模板用 Jinja2 渲染，遇到未定义变量直接报错。
  - **指纹**：protocol、environment、budget 各自计算指纹。D-10 的「同一 Harness、同一交互预算」要求三个指纹都相同；更换模型后端或采样参数不改变协议指纹。协议指纹由测试固定，完整协议文本在冻结指纹前交用户审阅。任务 19 取消强制 `THOUGHT:` 时使用新的 bash-block-v3 版本与指纹，同步更新示例、测试和 README；已有示范如依赖旧协议则重新生成，不把新旧协议的运行当作相同实验条件。
  - **依赖**：新增依赖（YAML 解析、Jinja2、rich 与 tokenizer 库）锁定精确版本。

  （Q10、Q15 及默认项）
- **D-27 — Task、提交快照与评分器。** Task 由稳定身份、指令、初始文件、执行环境标识和只在宿主持有的评分器组成，还可以声明反馈命令和实现路径，供分析层使用。提交快照沿用 v1 的限制并推广到整个工作区：只接受普通文件和目录，不跟随符号链接，拒绝特殊文件，受大小和文件数限制；不满足时结果为无效提交。首版实现两种评分器：
  - **函数调用评分**：给 MBPP 类任务使用。模块路径和函数名由 Task 指定，每个用例在新的沙箱进程中调用，JSON 结果在宿主比较，保留逐用例结果。
  - **测试命令评分**：只在评分时把隐藏文件复制进快照副本，退出码为 0 即通过。它的信任边界要写进 README：学生代码与测试在同一进程运行，恶意提交可以伪造通过。这个风险可以接受，因为 OPSD 的训练信号不使用评分结果（D-08），轨迹也都可以审计。

  其他 benchmark 的适配器等真正需要时再写。（Q16）
- **D-28 — 分析层。** 候选实现、验证和修复尝试都在事后计算，依据是轨迹、每步的工作区清单，以及 Task 声明的实现路径和反馈命令。它们不属于 Harness 规则，修改定义不需要升级协议。语义沿用 v1：
  - 候选按内容去重，回退到旧内容时沿用原编号；
  - 只有反馈材料未被修改、整条命令就是约定的反馈命令时，这一步才算验证；
  - 修复尝试从一次失败的验证开始，到验证一个不同的候选时结束。

  首次候选的精确定义仍按 P-04 待定。（Q8）
- **D-29 — CLI。** 提供三个命令：
  - `bashlet agent`：用指令和工作目录直接运行 Harness，不评分。工作目录先复制进运行目录，agent 只改副本。
  - `bashlet task`：经实验层运行一项任务，结束时显示评分。
  - `bashlet show`：用同一渲染器回放已保存的运行。

  实时视图用 rich 滚动输出。每步显示预算用量、可选的行动说明、高亮后的命令、模型实际看到的观察和格式错误；有说明时显示，缺少说明时仍正常显示步骤，不依赖 `THOUGHT:` 标签。原生 thinking 默认折叠，由任务 19 接入；结束时显示终止结果和评分。视图只能看、不能干预，没有 confirm 或 human 模式。Ctrl-C 以「中断」结束回合，并完整保存记录。批量运行的进度表随批量运行器另做。（Q14；2026-10-01 用户修订）

- **D-30 — W&B 实验监控。** 使用 `wandb` 统一监控 SFT、OPSD 小试、OPSD 主训练及配套基线、开发集和最终评测；project 默认 `bashlet`，entity 不硬编码，使用账号默认空间并记录实际解析结果。只上传指标、脱敏配置和 checkpoint 引用；断网时继续本地记录，恢复后同步。任务 07 接入并验证真实 W&B 链路，08 验证 OPSD 上报行为，09–11 复用；已交付任务保留历史记录。（2026-10-02 用户 Q1–Q4）

### W&B 监控约定

以下是 D-30 的实施与验收要求，由任务 07–11 共用；具体 SDK 版本和真实上传能力仍待实施验证。

- **接入位置与配置。** 在实验／训练组织层接入 W&B，复用既有评测汇总、轨迹分析和费用总账；Harness 不承担监控服务的认证或联网。任务 07 用 `uv` 增加并锁定 `wandb` 依赖。监控设置由顶层 `configs/` 中的实验配置或环境变量管理，与 D-26 的四块 Harness 配置分开；project、entity、在线／离线模式和日志频率不改变协议、环境、预算指纹或评测比较条件。普通行为测试使用监控替身，无需 W&B 密钥或联网。
- **凭证与启动。** 认证字段明确为 `WANDB_API_KEY`；接入时核对用户已配置的 `.env` 字段名，环境变量优先，缺失时使用现有 `python-dotenv` 依赖读取指定字段。不能假定 W&B 会自动加载 `.env`，也不能把其他服务的密钥用作 W&B 凭证。启动前检查密钥存在、配置和本地记录目录可用；在线时验证目标空间权限，缺失密钥、认证失败或配置错误须在付费调用前明确报错。若仅因网络不可达而改用离线记录，须显式记录该状态和仍待完成的在线验证。
- **运行身份与分组。** 每次训练或独立评测有对应的 W&B run，与本地 Experiment Run 身份相互关联；持久化 project、实际 entity、W&B run ID、可用时的 URL 和本地目录。同组比较用共同 group，阶段用 job type 区分（如 `baseline-eval`、`sft`、`dev-eval`、`opsd-pilot`、`opsd`、`held-out-eval`），开发评测关联其训练 run 和受评快照。脱敏配置记录代码版本、数据／split 身份、三个指纹、已有评测比较条件哈希、模型／tokenizer 身份、训练与采样参数；未知身份保持未知。已有基线可从原始评测记录导入，标明历史导入及来源身份，不冒充当时已有实时监控，也不为补监控重复付费运行。
- **上传范围。** 仅通过允许字段上传指标、脱敏配置和不含凭证的 checkpoint 标识／引用，分别注明 sampler 快照与完整训练状态。完整轨迹、逐 token 数据、工作区、参考实现、隐藏评分材料和模型权重保留在本地或原远端存储；不整目录上传 `runs/`、`.env` 或完整配置对象。关闭自动控制台内容与源码采集，避免终端轨迹或代码通过默认上传路径越过上述边界；凭证不进入配置快照、日志或 agent 沙箱。
- **记录频率与步数。** 默认每次确认完成的优化器更新记录一次训练指标，rollout／教师评分按批次汇总，开发与最终评测在进度变化和结束时记录结果；频率可配置并随 run 保存。明确区分优化器 step、交互步、episode 数与监控日志步，不能把尚在途或被拒绝的更新计为成功 step。为图表保留显式的优化器 step／评测进度字段，避免同一训练 step 下的不同事件相互覆盖。
- **断网与恢复。** 默认在线；W&B 网络故障不阻止训练继续，本地实验记录与 W&B 缓存落在 `runs/` 或 `outputs/` 下该运行目录，显示待同步状态。网络恢复后重试同步；仍未上传的缓存保留并给出 `uv run --env-file .env wandb sync <该次 W&B 缓存目录>` 的补同步方法，不能因 SDK 等待上传而无限阻塞退出。正常结束、异常和中断均保存状态及同步情况；恢复时校验 run 身份、已完成优化器 step 与已记录事件，不能重置进度、覆盖或重复统计。W&B 续记与模型／优化器的 Training State 恢复分别验证，离线续记／补同步语义以锁定 SDK 的验收为准。

各阶段至少记录以下指标，命名空间区分训练、开发评测和最终测试；分母、聚合口径及单位随运行保存：

| 范围 | 最低监控内容与口径 |
| --- | --- |
| SFT | 已完成的优化器 step、训练 loss、学习率、样本数和受监督 token 数；loss 注明其归一化／聚合方式。开发集选优结果关联对应快照，不能以 loss 代替任务成功 |
| OPSD | 更新 loss、蒸馏系数、受监督 token 数、sampled-token reverse KL（在受监督学生 token 上对 `rollout logprob - teacher logprob` 求均值）、有效／拒绝 batch 数及原因、学生与冻结教师快照身份；该采样统计不是完整词表 KL／JSD，也不强制截成非负值 |
| 基线、开发集与最终评测 | 预定／完成任务数、通过数、固定分母的最终通过率、首次实现结果、失败恢复分子与分母、终止原因及基础设施缺口；沿用既有分析定义，未完成评测明确标记 incomplete，空恢复分母记为不适用 |
| 用量与耗时 | 分阶段的输入／生成／教师评分／训练 token、rollout／评分／更新／评测耗时与可核对的吞吐；远程 GPU 指标仅在 PyTRIO 实际提供时记录，本地主机指标不得冒充远端训练资源指标 |
| 费用与预算 | 从同一费用总账提取运行及项目累计保守占用、在途预留、可用额度与已核实实际费用；区分上界和实扣，未知实扣保留未知／缺失标记，不填成 0。失败、重试及被丢弃工作仍计量，预算守卫不依赖 W&B 是否在线 |
| 结果与保存 | 运行完成／失败／中断状态、同步状态、开发集选优依据、选定快照及训练状态引用；报告和本地产物能反向定位对应 W&B run |

### 已确认方案的必要接口约束

以下是保持上述训练与评测语义所必需的约束，不是声称某个 SDK 或沙箱已经验证通过。

- **轨迹保真。** 保留学生实际生成的 token IDs、采样 logprob、行动位置、观察和学生策略版本。每个后续行动以前面的真实观察为条件；不能用 decode 后重新 tokenize 的近似历史静默替换已采样 token，也不能把未来工具结果插到先前行动的上下文中。任何采样后端都必须返回 token IDs 和逐 token 的采样 logprob，缺失即报错；回放等非采样来源须逐步标注，不进入 on-policy 更新。
- **监督范围。** SFT 的监督掩码和 OPSD 的训练目标只覆盖相应模型生成内容。用户输入、工具观察和其他纯上下文不承担学生预测目标；不同 teacher prompt 造成的位置偏移必须显式对齐。
- **教师对齐。** 教师对学生原样 token 序列评分，学生 token、rollout logprob 与对应教师 logprob 一一匹配；采样概率的定义须与训练接口要求一致。缺失或非有限数值、长度错位、不可兼容的 token 编码或损坏的前缀应阻止更新，而不是补零后继续假装监督有效。
- **策略版本。** 同一个用于更新的 rollout batch 具有明确的学生快照身份；远程更新确认完成后，后续采样使用该更新所对应的学生参数，不能把请求被接收当作更新完成。教师的参数身份保持固定，禁止误载未进行任务 SFT 的原始模型或自动切换到最新学生。
- **保存语义。** 供教师／评测使用的模型快照与恢复优化器进度所需的训练状态分别记录。仅保存 sampler 权重不等于已经能够恢复训练；从 SFT 进入 OPSD 必须明确实际加载了哪一个快照。
- **预算守卫。** 提交付费工作前考虑已发生支出和未结算请求的额度；失败请求和被丢弃 rollout 也需要计量。没有足够余额或无法约束请求成本时停止继续提交，不以异步排队或更换算法绕过总上限。
- **执行隔离。** 模型生成的 bash 不得继承服务凭证或获得宿主私有目录、参考实现及评分材料的访问权。独立工作目录和超时本身不构成隔离；执行器必须通过能力验证后才运行模型生成的命令。
- **可审计评分。** 所有阶段使用预先固定的任务清单、反馈规则、采样条件和交互预算。模型错误、协议错误和预算耗尽与服务／基础设施错误分开记录；不得静默丢弃失败样本来改变分母。

### 待确认建议：不冒充既定实现

以下事项尚未被访谈固定，不得把这些建议写成已获用户确认的结论。实施者可在已确认范围内选择、验证并记录例行配置，无需逐项重新访谈；若需改变主路线、资源上限或验收口径，则单独提出具体变更。

- **P-01 — Module 划分。** 建议围绕任务数据与评分、交互回合、学习更新、实验组织四项职责形成少量深 Module，隐藏各自的复杂实现。2026-09-29 起有两个公开入口：实验入口 `run_experiment`（主要测试 seam）和 Harness 入口 `run_episode`（供 CLI 使用）。模型后端由配置选择，只有一个执行后端，没有插件系统；其余内部拆分尚未定。
- **P-02 — 行动编码与终止协议。** 已于 2026-09-29 确定，并于 2026-10-01 取消强制 `THOUGHT:`，见 D-16–D-18 与 ADR-0004 修订。任务 19 实施 bash-block-v3，任务 16 的 v2 交付记录保留。具体文本在版本化配置中维护，生成 SFT 示范前冻结指纹，不同时兼容多套协议。
- **P-03 — 数值配置。** MBPP full／sanitized 的具体版本、任务子集规模、示范数量、训练步数、LoRA 配置、学习率、KL 系数、thinking 模式、采样参数、上下文和命令限制均待小试确定。已有的初始值为：单条命令超时 30 秒，观察首尾各保留 2000 字符，连续格式错误上限 3 次，首次小试关闭 thinking。两三次修复机会不能直接换算为三个模型调用；所有最终配置在正式测试前冻结。
- **P-04 — 提升解释。** 最低提升幅度、置信区间方法和首次候选实现的精确定义尚未约定；至少报告任务数、通过数和差值，不能凭单次 loss 下降宣称提升。建议在固定失败起点上补充少量配对恢复分析，避免直接比较两个模型各自不同的失败集合；不将额外训练消融设为首版硬要求。
- **P-05 — 执行器与记录工具。** 执行器已由任务 01 确定为 Bubblewrap，v2 补充 /etc、HOME 和工具环境（D-19）。外部实验监控已于 2026-10-02 确认为 W&B（D-30），不再属于待选工具；本地完整运行记录和费用总账继续保留，CLI 展示沿用 D-29。W&B 的依赖锁定与接入验收归入任务 07–11。

## Testing Decisions

### 已确认的主要 seam

通过调用者也会使用的小型实验 Interface 进行行为测试：输入明确任务、模型状态、运行模式与预算，观察真实执行结果、轨迹、提交给训练后端的请求、评分和运行记录。测试不依赖私有函数名、Module 数量或源文件布局。

付费模型计算是外部依赖，通过 Adapter 提供可控采样、教师评分、更新与费用响应。这个替身是测试工具，不是另一套生产训练后端；它能证明请求和行为契约，却不能证明真实模型已经学到能力。真实 SDK、权重保存／加载、一次参数更新和费用核对仍须通过 PyTRIO 小试完成。

规格最初编写时，项目中没有可复用的 Harness、训练实现或测试。当时的 prior art 是已核对的 PyTRIO SFT、OPSD 和多轮代码执行示例，以及 mini-swe-agent 的交互语义。这些是参考资料，不是项目中已通过的测试，也不应整套复制为新框架。任务 01、02 之后，项目内已有可沿用的 prior art，见「Harness v2 的测试 seam」。

### 行为场景

1. **生成与修复闭环。** Agent 写出一个失败实现，真实执行反馈测试，收到实际错误后修复；评分来自修复后提交文件，轨迹顺序与工作区变化可核对。
2. **正确性与反馈独立。** 一份实现通过反馈测试却未通过评分测试时，结果为任务失败；修改反馈测试或打印“全部通过”不能改变可信评分结果。不同但同样正确的实现不因与参考程序文本不同而被判错。
3. **数据家族隔离。** 同一原始题的重写、错误版本或新轨迹不能进入另一 split；MBPP+ 中属于原始训练范围的题目不能混入最终测试。数据过滤条件不依赖某个模型是否答对。
4. **特权信息隔离。** 教师可获得训练参考实现，学生请求和任务工作区不可获得；测试三个模型阶段时均无参考实现输入。轨迹和日志中的信息权限也须可检查。
5. **token 与掩码。** 用短且可人工核对的轨迹，包含命令、真实观察和修复，验证送往外部训练 Adapter 的监督仅落在原始学生生成位置。覆盖 teacher prompt 较长、首尾 token、特殊 token、空观察及多轮偏移；不以调用了哪个内部 helper 作为通过标准。
6. **拒绝无效监督。** 教师返回缺失概率、错位长度或不匹配编码时，不提交更新；记录已消耗的调用、错误和轨迹身份，不能通过静默补值产生伪训练。
7. **快照角色。** 学生更新后，下一批 rollout 使用新策略；冻结教师身份不变。检查将 SFT 快照误替换为原始模型、将 sampler 快照当完整训练状态等错误会被发现。
8. **预算与异步工作。** 测试预算不足、在途费用尚未结算、失败调用仍计费和额度耗尽等情形；不能在累计支出之外另起一份“评测预算”突破总上限。对真实价格归类的验证属于小试，不由测试替身保证。
9. **执行限制。** 验证任务进程无法读取服务凭证、宿主私有内容和评分材料，并按配置处理超时、输出过长、无效命令与未提交退出。隔离不可用时应停止真实模型代码执行，不把提示词限制当作隔离。
10. **评测完整性。** 固定清单下所有题目的状态都被保留；模型失败与基础设施错误分列。基础设施错误使运行不完整时，报告缺口或在相同协议及预算内重试，不悄悄缩小分母。
11. **产物可追溯。** 一个实验运行可追溯到配置、数据划分、学生／教师快照、轨迹、费用和结果；中断、续训和失败不会覆盖另一轮结果或伪装成完整运行。
12. **真实后端小试。** 核对账户可用模型、实际 SDK 契约、SFT 状态衔接、教师评分对齐、一次已完成的更新与账单。小试成功只证明链路成立，能力改善还须独立评测。

### Harness v2 的测试 seam（2026-09-30 用户确认）

- **主 seam：实验入口 `run_experiment`。** 协议、观察渲染与截断、执行环境与隔离、提交与快照、两种评分器、分析层、预算与终止结果、配置指纹，以及 B 级任务的端到端行为，都通过回放模型或可控 Adapter 在这里测试。真实 bash、真实沙箱和真实评分照常运行。
- **第二 seam：CLI 子进程。** `bashlet agent` 用于证明 Harness 不需要 Task 也能运行；`bashlet task` 与 `bashlet show` 用于检查实时视图与回放内容一致。测试只断言可见内容和运行记录，不依赖终端宽度、颜色或布局细节。
- **外部依赖替身。** 替身包括回放模型、可控 Adapter，以及一个在本地模拟 token 级响应的假 vLLM HTTP 服务。后者用来覆盖后端的错误映射，包括缺少 token 或 logprob、长度错位、瞬时失败重试和上下文耗尽。编码的正确性以官方聊天模板的输出作为独立参照，不用编码器自己的输出核对自己。
- **不进测试套件的真实后端。** 真实 vLLM 在计算节点验收；任务 20 保留开发证据，任务 21 分别验证原始 Qwen3.5-4B 与已有 SFT 导出权重的真实回合、加载证据与独立评分。假服务在既有 seam 检查身份错误、缺少 token／logprob、adapter 未加载和比较条件变化。真实 PyTRIO 由任务 04 的小试验证；冒烟不证明能力提升。
- **不设 Harness 入口 seam。** 除上面两个 seam 外，不直接调用 Harness 入口编写测试。解耦由 CLI 的 `bashlet agent` 路径和 B 级端到端测试共同证明。

好的测试只观察调用者可见的行为，即返回结果、运行记录、工作区和终端输出，不依赖私有函数、模块数量或文件布局。期望值来自独立来源：手写的字面值、官方模板输出或实际执行结果，不用被测代码的算法重新推导。

Prior art：
- 任务 01、02 的行为测试走同一实验入口，覆盖隔离与泄漏、执行限制、提交快照、终止结果与预算审计、token 上下文保真、候选与修复尝试。
- 可控 Adapter 的字节级玩具编码含整词 token，能发现 decode 后重新编码造成的替换。
- 用计数启动器模拟执行器在中途失效。

这些测试迁移到 v2 后应全部通过：断言按 v2 的协议文本和终止结果名称更新，行为语义保持不变。任务 19 升级到 v3 时再次更新协议文本和指纹断言，原有执行、提交与观察的行为语义保持不变。

从 v2 引入并由后续任务扩展的行为场景（接续上一节编号）：
13. **B 级任务端到端。** 工作区有多个模块和一个预置 bug，反馈命令运行公开测试，评分器用隐藏的测试命令。回放的回复先定位 bug、再修复、验证、提交，评分通过；另一条轨迹只通过公开测试，评分仍然失败。隐藏测试不会出现在工作区、观察或学生可达的日志中。
14. **解耦。** 同一份协议配置分别服务 MBPP 类任务和 B 级任务，协议指纹相同。系统提示中不出现任何任务集的文件名。`bashlet agent` 在没有 Task 的目录上完成一个回合。
15. **提交标记。** 只有输出首行为约定标记且返回码为 0 时才提交。以下情况都不算提交：标记不在首行、返回码非零、标记只出现在可选行动说明里、标记只出现在思考内容里。
16. **观察截断与超时。** 超长输出只保留首尾，并注明省略的字符数，命令照常执行完，文件写入不受影响。超过硬上限的输出不会让回合结束。超时有独立文案，后台进程在该步之后不能继续修改工作区。
17. **无状态 shell。** 第一步里的 `cd` 和 `export` 不影响第二步。第一步写入 HOME 和临时目录的内容，第二步看不到。
18. **执行环境。** awk、sed、grep、git、pytest 在沙箱内可用，HOME 可写。能力探针覆盖新增的挂载。宿主凭证、宿主私有文件和评分材料仍然不可读，也连不到宿主网络。
19. **来源标注与拒绝。** 回放的每一步都标注为回放来源。采样后端缺少 token 或 logprob 时，回合以模型错误结束，不补值。能力探针不通过时，不开始回合。
20. **thinking 解析。** 只解析思考结束标记之后的正文，思考中的 bash 块不执行，也不计入行动块数量。覆盖思考起始标记已在 prompt 中、采样文本不再包含该起始标记的情况。思考未结束就触及长度上限时，按格式错误处理。原始思考 token 保留在上下文中。
21. **配置与指纹。** 同一份配置得到相同的指纹。修改 protocol、environment、budget 中的任一块，只改变该块的指纹；更换模型后端不改变协议指纹。未定义的模板变量直接报错。运行记录保存合并后的完整配置。
22. **分析层。** 在 v2 轨迹上，候选、验证和修复尝试的结果与 v1 同类测试一致，依据的是 Task 声明的实现路径和反馈命令。
23. **CLI。** 实时视图与 `bashlet show` 显示的步骤内容一致，包括可选行动说明、命令、观察和格式错误；没有说明时仍正常显示步骤，不依赖 `THOUGHT:` 标签。Ctrl-C 以中断结束，运行记录完整。
24. **编码。** Qwen ChatML 编码的首个 prompt 与官方聊天模板的输出逐 token 一致。continuation 只追加 token，不重编码已有前缀。thinking 开关按预期改变生成提示。
25. **v3 行动说明可选。** thinking 关闭时，只含一个非空 bash 块的正文可以执行；开启时，思考结束后的正文可以直接给出 bash 块。两种模式均不要求 `THOUGHT:`，带可选说明的正文也合法。没有块、多个块、空块、块未闭合或因长度上限被截断时仍不执行。新版本和指纹进入运行记录，旧协议运行不与新协议运行混作同一实验条件。

### W&B 监控验收（2026-10-02）

继续通过实验入口验证监控的可见行为，日常测试替换 W&B 外部服务：完成的更新才增加训练 step；无效监督和失败调用仍留下指标与计量；开发／最终评测保留既有分母和缺失值；监控配置不改变实验比较条件。检查上传字段不含凭证或原始轨迹，断网时本地指标继续增长，补同步与中断恢复不覆盖历史或重复计数。

任务 07 另做真实 W&B 最小上报与补同步验证，关联其真实 SFT 和开发评测证据；任务 08 的监控替身不作为真实上传成功的证明。后续 09–11 复用同一接入，并抽核云端指标与本地评测、训练进度及费用总账一致。W&B 页面可访问、曲线下降或同步成功本身都不证明模型能力提升。

### 完成判据

- **工程完成：** 真实任务交互、SFT、OPSD 更新、保存与评测链路可执行；关键行为测试与真实后端小试有记录，结果和费用可追溯，运行产物与源码分离。W&B 监控满足 D-30，真实上报、断网记录和补同步有验收证据，运行链接可追溯到本地产物。通用 Harness 满足以下条件：B 级端到端测试通过；原有行为测试迁移后全部通过；补充执行环境后隔离探针仍然全部通过；任务 19 的 v3 行动说明可选、thinking 分离与协议指纹验证通过；在 CLI 中能看着开发用真实模型跑完一个回合。
- **实验完成：** 在固定协议上取得原始指令、SFT、OPSD 的有效结果，保留足够的任务数、通过数、失败状态、费用和方法配置供审阅。若未完成完整评测，应明确标注，不用局部样本替代完整结果。
- **项目达标：** 有效独立评测中 OPSD 的最终通过率高于 SFT，且没有通过改变任务清单、增加推理预算、暴露参考答案或污染测试集造成表面提升。最小差值和统计解释仍按 P-04 处理，不声称未检验的显著性。
- **未达标处理：** 跑通代码、降低 loss 或得到阴性结果均不单独构成项目成功；若实际指标没有提升，明确记录未达标，同时遵守时间与现金上限。

## Out of Scope

- 原始预训练 Base 模型的指令冷启动、大模型预训练、全参数训练或新算法框架。
- 实现每题一个镜像的执行环境（SWE-bench、Terminal-Bench 类）；Harness 与 Task 接口不排斥它们，但首版不提供。联网检索、运行时包管理、浏览器。
- 首版同时维护本地 verl／TRL、Tinker 与 PyTRIO 多套生产训练栈，或直接沿用 FARO2 的整套依赖。
- 类 Claude Code／Codex 的通用 agent 产品：多工具、插件系统、网页 UI、人工确认或人工接管模式、多种执行后端。Harness 与任务无关，但只有 bash 一种工具和一个执行后端。
- 其他 Harness 形态：持久 shell 会话、模型原生工具调用、同时兼容多套协议、Textual 全屏界面、首版中的批量运行进度表。
- 其他模型后端：HF transformers 本地推理后端、以只返回文本的聊天 API 作为采样后端、把未通过身份与条件验证的 vLLM 开发运行当作实验结果；新增本地生产训练栈。
- 为 HumanEval、LiveCodeBench、SWE-bench 等其他任务集编写适配器，除非后续任务真正需要。
- 在此规格内承诺 OPSD 天然优于独立教师 OPD、保证四天或 ¥200 内一定获得提升。
- 默认把 DeepSeek／Qwen 聊天 API 当作任意学生 token 的评分接口，或展开跨 tokenizer 蒸馏研究。
- 完整词表 JSD、额外 GRPO、系统性消融、多种子训练或规模扩展；有余量时需作为后续范围单独记录。
- 为每个库、文件名、容易调整的参数或显而易见的正确性要求分别创建 ADR。
- 在本次文档整理中安装训练环境、启动付费调用、充值或开始实验。

## Further Notes

### 已确认决定的来源

| 会话节点 | 保留的决定 |
| --- | --- |
| A1–A6 | RA／Research Intern 用途，研究与工程约 55%／45%，四至七天，每天约四小时；学习基础并真实跑实验，要求提升；既往工程失控是约束来源 |
| A7 | 主验收以任务 SFT checkpoint 为起点，额外消融与规模扩展后置 |
| A8、A11–A12 | Python 路线；参考 mini-swe-agent 但重新实现极简 bash Harness |
| A13–A16 | ¥200 上限；选择 PyTRIO；已有指令模型上做任务 SFT；需求生成后测试修复 |
| A17 | 优先验证 4B OPSD，9B OPD 作为备选 |
| A18 | 确认完整范围，包括冻结 SFT 教师、MBPP 起点、≤¥10 小试、¥150 常规计划与 ¥50 余量、三阶段结果及交付物 |
| 2026-09-28 仓库说明 | 项目根目录为本规格所列 bashlet 目录；保持本会话上下文；区分决定、建议与假设 |
| 本次测试 seam 回复 | 用户答复“符合，作为测试方案”，确认 D-14；不再将该 seam 标成待确认 |
| 2026-09-29 Harness grill（Q1–Q18） | Harness 与任务解耦，参考 mini-swe-agent 2.4.6（ADR-0004）；训练任务集从 MBPP 起步，以后可扩展。形成 D-15–D-29：两层结构、行动格式、提交、观察、执行环境、预算、轨迹、模型接口、后端范围、回放、thinking、配置与指纹、评分器、分析层、CLI。时间目标顺延两天。任务单先是手写的 12–14，2026-09-30 经 `/to-tickets` 替换为 12–20 |
| 2026-09-29 资源说明 | 本地实际可用一张 A100 40GB，由 AGENTS.md 约定，只用于开发推理 |
| 2026-09-30 测试 seam 确认 | 用户确认 v2 使用两个 seam：`run_experiment` 为主，CLI 子进程为第二个；外部依赖替身包括回放模型、可控 Adapter 和本地假 vLLM 服务；真实 vLLM 只做冒烟运行 |
| 2026-10-01 行动协议修订 | 用户确认取消强制 `THOUGHT:`，行动说明可选，原生 thinking 与行动正文分开。变更归入尚未完成的任务 19，并以 bash-block-v3 记录；18 调整展示要求，16 保留交付时的历史记录 |
| 2026-10-02 W&B grill（Q1–Q4） | 训练和配套评测统一使用 W&B；project=`bashlet`、entity 使用账号默认空间；仅上传指标、脱敏配置和 checkpoint 引用；运行中断网继续本地记录并在恢复后同步。接入和验收归入未开始的任务 07–11 |
| 2026-10-03 本地范围调整 | 任务 21 允许原始 Qwen3.5-4B 与已有 SFT 导出权重用于经验证的本地示范采样和评测；PyTRIO 仍是唯一生产训练后端，任务 20 历史证据不重标 |

### 待验证假设与外部依赖

以下条目没有被用户的范围确认自动变成事实；验证结果应关联对应实验运行。

| ID | 待验证内容 | 进入主实验前需要的证据 |
| --- | --- | --- |
| H-01 | PyTRIO 账户、支付准入、模型权限与 SDK 可用 | 在预算内完成模型清单和最小调用核对；不能假设最低充值或已有可用凭证 |
| H-02 | 4B 能执行所选 bash 协议，任务难度有学习空间 | 同一 Harness 下的初始指令与 SFT 开发集轨迹、错误类型和初步通过率 |
| H-03 | 正确参考实现能改善 4B 教师的监督 | 教师评分可取得、对齐可靠，以及受限更新后的开发集执行表现；只会复述答案或 loss 下降不够 |
| H-04 | 学生和教师实际使用相同 SFT 起点及兼容编码 | 记录实际模型 revision、tokenizer／模板、快照身份；验证特权 prompt 的偏移和原始学生 token 保留 |
| H-05 | 教师评分计费归类和缓存行为与预算模型一致 | 小额请求的使用量与实际账单；暂按未缓存前向估算，不能把零 loss 位置当免费 |
| H-06 | 本节点能可靠隔离 bash 执行 | 已发现 Bubblewrap 0.6.1 与 timeout，但 namespace 权限和完整隔离尚未验证；命令存在不代表可运行 |
| H-07 | 数据版本、参考解和增强测试适合首版 | 版本／题号清单、参考解执行验证、原始 split 映射和任务家族去重记录 |
| H-08 | 既有大模型 API 能提供有用且可负担的示范 | 对选定服务的实际访问、费用口径及通过测试的示范；普通聊天能力不证明 token 评分能力 |
| H-09 | 顺延后的时间目标（2026-10-03，上限 2026-10-06）与 ¥200 能容纳主实验 | 小试真实 token、吞吐与费用外推；将评测、失败调用和重跑一并计入 |
| H-10 | 开发用 vLLM 能在一张 A100 40GB 上以 token 形式的 prompt 返回 token 与逐 token logprob | 2026-10-02 已核对 vLLM 0.11.0 接口，并在 n30126 的 A100 40GB 上用 Qwen3-4B-Instruct-2507 完成真实 CLI 冒烟回合：3 步提交，48 个原始 token 与逐 token logprob 完整保存。详见[任务 20 的接口与运行证据](issues/20-dev-vllm-backend.md#comments)，仅作开发证据 |
| H-11 | Qwen3.5-4B 的聊天模板和思考标记与 Qwen ChatML 编码兼容 | 用真实 tokenizer 比对首个 prompt 与官方模板输出；核对多轮时思考内容的处理（任务 04） |
| H-12 | W&B 凭证、默认空间权限与所锁定 SDK 的上报／补同步能力可用 | 用户已在 `.env` 配置凭证，但不等于认证成功；任务 07 验证最小上报、实际 entity、运行身份、脱敏字段、断网记录及补同步，09–11 关联真实训练／评测记录 |

### 用户提供的价格记录

以下是本次会话提供、于 2026-09-28 记录的 PyTRIO 单价，单位均为人民币／百万 token；它们是预算输入，不是本项目已经发生的账单，也不是训练量预测。

| 模型 | prefill | cache | sample | train |
| --- | ---: | ---: | ---: | ---: |
| Qwen/Qwen3.5-4B | 1.49 | 0.30 | 4.54 | 4.54 |
| Qwen/Qwen3.5-9B | 4.48 | 0.90 | 13.53 | 9.93 |
| Qwen/Qwen3.8-27B | 8.41 | 1.68 | 25.29 | 25.29 |

4B 是主线学生与同源教师；9B 仅为备选，27B 未被选入首版。教师评分按未缓存 prefill 计费只是需要 H-05 验证的预算假设；OPSD 额外参考上下文会增加教师 token。此前对固定 token 用量的价格演算不能作为全项目报价。

### 工作次序与交付边界

1. 先验证数据划分、真实 bash 隔离、任务与评分闭环，以及 PyTRIO 的必要契约；优先发现会阻止实验的事实。
2. 构造并验证 SFT 示范，由任务 07 先接入并验证 W&B，再运行任务 SFT 和配套开发评测，保存可用于教师与评测的模型快照及相应训练状态。
3. 在固定 SFT 教师下做 ≤¥10 OPSD 小试；仅用训练与开发数据判断是否投入主实验或记录备选理由。
4. 根据真实费用和开发集表现确定主实验配置，运行所选路线；方案固定后执行最终测试并形成结果与失败分析。

上述次序服务于 D-01 的时间目标（2026-10-03），允许在上限（2026-10-06）前处理尚未解决的问题；它不承诺每一阶段已具备环境条件，也不授权突破现金上限。P-01–P-05 中仍待定的细节应在相应实施前具体化并记录；P-05 的监控工具已由 D-30 确定。不要用最终测试结果倒推更有利的协议。

### 仓库入口与长期记录

- [领域术语](../../CONTEXT.md)
- [ADR-0001：单一托管训练后端](../../docs/adr/0001-one-managed-training-backend.md)
- [ADR-0002：自写共用 Harness](../../docs/adr/0002-own-a-shared-minimal-bash-harness.md)
- [ADR-0003：冻结 SFT 教师的 OPSD](../../docs/adr/0003-frozen-sft-teacher-for-opsd.md)
- [ADR-0004：Harness 与任务解耦](../../docs/adr/0004-task-agnostic-harness.md)
- [本地任务跟踪约定](../../docs/agents/issue-tracker.md)

规格之外只保留以上四份 ADR：这些选择会影响后端衔接、行动分布或监督语义，改变它们需要重做实质验证。数据集版本、预算参数、测试工具和目录组织不另立 ADR；其状态保留在本规格或具体实验记录中。

### 方法参考与证据边界

- W&B 官方 SDK 的[环境变量定义](https://github.com/wandb/wandb/blob/main/wandb/env.py)、[Settings](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py) 和[同步 CLI](https://github.com/wandb/wandb/blob/main/wandb/cli/cli.py)：2026-10-02 核对 `WANDB_API_KEY`、project／entity、关闭控制台／源码采集及 `wandb sync` 的入口；实际接入版本和故障恢复行为由任务 07 锁定并验证，不把查阅源码当成真实账号验收。
- [OPSD 作者的方法说明](https://siyan-zhao.github.io/blog/2026/opsd/)：解释特权教师、自轨迹监督及其限制；论文实验不能当作本项目结果。
- [PyTRIO OPSD 示例](https://docs.pytrio.com/docs/example/opsd)：提供采样 token 监督和远程更新的参考；原示例的初始教师不自动等于本项目的 SFT 教师。
- [PyTRIO 计算与缓存说明](https://docs.pytrio.com/docs/prefill-cache)：用于区分训练监督位置、实际计算量与费用；实际收费需核账。
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)：提供极简交互设计参考；本项目已选择自行实现。参考版本为 2.4.6，借用内容与有意差异见 ADR-0004。
- [MBPP 原始资料](https://github.com/google-research/google-research/tree/master/mbpp) 与 [EvalPlus 数据实现](https://github.com/evalplus/evalplus/blob/master/evalplus/data/mbpp.py)：用于确认数据身份、划分和评分材料；使用前锁定实际版本。
- [FARO2 参考仓库](https://github.com/Camellia86/Full-Agent-RL-OPSD-with-Just-2-A800s)：访谈中的只读检查发现其可见脚本为标量奖励驱动的 PPO／group-normalized 路径，未见所需的特权教师蒸馏与工具执行闭环；它是项目灵感，不是本规格已验证的 OPSD 实现基础。

## Comments

- 2026-09-28：用户确认以完整小型实验为主要测试 seam，日常测试替换付费模型计算但保留真实执行、隔离与评分；远端契约以既定小试验证。该答复已合并到 D-14 与 Testing Decisions。
- 2026-09-29：用户认为 bash-block-v1 过度贴合 MBPP，经 grill 确认 Harness 与任务解耦（ADR-0004）。已修订 Problem Statement、Solution、User Stories 5 与 41–43、D-01、D-02、D-03、P-01、P-02、Out of Scope 和参考说明；新增任务单 12–14，并调整 03、04、06、08 的依赖。
- 2026-09-30：按 `/to-spec` 把 grill 的全部结论写入本规格：Problem Statement 和 Solution 补充了 v2 的动机与方案；新增 User Stories 44–89 和决定 D-15–D-29；改写 P-02、P-03、P-05；新增 Harness v2 的测试 seam 与场景；更新完成判据、Out of Scope，新增 H-10、H-11。
- 2026-09-30：按 `/to-tickets` 把 Harness v2 拆成 12–20，替换原来手写的 12–14；03、04、06、08 的依赖随之调整。用户允许修改本规格，使其中的任务编号指向新任务单。
- 2026-10-01：用户确认取消强制 `THOUGHT:`，要求不把此次变更放回已经结束的任务。已修订 User Stories 46、76、D-16、D-25、D-26、D-29、P-02、相关行为场景与完成判据；实施归入任务 19，协议升级为 bash-block-v3，任务 18 调整可选行动说明的展示，任务 16 的交付记录保留。
- 2026-10-02：用户在 W&B grill 的四项问题中确认训练与评测范围、`bashlet`／账号默认空间、上传边界及断网继续记录。新增 D-30、共同监控与验收约定、H-12，并修正 P-05 的工具未定表述；同步 plan 和未开始的任务 07–11。已有基线的监控导入归入 07，05 的独立实扣费用缺口及已交付任务历史保持原状；本次未安装依赖或开始训练。
- 2026-10-02：任务 07 出现 SFT 回退后，经新一轮 grill 明确 SFT 的 Harness 适配目标。按用户要求，仅修订 Solution 与 D-06 中的长期目标和验收边界；数据规模、监督策略、推理安排、预算和小试配置另列新任务。任务 06、07 的既有示范与选优记录保留为历史证据，工程交付不代表通过本轮 SFT 验收。
