# Harness 数据角色 v2

v2 将来源身份与本轮用途分开。`origin` 保留来源、来源版本、内容哈希、原题身份、原始家族和原始划分。`assignment` 记录数据版本、cohort 内容哈希、本轮家族和 `train_candidate`、`dev` 或 `test` 角色。任务清单中的 `split=train` 是训练入口的兼容选择值；它不会覆盖 `origin.split`。

`harness-cohort-v3.json` 是当前质量验收样本：5 个训练候选家族、3 个 dev 家族和 3 个 test 家族，覆盖三类任务。候选池复用两个旧 train、两个旧 dev，并加入一个短程序家族。`license-audit/policy-green` 随旧 dev 家族进入候选池，不增加家族数。该样本不代替后续 536/100/200 家族的完整冻结。

```bash
uv run --locked bashlet prepare-harness \
  --cohort configs/harness-cohort-v3.json \
  --output-dir runs/harness-role-quality-v3 \
  -c configs/bash-block-v6.yaml
```

每个任务都用现有 Harness 执行参考实现和代表性错误实现。参考实现须通过反馈与独立评分；错误实现须实际写入提交并被评分拒绝。test 角色也只做这一质量流程。证据标记为 `reference-and-mutation-replay`，`model_baseline=false`。这些结果不能用作 Baseline 或 Student 的能力分数。

新增来源先在 `task-sources-v2.json` 登记内容锁。公开、人工编写和模型辅助来源共用准入检查；作者标签不提供信任。catalog 基本格式见 `harness-supplement-v2.json`，生成身份与评分修订见 `harness-supplement-v3.json`。登记原题、家族、原始划分、类型、变体、参考实现和错误实现；模型辅助来源还须记录生成模型及版本，无法核实时明确为 `unknown`／`null`。未划分来源保留 `unsplit`。修改来源内容或新增变体时，须登记相应来源版本和内容锁，再运行相同质量验收。

`supplement/cache-expiry/v3` 用命令评分检查 `False`、`0`、到期边界及输入未被修改。它与旧版本同属 `supplement/cache-expiry`，不能重复计数或跨角色。v2 来源字节和旧评分语义保留；新 cohort 使用修正后的来源版本。

复用 MBPP 时，另传 `--mbpp-manifest runs/mbpp-sanitized-v1-verified/manifest.json`，并在 cohort 中列出任务。v2 核对登记的来源版本、原始源字节及原始任务内容。`mbpp/625` 与 `mbpp/591` 的本轮家族均为 `mbpp/591`；只能取一个独立代表，或将登记的变体保留在同一角色中。旧 v1 的开发家族保留规则继续生效。

输出目录必须是新目录。`private/cohort.json`、`private/source-registry.json`、`private/family-guard.json` 和 `private/sources/` 保存冻结字节；manifest 记录对应哈希及逐任务执行证据。`private/tasks/` 含参考实现与隐藏评分，只供宿主加载；Student 仅获得题面和初始工作区。整个候选池都会参与隔离检查，尚未抽中的候选也不能进入独立评估。

`assignment.manifest_path` 绑定本地冻结清单；单独运行 Task 也须能读取该清单和来源副本。角色、版本或任务内容与清单不符时拒绝执行。准备中的任务仅供本次质量校验使用；训练必须读取已完成清单。换到其他存储路径时，用同一 cohort 准备新的数据目录并保留旧证据。

`harness-dev.json` 可交给现有 `evaluate`。`prepare-sft` 与 `train-opsd` 仍用 `train` 选择训练候选；`task --split train_candidate` 也可显式选择新版候选。SFT 导出、训练加载和实验记录保留本轮角色与历史来源。v1 数据及已有实验文件不需要迁移。
