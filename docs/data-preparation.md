# OPSD 数据准备

`data/` 只维护三份定义：`authored.json` 保存自编任务正文，`sources.json` 保存来源登记、来源锁和公开任务导入计划，`cohort.json` 保存任务划分、生成计划、家族隔离规则和审计身份。三个角色分别包含 536、100、200 个独立家族；短程序／代码修改／工作区配额分别为 214/161/161、40/30/30、80/60/60。内容中的版本、revision、task ID 和哈希继续标识数据。旧来源中确认的近重复在此版本只保留一个代表，具体取舍见冻结目录中的审计材料。

`cohort.json` 的 `generation_plan` 保存生成 cohort 使用的角色配额、难度阈值、抽样与训练种子和访问轮数，并锁定原始计划哈希。`expanded_cohort` 从中恢复计划，再结合质量清单、历史清单和家族审阅结果生成冻结配置。已冻结的数据身份及顺序继续以权威 cohort 为准。新实验的轮数、种子和交互预算由 `configs/opsd.yaml` 决定；当前正式训练为两轮，数据计划中的四轮是历史生成记录。

本版复核 191 个历史家族，保留 174 个；15 个因同题或模板近重复不再独立计数，另两个旧任务存在参考错误或题面／接口不一致。新增 662 个家族补齐总数。25 项公开来源的确定质量或题面／接口问题均有执行反例，已剔除。即使历史参考被剔除，它所在语义组的正确替代代表也只进入候选池。冻结目录的审计副本记录这些取舍与证据身份。

`sources.json` 的 `locks` 固定来源提交、原始文件哈希及作者；`imports` 固定转换类型。内嵌计划按固定 JSON 格式恢复原文，继续核对原 SHA-256。计划中的旧锁文件名作为内部别名解析，不再对应独立文件。转换不会增加原题家族数。短程序保留原始需求；代码修改提供实际错误实现；工作区任务必须读取本地需求和输入，并生成结果文件。工作区评分会换用独立输入，拒绝只记住可见输出的实现。

```bash
uv run --locked bashlet prepare-mbpp \
  --source-lock data/sources.json \
  --output-dir runs/data/mbpp -c configs/opsd.yaml

uv run --locked bashlet import-public-tasks data/sources.json \
  --catalog public-python --output-dir runs/data/public-python

uv run --locked bashlet import-public-tasks data/sources.json \
  --catalog p3 --output-dir runs/data/p3

uv run --locked bashlet prepare-harness \
  --cohort data/cohort.json \
  --mbpp-manifest runs/data/mbpp/manifest.json --artifact-root runs/data \
  --workers 16 --output-dir runs/data/cohort \
  -c configs/opsd.yaml
```

这些命令在具备 Bubblewrap 的计算节点上执行；仅回放参考和错误实现，不访问模型。输出目录必须尚不存在。离线导入可传 `--source-dir`，其中包含来源锁指定的原始文件。生成 catalog 位于对应导入目录的 `private/`，其预期哈希由 `sources.json` 登记。冻结目录保存所需来源副本；原导入目录缺失时，加载器使用副本并继续核对登记哈希。

来源 catalog 与审计材料共用产物根目录，默认是仓库的 `runs/data/`；`--artifact-root` 可以指定其他根目录。审计材料放在该根目录的 `audits/<sha256>.json`，已有冻结目录的同名审计副本可以恢复到这里。加载器限制路径落在指定根目录内，并逐份核对原哈希。当前 13 份审计中，生成计划已内嵌，其余 12 份历史材料未随仓库提供。缺失材料会在创建准备输出前一次性列出；必须补齐真实材料后才能冻结，不能用空记录替代。

最终测试名单由实际训练 manifest 中的完整 `test` 划分生成，保留冻结顺序；运行记录保存在 `evaluation/heldout-plan.json`。不再维护独立的 `data/test.json`，完整训练结束后才能进入最终测试。

`configs/opsd.yaml` 的 `preparation` 节只用于参考及错误实现回放，允许完整写入较长参考程序。`prepare-mbpp` 和 `prepare-harness` 读取此节；训练与评测读取 `condition` 节。`--collect-quality` 用于来源清点：继续记录质量失败，任何拒绝都会使清单保持 `incomplete`，不会缩小冻结分母。`--workers` 范围为 1–16；默认串行。

冻结产物包括：

- `manifest.json`：全部任务、来源、角色、内容哈希和执行证据。
- `training-selection.json`：固定 268 个家族、内容哈希、难度分层、独立种子和四轮顺序。配额为 107/80/81；最大余数相同时依次优先工作区、代码修改、短程序。各难度层也用最大余数法分配，依次优先 easy、medium、hard。
- `harness-dev.json`：现有 `evaluate` 可读取的开发计划。最终测试任务通过清单的 `test` 角色冻结，只允许准备流程做质量验收，正式比较由后续评估任务实施。
- `private/quality-report.json`、`private/audits/`：执行、隔离、来源审阅和旧数据迁移证据。
- `private/tasks/`、`private/sources/`、`private/validation/`：可信宿主持有的任务、参考、隐藏评分和原始执行记录。

四轮始终访问相同的 268 个任务；顺序为排序后的子集，用 `epoch_shuffle_seed + epoch - 1` 分别洗牌。训练端可将 `task_ids`、`epochs=4`、`shuffle_seed=epoch_shuffle` 接入现有 OPSD 编排。1,072 是计划访问数，数据准备不产生优化器更新。抽样、训练、顺序与五个评估种子均是执行配置选择，不是用户指定数值。

普通 `task` CLI 支持 `--split train_candidate`、`dev` 和 `test`。训练子集之外的 268 个候选仍保留候选角色。加载时核对整个候选／dev／test 集、来源、审计、固定子集和顺序；任务内容或划分变化会被拒绝。

近重复检查同时比较去名后的参考 AST、常量模板和需求词汇，再记录语义审阅决定。该审计不能数学上证明不存在任何语义等价。静态难度以参考程序语句和分支数分层，不含公开来源的通用适配代码；它不代表模型实测难度。执行验收和人工可复核的原始记录是数据质量证据，不是模型能力基线。

公开来源的评分随机种子固定为 0，并与训练和模型采样种子分开。函数在子进程内执行，可信评分进程检查实际返回值；成功退出但没有结果不能通过。字典、列表、元组和集合按值传递，`Counter` 等字典子类保留其键值。

P3 补入 Hanoi 和 Sudoku 两个原生家族。每题的五个参数实例只计一个家族；HumanEval 派生题不会重复导入。公开的 `sat` 谓词属于题目规范，参考 `sol` 和其余参数实例由可信流程持有。Hanoi 额外检查 peg 索引范围，Sudoku 检查恰好 81 个字符。原生约束会拒绝固定求解首例的实现。
