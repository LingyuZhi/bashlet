# OPSD 扩展数据 v6

权威 cohort 为 `opsd-expanded-cohort-v6.json`。三个角色分别包含 536、100、200 个独立家族；短程序／代码修改／工作区配额分别为 214/161/161、40/30/30、80/60/60。历史来源和历史文件保留。旧来源中确认的近重复在此版本只保留一个代表，具体取舍见冻结目录中的审计材料。

`mbpp-full-v1.json` 和 `humaneval-v1.json` 固定公开来源的提交、原始文件哈希及作者。`opsd-public-import-v5.json` 固定转换类型。转换不会增加原题家族数。短程序保留原始需求；代码修改提供实际错误实现；工作区任务必须读取本地需求和输入，并生成结果文件。工作区评分会换用独立输入，拒绝只记住可见输出的实现。

```bash
uv run --locked bashlet import-public-tasks configs/opsd-public-import-v5.json \
  --output-dir runs/issue31-public-catalog-v5-20261009

uv run --locked bashlet prepare-harness \
  --cohort configs/opsd-expanded-cohort-v6.json \
  --mbpp-manifest runs/mbpp-sanitized-v1-verified/manifest.json \
  --workers 16 --output-dir runs/issue31-expanded-freeze-v6-20261009 \
  -c configs/bash-block-v6.yaml -c configs/cohort-quality-v1.yaml
```

输出目录必须尚不存在。离线导入可传 `--source-dir`，其中包含来源锁指定的原始文件。生成 catalog 位于运行目录的 `private/`，由 `public-python-v5.lock.json` 登记，未生成时不会影响旧来源的加载。冻结目录保存所需来源副本；原导入目录缺失时，加载器使用副本并继续核对登记哈希。cohort 引用的审计文件也必须存在且与记录哈希一致。

`cohort-quality-v1.yaml` 只用于参考及错误实现回放，允许完整写入较长参考程序。它不定义 Student 或 Baseline 的实验预算。`--collect-quality` 用于来源清点：继续记录质量失败，任何拒绝都会使清单保持 `incomplete`，不会缩小冻结分母。`--workers` 范围为 1–16；默认串行。

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
