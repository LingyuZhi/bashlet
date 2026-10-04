# 任务 27 两轴审查

基点由用户确认：`e9a9a0c775423f0177eaca25b7f60abd0d85ee47`。
两名独立审查 agent 并行检查已暂存的 WIP，实现审查后才提交。
固定差异命令为 `git diff --cached e9a9a0c775423f0177eaca25b7f60abd0d85ee47`；
当时 `git log e9a9a0c775423f0177eaca25b7f60abd0d85ee47..HEAD --oneline` 为空。
已有未暂存 `.gitignore`、`README.md` 和主规格修改没有进入本轮审查或提交。
原始采样、回放、token、blobs 和进程日志按实验数据审查，不按维护源码改写。

## Standards

已审查相对 `e9a9a0c` 的暂存差异。没有发现需要修正的实际规范问题。

硬性规范：0 项。核心实现放在 `src/bashlet/curation/`，配置放在 `configs/`，
实验记录放在 `runs/`，符合 `AGENTS.md`。沿用已有 Harness、真实执行与训练
加载器，符合 ADR-0001、0002、0004。

判断建议：1 项 possible **Duplicated Code**。`pilot.py:59` 中的
`if _sha(run_path) != attempt["run_sha256"]`、目录边界及运行来源校验，与
既有 `freeze_demonstrations` 的检查重复。按 `AGENTS.md` 的 Consolidation-first
原则，后续可将共同校验提取为 curation 内的私有函数，降低两处准入规则
发生偏差的风险。本轮报告处理不完整 collection，独立模块合理；该建议
不阻断交付。

处理：保留为后续合并建议。本轮没有扩大既有冻结入口的重构范围；两个
入口已共享 `_plan` 与 `_sha`，实际旧冻结入口测试通过。

## Spec

Spec：0 项需修正问题。未发现规格缺失、越界行为或错误实现。

已核查 9 个训练家族、18 次真实配对尝试、102 条原回复审阅、token/logprob
与追加历史、固定推理条件及冻结哈希。离线重新加载成功：v3 为 1 条，
v4 为 4 条，参考辅助为 1 条；输入及监督 token 与报告一致。

时间戳回放候选按要求拒绝，原始观察保留。工作区模型示范缺口、辅助来源、
540 秒 GPU 用量及 ¥0 新增付费均明确记录。后续采集建议与批准范围分开；
未执行正式采集或新训练。

Standards：0 项硬性问题、1 项非阻断重复校验建议；Spec：0 项问题。
