# bashlet

A Tiny Coding Agent with On-Policy Self-Distillation。

当前实现任务 01：一个小型实验入口，贯通手工任务、隔离 bash、真实反馈、提交快照、独立评分和运行记录。后续 Harness、数据与训练切片扩展 `bashlet.run_experiment`。此入口当前接受受控行动序列；模型采样与训练将在后续任务接入。

## 运行

宿主使用 Python 3.13.12 和 uv；依赖版本锁定在 `uv.lock`，默认使用清华 PyPI 镜像。
执行节点需提供 Linux user/mount/PID/network namespaces、Bubblewrap、`/usr/bin/prlimit`、`/bin/bash` 和 `/usr/bin/python3`。

```bash
uv sync --locked
uv run bashlet examples/sum.json
```

也可以用 `uv run python -m bashlet examples/sum.json`。终端返回状态和运行目录：退出码 0 表示提交通过，1 表示任务未通过或未提交，2 表示基础设施或配置错误。
用 `--output-dir <目录>` 指定产物根目录，每次都会建立新的运行目录。

JSON 请求是可信实验配置，包含 `task`、`actions` 和可选的 `limits`，结构见示例。`bash` 行动携带 `command`；`submit` 提交固定的 `solution.py` 并立即结束回合。命令失败、超时和超量输出保留真实观察，后续受控命令可继续执行。任务 02 将在同一入口接入根据观察采取下一行动的策略及模型协议。

Python 调用者直接使用相同入口：

```python
from pathlib import Path
from bashlet import Bash, GradingCase, Submit, Task, run_experiment

result = run_experiment(
    task=Task(
        task_id="manual/double-v1",
        prompt="Write solution.py with solve(n), returning twice n.",
        feedback_tests="from solution import solve\nassert solve(2) == 4\n",
        grading_cases=(GradingCase(args=(-3,), expected=-6),),
    ),
    actions=[
        Bash("printf 'def solve(n): return n * 2\\n' > solution.py"),
        Bash("python3 feedback_tests.py"),
        Submit(),
    ],
    output_dir=Path("runs"),
)
print(result.status, result.grade)
```

## 隔离与评分边界

每次实验先运行真实访问探针：验证 namespace 已分离、环境变量已清理、宿主私有文件和包含参考实现／评分材料的配置无法读取，以及任务目录可写。探针不通过时记录 `infrastructure_error`，不执行任何任务命令，也不退回宿主直接执行。缺少执行器可用 `--bwrap /does-not-exist/bwrap` 重现。

沙箱只挂载只读的 `/usr/bin`、`/usr/lib`、`/usr/lib64` 和当前任务目录，使用独立 `/proc`、`/dev`、`/tmp`，清理环境和继承的文件描述符，断开宿主网络，移除 capabilities 并建立新会话。实现依据 [Bubblewrap 0.6.1 的说明](https://github.com/containers/bubblewrap/blob/v0.6.1/README.md)及[参数文档](https://github.com/containers/bubblewrap/blob/v0.6.1/bwrap.xml)。节点内实际任务 Python 来自 `/usr/bin/python3`，与宿主 Python 分别记录；当前节点为 Python 3.10.12。首版任务仅使用该解释器的标准库。

`json-function-v1` 评分调用 `Task.function_name`（默认 `solve`），接受 JSON 位置参数，返回值须可表示为 JSON。每个评分用例在新沙箱中执行只读的提交快照，宿主比较真实返回值与期望值。参考实现只用于可信材料记录，不执行、不做源码相等比较；评分期望值始终留在宿主。学生的退出码 0、自报通过或被修改的反馈测试均不能代替评分。此切片尚未实现任意 MBPP 测试程序的适配。

限制覆盖墙钟超时、stdout 与 stderr 合计字节、提交大小、每进程地址空间、CPU 时间、单文件大小及打开文件数。超时或输出超限会终止整个沙箱及后代进程。限制值随运行记录保存；它们不是 cgroup 总量配额，不能视为完整的多租户资源隔离。安全边界依赖节点内核和系统运行库。

## 运行产物

`runs/<唯一运行 ID>/` 与源码分离，权限为当前用户私有：

- `run.json`：任务身份与材料摘要、配置、隔离检查、行动计划、实际轨迹、提交哈希和评分。`completed` 表示评分流程完成；任务成功由 `grade.passed` 决定。评分基础设施故障单独记录，未执行的用例不会计为通过。
- `private/task.json`：完整可信任务配置，包括参考实现和评分材料。
- `workspace/`：学生可见的需求、反馈测试和实际文件状态。
- `submission/solution.py`：只读提交快照；只接受受大小限制的普通文件，拒绝符号链接和 FIFO。
- `probe/`：能力探针使用的临时工作区。

`run_experiment` 返回值及整个运行目录供实验操作者使用；未来模型上下文只能消费公开任务与 `trajectory` 中的真实观察。评分结果、参考实现、隐藏输入及期望值均不回写本回合工作区。

## 验证

```bash
uv run mypy
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

测试均通过小型实验入口保留真实隔离、bash 和评分，不调用付费模型。需要在允许 Bubblewrap 创建 namespaces 的节点执行；受限外层沙箱中的隔离测试应失败，不能靠跳过测试宣称节点隔离已验证。
