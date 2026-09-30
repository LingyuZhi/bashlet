"""Run a trusted replay experiment request through the shared interface."""

import argparse
import json
from pathlib import Path

from . import (
    GradingCase,
    ReplayModel,
    Task,
    Utf8Encoding,
    run_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated bashlet experiment")
    parser.add_argument("request", type=Path, help="Trusted task/replies JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        default=[],
        help="YAML layered after the built-in default; later files win",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        metavar="key=value",
        help="Override one merged config value; applied after --config",
    )
    options = parser.parse_args()
    try:
        request = json.loads(options.request.read_text())
        if "limits" in request or "budget" in request:
            raise ValueError(
                "Put execution limits and the interaction budget in the experiment "
                "config, not the request"
            )
        task_data = request["task"]
        cases = tuple(
            GradingCase(args=tuple(case["args"]), expected=case["expected"])
            for case in task_data.pop("grading_cases")
        )
        task = Task(**task_data, grading_cases=cases)
        if "actions" in request:
            raise ValueError("Use replay replies instead of actions")
        replies = request["replies"]
        if not isinstance(replies, list) or not all(
            isinstance(reply, str) for reply in replies
        ):
            raise ValueError("Replay replies must be a list of strings")
        result = run_experiment(
            task=task,
            model=ReplayModel(replies, encoding=Utf8Encoding()),
            output_dir=options.output_dir,
            config=options.config,
            overrides=options.overrides,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "status": result.status,
                "passed": result.grade.passed if result.grade is not None else None,
                "reason": result.reason,
            },
            ensure_ascii=False,
        )
    )
    if result.status == "infrastructure_error":
        raise SystemExit(2)
    if result.grade is None or not result.grade.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
