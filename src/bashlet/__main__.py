"""Run a trusted, scripted experiment request through the shared interface."""

import argparse
import json
from pathlib import Path

from . import (
    Bash,
    GradingCase,
    InteractionBudget,
    Limits,
    Submit,
    Task,
    run_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated bashlet experiment")
    parser.add_argument("request", type=Path, help="Trusted task/actions JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--bwrap", default="/usr/bin/bwrap")
    options = parser.parse_args()
    try:
        request = json.loads(options.request.read_text())
        task_data = request["task"]
        cases = tuple(
            GradingCase(args=tuple(case["args"]), expected=case["expected"])
            for case in task_data.pop("grading_cases")
        )
        task = Task(**task_data, grading_cases=cases)
        actions: list[Bash | Submit] = []
        for action in request["actions"]:
            kind = action.pop("kind")
            if kind == "bash":
                actions.append(Bash(**action))
            elif kind == "submit":
                actions.append(Submit(**action))
            else:
                raise ValueError(f"Unknown action kind: {kind!r}")
        result = run_experiment(
            task=task,
            actions=actions,
            output_dir=options.output_dir,
            limits=Limits(**request.get("limits", {})),
            budget=InteractionBudget(**request.get("budget", {})),
            bwrap=options.bwrap,
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
