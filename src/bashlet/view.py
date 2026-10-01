"""One scrolling renderer for live progress and saved run records."""

import re
import signal
from collections.abc import Mapping
from typing import Any

from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

_BLOCK_START = re.compile(r"^```bash[ \t]*$", re.MULTILINE)


class RunView:
    """Render only newly completed steps; never read input or modify a run."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.steps = 0
        self.tokens = 0
        self.invalid = 0
        self.seconds = 0.0
        self.started = False

    def update(self, record: Mapping[str, Any]) -> None:
        # Finish one live update before delivering Ctrl-C. Otherwise cleanup
        # could skip a step whose terminal output was only partially written.
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            self._update(record)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def _update(self, record: Mapping[str, Any]) -> None:
        if not self.started:
            self.console.print(
                Text(f"Run directory: {record['run_dir']}"), soft_wrap=True
            )
            self.started = True
        budget = record["budget"]
        for step in record["trajectory"][self.steps :]:
            self.steps += 1
            call = step.get("model_call") or {}
            self.tokens += len(call.get("generation", {}).get("tokens", []))
            self.seconds += call.get("duration_seconds", 0.0)
            observation = step.get("observation") or {}
            self.seconds += observation.get("duration_seconds", 0.0)
            self.invalid += step["action"]["kind"] == "invalid"
            usage = step.get("usage") or {
                "generated_tokens": self.tokens,
                "invalid_actions": self.invalid,
                "elapsed_seconds": self.seconds,
            }
            self.console.print(Text(f"Step {self.steps}", style="bold cyan"))
            self.console.print(
                Text(
                    f"Budget: steps {self.steps}/{budget['max_steps']} | "
                    f"tokens {usage['generated_tokens']}/"
                    f"{budget['max_generated_tokens']} | "
                    f"time {usage['elapsed_seconds']:.2f}/"
                    f"{budget['max_seconds']:.2f}s | "
                    f"invalid {usage['invalid_actions']}"
                ),
                soft_wrap=True,
            )
            text = call.get("text", "")
            if step["action"]["kind"] == "bash":
                explanation = _BLOCK_START.split(text, maxsplit=1)[0].strip()
                if explanation:
                    self.console.print(Text("Explanation:", style="bold"))
                    self.console.print(Text(explanation), soft_wrap=True)
                self.console.print(Text("Command:", style="bold"))
                self.console.print(
                    Syntax(
                        step["action"]["command"],
                        "bash",
                        background_color="default",
                        word_wrap=True,
                    )
                )
            elif text:
                self.console.print(Text("Reply:", style="bold"))
                self.console.print(Text(text), soft_wrap=True)
            if step.get("feedback") is not None:
                label = (
                    "Format error:"
                    if step["action"]["kind"] == "invalid"
                    else "Observation:"
                )
                self.console.print(Text(label, style="bold"))
                # Text avoids interpreting model output as Rich markup or emoji.
                self.console.print(Text(step["feedback"]), soft_wrap=True)
            elif observation:
                self.console.print(
                    Text(
                        f"Command result: {observation['status']} "
                        f"(return code {observation['exit_code']})"
                    )
                )
                self.console.print(Text(observation.get("output", "")), soft_wrap=True)

    def finish(self, record: Mapping[str, Any]) -> None:
        self.update(record)
        self.console.print(Text(f"Outcome: {record['status']}", style="bold"))
        if record.get("reason"):
            self.console.print(Text(record["reason"]), soft_wrap=True)
        if record.get("submission_text"):
            self.console.print(Text("Submission:", style="bold"))
            self.console.print(Text(record["submission_text"]), soft_wrap=True)
        grade = record.get("grade")
        if grade is None:
            self.console.print(Text("Grade: not graded"))
        else:
            outcome = "passed" if grade["passed"] else "failed"
            self.console.print(
                Text(
                    f"Grade: {outcome} "
                    f"({grade['passed_cases']}/{grade['total_cases']})",
                    style="bold",
                )
            )
