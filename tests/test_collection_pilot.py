"""Paired collection reporting through the CLI and real execution seam."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_cli import cli
from test_curation import collection_plan
from test_demonstrations import REPAIR, SUBMIT, SampledTrajectory, executed_task

from bashlet import QwenChatMLEncoding, run_experiment
from bashlet.config import config_record, load_config
from bashlet.models import JSONValue


class ContextBoundTrajectory(SampledTrajectory):
    """A bounded model boundary; the command and observation stay real."""

    max_context_tokens: int

    def probe(self) -> None:
        pass

    def describe(self) -> dict[str, JSONValue]:
        return {"max_model_len": self.max_context_tokens}


@pytest.mark.parametrize("review_fault", ["none", "wrong_reply", "context_exhausted"])
def test_paired_report_keeps_the_planned_denominator_and_binds_reply_reviews(
    tmp_path: Path,
    review_fault: str,
) -> None:
    plan_path = collection_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    config = load_config()
    encoding = QwenChatMLEncoding(
        "Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        cache_dir="runs/tokenizers",
    )
    if review_fault == "context_exhausted":
        plan["max_context_tokens"] = (
            len(encoding.prompt(config.protocol.system_prompt, executed_task().prompt))
            + 100
        )
        plan_path.write_text(json.dumps(plan))
    arguments: list[str] = []
    for version in ("v3", "v4"):
        root = tmp_path / version
        root.mkdir()
        (root / "plan.json").write_bytes(plan_path.read_bytes())
        condition = replace(
            config,
            protocol=replace(config.protocol, name=f"bash-block-{version}"),
        )
        attempts = []
        decisions: list[dict[str, Any]] = []
        if version == "v3":
            model = SampledTrajectory(encoding, [REPAIR, SUBMIT])
            roles: tuple[str, ...] = ("repair", "submit")
            if review_fault == "context_exhausted":
                model = ContextBoundTrajectory(
                    encoding, ["```bash\nprintf '%05000d\\n' 0\n```"]
                )
                model.max_context_tokens = plan["max_context_tokens"]
                roles = ("inspect",)
            result = run_experiment(
                task=executed_task(),
                model=model,
                config=condition,
                output_dir=root / "episodes",
            )
            assert result.status == (
                "context_exhausted"
                if review_fault == "context_exhausted"
                else "completed"
            )
            attempts.append(
                dict(
                    plan["tasks"][0],
                    attempt=1,
                    status=result.status,
                    passed=result.passed,
                    run_dir=str(result.run_dir.relative_to(root)),
                    run_sha256=hashlib.sha256(
                        (result.run_dir / "run.json").read_bytes()
                    ).hexdigest(),
                )
            )
            decisions.append(
                {
                    "attempt": 1,
                    "accept": result.passed,
                    "reason": "Useful repair and independently graded submission",
                    "actions": [
                        {
                            "supervise": True,
                            "reason": "Valid real action without invented results",
                            "author": "model",
                            "action_type": role,
                            "observation_review": {
                                "invented": False,
                                "reason": "The reply contains an action only",
                                "reply_sha256": hashlib.sha256(
                                    step.model_call.text.encode()
                                ).hexdigest(),
                            },
                        }
                        for role, step in zip(roles, result.trajectory, strict=True)
                        if step.model_call is not None
                    ],
                }
            )
        collection = root / "collection.json"
        collection.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete" if attempts else "incomplete",
                    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    "config": config_record(condition),
                    "attempts": attempts,
                }
            )
        )
        review = root / "review.json"
        if decisions and review_fault == "wrong_reply":
            decisions[0]["actions"][0]["observation_review"]["reply_sha256"] = "stale"
        review.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "collection_sha256": hashlib.sha256(
                        collection.read_bytes()
                    ).hexdigest(),
                    "attempts": decisions,
                }
            )
        )
        arguments.extend(("--collection", str(collection), "--review", str(review)))
    output = tmp_path / "report"
    process = cli(
        "report-collection-pilot",
        str(plan_path),
        *arguments,
        "--output-dir",
        str(output),
        cwd=Path(__file__).parents[1],
    )
    if review_fault == "wrong_reply":
        assert process.returncode != 0
        assert "bind the original reply" in process.stdout + process.stderr
        assert not output.exists()
        return
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["planned_attempts"] == 2
    if review_fault == "context_exhausted":
        assert report["protocols"]["bash-block-v3"]["context_exhausted"] == 1
        assert report["protocols"]["bash-block-v3"]["history_over_limit"] == 1
        row = report["paired"][0]["bash-block-v3"]
        assert row["history_tokens"] > report["max_context_tokens"]
        return
    assert report["protocols"]["bash-block-v3"]["grading_passes"] == 1
    assert report["protocols"]["bash-block-v3"]["valid_actions"] == 2
    assert report["protocols"]["bash-block-v4"]["unattempted"] == 1
    assert report["paired"][0]["bash-block-v4"]["status"] == "unattempted"
