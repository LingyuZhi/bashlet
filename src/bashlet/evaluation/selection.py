"""Paired task-family acceptance; a run may have no eligible new snapshot."""

import json
from math import comb
from pathlib import Path
from typing import Any

from ..monitoring import MonitorConfig, RunMonitor
from .harness import _digest, _require_verified_local, _write, effective_comparison
from .suite import VERSION, effective_conditions, file_hash, load_evaluation


def _legacy(control: dict[str, Any], paths: list[Path]) -> None:
    """Keep historical identity checks, but never select from one-track evidence."""
    records = [(path, json.loads(path.read_text())) for path in paths]
    local = control.get("snapshot_details", {}).get("backend") == "vllm"
    if (local and control["snapshot_details"].get("snapshot_kind") != "base") or (
        not local and not control.get("snapshot", "").endswith("@unversioned-base")
    ):
        raise ValueError("SFT baseline must identify the original base model")
    for path, record in records:
        if control.get("snapshot_details", {}).get("backend") == "vllm":
            _require_verified_local(record, path)
        if record["plan"]["split"] != "dev":
            raise ValueError("SFT selection requires development data only")
        if record["comparison_sha256"] != control["comparison_sha256"]:
            raise ValueError(
                "Changed comparison conditions require a matching baseline"
            )
    raise ValueError("SFT selection requires frozen dual-track acceptance evidence")


def _load_suite(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(path.read_text())
    if record.get("version") != VERSION:
        raise ValueError("SFT selection requires frozen dual-track acceptance evidence")
    if record["frozen_sha256"] != _digest(record["frozen"]):
        raise ValueError("Frozen acceptance policy hash differs")
    return record


def _evidence(
    record: dict[str, Any],
    expected_kind: str,
) -> tuple[dict[str, dict[int, dict[str, Any]]], list[str]]:
    tracks: dict[str, dict[int, dict[str, Any]]] = {}
    gaps = []
    if record["evidence_kind"] != expected_kind:
        gaps.append("evidence_kind_mismatch")
    if record["status"] != "complete":
        gaps.append("incomplete_evaluation")
    expected_seeds = record["frozen"]["policy"]["seeds"]
    runtimes = set()
    for track, frozen in record["frozen"]["inputs"].items():
        frozen_tasks = {task["task_id"]: task for task in frozen["tasks"]}
        tracks[track] = {}
        for seed in expected_seeds:
            reference = record["runs"].get(track, {}).get(str(seed))
            if reference is None:
                gaps.append(f"missing:{track}:seed-{seed}")
                continue
            try:
                path, evaluation = load_evaluation(reference)
                if (
                    evaluation["snapshot"] != record["snapshot"]
                    or evaluation["evidence_kind"] != expected_kind
                    or evaluation["config"]["model"]["sampling"]["seed"] != seed
                    or evaluation["plan"]["task_ids"] != frozen["task_ids"]
                    or evaluation["plan"]["split"] != "dev"
                ):
                    raise ValueError("Track, snapshot or seed differs")
                if evaluation["status"] != "complete":
                    gaps.append(f"incomplete:{track}:seed-{seed}")
                runtimes.add(
                    _digest(
                        {
                            key: evaluation["comparison"].get(key)
                            for key in ("inference", "encoding", "environment")
                        }
                    )
                )
                if expected_kind == "experimental":
                    _require_verified_local(evaluation, path)
                for row in evaluation["tasks"]:
                    if row["outcome"] in {
                        "not_started",
                        "running",
                        "infrastructure_error",
                        "interrupted",
                    }:
                        gaps.append(
                            f"missing_task_evidence:{track}:seed-{seed}:{row['task_id']}"
                        )
                    if row["outcome"] == "model_error":
                        gaps.append(f"backend_token_evidence_error:{track}:seed-{seed}")
                    if not row.get("run"):
                        continue
                    episode_path = path.parent / row["run"] / "run.json"
                    if file_hash(episode_path) != row["episode_sha256"]:
                        raise ValueError("Episode hash differs")
                    episode = json.loads(episode_path.read_text())
                    task_path = episode_path.parent / "private/task.json"
                    frozen_task = frozen_tasks[row["task_id"]]
                    task = json.loads(task_path.read_text())
                    if (
                        _digest(task) != frozen_task["task_sha256"]
                        or row["task_origin"]["family_id"] != frozen_task["family_id"]
                        or row["task_type"]
                        != (task.get("task_type") or "short_program")
                    ):
                        raise ValueError("frozen_task_identity_mismatch")
                    if (
                        episode["policy"] != record["snapshot"]
                        or episode["task_sha256"] != row["task_sha256"]
                        or episode["passed"] != (row["outcome"] == "passed")
                    ):
                        raise ValueError("Episode outcome or identity differs")
                    if (
                        track == "coding"
                        and episode.get("evaluation_track") != "coding"
                    ):
                        raise ValueError("Independent coding evidence is missing")
                    if track == "coding" and (
                        len(episode["trajectory"]) > 1
                        or any(
                            step.get("observation") is not None
                            or step["action"]["kind"] == "bash"
                            for step in episode["trajectory"]
                        )
                    ):
                        raise ValueError(
                            "Independent coding needs one answer and no tool feedback"
                        )
                    row["episode_path"] = str(episode_path)
                tracks[track][seed] = evaluation
            except (OSError, ValueError, KeyError, TypeError) as error:
                gaps.append(f"invalid:{track}:seed-{seed}:{error}")
    if len(runtimes) > 1:
        gaps.append("inconsistent_inference_or_encoding_within_suite")
    return tracks, gaps


def _paired(
    control: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
    *,
    task_type: str | None = None,
) -> dict[str, Any]:
    differences: dict[str, list[int]] = {}
    family_seeds: dict[str, dict[str, int]] = {}
    by_seed = {}
    total_passed = 0
    for seed in control.keys() & candidate.keys():
        base = control[seed]
        current = candidate[seed]
        if _digest(effective_comparison(base)) != _digest(
            effective_comparison(current)
        ):
            raise ValueError("comparison_conditions_mismatch")
        if len(base["tasks"]) != len(current["tasks"]):
            raise ValueError("task_denominator_mismatch")
        delta = 0
        for before, after in zip(base["tasks"], current["tasks"], strict=True):
            if (
                before["task_id"] != after["task_id"]
                or before["task_sha256"] != after["task_sha256"]
            ):
                raise ValueError("task_identity_mismatch")
            if task_type is not None and before["task_type"] != task_type:
                continue
            family = before["task_origin"]["family_id"]
            value = int(after["outcome"] == "passed") - int(
                before["outcome"] == "passed"
            )
            differences.setdefault(family, []).append(value)
            family_seeds.setdefault(family, {})[str(seed)] = value
            delta += value
            total_passed += after["outcome"] == "passed"
        by_seed[str(seed)] = delta
    wins = sum(sum(values) > 0 for values in differences.values())
    losses = sum(sum(values) < 0 for values in differences.values())
    discordant = wins + losses
    # Each family contributes one sign after aggregating its correlated seeds.
    loss_p = (
        sum(comb(discordant, index) for index in range(losses, discordant + 1))
        / 2**discordant
        if discordant
        else 1.0
    )
    return {
        "by_seed": by_seed,
        "net_successes": sum(by_seed.values()),
        "candidate_successes": total_passed,
        "task_families": len(differences),
        "gained_families": wins,
        "lost_families": losses,
        "one_sided_family_loss_p": loss_p,
        "family_differences": differences,
        "family_by_seed": family_seeds,
    }


def _reviewed_gains(
    review_paths: list[Path],
    baseline_path: Path,
    candidate_path: Path,
    tracks: dict[str, dict[int, dict[str, Any]]],
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    reviewed: dict[str, set[int]] = {}
    evidence = []
    for path in review_paths:
        review = json.loads(path.read_text())
        if review.get("candidate_sha256") != file_hash(candidate_path):
            continue
        if (
            review.get("baseline_sha256") != file_hash(baseline_path)
            or review.get("version") != "workspace-benefit-review-v1"
        ):
            raise ValueError("Workspace review must identify this paired comparison")
        for item in review["items"]:
            seed = item["seed"]
            evaluation = tracks["harness"][seed]
            row = next(
                row for row in evaluation["tasks"] if row["task_id"] == item["task_id"]
            )
            if row["task_type"] != "workspace" or row["outcome"] != "passed":
                raise ValueError(
                    "Reviewed workspace gain requires independent task success"
                )
            episode_path = Path(row["episode_path"])
            episode = json.loads(episode_path.read_text())
            observation_index, implementation_index = (
                item["observation_step"],
                item["implementation_step"],
            )
            steps = episode["trajectory"]
            if not (
                type(observation_index) is int
                and type(implementation_index) is int
                and 0 <= observation_index < implementation_index < len(steps)
            ):
                raise ValueError(
                    "Review must place the real observation before the implementation"
                )
            observation = steps[observation_index]
            implementation = steps[implementation_index]
            if (
                not observation.get("observation")
                or observation["observation"]["status"] != "ok"
                or not observation.get("feedback")
                or not item["observed_paths"]
                or not item["reason"].strip()
                or not item["author"].strip()
            ):
                raise ValueError(
                    "Review needs a visible observation and explained state dependency"
                )
            task = json.loads((episode_path.parent / "private/task.json").read_text())
            for name in item["observed_paths"]:
                content = task["initial_files"][name]
                if (
                    not content.strip()
                    or content.strip() not in observation["feedback"]
                    or observation["workspace_manifest"][name]
                    != episode["initial_workspace_manifest"][name]
                ):
                    raise ValueError(
                        "Reviewed state was not visible or changed before observation"
                    )
            if not any(
                implementation["workspace_manifest"].get(name)
                != observation["workspace_manifest"].get(name)
                for name in task["implementation_paths"]
            ):
                raise ValueError("Review must identify a later implementation change")
            family = row["task_origin"]["family_id"]
            reviewed.setdefault(family, set()).add(seed)
            evidence.append(
                {
                    "review": str(path.resolve()),
                    "review_sha256": file_hash(path),
                    "episode": str(episode_path),
                    "episode_sha256": row["episode_sha256"],
                    **item,
                }
            )
    return reviewed, evidence


def _assess(
    control: dict[str, Any],
    candidate: dict[str, Any],
    baseline_path: Path,
    candidate_path: Path,
    evidence_kind: str,
    reviews: list[Path],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "decision": "insufficient",
        "reasons": [],
        "gaps": [],
        "metrics": {},
    }
    if control["frozen_sha256"] != candidate["frozen_sha256"] or _digest(
        effective_conditions(control)
    ) != _digest(effective_conditions(candidate)):
        result["gaps"].append("comparison_conditions_mismatch")
        return result
    base_tracks, base_gaps = _evidence(control, evidence_kind)
    tracks, gaps = _evidence(candidate, evidence_kind)
    result["gaps"] = [*base_gaps, *gaps]
    rules = control["frozen"]["policy"]["rules"]
    try:
        harness = _paired(base_tracks["harness"], tracks["harness"])
        coding = _paired(base_tracks["coding"], tracks["coding"])
        classes = {
            kind: _paired(base_tracks["harness"], tracks["harness"], task_type=kind)
            for kind in control["frozen"]["policy"]["harness_counts"]
        }
        result["metrics"] = {
            "harness": harness,
            "coding": coding,
            "by_task_type": classes,
        }
        if "variant" in tracks:
            result["metrics"]["variant"] = _paired(
                base_tracks["variant"], tracks["variant"]
            )
    except ValueError as error:
        result["gaps"].append(str(error))
        return result
    workspace = classes.get("workspace", {"net_successes": 0})
    result["screening"] = {
        "promising": (
            harness["net_successes"] > 0
            and coding["net_successes"] >= 0
            and workspace["net_successes"] > 0
            and all(metric["net_successes"] >= 0 for metric in classes.values())
            and all(gap.startswith("missing:") for gap in result["gaps"])
        ),
        "paired_seeds": sorted(
            base_tracks["harness"].keys() & tracks["harness"].keys()
        ),
        "confirmation_required": bool(result["gaps"]),
        "is_qualification": False,
    }
    if result["gaps"]:
        return result
    repeated = rules["repeated_seeds"]
    for label, metric in [("coding", coding), *classes.items()]:
        if sum(delta < 0 for delta in metric["by_seed"].values()) >= repeated:
            if metric["one_sided_family_loss_p"] <= rules["regression_sign_test_alpha"]:
                result["reasons"].append(f"confirmed_regression:{label}")
            else:
                result["gaps"].append(f"unconfirmed_regression:{label}")
    if result["reasons"]:
        result["decision"] = "ineligible"
        return result
    if result["gaps"]:
        return result
    if harness["net_successes"] <= 0:
        result.update(decision="ineligible", reasons=["no_task_gain"])
        return result
    if any(delta < 0 for delta in coding["by_seed"].values()):
        result["gaps"].append("coding_preservation_not_established")
        return result
    if "variant" in result["metrics"] and any(
        delta < 0 for delta in result["metrics"]["variant"]["by_seed"].values()
    ):
        result["gaps"].append("workspace_variant_regression")
        return result
    try:
        reviewed, evidence = _reviewed_gains(
            reviews, baseline_path, candidate_path, tracks
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        result["gaps"].append(f"invalid_workspace_review:{error}")
        return result
    workspace = classes.get(rules["scene_task_type"], {"family_differences": {}})
    gained = [
        family
        for family, values in workspace["family_differences"].items()
        if sum(value > 0 for value in values) >= repeated
        and not any(value < 0 for value in values)
        and sum(
            workspace["family_by_seed"][family].get(str(seed), 0) > 0
            for seed in reviewed.get(family, set())
        )
        >= repeated
    ]
    result["workspace_gain"] = {
        "gained_family_ids": gained,
        "evidence": evidence,
        "counting_unit": "task family; seeds and variants are correlated",
    }
    if len(gained) < rules["scene_min_gained_families"]:
        result["gaps"].append("reviewed_workspace_task_gain_not_established")
        return result
    result.update(
        decision="qualified", reasons=["coding_preserved_and_reviewed_workspace_gain"]
    )
    return result


def _training_record_matches(
    digest: Any,
    training_path: Path,
    training: dict[str, Any],
    checkpoint: dict[str, Any],
) -> bool:
    """Bind an export to this run, including journals captured before a resume."""
    if digest == file_hash(training_path):
        return True
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        return False
    path = training_path.parent / "training-records" / f"{digest}.json"
    try:
        if file_hash(path) != digest:
            return False
        captured = json.loads(path.read_text())
        return (
            bool(captured["run_id"])
            and all(
                captured[key] == training[key]
                for key in ("run_id", "config", "metadata", "code_revision")
            )
            and checkpoint in captured["checkpoints"]
            and captured["checkpoints"]
            == training["checkpoints"][: len(captured["checkpoints"])]
            and captured["updates"] == training["updates"][: len(captured["updates"])]
            and checkpoint["optimizer_step"]
            <= captured["optimizer_step"]
            <= training["optimizer_step"]
            and captured["optimizer_step"] == len(captured["updates"])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def select_sft(
    run_dir: Path,
    baseline: Path,
    evaluations: list[Path],
    *,
    evidence_kind: str = "experimental",
    reviews: list[Path] | None = None,
    output_dir: Path | None = None,
    monitoring: MonitorConfig | None = None,
) -> Path:
    """Keep every decision; only qualified real evidence may initialize OPSD."""
    training_path = run_dir / "training.json"
    training = json.loads(training_path.read_text())
    control = json.loads(baseline.read_text())
    if control.get("version") != VERSION:
        _legacy(control, [baseline, *evaluations])
    control = _load_suite(baseline)
    if not evaluations or len({path.resolve() for path in evaluations}) != len(
        evaluations
    ):
        raise ValueError("Select from distinct dual-track candidate evaluations")
    if control["evidence_kind"] != evidence_kind:
        raise ValueError(
            "Baseline evidence kind differs; test doubles cannot initialize OPSD"
        )
    base_model = training["metadata"]["base_model"]
    if evidence_kind == "experimental":
        revision = control["snapshot"].removeprefix(f"vllm/{base_model}@")
        if (
            not control["snapshot"].startswith(f"vllm/{base_model}@")
            or len(revision) != 40
            or any(char not in "0123456789abcdef" for char in revision)
        ):
            raise ValueError("SFT baseline must load the original local base snapshot")
    elif control["snapshot"] != f"pytrio/{base_model}@unversioned-base":
        raise ValueError("Test baseline must identify the original base model")
    checkpoints = {
        item["sampler"]["path"]: item
        for item in training["checkpoints"]
        if item["sampler"]["kind"] == "sampler_weights"
        and item["training_state"]["kind"] == "training_state"
    }
    candidates = []
    seen = set()
    for path in evaluations:
        current = _load_suite(path)
        if path.resolve() == baseline.resolve():
            raise ValueError("A baseline cannot be an SFT candidate")
        if evidence_kind == "experimental":
            matches = [
                key
                for key in checkpoints
                if current["snapshot"].startswith(f"vllm/{base_model}@{key}:")
            ]
            snapshot = matches[0] if len(matches) == 1 else current["snapshot"]
        else:
            prefix = f"pytrio/{base_model}@"
            snapshot = current["snapshot"].removeprefix(prefix)
        if snapshot in seen:
            raise ValueError(
                "Candidate must identify a distinct saved SFT sampler snapshot"
            )
        seen.add(snapshot)
        assessment = _assess(
            control, current, baseline, path, evidence_kind, reviews or []
        )
        if snapshot not in checkpoints:
            assessment.update(decision="insufficient")
            assessment["gaps"].append("unregistered_sft_sampler_snapshot")
        if evidence_kind == "experimental":
            for track, refs in current["runs"].items():
                for seed, ref in refs.items():
                    try:
                        _, evaluation = load_evaluation(ref)
                        adapter = evaluation["snapshot_details"].get("adapter") or {}
                        checkpoint = checkpoints.get(snapshot, {})
                        if (
                            not _training_record_matches(
                                adapter.get("training_record_sha256"),
                                training_path,
                                training,
                                checkpoint,
                            )
                            or adapter.get("checkpoint", {}).get("path") != snapshot
                            or adapter.get("optimizer_step")
                            != checkpoint.get("optimizer_step")
                        ):
                            assessment["gaps"].append(
                                f"sft_training_identity_mismatch:{track}:seed-{seed}"
                            )
                            assessment["decision"] = "insufficient"
                    except (OSError, ValueError, KeyError, TypeError):
                        # The evidence validator already retained this failed reference.
                        pass
        candidates.append(
            {
                "evaluation": str(path.resolve()),
                "evaluation_sha256": file_hash(path),
                "snapshot_identity": current["snapshot"],
                **checkpoints.get(
                    snapshot,
                    {"optimizer_step": None, "sampler": None, "training_state": None},
                ),
                **assessment,
            }
        )
    qualified = [
        candidate for candidate in candidates if candidate["decision"] == "qualified"
    ]
    selected = (
        max(
            qualified,
            key=lambda candidate: (
                candidate["metrics"]["harness"]["candidate_successes"],
                -candidate["optimizer_step"],
            ),
        )
        if qualified
        else None
    )
    real = selected is not None and evidence_kind == "experimental"
    selection: dict[str, Any] = {
        "version": VERSION,
        "evidence_kind": evidence_kind,
        "status": "selected"
        if real
        else "test_double_selection"
        if selected
        else "no_qualified_snapshot",
        "rule": control["frozen"]["policy"]["rules"],
        "frozen_sha256": control["frozen_sha256"],
        "baseline": str(baseline.resolve()),
        "baseline_sha256": file_hash(baseline),
        "candidates": candidates,
        "optimizer_step": selected["optimizer_step"] if real and selected else None,
        "student_snapshot": selected["sampler"]["path"] if real and selected else None,
        "frozen_teacher_snapshot": selected["sampler"]["path"]
        if real and selected
        else None,
        "sft_training_state": selected["training_state"] if real and selected else None,
        "simulation_selected_snapshot": selected["sampler"]["path"]
        if selected and not real
        else None,
        "optimizer_transition": "reset-for-opsd" if real else None,
        "teacher_scoring_verified": False,
    }
    destination = (output_dir or run_dir).resolve() / "selection.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and json.loads(destination.read_text()) != selection:
        raise ValueError(
            "Selection already exists; preserve it and use a new output directory"
        )
    _write(destination, selection)
    if monitoring is not None:
        monitor = RunMonitor(
            destination.parent,
            monitoring,
            phase="acceptance-selection",
            metadata={
                "training_run": str(run_dir.resolve()),
                "snapshot": selection["student_snapshot"],
                "evidence_kind": evidence_kind,
                "comparison_sha256": control["frozen_sha256"],
            },
        )
        metrics = {
            "checkpoint/acceptance_status": selection["status"],
            "checkpoint/qualified_candidates": len(qualified),
        }
        if selection["student_snapshot"] is not None:
            metrics["checkpoint/selected_sampler"] = selection["student_snapshot"]
        try:
            monitor.log("acceptance-selection", metrics)
        finally:
            monitor.finish()
    return destination
