"""Deterministic stratification and integrity checks for a verified cohort."""

import ast
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from bashlet.tasks.sources import TASK_TYPES

# Stable remainder tie order, recorded in every frozen selection.
TYPE_REMAINDER_ORDER = ("workspace", "code_edit", "short_program")
DIFFICULTIES = ("easy", "medium", "hard")


def validate_plan(cohort: dict[str, Any]) -> None:
    if "freeze" not in cohort:
        return
    rules = cohort["freeze"]
    required = {
        "version",
        "epochs",
        "subset_size",
        "sampling_seed",
        "training_seed",
        "epoch_shuffle_seed",
        "evaluation_seeds",
        "difficulty_basis",
        "difficulty",
    }
    if not isinstance(rules, dict) or set(rules) != required:
        raise ValueError("Incomplete or unknown cohort freeze settings")
    if rules["version"] != "opsd-cohort-freeze-v1":
        raise ValueError("Cohort freeze requires version v1")
    if type(rules["epochs"]) is not int or rules["epochs"] < 1:
        raise ValueError("Cohort epochs must be a positive integer")
    seeds = [rules[k] for k in ("sampling_seed", "training_seed", "epoch_shuffle_seed")]
    evaluation = rules["evaluation_seeds"]
    if (
        not isinstance(evaluation, list)
        or not evaluation
        or any(type(seed) is not int or seed < 0 for seed in seeds + evaluation)
        or len(set(seeds + evaluation)) != len(seeds + evaluation)
        or not isinstance(rules["difficulty_basis"], str)
        or not rules["difficulty_basis"].strip()
    ):
        raise ValueError("Freeze requires separate valid seeds and a difficulty basis")
    ids = cohort["training_candidates"]["task_ids"]
    difficulty = rules["difficulty"]
    if (
        not isinstance(difficulty, dict)
        or set(difficulty) != set(ids)
        or any(v not in DIFFICULTIES for v in difficulty.values())
    ):
        raise ValueError("Freeze requires pre-recorded difficulty for every candidate")
    if type(rules["subset_size"]) is not int or not 0 < rules["subset_size"] <= len(
        ids
    ):
        raise ValueError("Frozen subset size must be positive and fit its pool")


def _allocate(
    size: int, counts: dict[str, int], order: tuple[str, ...]
) -> dict[str, int]:
    total = sum(counts.values())
    if type(size) is not int or not 0 < size <= total:
        raise ValueError("Frozen subset size must be positive and fit its pool")
    result = {kind: size * count // total for kind, count in counts.items()}
    ranked = sorted(order, key=lambda kind: -(size * counts[kind] % total))
    for kind in ranked[: size - sum(result.values())]:
        result[kind] += 1
    return result


def selection(cohort: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Select only verified candidates, using difficulty fixed before execution."""
    validate_plan(cohort)
    rules = cohort["freeze"]
    seeds = [rules[k] for k in ("sampling_seed", "training_seed", "epoch_shuffle_seed")]
    evaluation = rules["evaluation_seeds"]
    by_id = {entry["task_id"]: entry for entry in entries}
    ids = cohort["training_candidates"]["task_ids"]
    difficulty = rules["difficulty"]
    candidates = [by_id[identity] for identity in sorted(ids)]
    if any(entry["status"] != "verified" for entry in candidates):
        raise ValueError("Freeze requires the whole candidate pool to be verified")
    counts = {
        kind: sum(e["task_type"] == kind for e in candidates) for kind in TASK_TYPES
    }
    quotas = _allocate(rules["subset_size"], counts, TYPE_REMAINDER_ORDER)
    rng = random.Random(rules["sampling_seed"])
    selected: list[str] = []
    strata: dict[str, Any] = {}
    for kind in TASK_TYPES:
        pool = [e["task_id"] for e in candidates if e["task_type"] == kind]
        sizes = {
            level: sum(difficulty[i] == level for i in pool) for level in DIFFICULTIES
        }
        allocation = (
            _allocate(quotas[kind], sizes, DIFFICULTIES)
            if quotas[kind]
            else dict.fromkeys(DIFFICULTIES, 0)
        )
        strata[kind] = {"pool": sizes, "selected": allocation}
        for level in DIFFICULTIES:
            selected.extend(
                rng.sample(
                    [i for i in pool if difficulty[i] == level], allocation[level]
                )
            )
    selected.sort()
    epochs = []
    for epoch in range(rules["epochs"]):
        order = list(selected)
        random.Random(rules["epoch_shuffle_seed"] + epoch).shuffle(order)
        epochs.append(order)
    return {
        "version": "opsd-training-selection-v1",
        "data_version": cohort["name"],
        "task_ids": selected,
        "family_ids": [by_id[i]["family_id"] for i in selected],
        "task_sha256": {i: by_id[i]["task_sha256"] for i in selected},
        "per_type": quotas,
        "strata": strata,
        "epochs": epochs,
        "planned_visits": rules["epochs"] * len(selected),
        "seeds": {
            "sampling": seeds[0],
            "training": seeds[1],
            "epoch_shuffle": seeds[2],
            "evaluation": evaluation,
            "chosen_by": "execution-config",
        },
        "rules": {
            "allocation": "Largest remainder, proportional to the verified pool.",
            "type_remainder_order": list(TYPE_REMAINDER_ORDER),
            "difficulty_remainder_order": list(DIFFICULTIES),
            "sampling": "Python random.Random; sorted IDs; type/difficulty strata.",
            "epoch_order": "Sorted subset; seed = epoch_shuffle_seed + epoch - 1.",
            "difficulty_basis": rules["difficulty_basis"],
        },
    }


def publish_selection(root: Path, manifest: dict[str, Any]) -> None:
    """Called only after every reference and targeted mutation was checked."""
    if "freeze" not in manifest["cohort"]:
        return
    document = selection(manifest["cohort"], manifest["tasks"])
    path = root / "training-selection.json"
    path.write_text(json.dumps(document, indent=2) + "\n")
    manifest["training_selection"] = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    manifest["family_counts"] = dict(
        Counter(e["split"] for e in manifest["tasks"] if e["variant_id"] == "base")
    )
    report = quality_report(root, manifest)
    report_path = root / "private/quality-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest["quality_report"] = {
        "file": "private/quality-report.json",
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }


def quality_report(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Summarize actual executions, type behavior and Student-visible material."""
    rows = []
    for entry in manifest["tasks"]:
        task = json.loads((root / entry["task_file"]).read_text())
        visible = task["initial_files"]
        implementation = task["implementation_paths"][0]
        if task["reference_implementation"].strip() in "\n".join(visible.values()):
            raise ValueError(
                "Reference implementation leaked into the initial workspace"
            )
        if any(path.startswith("_grading/") for path in visible):
            raise ValueError(
                "Hidden grading material leaked into the initial workspace"
            )
        if task["task_type"] == "code_edit" and (
            implementation not in visible
            or visible[implementation] == task["reference_implementation"]
        ):
            raise ValueError(
                "Code editing requires an existing incorrect implementation"
            )
        runs = {}
        for label, passed in (("reference", True), ("wrong", False)):
            path = root / entry[label + "_run"] / "run.json"
            raw = path.read_bytes()
            run = json.loads(raw)
            if (
                hashlib.sha256(raw).hexdigest() != entry[label + "_run_sha256"]
                or run["status"] != "completed"
                or run["grade"]["passed"] is not passed
                or any(
                    path.startswith("_grading/")
                    for path in run["initial_workspace_manifest"]
                )
            ):
                raise ValueError("Quality execution evidence is incomplete or changed")
            observations = json.dumps(
                [step["observation"] for step in run["trajectory"]]
            )
            if task["reference_implementation"] in observations:
                raise ValueError("Reference implementation leaked into observations")
            runs[label] = {
                "file": entry[label + "_run"] + "/run.json",
                "sha256": entry[label + "_run_sha256"],
                "passed": passed,
            }
        rows.append(
            {
                "task_id": entry["task_id"],
                "family_id": entry["family_id"],
                "role": entry["assignment"]["role"],
                "task_sha256": entry["task_sha256"],
                "executions": runs,
            }
        )
    groups = {
        role: {
            e["family_id"]
            for e in manifest["tasks"]
            if e["assignment"]["role"] == role and e["variant_id"] == "base"
        }
        for role in ("train_candidate", "dev", "test")
    }
    intersections = {
        "candidate_dev": sorted(groups["train_candidate"] & groups["dev"]),
        "candidate_test": sorted(groups["train_candidate"] & groups["test"]),
        "dev_test": sorted(groups["dev"] & groups["test"]),
    }
    if any(intersections.values()):
        raise ValueError("Quality audit found cross-role families")
    return {
        "version": "frozen-cohort-quality-v1",
        "data_version": manifest["cohort"]["name"],
        "evidence_kind": "reference-and-mutation-replay",
        "model_baseline": False,
        "counts": manifest["counts"],
        "independent_families": sum(len(g) for g in groups.values()),
        "task_variants": len(rows),
        "reference_executions": len(rows),
        "wrong_executions": len(rows),
        "family_intersections": intersections,
        "student_material_isolation": "passed",
        "paid_model_calls": 0,
        "tasks": rows,
    }


def validate_selection(root: Path, manifest: dict[str, Any]) -> None:
    """A loader rejects changed selections, orders and any pool task bytes."""
    if "freeze" not in manifest["cohort"]:
        if "training_selection" in manifest:
            raise ValueError("Unexpected frozen training selection")
        return
    artifact = manifest.get("training_selection", {})
    path = root / "training-selection.json"
    if artifact.get("file") != path.name or not path.is_file():
        raise ValueError("Frozen training selection is missing")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != artifact.get("sha256") or json.loads(
        raw
    ) != selection(manifest["cohort"], manifest["tasks"]):
        raise ValueError("Frozen training selection or epoch order changed")
    for entry in manifest["tasks"]:
        task_file = (root / entry["task_file"]).resolve()
        if (
            not task_file.is_relative_to((root / "private/tasks").resolve())
            or hashlib.sha256(task_file.read_bytes()).hexdigest()
            != entry["task_sha256"]
        ):
            raise ValueError("Frozen pool task content changed")
        if entry["status"] != "verified":
            raise ValueError("Frozen pool contains an unverified task")
    report = manifest.get("quality_report", {})
    report_path = root / "private/quality-report.json"
    if (
        report.get("file") != "private/quality-report.json"
        or not report_path.is_file()
        or hashlib.sha256(report_path.read_bytes()).hexdigest() != report.get("sha256")
    ):
        raise ValueError("Frozen quality report changed")
    report_record = json.loads(report_path.read_text())
    if {row["task_id"]: row["task_sha256"] for row in report_record["tasks"]} != {
        entry["task_id"]: entry["task_sha256"] for entry in manifest["tasks"]
    }:
        raise ValueError("Frozen quality evidence or pool task content changed")
    by_id = {entry["task_id"]: entry for entry in manifest["tasks"]}
    for group, role in (
        ("training_candidates", "train_candidate"),
        ("development", "dev"),
        ("test", "test"),
    ):
        actual = {
            kind: sum(
                by_id[i]["task_type"] == kind
                for i in manifest["cohort"][group]["task_ids"]
            )
            for kind in TASK_TYPES
        }
        target = manifest["cohort"][group]["per_type"]
        expected = dict.fromkeys(TASK_TYPES, target) if type(target) is int else target
        if (
            actual != expected
            or manifest["counts"][role] != actual
            or report_record["counts"][role] != actual
        ):
            raise ValueError("Frozen cohort quotas changed")


def expanded_cohort(
    *,
    inventory_path: Path,
    historical_path: Path,
    review_path: Path,
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a configured freeze plan to actual quality receipts and review."""
    inventory = json.loads(inventory_path.read_text())
    historical = json.loads(historical_path.read_text())
    review = json.loads(review_path.read_text())
    plan = json.loads(plan_path.read_text())
    if plan["version"] != "opsd-expanded-plan-v1":
        raise ValueError("Unsupported expanded cohort plan")
    quotas = plan["quotas"]
    candidate_quotas = quotas["training_candidates"]
    dev_quotas = quotas["development"]
    test_quotas = quotas["test"]
    if any(
        set(role) != set(TASK_TYPES)
        or any(type(value) is not int or value < 0 for value in role.values())
        for role in (candidate_quotas, dev_quotas, test_quotas)
    ):
        raise ValueError("Expanded cohort quotas require nonnegative type counts")
    split_seed = plan["split_seed"]
    if (
        inventory.get("model_baseline") is not False
        or inventory.get("evidence_kind") != "reference-and-mutation-replay"
    ):
        raise ValueError("Only reference/mutation quality evidence may select data")
    old_order = {e["family_id"]: n for n, e in enumerate(historical["tasks"])}
    component = {
        member: tuple(group["members"])
        for group in review["independence_guard"]
        for member in group["members"]
    }
    quality_exclusions = {
        row["family_id"]: row["reason"] for row in review.get("quality_exclusions", [])
    }
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for entry in inventory["tasks"]:
        if (
            entry["status"] == "verified"
            and entry["variant_id"] == "base"
            and entry["family_id"] not in quality_exclusions
        ):
            key = component.get(entry["family_id"], (entry["family_id"],))
            groups.setdefault(key, []).append(entry)
    retained = []
    reserved = []
    novel = {}
    for key, entries in groups.items():
        old = [e for e in entries if e["family_id"] in old_order]
        if any(member in old_order for member in key):
            representative = min(
                old or entries,
                key=lambda e: (
                    e["task_type"] == "short_program",
                    old_order.get(e["family_id"], len(old_order)),
                    e["task_id"],
                ),
            )
            reserved.append(representative)
            if old:
                retained.append(representative)
        else:
            novel[key] = entries
    old_counts = Counter(e["task_type"] for e in retained)
    reserved_counts = Counter(e["task_type"] for e in reserved)
    total_quotas = {
        kind: candidate_quotas[kind] + dev_quotas[kind] + test_quotas[kind]
        for kind in TASK_TYPES
    }
    if any(reserved_counts[kind] > candidate_quotas[kind] for kind in TASK_TYPES):
        raise ValueError("Historical candidates exceed the configured quotas")
    needed = {kind: total_quotas[kind] - reserved_counts[kind] for kind in TASK_TYPES}
    available = {
        kind: sum(
            any(e["task_type"] == kind for e in entries) for entries in novel.values()
        )
        for kind in TASK_TYPES
    }
    rng = random.Random(split_seed)
    assignments: dict[str, list[dict[str, Any]]] = {}
    for kind in sorted(TASK_TYPES, key=lambda k: available[k] - needed[k]):
        keys = [
            key
            for key, entries in novel.items()
            if any(e["task_type"] == kind for e in entries)
        ]
        rng.shuffle(keys)
        keys.sort(key=lambda key: len({e["task_type"] for e in novel[key]}))
        if len(keys) < needed[kind]:
            raise ValueError(
                f"Independent verified {kind} deficit: {needed[kind] - len(keys)}"
            )
        assignments[kind] = []
        for key in keys[: needed[kind]]:
            entry = min(
                (e for e in novel.pop(key) if e["task_type"] == kind),
                key=lambda e: e["task_id"],
            )
            assignments[kind].append(entry)
    roles = {"training_candidates": list(reserved), "development": [], "test": []}
    for kind in TASK_TYPES:
        pool = assignments[kind]
        rng.shuffle(pool)
        train = candidate_quotas[kind] - reserved_counts[kind]
        dev = dev_quotas[kind]
        roles["training_candidates"].extend(pool[:train])
        roles["development"].extend(pool[train : train + dev])
        roles["test"].extend(pool[train + dev :])
    difficulty = {}
    public_rows = {}
    for source in inventory["source_artifacts"]:
        if source["dataset"] == "bashlet/public-python":
            catalog = json.loads((inventory_path.parent / source["file"]).read_text())
            public_rows.update(
                {
                    family["family_id"]: family["provenance"]["upstream_record"]
                    for family in catalog["families"]
                }
            )
    for entry in roles["training_candidates"]:
        task = json.loads((inventory_path.parent / entry["task_file"]).read_text())
        reference = task["reference_implementation"]
        if entry["family_id"] in public_rows:
            row = public_rows[entry["family_id"]]
            reference = row.get(
                "code", row.get("prompt", "") + row.get("canonical_solution", "")
            )
        tree = ast.parse(reference)
        statements = sum(isinstance(n, ast.stmt) for n in ast.walk(tree))
        branches = sum(
            isinstance(n, (ast.If, ast.For, ast.While, ast.comprehension))
            for n in ast.walk(tree)
        )
        difficulty[entry["task_id"]] = next(
            (
                level
                for level in ("easy", "medium")
                if statements <= plan["difficulty_thresholds"][level]["statements"]
                and branches <= plan["difficulty_thresholds"][level]["branches"]
            ),
            "hard",
        )
    cohort = {
        "version": "harness-task-cohort-v2",
        "name": plan["name"],
        **{
            group: {
                "task_ids": sorted(e["task_id"] for e in entries),
                "per_type": quotas,
            }
            for (group, entries), quotas in zip(
                roles.items(), (candidate_quotas, dev_quotas, test_quotas), strict=True
            )
        },
        "environment_variants": [],
        "independence_guard": review["independence_guard"],
        "split_seed": split_seed,
        "freeze": plan["freeze"] | {"difficulty": difficulty},
        "rules": plan["rules"],
    }
    validate_plan(cohort)
    kept = {e["family_id"] for e in retained}
    exclusions = [
        {
            "family_id": e["family_id"],
            "reason": quality_exclusions.get(
                e["family_id"],
                "Reviewed duplicate/template variant; historical bytes retained.",
            ),
            "component": list(component.get(e["family_id"], (e["family_id"],))),
        }
        for e in historical["tasks"]
        if e["family_id"] not in kept
    ]
    report = {
        "historical_manifest_sha256": hashlib.sha256(
            historical_path.read_bytes()
        ).hexdigest(),
        "quality_inventory_sha256": hashlib.sha256(
            inventory_path.read_bytes()
        ).hexdigest(),
        "family_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "generation_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "historical_entries": len(historical["tasks"]),
        "historical_retained": len(retained),
        "historical_exclusions": exclusions,
        "historical_per_type": dict(old_counts),
        "historical_components_reserved_to_candidates": len(reserved),
        "new_families": sum(total_quotas.values()) - len(retained),
        "roles": {k: len(v) for k, v in roles.items()},
        "source_quality_rejections": [
            e for e in inventory["tasks"] if e["status"] != "verified"
        ],
        "review_quality_exclusions": review.get("quality_exclusions", []),
        "model_baseline": False,
    }
    return cohort, report
