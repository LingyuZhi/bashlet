"""Consolidated data preserves source identities and frozen evaluation evidence."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from test_cli import cli

from bashlet.config import load_config
from bashlet.evaluation.checkpoints import _heldout_plan
from bashlet.tasks import sources
from bashlet.tasks.preparation import prepare_harness
from bashlet.tasks.public import prepare_public_catalog


def _pinned(value: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(value, indent=2) + "\n").encode()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "definition": value}


def test_combined_import_keeps_catalog_bytes_and_admits_custom_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {
        "task_id": 11,
        "text": "Return twice an integer.",
        "code": "def double(n):\n    return n * 2\n",
        "test_list": [
            "assert double(2) == 4",
            "assert double(-3) == -6",
            "assert double(0) == 0",
        ],
    }
    raw = (json.dumps(row) + "\n").encode()
    (tmp_path / "source.jsonl").write_bytes(raw)
    lock = {
        "dataset": "synthetic/mbpp",
        "revision": "fixture-v1",
        "format": "mbpp",
        "file": "source.jsonl",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (tmp_path / "upstream-v1.json").write_text(json.dumps(lock))
    plan: dict[str, Any] = {
        "version": "public-python-plan-v1",
        "dataset": "synthetic/public",
        "revision": "fixture-v1",
        "sources": [
            {"lock": "upstream-v1.json", "task_types": {"11": "short_program"}}
        ],
    }
    standalone = tmp_path / "standalone.json"
    standalone.write_bytes(sources.definition_bytes(plan))
    expected = prepare_public_catalog(
        plan_path=standalone, source_dir=tmp_path, output_dir=tmp_path / "standalone"
    )
    bundle: dict[str, Any] = {
        "version": "task-source-registry-v3",
        "sources": [],
        "locks": {"upstream-v1.json": lock},
        "imports": {"fixture": _pinned(plan)},
    }
    combined = tmp_path / "sources.json"
    combined.write_text(json.dumps(bundle))
    root = tmp_path / "artifacts"
    output = root / "imported"
    result = cli(
        "import-public-tasks",
        str(combined),
        "--catalog",
        "fixture",
        "--source-dir",
        str(tmp_path),
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    actual = output / "private/catalog.json"
    assert json.loads(actual.read_text())["families"]
    assert actual.read_bytes() == expected.read_bytes()
    assert (output / "private/plan.json").read_bytes() == standalone.read_bytes()

    bundle["locks"]["catalog"] = {
        "dataset": "synthetic/public",
        "revision": "fixture-v1",
        "artifact": "imported/private/catalog.json",
        "sha256": hashlib.sha256(actual.read_bytes()).hexdigest(),
    }
    bundle["sources"].append(
        {
            "format": "catalog",
            "kind": "public",
            "lock": "catalog",
            "optional_artifact": True,
        }
    )
    combined.write_text(json.dumps(bundle))
    monkeypatch.setattr(sources, "_DATA", tmp_path)
    monkeypatch.setattr(sources, "SOURCE_REGISTRY", combined)
    assert sources.admitted_sources() == {}
    with sources.preparing_sources(root):
        assert len(sources.admitted_sources()) == 1
        assert sources.source_file(bundle["locks"]["catalog"]) == actual
    assert sources.admitted_sources() == {}
    bundle["imports"]["fixture"]["definition"]["revision"] = "changed"
    combined.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="definition SHA-256"):
        prepare_public_catalog(
            plan_path=combined,
            catalog="fixture",
            source_dir=tmp_path,
            output_dir=tmp_path / "invalid",
        )
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("state", ["valid", "missing", "changed", "escape"])
def test_audit_artifacts_require_original_bytes_inside_the_selected_root(
    tmp_path: Path, state: str
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    content = b'{"review": "fixture"}\n'
    generation = {"version": "fixture-plan"}
    external = root / "review.json"
    if state == "escape":
        external = tmp_path / "outside.json"
    if state != "missing":
        external.write_bytes(b"changed" if state == "changed" else content)
    cohort = {
        "generation_plan": _pinned(generation),
        "audit_documents": [
            {"embedded": "generation_plan", "sha256": _pinned(generation)["sha256"]},
            {
                "artifact": "../outside.json" if state == "escape" else "review.json",
                "sha256": hashlib.sha256(content).hexdigest(),
            },
        ],
    }
    if state == "missing":
        with pytest.raises(FileNotFoundError, match="Missing cohort audit documents"):
            sources.audit_contents(cohort, root)
    elif state == "changed":
        with pytest.raises(ValueError, match="audit document content changed"):
            sources.audit_contents(cohort, root)
    elif state == "escape":
        with pytest.raises(ValueError, match="artifact root"):
            sources.audit_contents(cohort, root)
    else:
        verified = sources.audit_contents(cohort, root)
        assert [body for _, body in verified] == [
            sources.definition_bytes(generation),
            content,
        ]


def test_published_cohort_reports_all_missing_audits_before_creating_output(
    tmp_path: Path,
) -> None:
    path = Path(__file__).parents[1] / "data/cohort.json"
    cohort = json.loads(path.read_bytes())
    missing = [d for d in cohort["audit_documents"] if "artifact" in d]
    output = tmp_path / "prepared"
    with pytest.raises(FileNotFoundError) as error:
        prepare_harness(
            cohort_path=path,
            mbpp_manifest=None,
            config=load_config(),
            output_dir=output,
            artifact_root=tmp_path / "missing",
        )
    assert f"({len(missing)})" in str(error.value)
    assert all(d["artifact"] in str(error.value) for d in missing)
    assert not output.exists()


def test_heldout_plan_uses_frozen_order_and_rejects_overwritten_routing(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "complete",
                "source": {"revision": "frozen-fixture"},
                "cohort": {"test": {"task_ids": ["task/z", "task/a"]}},
            }
        )
    )
    output = tmp_path / "evaluation"
    path = _heldout_plan(manifest, output)
    plan = json.loads(path.read_bytes())
    assert plan["task_ids"] == ["task/z", "task/a"] and plan["split"] == "test"
    assert (path.parent / plan["manifest"]).resolve() == manifest
    original = path.read_bytes()
    assert _heldout_plan(manifest, output).read_bytes() == original
    plan["task_ids"].reverse()
    changed = json.dumps(plan).encode()
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="held-out evaluation plan changed"):
        _heldout_plan(manifest, output)
    assert path.read_bytes() == changed


@pytest.mark.parametrize(
    "status,identities",
    [("incomplete", ["task/a"]), ("complete", []), ("complete", ["task/a", "task/a"])],
)
def test_heldout_plan_refuses_incomplete_or_invalid_frozen_denominators(
    tmp_path: Path, status: str, identities: list[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": status,
                "source": {"revision": "fixture"},
                "cohort": {"test": {"task_ids": identities}},
            }
        )
    )
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError):
        _heldout_plan(manifest, output)
    assert not output.exists()
