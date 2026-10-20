"""Convert pinned Python problems without evaluating source code on the host."""

import ast
import gzip
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

from bashlet.harness.protocol import fingerprint
from bashlet.tasks.mbpp import _download, _split
from bashlet.tasks.sources import TASK_TYPES, WORKSPACE_GRADING_DRIVER

HEADER = "from typing import *\n"
FUNCTION_RUNNER = """import ast, contextlib, importlib.util, io, random, sys
from pathlib import Path
random.seed(0)
args, kwargs = ast.literal_eval(sys.stdin.read())
path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location('_student_implementation', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(module)
    result = getattr(module, sys.argv[2])(*args, **kwargs)
print(_literal(result))
"""


def _literal(value: Any) -> str:
    """Preserve Python types without process-dependent set representations."""
    if isinstance(value, set):
        return (
            "{" + ", ".join(sorted(_literal(v) for v in value)) + "}"
            if value
            else "set()"
        )
    if isinstance(value, tuple):
        return (
            "("
            + ", ".join(_literal(v) for v in value)
            + ("," if len(value) == 1 else "")
            + ")"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(_literal(k) + ": " + _literal(v) for k, v in value.items())
            + "}"
        )
    return repr(value)


def _example(assertion: str, name: str) -> tuple[tuple[Any, ...], Any]:
    node = ast.parse(assertion).body[0]
    if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
        raise ValueError("Expected a literal equality example")
    comparison = node.test
    call = comparison.left
    if (
        len(comparison.ops) != 1
        or not isinstance(comparison.ops[0], ast.Eq)
        or not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != name
        or call.keywords
    ):
        raise ValueError("Expected one tested function with literal arguments")
    try:
        return tuple(ast.literal_eval(arg) for arg in call.args), ast.literal_eval(
            comparison.comparators[0]
        )
    except (ValueError, TypeError) as error:
        raise ValueError(
            "Non-literal source example: " + ast.unparse(node.test)
        ) from error


def _broken(reference: str, name: str) -> str:
    tree = ast.parse(reference)
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    returns = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Return) and n.value is not None
    ]
    if not returns:
        raise ValueError("No return path to mutate in the existing implementation")
    returns[0].value = ast.Constant(None)
    return ast.unparse(tree) + "\n"


def _problem(
    row: dict[str, Any], source: dict[str, Any]
) -> tuple[str, str, str, str, tuple[Any, ...], Any]:
    if source["format"] == "mbpp":
        tests = row["test_list"]
        if len(tests) < 3:
            raise ValueError("At least three source assertions are required")
        call = ast.parse(tests[0]).body[0]
        assert isinstance(call, ast.Assert) and isinstance(call.test, ast.Compare)
        function = call.test.left
        if not isinstance(function, ast.Call) or not isinstance(
            function.func, ast.Name
        ):
            raise ValueError("Unsupported source assertion")
        name = function.func.id
        examples = [_example(test, name) for test in tests]
        if len({repr(args) for args, _ in examples}) != len(examples):
            raise ValueError("Source assertions must use independent inputs")
        args, expected = examples[0]
        reference = HEADER + row["code"].replace("\r\n", "\n") + "\n"
        hidden = (
            row.get("test_setup_code", "")
            + "\n"
            + "\n".join(tests[1:] + row.get("challenge_test_list", []))
            + "\n"
        )
        return name, row["text"], reference, hidden, args, expected
    if source["format"] != "humaneval":
        raise ValueError("Unsupported public Python source format")
    name = row["entry_point"]
    tree = ast.parse(row["test"])
    assertions = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    examples = []
    for node in assertions:
        try:
            examples.append(_example(ast.unparse(node), "candidate"))
        except (ValueError, TypeError, SyntaxError):
            continue
    if not examples:
        raise ValueError("No literal feedback example in the native tests")
    args, expected = examples[0]
    reference = row["prompt"] + row["canonical_solution"]
    return (
        name,
        row["prompt"],
        reference,
        row["test"] + "\ncheck(candidate)\n",
        args,
        expected,
    )


def _function_proxy(name: str) -> str:
    renderer = ast.unparse(ast.parse(inspect.getsource(_literal))) + "\n"
    runner = "from typing import Any\n" + renderer + FUNCTION_RUNNER
    return (
        "import ast as _grading_ast, subprocess as _grading_subprocess\n"
        "import sys as _grading_sys, random as _grading_random\n"
        "_grading_random.seed(0)\n"
        "def _grading_candidate(*args, **kwargs):\n"
        "    result = _grading_subprocess.run([_grading_sys.executable, '-I', '-B', "
        f"'-c', {runner!r}, 'solution.py', {name!r}], "
        "input=repr((args, kwargs)), text=True, capture_output=True, "
        "timeout=20, check=True)\n"
        "    return _grading_ast.literal_eval(result.stdout)\n"
        f"{name} = candidate = _grading_candidate\n"
    )


def _puzzle(
    instances: list[dict[str, Any]],
    source: dict[str, Any],
    kind: str,
    requirement: str,
    contract: str,
) -> dict[str, Any]:
    """Group native P3 parameter instances into one construction family."""
    if kind != "short_program" or len(instances) < 3:
        raise ValueError("P3 requires a short program and three native instances")
    original = instances[0]["name"].split(":")[0]
    if any(
        row["name"].split(":")[0] != original or row["module"] == "human_eval.py"
        for row in instances
    ):
        raise ValueError("P3 families cannot mix identities or duplicate HumanEval")
    cases = []
    parameter_names = None
    for row in instances:
        function = ast.parse(row["sol_header"] + "\n    pass\n").body[0]
        assert isinstance(function, ast.FunctionDef)
        names = [arg.arg for arg in function.args.args]
        if len(names) != len(function.args.defaults) or (
            parameter_names is not None and names != parameter_names
        ):
            raise ValueError("P3 instances require a consistent literal interface")
        parameter_names = names
        cases.append(tuple(ast.literal_eval(n) for n in function.args.defaults))
    if len({_literal(args) for args in cases}) < 3:
        raise ValueError("P3 quality checks require distinct parameter instances")
    first = instances[0]
    reference = (
        HEADER
        + first["sol_header"].replace("def sol(", "def solve(", 1)
        + "\n"
        + first["sol_bodies"][0]
        + "\n"
    )
    feedback = (
        HEADER
        + first["sat"]
        + "\n"
        + contract
        + "\nfrom solution import solve\n"
        + f"args = {_literal(cases[0])}\nanswer = solve(*args)\n"
        + "_grading_contract(answer, args)\nassert sat(answer, *args) is True\n"
    )
    hidden = HEADER + _function_proxy("solve") + contract + "\n"
    for row, args in zip(instances, cases, strict=True):
        hidden += (
            row["sat"]
            + "\n"
            + f"args = {_literal(args)}\n"
            + "answer = candidate(*args)\n_grading_contract(answer, args)\n"
            + "assert sat(answer, *args) is True\n"
        )
    family = "p3/" + original.lower()
    return {
        "original_task_id": original,
        "family_id": family,
        "split": "unsplit",
        "task_type": kind,
        "provenance": {
            "source": source,
            "upstream_instances": instances,
            "upstream_instances_sha256": fingerprint(instances),
            "conversion": "p3-construction-v1",
            "grading_seed": 0,
            "author": "P3 upstream authors; interface conversion by Codex",
            "generator": {"model": "gpt-6", "model_revision": None},
        },
        "variants": [
            {
                "variant_id": "base",
                "task": {
                    "task_id": "public/" + family,
                    "prompt": requirement
                    + "\n\nImplement solve("
                    + ", ".join(parameter_names or [])
                    + ") in solution.py. "
                    "The public predicate and contract in feedback_tests.py define "
                    "valid outputs for the supplied parameters. "
                    "Run python3 feedback_tests.py.",
                    "initial_files": {"feedback_tests.py": feedback},
                    "grader": {
                        "command": "python3 _grading/check.py",
                        "hidden_files": {"_grading/check.py": hidden},
                    },
                    "reference_implementation": reference,
                    "implementation_paths": ["solution.py"],
                    "feedback_command": "python3 feedback_tests.py",
                },
                "wrong_files": {
                    "solution.py": reference + "\n_original_solve = solve\n"
                    "def solve(*args):\n    return _original_solve()\n"
                },
                "wrong_reason": (
                    "Always solves the first public instance; "
                    "independent native parameter instances reject it."
                ),
            }
        ],
    }


def convert_problem(
    row: dict[str, Any], source: dict[str, Any], kind: str
) -> dict[str, Any]:
    """Each conversion retains one upstream family, regardless of task type."""
    if kind not in TASK_TYPES:
        raise ValueError("Unknown public task type")
    name, description, reference, hidden, args, expected = _problem(row, source)
    renderer = ast.unparse(ast.parse(inspect.getsource(_literal))) + "\n"
    arguments, answer = _literal(args), _literal(expected)
    original = str(row["task_id"])
    family = f"mbpp/{original}" if source["format"] == "mbpp" else original.lower()
    split = _split(int(original)) if source["format"] == "mbpp" else "test"
    if split == "prompt":
        raise ValueError("Upstream few-shot prompts remain reserved")
    identity = "public/" + family
    implementation = "solution.py"
    prompt = (
        description
        + f"\n\nImplement {name} in solution.py. Run python3 feedback_tests.py."
    )
    initial = {
        "feedback_tests.py": (
            f"from solution import {name}\nassert {name}(*{arguments}) == {answer}\n"
        )
    }
    grader = _function_proxy(name) + hidden
    wrong = (
        f"def {name}(*args, **kwargs):\n"
        f"    return {answer} if args == {arguments} else None\n"
    )
    reason = "Memorizes the public input; independent native assertions reject it."
    if kind == "code_edit":
        initial[implementation] = _broken(reference, name)
        prompt = (
            description
            + f"\n\nRepair the existing {name} implementation in solution.py. "
            "Run python3 feedback_tests.py."
        )
        wrong = initial[implementation]
        reason = (
            "The first return path in the existing implementation loses its result."
        )
    elif kind == "workspace":
        implementation = "app/processor.py"
        interface = (
            "Read data/work_items.txt with ast.literal_eval. "
            "It contains a list of argument tuples. "
            "Apply the requirement in docs/requirement.md to every tuple, in order. "
            "Write repr(the_list_of_results) to output/results.txt. "
            "Keep the task inputs and feedback check unchanged."
        )
        prompt = (
            "Implement app/processor.py using the local requirements and work items. "
            + interface
            + " Run python3 feedback_tests.py."
        )
        reference += (
            "\nfrom typing import Any\n"
            + renderer
            + (
                "\nif __name__ == '__main__':\n"
                "    import ast\n    from pathlib import Path\n"
                "    items = ast.literal_eval("
                "Path('data/work_items.txt').read_text())\n"
                f"    results = [{name}(*args) for args in items]\n"
                "    Path('output').mkdir(exist_ok=True)\n"
                "    Path('output/results.txt').write_text(_literal(results))\n"
            )
        )
        initial = {
            "docs/requirement.md": description + "\n\n" + interface + "\n",
            "data/work_items.txt": _literal([args]) + "\n",
            "feedback_tests.py": (
                "import ast, runpy\nfrom pathlib import Path\n"
                "runpy.run_path('app/processor.py', run_name='__main__')\n"
                "assert ast.literal_eval(Path('output/results.txt').read_text()) "
                f"== {_literal([expected])}\n"
            ),
        }
        grader = (
            "import ast, subprocess, sys, random\nfrom pathlib import Path\n"
            "random.seed(0)\n"
            "def candidate(*args):\n"
            "    Path('data/work_items.txt').write_text(repr([args]))\n"
            "    Path('output/results.txt').unlink(missing_ok=True)\n"
            "    subprocess.run([sys.executable, 'app/processor.py'], "
            "check=True, capture_output=True, timeout=20)\n"
            "    result = ast.literal_eval(Path('output/results.txt').read_text())\n"
            "    assert isinstance(result, list) and len(result) == 1\n"
            "    return result[0]\n"
            f"{name} = candidate\n"
        ) + hidden
        wrong = (
            "from pathlib import Path\nPath('output').mkdir(exist_ok=True)\n"
            f"Path('output/results.txt').write_text({_literal([expected])!r})\n"
        )
        reason = "Hardcodes the visible work item; fails when local task inputs change."
    return {
        "original_task_id": family,
        "family_id": family,
        "split": split,
        "task_type": kind,
        "provenance": {
            "source": source,
            "upstream_task_id": original,
            "upstream_record_sha256": fingerprint(row),
            "upstream_record": row,
            "conversion": "public-python-v5",
            "grading_seed": 0,
            "author": "upstream authors; conversion by Codex",
            "generator": {
                "model": "gpt-6",
                "model_revision": None,
                "identity_status": (
                    "system model family only; deployment revision unavailable"
                ),
            },
        },
        "variants": [
            {
                "variant_id": "base",
                "task": {
                    "task_id": identity,
                    "prompt": prompt,
                    "initial_files": initial,
                    "grader": {
                        "command": "python3 _grading/check.py",
                        "hidden_files": {
                            "_grading/check.py": (
                                WORKSPACE_GRADING_DRIVER
                                if kind == "workspace"
                                else (
                                    "import runpy\n"
                                    "runpy.run_path('_grading/cases.py', "
                                    "run_name='__main__')\n"
                                )
                            ),
                            "_grading/cases.py": grader,
                        },
                    },
                    "reference_implementation": reference,
                    "implementation_paths": [implementation],
                    "feedback_command": "python3 feedback_tests.py",
                },
                "wrong_files": {implementation: wrong},
                "wrong_reason": reason,
            }
        ],
    }


def prepare_public_catalog(
    *,
    plan_path: Path,
    source_dir: Path | None,
    output_dir: Path,
    catalog: str | None = None,
) -> Path:
    """Materialize a private, reproducible catalog; execution is prepare-harness."""
    from bashlet.tasks.sources import import_definition

    plan, plan_bytes, locks = import_definition(plan_path, catalog)
    if plan["version"] != "public-python-plan-v1":
        raise ValueError("Unsupported public import plan")
    output_dir.mkdir(parents=True, mode=0o700)
    private = output_dir / "private"
    private.mkdir(mode=0o700)
    upstream = private / "upstream"
    upstream.mkdir(mode=0o700)
    families, rejected, sources = [], [], []
    for item in plan["sources"]:
        source = (
            locks[item["lock"]]
            if locks
            else json.loads((plan_path.parent / item["lock"]).read_text())
        )
        raw = (
            _download(source["url"])
            if source_dir is None
            else (source_dir / source["file"]).read_bytes()
        )
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("Public source content SHA-256 mismatch")
        (upstream / source["file"]).write_bytes(raw)
        content = gzip.decompress(raw) if source["file"].endswith(".gz") else raw
        rows: dict[str, Any] = {}
        if source["format"] == "p3":
            for row in json.loads(content):
                rows.setdefault(row["name"].split(":")[0], []).append(row)
        else:
            rows = {
                str(row["task_id"]): row
                for line in content.decode().splitlines()
                if (row := json.loads(line))
            }
        sources.append(source)
        for original, kind in item["task_types"].items():
            try:
                family = (
                    _puzzle(
                        rows[original],
                        source,
                        kind,
                        item["requirements"][original],
                        item["contracts"][original],
                    )
                    if source["format"] == "p3"
                    else convert_problem(rows[original], source, kind)
                )
                family["variants"][0]["task"]["task_id"] += plan.get(
                    "task_id_suffix", ""
                )
                families.append(family)
            except (
                ValueError,
                TypeError,
                SyntaxError,
                AssertionError,
                StopIteration,
            ) as error:
                rejected.append(
                    {
                        "source": source["dataset"],
                        "task_id": original,
                        "reason": str(error) or type(error).__name__,
                    }
                )
    catalog_record = {
        "dataset": plan["dataset"],
        "revision": plan["revision"],
        "families": families,
        "upstream_sources": sources,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
    }
    usage = {
        "conversion_attempts": sum(len(s["task_types"]) for s in plan["sources"]),
        "converted_families": len(families),
        "rejected": rejected,
        "paid_model_calls": 0,
    }
    destination = private / "catalog.json"
    destination.write_text(
        json.dumps(catalog_record, indent=2, ensure_ascii=True) + "\n"
    )
    (private / "usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    (private / "plan.json").write_bytes(plan_bytes)
    return destination
