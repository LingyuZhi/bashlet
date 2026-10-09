"""Content-based family candidates and version-scoped independence guards."""

import ast
import builtins
import difflib
import re
from collections import defaultdict
from itertools import combinations
from typing import Any


class _Names(ast.NodeTransformer):
    def __init__(self, *, constants: bool) -> None:
        self.names: dict[str, str] = {}
        self.constants = constants

    def _name(self, name: str) -> str:
        if name in vars(builtins):
            return name
        return self.names.setdefault(name, f"v{len(self.names)}")

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._name(node.arg)
        node.annotation = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = self._name(node.name)
        node.returns = None
        node.body = [
            n
            for n in node.body
            if not (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            )
        ]
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if self.constants:
            node.value = type(node.value).__name__
        return node


def family_candidates(records: list[dict[str, str]]) -> dict[str, Any]:
    """Surface aliases and template reuse; similarity is not a semantic verdict."""
    normalized: dict[str, list[str]] = defaultdict(list)
    templates: dict[str, list[str]] = defaultdict(list)
    words = {}
    for row in records:
        identity = row["family_id"]
        for replace_constants, target in ((False, normalized), (True, templates)):
            try:
                tree = _Names(constants=replace_constants).visit(
                    ast.parse(row["reference"])
                )
                target[ast.dump(tree, include_attributes=False)].append(identity)
            except SyntaxError:
                pass
        text = row["description"].lower()
        for pattern, replacement in (
            (r"greatest common divisor|highest common factor", "gcd"),
            (r"least common multiple|lowest common multiple", "lcm"),
            (r"fibbonacci", "fibonacci"),
            (r"\barray\b", "list"),
            (r"\bdistinct\b", "unique"),
            (r"\bsmallest\b", "minimum"),
            (r"\blargest\b", "maximum"),
        ):
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"write a (?:python )?function (?:to |that |which )?", "", text)
        tokens = {
            word[:-1] if len(word) > 4 and word.endswith("s") else word
            for word in re.findall(r"[a-z]{3,}", text)
        }
        words[identity] = tokens - {
            "given",
            "list",
            "string",
            "number",
            "find",
            "return",
            "input",
            "output",
            "sample",
            "example",
            "the",
            "and",
            "for",
            "with",
            "python",
            "function",
            "using",
            "integer",
            "value",
            "check",
            "whether",
            "not",
            "calculate",
            "compute",
            "get",
            "determine",
            "true",
            "false",
        }
    near = []
    by_id = {row["family_id"]: row for row in records}
    for left, right in combinations(by_id, 2):
        union = words[left] | words[right]
        overlap = len(words[left] & words[right]) / len(union) if union else 0
        if overlap >= 0.55:
            similarity = difflib.SequenceMatcher(
                None,
                by_id[left]["description"].lower(),
                by_id[right]["description"].lower(),
            ).ratio()
            near.append(
                {
                    "members": [left, right],
                    "word_overlap": overlap,
                    "text_similarity": similarity,
                }
            )
    return {
        "version": "family-content-audit-v2",
        "families_examined": len(by_id),
        "normalized_reference_groups": [v for v in normalized.values() if len(v) > 1],
        "template_groups": [v for v in templates.values() if len(v) > 1],
        "near_descriptions": sorted(near, key=lambda row: -row["word_overlap"]),
        "limitations": (
            "Lexical and syntax checks surface candidates; semantic review must "
            "record decisions. Public adapters do not create families."
        ),
    }


def validate_independence(cohort: dict[str, Any], families: list[str]) -> None:
    """Apply newly reviewed equivalences only to this version, preserving history."""
    for group in cohort.get("independence_guard", []):
        members = set(group["members"])
        if sum(identity in members for identity in families) > 1:
            raise ValueError("Known equivalent families cannot increase a frozen pool")
