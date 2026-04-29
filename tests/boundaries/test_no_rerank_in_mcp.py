"""Boundary gate for task #32 / task #35: MCP must not expose rerank.

Rerank was removed because ApeRAG now exposes separate MCP tools for
vector, fulltext, and graph retrieval. Each index owns its native
ranking semantics; there is no cross-index fusion layer where rerank
would be a natural default. This test keeps the public MCP surface and
search request schema from drifting back to the old per-index rerank
contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_ROOT = REPO_ROOT / "aperag" / "mcp"
MCP_TOOLS_ROOT = MCP_ROOT / "tools"
RETRIEVAL_SCHEMAS = REPO_ROOT / "aperag" / "domains" / "retrieval" / "schemas.py"


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _format_paths(paths: list[Path]) -> str:
    return "\n".join(f"- {path.relative_to(REPO_ROOT)}" for path in paths)


def _search_request_field_names() -> set[str]:
    source = RETRIEVAL_SCHEMAS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SearchRequest":
            fields: set[str] = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.add(stmt.target.id)
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            fields.add(target.id)
            return fields
    raise AssertionError(f"SearchRequest class not found in {RETRIEVAL_SCHEMAS}")


def test_mcp_python_surface_has_no_rerank_token():
    """No MCP module should mention rerank in signatures, payloads, or docs."""

    offenders = [path for path in _python_files(MCP_ROOT) if "rerank" in path.read_text(encoding="utf-8").lower()]
    assert not offenders, (
        "MCP modules must not expose or document rerank. Rerank belongs to no "
        "single-index MCP tool after task #35 removal. Offending files:\n"
        f"{_format_paths(offenders)}"
    )


def test_mcp_tool_functions_do_not_accept_rerank_parameter():
    """AST-level guard: MCP tool functions must not reintroduce rerank args."""

    offenders: list[str] = []
    for path in _python_files(MCP_TOOLS_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            arg_names = [arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)]
            if "rerank" in arg_names:
                offenders.append(f"{path.relative_to(REPO_ROOT)}::{node.name}")

    assert not offenders, (
        "MCP tool signatures must not accept `rerank`; single-index search tools "
        "use their native ranking semantics. Offenders:\n" + "\n".join(f"- {item}" for item in offenders)
    )


def test_search_request_schema_has_no_rerank_field():
    """Backend search schema must not keep a hidden rerank passthrough."""

    fields = _search_request_field_names()
    assert "rerank" not in fields, (
        "SearchRequest.rerank must stay removed. If MCP tools do not expose "
        "rerank but the backend schema keeps accepting it, the public contract "
        "can drift back through generated clients or ad-hoc callers."
    )
