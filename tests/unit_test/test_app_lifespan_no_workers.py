# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""task #17 hard gate #1 — API 进程不启动重型 indexing 执行面.

Pin the API/Worker hard cut invariant at the source level so a future
PR cannot regress by re-adding ``asyncio.create_task(run_*_worker)``
inside ``aperag/app.py:combined_lifespan``. The runtime hard cut is
the architectural promise that solved the Singapore 503 (huangzhangshu
task #13): the API process only handles HTTP routing + lightweight
enqueue, while ``indexing-worker`` deployment runs the actual workers
out-of-process via ``python -m aperag.cli.indexing_worker``.

These tests are the source-level grep gate that backs the deployment
runbook's hard gate #1 ("API readiness probe must remain stable while
graph indexing is under pressure"). They run in unit_test so they
execute in the standard PR-gate suite without any deployment.

The companion ``tests/unit_test/test_app_lifespan_no_workers.py``
asserts the *positive* contract on ``aperag/cli/indexing_worker.py``
— that file MUST start the workers + parse + reconciler + cleanup
loops, otherwise the worker deployment would boot but consume nothing.

Owners: nominally @Bryce per @huangheng task-17-cr-review-checklist
hard-gate-to-test mapping; @chenyexuan co-implementer covers this
slice (msg=6627eb69 in #indexing优化:5e959a2d) so Bryce can focus on
the task #17.4 ``document_service.delete`` cleanup migration.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PY = REPO_ROOT / "aperag" / "app.py"
CLI_WORKER_PY = REPO_ROOT / "aperag" / "cli" / "indexing_worker.py"

# Symbols that, if invoked from ``aperag/app.py``, would re-introduce
# the API-side worker startup that the hard cut removed. The list is
# the same set ``run_*`` entrypoints that ``aperag/cli/indexing_worker.py``
# now owns.
_BANNED_LIFESPAN_INVOCATIONS: tuple[str, ...] = (
    "run_vector_worker",
    "run_fulltext_worker",
    "run_graph_worker",
    "run_graph_facts_worker",
    "run_graph_vectors_worker",
    "run_summary_worker",
    "run_vision_worker",
    "run_parse_worker",
    "run_reconcile_loop",
    "run_cleanup_loop",
)

# ``ProductionWorkerFactory`` materialises real backends (Qdrant /
# Elasticsearch / embedders / completion model). Constructing it from
# the API process pulls those clients into the API event loop, which
# is the exact scenario the hard cut fixes. The CLI worker still
# constructs it (positive contract checked below).
_BANNED_FACTORY_CONSTRUCTOR = "ProductionWorkerFactory("

# ``IndexingRuntime.cleanup_worker_factory=...`` injection of a real
# factory would let service-layer ``Document.delete`` run heavy
# backend cleanup synchronously inside an API request — the exact
# behaviour ziang msg=cecb0d88 + huangheng msg=f97b7c5f #6 banned.
# After the hard cut the API constructs ``IndexingRuntime`` with
# ``cleanup_worker_factory=None``; this regex catches any other
# concrete value being assigned in app.py.
_CLEANUP_FACTORY_ASSIGNMENT_RE = re.compile(
    r"cleanup_worker_factory\s*=\s*(?!None\b)([A-Za-z_]\w*)",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------
# Negative contract — aperag/app.py must not start workers.
# ---------------------------------------------------------------------


def test_app_lifespan_does_not_invoke_worker_entrypoints():
    """No ``run_*_worker`` / ``run_*_loop`` call inside ``aperag/app.py``.

    Catches accidental reintroduction of any of the ten worker
    entrypoints into the lifespan. A regression here would re-merge
    the API and worker runtimes — the Singapore 503 root cause.
    """
    src = _read(APP_PY)
    offenders = [name for name in _BANNED_LIFESPAN_INVOCATIONS if f"{name}(" in src]
    assert not offenders, (
        f"aperag/app.py invokes worker entrypoints that the task #17 "
        f"hard cut moved to aperag/cli/indexing_worker.py: {offenders}. "
        "If you intentionally re-introduce a worker into the API "
        "process, the API/worker isolation contract is broken — "
        "see docs/zh-CN/architecture/task-system-hard-cut-v8.md."
    )


def test_app_lifespan_does_not_construct_production_worker_factory():
    """API process must not construct ``ProductionWorkerFactory``.

    The factory pulls in heavy backend clients; constructing it from
    the API event loop is what the hard cut prevents.
    """
    src = _read(APP_PY)
    assert _BANNED_FACTORY_CONSTRUCTOR not in src, (
        "aperag/app.py constructs ProductionWorkerFactory(...). The "
        "task #17 hard cut requires this construction to live only "
        "in aperag/cli/indexing_worker.py — see "
        "docs/zh-CN/architecture/task-system-hard-cut-v8.md."
    )


def test_app_lifespan_runtime_uses_no_cleanup_factory():
    """``IndexingRuntime(cleanup_worker_factory=...)`` in ``app.py``
    must be ``None`` so service-layer ``Document.delete`` is forced
    onto the worker-side cleanup loop (ziang msg=cecb0d88 / huangheng
    msg=f97b7c5f #6 hard gate)."""
    src = _read(APP_PY)
    matches = _CLEANUP_FACTORY_ASSIGNMENT_RE.findall(src)
    assert not matches, (
        "aperag/app.py assigns cleanup_worker_factory to a concrete "
        f"value ({matches}). After the task #17 hard cut the API must "
        "use cleanup_worker_factory=None so the API request path "
        "cannot run heavy backend cleanup."
    )


# ---------------------------------------------------------------------
# Positive contract — aperag/cli/indexing_worker.py must start the
# workers the hard cut moved off the API.
# ---------------------------------------------------------------------


def test_cli_worker_starts_every_runtime_loop():
    """The hard cut is symmetrical: every entrypoint that ``app.py``
    no longer invokes must be invoked by the CLI worker, otherwise
    the worker deployment boots but consumes no queues."""
    src = _read(CLI_WORKER_PY)
    missing = [name for name in _BANNED_LIFESPAN_INVOCATIONS if f"{name}(" not in src]
    assert not missing, (
        f"aperag/cli/indexing_worker.py is missing invocations of "
        f"{missing}. The hard cut moved these off the API but they "
        "must still run somewhere — the CLI worker is that home. "
        "Without them the indexing-worker deployment would not "
        "consume the corresponding queues."
    )


def test_cli_worker_constructs_production_worker_factory():
    """Symmetric positive: the CLI worker must construct the factory
    the API no longer constructs."""
    src = _read(CLI_WORKER_PY)
    assert _BANNED_FACTORY_CONSTRUCTOR in src, (
        "aperag/cli/indexing_worker.py must construct "
        "ProductionWorkerFactory(...) — it is the worker-side owner "
        "of the heavy backend clients after the task #17 hard cut."
    )
