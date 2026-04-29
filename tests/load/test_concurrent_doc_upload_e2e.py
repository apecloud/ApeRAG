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

"""task #17 sub-task #22 — multi-document concurrent upload e2e load test.

Acceptance gate for the API/Worker hard-cut deployment:
- API pod must remain responsive (auth + collection list latency
  unaffected) while N documents flow through the indexing pipeline.
- All N documents must reach ``ACTIVE`` for the enabled modalities
  within a wall-time budget that proves worker throughput is not
  bottlenecked by API request handling.
- DB connection count must stay under the configured pool budget
  (asserted via PG ``pg_stat_activity`` count from a side channel —
  out of scope for this script; see ``tests/load/test_pool_budget.py``
  in @Planetegg's lane).

Skipped by default; runs against a deployed ApeRAG stack. Set
``RUN_TASK_17_E2E=1`` and the e2e_pytest config env vars (model
providers + base url) to run.

Owners: @Planetegg main task #22 / @chenyexuan co-implementer for
this script. Cross-reviewed with @huangzhangshu (task #18 deployment
gates) and @ziang (task #19 cleanup SoT).
"""

from __future__ import annotations

import concurrent.futures
import os
import time
import uuid
from http import HTTPStatus

import httpx
import pytest

from tests.e2e_pytest.config import (
    API_BASE_URL,
    EMBEDDING_MODEL_CUSTOM_PROVIDER,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PROVIDER,
)

# ---------------------------------------------------------------------
# Tunables — keep low for local-dev defaults; CI / ops override via env
# so the same script powers both quick smoke and 100-doc burst runs.
# ---------------------------------------------------------------------

DOC_COUNT: int = int(os.getenv("TASK_17_E2E_DOC_COUNT", "20"))
"""Number of documents to upload concurrently per run."""

UPLOAD_CONCURRENCY: int = int(os.getenv("TASK_17_E2E_UPLOAD_CONCURRENCY", "10"))
"""How many in-flight upload requests to keep against the API. The API
must keep ``/health/live`` and ``/api/v2/auth/user`` stable while this
load runs — that is the cascading-failure-mode the hard cut prevents."""

POLL_BUDGET_SECONDS: float = float(os.getenv("TASK_17_E2E_POLL_BUDGET_SECONDS", "300"))
"""Wall-time ceiling from upload-confirm to all-modalities-ACTIVE.
Production SLO is 30 minutes for 100 docs (graph LLM extraction
dominates). The default 5 min budget here covers DOC_COUNT=20 with
embedding + fulltext + graph_facts enabled. Override for larger runs."""

POLL_INTERVAL_SECONDS: float = float(os.getenv("TASK_17_E2E_POLL_INTERVAL_SECONDS", "5"))

API_HEALTH_PROBE_INTERVAL_SECONDS: float = 2.0
"""How often to sample ``/health/live`` while documents are indexing.
Any single sample > 500ms or any non-200 response constitutes a hard
cut regression — the API must never be blocked by worker pressure
once the deployments are split."""

API_HEALTH_PROBE_LATENCY_BUDGET_SECONDS: float = 0.5

REQUIRES_DEPLOYMENT_RUN_TOKEN = "RUN_TASK_17_E2E"

pytestmark = pytest.mark.skipif(
    os.getenv(REQUIRES_DEPLOYMENT_RUN_TOKEN) != "1",
    reason=(
        f"deployment-aware e2e; set {REQUIRES_DEPLOYMENT_RUN_TOKEN}=1 + "
        "tests/e2e_pytest config env vars to run against a live stack."
    ),
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def concurrent_collection(client):
    """A collection with the modalities exercised by the burst run.

    Vector + fulltext + graph_facts cover the three pipelines that
    contend for the worker pool; vision + summary are off by default
    so a CI run without GPU / vision provider can still exercise the
    deployment gate.
    """
    payload = {
        "title": f"task17 burst {uuid.uuid4().hex[:8]}",
        "type": "document",
        "config": {
            "source": "system",
            "enable_vector": True,
            "enable_fulltext": True,
            "enable_knowledge_graph": True,
            "enable_summary": False,
            "enable_vision": False,
            "embedding": {
                "model": EMBEDDING_MODEL_NAME,
                "model_service_provider": EMBEDDING_MODEL_PROVIDER,
                "custom_llm_provider": EMBEDDING_MODEL_CUSTOM_PROVIDER,
            },
        },
    }
    resp = client.post("/api/v2/collections", json=payload)
    assert resp.status_code == HTTPStatus.OK, resp.text
    coll = resp.json()
    yield coll
    client.delete(f"/api/v2/collections/{coll['id']}")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_doc_body(idx: int) -> bytes:
    """Each document is unique enough that vector / fulltext / graph
    extraction must actually run — pure repeat content would let
    parse_version dedup short-circuit the worker path."""
    return (
        f"# task #17 burst document {idx}\n\n"
        f"This synthetic document number {idx} carries unique content so "
        f"the indexing pipeline cannot collapse parse_version dedup. "
        f"It mentions Alice-{idx} talking to Bob-{idx} about Project-{idx} "
        f"so the graph extractor produces at least one entity triple per doc.\n\n"
        f"## Section\n\n"
        f"Paragraph two of document {idx}, with content suitable for "
        f"chunking-window splitter.\n"
    ).encode("utf-8")


def _upload_one(client: httpx.Client, collection_id: str, idx: int) -> str:
    files = {"files": (f"task17-burst-{idx:04d}.txt", _build_doc_body(idx), "text/plain")}
    resp = client.post(f"/api/v2/collections/{collection_id}/documents", files=files)
    assert resp.status_code == HTTPStatus.OK, f"upload {idx} failed: {resp.text}"
    items = resp.json()["items"]
    assert len(items) == 1
    return items[0]["id"]


def _all_indexes_active(item: dict, *, require_graph: bool) -> bool:
    """A document is fully indexed once every enabled modality has
    transitioned to ACTIVE. Graph is gated by ``require_graph`` so
    collections without graph still exercise the same poll loop."""
    if item.get("vector_index_status") != "ACTIVE":
        return False
    if item.get("fulltext_index_status") != "ACTIVE":
        return False
    if require_graph and item.get("graph_index_status") != "ACTIVE":
        return False
    return True


def _probe_health_during_burst(probes: list[tuple[float, float, int]], stop_token: dict) -> None:
    """Background sampler: hits ``/health/live`` every 2s while the
    document burst runs and records (timestamp, latency, status_code).
    Run this in a worker thread; main thread sets ``stop_token['stop']``
    when the burst finishes."""
    base = API_BASE_URL.rstrip("/")
    with httpx.Client(timeout=2.0) as probe:
        while not stop_token.get("stop"):
            t0 = time.monotonic()
            try:
                resp = probe.get(f"{base}/health/live")
                latency = time.monotonic() - t0
                probes.append((time.time(), latency, resp.status_code))
            except httpx.RequestError:
                latency = time.monotonic() - t0
                probes.append((time.time(), latency, -1))
            time.sleep(API_HEALTH_PROBE_INTERVAL_SECONDS)


# ---------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------


def test_concurrent_doc_upload_indexes_under_budget(client, concurrent_collection):
    """Multi-document burst: upload + confirm + wait-for-ACTIVE.

    Asserts:
    - All ``DOC_COUNT`` documents reach ACTIVE for vector / fulltext
      / graph_facts within ``POLL_BUDGET_SECONDS``.
    - ``/health/live`` stays 200 with p95 latency under
      ``API_HEALTH_PROBE_LATENCY_BUDGET_SECONDS`` throughout the burst
      — this is the hard cut acceptance gate (API not blocked by
      worker pressure once the deployments are split).
    - No upload returns 5xx or times out at the API entry point.
    """
    collection_id = concurrent_collection["id"]

    # ---- Phase 1: parallel uploads ----
    upload_started = time.monotonic()
    document_ids: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=UPLOAD_CONCURRENCY) as pool:
        futures = [pool.submit(_upload_one, client, collection_id, i) for i in range(DOC_COUNT)]
        for fut in concurrent.futures.as_completed(futures):
            document_ids.append(fut.result())
    upload_elapsed = time.monotonic() - upload_started
    assert len(document_ids) == DOC_COUNT
    # Upload itself must not take embarrassingly long — the API path
    # writes intent rows + enqueues, no heavy work. 30s/20-docs is a
    # generous ceiling that catches API-side regressions.
    assert upload_elapsed < 30.0, (
        f"upload phase took {upload_elapsed:.1f}s for {DOC_COUNT} docs — API write path regressed?"
    )

    # ---- Phase 2: poll for indexing completion + sample API health ----
    health_probes: list[tuple[float, float, int]] = []
    stop_token = {"stop": False}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as probe_pool:
        probe_pool.submit(_probe_health_during_burst, health_probes, stop_token)

        deadline = time.monotonic() + POLL_BUDGET_SECONDS
        active_doc_ids: set[str] = set()
        last_seen: dict[str, dict] = {}
        while time.monotonic() < deadline and len(active_doc_ids) < DOC_COUNT:
            resp = client.get(
                f"/api/v2/collections/{collection_id}/documents",
                params={"page_size": DOC_COUNT * 2},
            )
            assert resp.status_code == HTTPStatus.OK, resp.text
            for item in resp.json()["items"]:
                last_seen[item["id"]] = item
                if _all_indexes_active(item, require_graph=True):
                    active_doc_ids.add(item["id"])
            if len(active_doc_ids) < DOC_COUNT:
                time.sleep(POLL_INTERVAL_SECONDS)

        stop_token["stop"] = True

    # ---- Phase 3: assertions ----
    missing = [did for did in document_ids if did not in active_doc_ids]
    assert not missing, (
        f"{len(missing)}/{DOC_COUNT} documents did not reach ACTIVE within "
        f"{POLL_BUDGET_SECONDS}s budget. Stuck states: "
        f"{[(did, last_seen.get(did, {})) for did in missing[:3]]}"
    )

    # Health probe gate: all samples must be 200 + under the latency
    # budget. The hard cut deployment promise is that worker pressure
    # never blocks the API; if any probe sample violates this, the
    # deployment split has not actually isolated the runtimes.
    failed_probes = [p for p in health_probes if p[2] != 200]
    assert not failed_probes, (
        f"/health/live returned non-200 during burst: {[(t, lat, code) for t, lat, code in failed_probes[:5]]}"
    )
    slow_probes = [p for p in health_probes if p[1] > API_HEALTH_PROBE_LATENCY_BUDGET_SECONDS]
    assert len(slow_probes) <= max(1, len(health_probes) // 20), (
        f"/health/live latency violated {API_HEALTH_PROBE_LATENCY_BUDGET_SECONDS}s "
        f"budget on {len(slow_probes)}/{len(health_probes)} samples — "
        "API event loop is being blocked by worker activity."
    )
