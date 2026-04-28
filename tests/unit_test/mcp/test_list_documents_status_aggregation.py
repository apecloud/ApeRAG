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

"""Pin the MCP-side index-status aggregation contract.

Why this test exists: ``Document.status`` is set to ``PENDING`` at
``confirm_documents`` time and never updated by the indexing
pipeline (no writer in reconciler / index workers — verified by
grep). Reading the raw column directly causes
``list_documents`` / ``get_document_metadata`` to forever report
``"pending"`` even when every modality is ``ACTIVE+is_serving``.

The FE goes through ``_index_statuses_to_document_status``
(document_service.py:105) which aggregates the per-modality
``DocumentIndex`` rows. The MCP tool now does the same via
``_aggregate_index_status`` so both surfaces (FE list + MCP list +
agent activity narration) report the same truth.

earayu2 bug report (msg=dec8bcff): UI shows complete, agent says
"all 8 docs are pending". huangheng grep evidence (msg=ed0db20c).
"""

from __future__ import annotations

from dataclasses import dataclass

from aperag.domains.knowledge_base.db.models import DocumentStatus
from aperag.indexing.models import IndexStatus
from aperag.mcp.tools.list_documents import _aggregate_index_status


@dataclass
class _FakeIndex:
    """Stand-in for ``DocumentIndex`` ORM rows — only the two fields
    the aggregator reads, no DB session required."""

    status: str
    is_serving: bool = False


def test_aggregate_returns_complete_when_all_modalities_active_and_serving():
    """The canonical "knowledge base looks healthy" case — every
    modality has its serving row in ACTIVE state. The pre-fix code
    would have returned the stale ``Document.status`` (PENDING)
    instead, which is what made the agent say "待索引" for fully
    indexed docs."""

    indexes = [
        _FakeIndex(status=IndexStatus.ACTIVE.value, is_serving=True),
        _FakeIndex(status=IndexStatus.ACTIVE.value, is_serving=True),
        _FakeIndex(status=IndexStatus.ACTIVE.value, is_serving=True),
    ]
    assert _aggregate_index_status(indexes, fallback=DocumentStatus.PENDING) is DocumentStatus.COMPLETE


def test_aggregate_returns_running_when_any_modality_pending_or_running():
    indexes_pending = [
        _FakeIndex(status=IndexStatus.ACTIVE.value, is_serving=True),
        _FakeIndex(status=IndexStatus.PENDING.value),
    ]
    assert _aggregate_index_status(indexes_pending, fallback=DocumentStatus.COMPLETE) is DocumentStatus.RUNNING

    indexes_running = [
        _FakeIndex(status=IndexStatus.ACTIVE.value, is_serving=True),
        _FakeIndex(status=IndexStatus.RUNNING.value),
    ]
    assert _aggregate_index_status(indexes_running, fallback=DocumentStatus.COMPLETE) is DocumentStatus.RUNNING


def test_aggregate_returns_failed_when_any_modality_failed():
    """FAILED dominates all other statuses — surface the failure even
    if other modalities are healthy, so the user sees something is
    wrong rather than a silent partial-index state."""

    indexes = [
        _FakeIndex(status=IndexStatus.ACTIVE.value, is_serving=True),
        _FakeIndex(status=IndexStatus.ACTIVE.value, is_serving=True),
        _FakeIndex(status=IndexStatus.FAILED.value),
    ]
    assert _aggregate_index_status(indexes, fallback=DocumentStatus.COMPLETE) is DocumentStatus.FAILED


def test_aggregate_active_but_not_serving_returns_fallback():
    """The "cutover transit" edge case from
    ``Document.get_overall_index_status`` — modality is ACTIVE but
    not yet flipped to ``is_serving=True``. We don't claim COMPLETE
    yet; we hand back to the caller's fallback (which is
    ``Document.status`` from the row, typically PENDING)."""

    indexes = [
        _FakeIndex(status=IndexStatus.ACTIVE.value, is_serving=False),
    ]
    assert _aggregate_index_status(indexes, fallback=DocumentStatus.PENDING) is DocumentStatus.PENDING


def test_aggregate_no_indexes_returns_fallback():
    """An empty list means no modality was ever scheduled for this
    document — return the caller's fallback (typically the raw
    ``Document.status``, which is the most informative thing we have
    in that scenario)."""

    assert _aggregate_index_status([], fallback=DocumentStatus.UPLOADED) is DocumentStatus.UPLOADED
    assert _aggregate_index_status([], fallback=DocumentStatus.PENDING) is DocumentStatus.PENDING


def test_failed_takes_precedence_over_running():
    """Pin the priority order: FAILED beats RUNNING beats COMPLETE.
    Without this ordering, a document with one FAILED modality and
    one PENDING modality would appear as RUNNING and the user
    wouldn't know to retry the failed one."""

    indexes = [
        _FakeIndex(status=IndexStatus.RUNNING.value),
        _FakeIndex(status=IndexStatus.FAILED.value),
    ]
    assert _aggregate_index_status(indexes, fallback=DocumentStatus.PENDING) is DocumentStatus.FAILED
