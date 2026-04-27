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

"""Per-document index state read helper — celery T3.2.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §G.5,
search results carry an ``index_state_per_modality`` map describing
the current indexing state of every modality for the document the
hit belongs to. Clients use this to reason about the §F.4 short
inconsistency window — a vector hit that ships
``index_state_per_modality["fulltext"] == "INDEXING"`` tells the
agent layer "fulltext for this document is behind; we may have
stale recall and should re-issue once it lands".

The helper here translates the raw :class:`DocumentIndex` table
state into the locked :data:`IndexStateValue` enum:

* ``ACTIVE``      — row is ``status=ACTIVE`` AND ``is_serving=TRUE``.
* ``FAILED``      — row is ``status=FAILED`` (after retry budget).
* ``INDEXING``    — row exists but is ``PENDING`` / ``RUNNING`` /
                    ``ACTIVE-but-not-serving`` (in the §F.3 cutover
                    transit window).
* ``NOT_ENABLED`` — no row exists for this ``(document_id, modality)``
                    pair, i.e., the modality was never enqueued for
                    this document.

The helper is a single-batch read keyed by ``(collection_id,
document_ids)`` so the search pipeline can hydrate metadata for an
entire result page in one DB round-trip rather than N+1 queries.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

from sqlalchemy import and_, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from aperag.indexing.models import DocumentIndex, IndexStatus, Modality

logger = logging.getLogger(__name__)


IndexStateValue = Literal["ACTIVE", "FAILED", "NOT_ENABLED", "INDEXING"]
"""Mirror of :class:`aperag.domains.retrieval.schemas.IndexStateValue`.

Re-declared here so :mod:`aperag.indexing` does not depend on
``aperag.domains.retrieval`` (the dependency runs in the other
direction). The two literals MUST stay in sync; if a new state is
added the §G.5 amendment must be done at both call sites in lockstep.
"""


# All modalities currently advertised to clients. Keeping this list
# explicit (rather than ``[m.value for m in Modality]``) lets a future
# private-API modality stay out of public search metadata without a
# schema flag flip.
PUBLIC_MODALITY_VALUES: tuple[str, ...] = tuple(m.value for m in Modality)


def _state_for_row(status: str, is_serving: bool) -> IndexStateValue:
    """Translate a single :class:`DocumentIndex` row's
    ``(status, is_serving)`` pair into the §G.5 enum.

    The §F.3 cutover model means a row may be ``ACTIVE`` but not yet
    serving (the brief window between worker-side
    ``_finalize_active_with_cutover`` and the end of the same TX).
    Per §F.4 we treat that intermediate as ``INDEXING`` for client
    purposes — hits served from a different modality should be told
    "this one is in transit, retry shortly" rather than "this one is
    ready".
    """
    if status == IndexStatus.ACTIVE.value and is_serving:
        return "ACTIVE"
    if status == IndexStatus.FAILED.value:
        return "FAILED"
    return "INDEXING"


def query_index_state_for_documents(
    *,
    engine: Engine,
    collection_id: str,
    document_ids: Iterable[str],
) -> dict[str, dict[str, IndexStateValue]]:
    """Return ``{document_id: {modality: state}}`` for every
    ``(document_id, modality)`` enumerated by ``document_ids`` and the
    canonical modality list.

    Implementation note: the result map is dense over ``Modality``
    values — every document_id key maps to an inner dict that has
    ALL modalities present, defaulting to ``NOT_ENABLED`` when no
    DocumentIndex row exists for that ``(doc, modality)``. This
    makes the §G.5 contract symmetric: clients get a stable shape
    and don't have to reason about "field missing means what?".

    The function is synchronous because the search pipeline calls it
    inside a worker-thread offload in :mod:`aperag.domains.retrieval.
    pipeline`. Callers in async contexts should wrap with
    :func:`asyncio.to_thread`.
    """
    document_id_list = list(document_ids)
    if not document_id_list:
        return {}

    # Initialize a dense result map: every doc_id → every modality →
    # NOT_ENABLED. Subsequent rows from DB overwrite where data exists.
    result: dict[str, dict[str, IndexStateValue]] = {
        doc_id: {modality: "NOT_ENABLED" for modality in PUBLIC_MODALITY_VALUES} for doc_id in document_id_list
    }

    with Session(engine) as session:
        rows = list(
            session.scalars(
                select(DocumentIndex).where(
                    and_(
                        DocumentIndex.collection_id == collection_id,
                        DocumentIndex.document_id.in_(document_id_list),
                    )
                )
            )
        )

    # When multiple rows exist for the same (doc, modality) — e.g.,
    # an old PENDING row plus a new ACTIVE+serving row — pick the
    # serving row first (most user-relevant), then fall through to
    # any non-serving row's state. The §F.1 partial unique index
    # guarantees at most one is_serving=TRUE per (doc, modality), so
    # the precedence is deterministic.
    serving_seen: dict[tuple[str, str], bool] = {}
    for row in rows:
        key = (row.document_id, row.modality)
        state = _state_for_row(row.status, row.is_serving)
        if not serving_seen.get(key) and (state != "ACTIVE"):
            # No serving row yet observed for this (doc, modality);
            # write the row's state but don't lock it in.
            current = result.get(row.document_id, {}).get(row.modality, "NOT_ENABLED")
            if current == "NOT_ENABLED" or state == "FAILED":
                result.setdefault(row.document_id, {})[row.modality] = state
        if state == "ACTIVE":
            serving_seen[key] = True
            result.setdefault(row.document_id, {})[row.modality] = "ACTIVE"

    return result


__all__ = [
    "IndexStateValue",
    "PUBLIC_MODALITY_VALUES",
    "query_index_state_for_documents",
]
