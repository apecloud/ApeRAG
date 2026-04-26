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

"""``Modality`` abstract base — celery T1.1 Foundation.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §C/§D, every
modality (``vector`` / ``fulltext`` / ``graph`` / ``summary`` / ``vision``)
implements two operations:

- ``derive(document_id, parse_version, source_path) -> derived_artifact_path``
  Reads the source / parser output, calls any expensive LLM /
  embedding / vision pipelines, writes the canonical artifact under
  ``derived/parse_<version>/<modality_file>`` using the
  ``ObjectStore`` write-then-rename / multipart-then-complete contract
  (§C.7) so partial writes are never visible.

- ``sync(document_id, parse_version, derived_artifact_path) -> None``
  Reads the derived artifact and applies the §D.1 two-phase
  replace-idempotent contract: DELETE all backend entries WHERE
  (document_id=X, parse_version=Y), then INSERT all entries from the
  artifact. Cheap to retry; never re-runs ``derive``.

The graph modality is the one exception to the simple
DELETE-by-(doc, parse_version) shape — see §D.3 lineage model. The
ABC accepts that variation: ``sync`` is just "make backend
byte-equivalent to the artifact for this (doc, version) slot",
however that has to be done.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from aperag.indexing.models import Modality


@dataclass(frozen=True)
class DeriveResult:
    """Outcome of ``Modality.derive``.

    The ``derived_artifact_path`` is what ``sync`` will read on this
    and any subsequent retry. It is opaque to callers — only the
    matching ``Modality.sync`` knows how to interpret it.
    """

    derived_artifact_path: str


class ModalityWorker(ABC):
    """Abstract base for the 5 per-modality workers.

    Implementations live in ``aperag/indexing/{vector,fulltext,
    graph,summary,vision}.py`` and are instantiated by the
    orchestrator (T2.1). The ABC enforces the (derive, sync) split so
    the orchestrator can route retries to ``sync`` only — never
    re-charging the LLM / embedding cost.
    """

    #: The modality this worker owns. Subclasses MUST override.
    modality: Modality

    @abstractmethod
    async def derive(
        self,
        *,
        document_id: str,
        parse_version: str,
        source_path: str,
    ) -> DeriveResult:
        """Produce the canonical derived artifact for this modality.

        Implementations must use the ``ObjectStore`` write-then-rename
        / multipart-then-complete contract (§C.7) so a partial write
        is never visible. Implementations MUST be idempotent at the
        artifact level: re-running ``derive`` for the same
        ``(document_id, parse_version)`` produces a byte-equivalent
        artifact (modulo non-deterministic LLM noise, which is the
        whole reason the artifact gets persisted in the first place).
        """

    @abstractmethod
    async def sync(
        self,
        *,
        document_id: str,
        parse_version: str,
        derived_artifact_path: str,
    ) -> None:
        """Apply the §D.1 replace-idempotent contract to the backend.

        DELETE all backend entries WHERE
        ``(document_id=X, parse_version=Y)`` then INSERT the entries
        encoded in ``derived_artifact_path``. Re-running this method
        produces a backend state byte-equivalent to a fresh sync (§D.4).

        Graph modality reinterprets DELETE as the §D.3 lineage-level
        DELETE+INSERT — ``sync`` keeps the same signature; the §D.3
        algorithm is internal to the graph implementation.
        """


__all__ = ["ModalityWorker", "DeriveResult"]
