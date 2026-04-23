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

"""Residual v1 graph routes.

The write-side graph routes (``nodes/merge``, ``merge-suggestions``,
``merge-suggestions/{id}/action``) and the read-side labels / subgraph
routes that used to live here were hard-cut by the Phase 2
``knowledge_graph`` domain split and now live under
``aperag/domains/knowledge_graph/api/routes.py`` (mounted at
``/api/v2/``). See
``docs/modularization/breaking-changes/phase2-retrieval-knowledge_graph.md``.

The one handler left below is the historical ``GET /collections/{id}/graphs/export/kg-eval``
endpoint that was deleted with the LightRAG-era graph workflow. Kept
as a 410 Gone so out-of-tree scripts / notebooks that still hit it
receive a discoverable explanation instead of a 404.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter()

_KG_EVAL_REMOVAL_DETAIL = (
    "This legacy KG-eval export endpoint was removed together with the "
    "LightRAG-era graph workflow. See docs/zh-CN/design/graphindex_rewrite.md."
)


def _gone() -> HTTPException:
    """Uniform 410 response for the removed KG-eval route."""
    return HTTPException(status_code=410, detail=_KG_EVAL_REMOVAL_DETAIL)


@router.get("/collections/{collection_id}/graphs/export/kg-eval", tags=["graph"])
async def export_kg_eval_view(request: Request, collection_id: str) -> dict:
    raise _gone()
