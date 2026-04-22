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

"""Graph-index HTTP routes.

After the LightRAG removal the knowledge-graph surface exposed to the
web frontend is intentionally minimal:

* Labels list and subgraph-for-visualisation live on
  ``aperag/views/collections.py`` because they are collection-scoped
  read operations and share auth / quota plumbing there.
* The three LightRAG-era curation endpoints — merge suggestions,
  merge execution, KG-Eval export — used to live here. They were
  removed as part of the graphindex v2 rewrite (see
  ``docs/zh-CN/design/graphindex_rewrite.md`` §1). We keep the routes
  mounted and returning **HTTP 410 Gone** so:
  * existing frontend bundles get an explicit, debuggable error on
    click, rather than a confusing backend 404;
  * cleaning up the web UI can happen as a dedicated follow-up PR
    without racing against this backend removal;
  * operators grepping access logs see a single, distinctive status
    code for "curation was removed" versus "route doesn't exist".

Once the frontend drops the merge-suggestion UI the whole router can
be deleted.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter()

_REMOVAL_DETAIL = (
    "Graph-curation endpoints (merge suggestions, merge execution, "
    "KG-Eval export) were removed together with the LightRAG-based "
    "graph index in graphindex v2. See "
    "docs/zh-CN/design/graphindex_rewrite.md."
)


def _gone() -> HTTPException:
    """Uniform 410 response for every removed curation route."""
    return HTTPException(status_code=410, detail=_REMOVAL_DETAIL)


@router.post("/collections/{collection_id}/graphs/nodes/merge", tags=["graph"])
async def merge_nodes_view(request: Request, collection_id: str) -> dict:
    raise _gone()


@router.post(
    "/collections/{collection_id}/graphs/merge-suggestions/{suggestion_id}/action",
    tags=["graph"],
)
async def handle_suggestion_action_view(request: Request, collection_id: str, suggestion_id: str) -> dict:
    raise _gone()


@router.post("/collections/{collection_id}/graphs/merge-suggestions", tags=["graph"])
async def merge_suggestions_view(request: Request, collection_id: str) -> dict:
    raise _gone()


@router.get("/collections/{collection_id}/graphs/export/kg-eval", tags=["graph"])
async def export_kg_eval_view(request: Request, collection_id: str) -> dict:
    raise _gone()
