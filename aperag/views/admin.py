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

import json
import logging
from http import HTTPStatus

from ninja import Router

from config import settings
from aperag.context.context import ContextManager
from aperag.db.ops import query_collection_without_user
from aperag.embed.base_embedding import get_embedding_model
from aperag.rank.reranker import rerank
from aperag.source.utils import async_run
from aperag.utils.request import get_user
from aperag.utils.utils import generate_vector_db_collection_name
from aperag.views.utils import fail, success

logger = logging.getLogger(__name__)


router = Router()


@router.post("/collections/{collection_id}/query")
async def query_similar(request, collection_id, topk=3, score_threshold=0.5):
    user = get_user(request)
    if user != settings.ADMIN_USER:
        return fail(HTTPStatus.FORBIDDEN, "Permission denied")

    collection = await query_collection_without_user(collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")

    query = request.body.decode('utf-8')
    log_prefix = "query_similar|collection_id: %s|query: %s" % (collection_id, query)
    config = json.loads(collection.config)
    embedding_model_name = config.get("embedding_model", settings.EMBEDDING_MODEL)
    embedding_model, vector_size = get_embedding_model(embedding_model_name)

    collection_name = generate_vector_db_collection_name(collection_id)
    vectordb_ctx = json.loads(settings.VECTOR_DB_CONTEXT)
    vectordb_ctx["collection"] = collection_name
    context_manager = ContextManager(collection_name, embedding_model, settings.VECTOR_DB_TYPE, vectordb_ctx)
    vector = embedding_model.embed_query(query)
    results = await async_run(context_manager.query, query,
                              score_threshold=score_threshold, topk=topk * 6, vector=vector)

    if len(results) > 1:
        results = await rerank(query, results)
        logger.info("[%s] rerank candidates end", log_prefix)
    else:
        logger.info("[%s] don't need to rerank ", log_prefix)

    candidates = results[:topk]
    result = []
    for item in candidates:
        if item.score == 0:
            continue
        result.append({
            "score": item.score,
            "text": item.text,
            "metadata": item.metadata
        })
    return success({"result": result})
