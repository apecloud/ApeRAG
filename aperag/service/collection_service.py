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

from datetime import datetime
from http import HTTPStatus

from sqlalchemy import desc, select

from aperag.apps import QuotaType
from aperag.config import SessionDep, settings
from aperag.db import models as db_models
from aperag.db.models import SearchTestHistory
from aperag.db.ops import PagedQuery, query_collection, query_collections, query_collections_count, query_user_quota
from aperag.flow.base.models import Edge, FlowInstance, NodeInstance
from aperag.flow.engine import FlowEngine
from aperag.graph.lightrag_holder import delete_lightrag_holder, reload_lightrag_holder
from aperag.schema import view_models
from aperag.schema.utils import dumpCollectionConfig, parseCollectionConfig
from aperag.schema.view_models import (
    Collection,
    CollectionList,
    SearchTestResult,
    SearchTestResultItem,
    SearchTestResultList,
)
from aperag.source.base import get_source
from aperag.tasks.collection import delete_collection_task, init_collection_task
from aperag.tasks.scan import delete_sync_documents_cron_job, update_sync_documents_cron_job
from aperag.views.utils import fail, success, validate_source_connect_config


def build_collection_response(instance: db_models.Collection) -> view_models.Collection:
    """Build Collection response object for API return."""
    return Collection(
        id=instance.id,
        title=instance.title,
        description=instance.description,
        type=instance.type,
        status=getattr(instance, "status", None),
        config=parseCollectionConfig(instance.config),
        created=instance.gmt_created.isoformat(),
        updated=instance.gmt_updated.isoformat(),
    )


async def create_collection(
    session: SessionDep, user: str, collection: view_models.CollectionCreate
) -> view_models.Collection:
    collection_config = collection.config
    if collection.type == db_models.CollectionType.DOCUMENT:
        is_validate, error_msg = validate_source_connect_config(collection_config)
        if not is_validate:
            return fail(HTTPStatus.BAD_REQUEST, error_msg)

    # there is quota limit on collection
    if settings.max_collection_count:
        collection_limit = await query_user_quota(session, user, QuotaType.MAX_COLLECTION_COUNT)
        if collection_limit is None:
            collection_limit = settings.max_collection_count
        if collection_limit and await query_collections_count(session, user) >= collection_limit:
            return fail(HTTPStatus.FORBIDDEN, f"collection number has reached the limit of {collection_limit}")

    instance = db_models.Collection(
        user=user,
        type=collection.type,
        status=db_models.CollectionStatus.INACTIVE,
        title=collection.title,
        description=collection.description,
    )

    if collection.config is not None:
        instance.config = dumpCollectionConfig(collection_config)
    session.add(instance)
    await session.commit()
    await session.refresh(instance)
    if getattr(collection_config, "enable_knowledge_graph", False):
        await reload_lightrag_holder(instance)

    if instance.type == db_models.CollectionType.DOCUMENT:
        document_user_quota = await query_user_quota(session, user, QuotaType.MAX_DOCUMENT_COUNT)
        init_collection_task.delay(instance.id, document_user_quota)
    else:
        return fail(HTTPStatus.BAD_REQUEST, "unknown collection type")

    return success(build_collection_response(instance))


async def list_collections(session: SessionDep, user: str, pq: PagedQuery) -> view_models.CollectionList:
    pr = await query_collections(session, [user, settings.admin_user], pq)
    response = []
    for collection in pr.data:
        response.append(build_collection_response(collection))
    return success(CollectionList(items=response), pr=pr)


async def get_collection(session: SessionDep, user: str, collection_id: str) -> view_models.Collection:
    collection = await query_collection(session, user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    return success(build_collection_response(collection))


async def update_collection(
    session: SessionDep, user: str, collection_id: str, collection: view_models.CollectionUpdate
) -> view_models.Collection:
    instance = await query_collection(session, user, collection_id)
    if instance is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    instance.title = collection.title
    instance.description = collection.description
    instance.config = dumpCollectionConfig(collection.config)
    session.add(instance)
    await session.commit()
    await session.refresh(instance)
    await reload_lightrag_holder(instance)
    source = get_source(collection.config)
    if source.sync_enabled():
        await update_sync_documents_cron_job(instance.id)
    return success(build_collection_response(instance))


async def delete_collection(session: SessionDep, user: str, collection_id: str) -> view_models.Collection:
    collection = await query_collection(session, user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    collection_bots = await collection.bots(session, only_ids=True)
    if len(collection_bots) > 0:
        return fail(
            HTTPStatus.BAD_REQUEST, f"Collection has related to bots {','.join(collection_bots)}, can not be deleted"
        )
    await delete_sync_documents_cron_job(collection.id)
    collection.status = db_models.CollectionStatus.DELETED
    collection.gmt_deleted = datetime.utcnow()
    session.add(collection)
    await session.commit()
    await delete_lightrag_holder(collection)
    delete_collection_task.delay(collection_id)
    return success(build_collection_response(collection))


async def create_search_test(
    session: SessionDep, user: str, collection_id: str, data: view_models.SearchTestRequest
) -> view_models.SearchTestResult:
    collection = await query_collection(session, user, collection_id)
    if not collection:
        return fail(404, "Collection not found")
    nodes = {}
    edges = []
    end_node_id = "merge"
    end_node_values = {
        "merge_strategy": "union",
        "deduplicate": True,
    }
    query = data.query
    if data.vector_search:
        node_id = "vector_search"
        nodes[node_id] = NodeInstance(
            id=node_id,
            type="vector_search",
            input_values={
                "query": query,
                "top_k": data.vector_search.topk if data.vector_search else 5,
                "similarity_threshold": data.vector_search.similarity if data.vector_search else 0.7,
                "collection_ids": [collection_id],
            },
        )
        end_node_values["vector_search_docs"] = "{{ nodes.vector_search.output.docs }}"
        edges.append(Edge(source=node_id, target=end_node_id))
    if data.fulltext_search:
        node_id = "fulltext_search"
        nodes[node_id] = NodeInstance(
            id=node_id,
            type="fulltext_search",
            input_values={
                "query": query,
                "top_k": data.vector_search.topk if data.vector_search else 5,
                "collection_ids": [collection_id],
            },
        )
        end_node_values["fulltext_search_docs"] = "{{ nodes.fulltext_search.output.docs }}"
        edges.append(Edge(source=node_id, target=end_node_id))
    if data.graph_search:
        nodes["graph_search"] = NodeInstance(
            id="graph_search",
            type="graph_search",
            input_values={
                "query": query,
                "top_k": data.graph_search.topk if data.graph_search else 5,
                "collection_ids": [collection_id],
            },
        )
        end_node_values["graph_search_docs"] = "{{ nodes.graph_search.output.docs }}"
        edges.append(Edge(source="graph_search", target=end_node_id))
    nodes[end_node_id] = NodeInstance(
        id=end_node_id,
        type="merge",
        input_values=end_node_values,
    )
    flow = FlowInstance(
        name="search_test",
        title="Search Test",
        nodes=nodes,
        edges=edges,
    )
    engine = FlowEngine()
    initial_data = {"query": query, "user": user}
    result, _ = await engine.execute_flow(flow, initial_data)
    if not result:
        return fail(400, "Failed to execute flow")
    docs = result.get(end_node_id, {}).docs
    items = []
    for idx, doc in enumerate(docs):
        items.append(
            SearchTestResultItem(
                rank=idx + 1,
                score=doc.score,
                content=doc.text,
                source=doc.metadata.get("source", ""),
                recall_type=doc.metadata.get("recall_type", ""),
            )
        )
    record = SearchTestHistory(
        user=user,
        query=data.query,
        collection_id=collection_id,
        vector_search=data.vector_search.dict() if data.vector_search else None,
        fulltext_search=data.fulltext_search.dict() if data.fulltext_search else None,
        graph_search=data.graph_search.dict() if data.graph_search else None,
        items=[item.dict() for item in items],
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    result = SearchTestResult(
        id=record.id,
        query=record.query,
        vector_search=record.vector_search,
        fulltext_search=record.fulltext_search,
        graph_search=record.graph_search,
        items=items,
        created=record.gmt_created.isoformat(),
    )
    return success(result)


async def list_search_tests(session: SessionDep, user: str, collection_id: str) -> view_models.SearchTestResultList:
    stmt = (
        select(SearchTestHistory)
        .where(
            SearchTestHistory.user == user,
            SearchTestHistory.collection_id == collection_id,
            SearchTestHistory.gmt_deleted is None,
        )
        .order_by(desc(SearchTestHistory.gmt_created))
        .limit(50)
    )
    result = await session.execute(stmt)
    records = result.scalars().all()
    resultList = []
    for record in records:
        items = []
        for item in record.items:
            items.append(
                SearchTestResultItem(
                    rank=item["rank"],
                    score=item["score"],
                    content=item["content"],
                    source=item["source"],
                    recall_type=item["recall_type"],
                )
            )
        result = SearchTestResult(
            id=record.id,
            query=record.query,
            vector_search=record.vector_search,
            fulltext_search=record.fulltext_search,
            graph_search=record.graph_search,
            items=items,
            created=record.gmt_created.isoformat(),
        )
        resultList.append(result)
    return success(SearchTestResultList(items=resultList))


async def delete_search_test(session: SessionDep, user: str, collection_id: str, search_test_id: str):
    stmt = select(SearchTestHistory).where(
        SearchTestHistory.user == user,
        SearchTestHistory.id == search_test_id,
        SearchTestHistory.collection_id == collection_id,
        SearchTestHistory.gmt_deleted is None,
    )
    result = await session.execute(stmt)
    record = result.scalars().first()
    if record:
        record.gmt_deleted = datetime.utcnow()
        session.add(record)
        await session.commit()
    return success({})
