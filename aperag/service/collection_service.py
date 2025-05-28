from http import HTTPStatus

from django.utils import timezone

from aperag.apps import QuotaType
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
from config import settings


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


async def create_collection(user: str, collection: view_models.CollectionCreate) -> view_models.Collection:
    collection_config = collection.config
    if collection.type == db_models.Collection.Type.DOCUMENT:
        is_validate, error_msg = validate_source_connect_config(collection_config)
        if not is_validate:
            return fail(HTTPStatus.BAD_REQUEST, error_msg)

    # there is quota limit on collection
    if settings.MAX_COLLECTION_COUNT:
        collection_limit = await query_user_quota(user, QuotaType.MAX_COLLECTION_COUNT)
        if collection_limit is None:
            collection_limit = settings.MAX_COLLECTION_COUNT
        if collection_limit and await query_collections_count(user) >= collection_limit:
            return fail(HTTPStatus.FORBIDDEN, f"collection number has reached the limit of {collection_limit}")

    instance = db_models.Collection(
        user=user,
        type=collection.type,
        status=db_models.Collection.Status.INACTIVE,
        title=collection.title,
        description=collection.description,
    )

    if collection.config is not None:
        instance.config = dumpCollectionConfig(collection_config)
    await instance.asave()
    if getattr(collection_config, "enable_knowledge_graph", False):
        await reload_lightrag_holder(instance)

    if instance.type == db_models.Collection.Type.DOCUMENT:
        document_user_quota = await query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
        init_collection_task.delay(instance.id, document_user_quota)
    else:
        return fail(HTTPStatus.BAD_REQUEST, "unknown collection type")

    return success(build_collection_response(instance))


async def list_collections(user: str, pq: PagedQuery) -> view_models.CollectionList:
    pr = await query_collections([user, settings.ADMIN_USER], pq)
    response = []
    async for collection in pr.data:
        response.append(build_collection_response(collection))
    return success(CollectionList(items=response), pr=pr)


async def get_collection(user: str, collection_id: str) -> view_models.Collection:
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    return success(build_collection_response(collection))


async def update_collection(
    user: str, collection_id: str, collection: view_models.CollectionUpdate
) -> view_models.Collection:
    instance = await query_collection(user, collection_id)
    if instance is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    instance.title = collection.title
    instance.description = collection.description
    instance.config = dumpCollectionConfig(collection.config)
    await instance.asave()
    await reload_lightrag_holder(instance)
    source = get_source(collection.config)
    if source.sync_enabled():
        await update_sync_documents_cron_job(instance.id)
    return success(build_collection_response(instance))


async def delete_collection(user: str, collection_id: str) -> view_models.Collection:
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    collection_bots = await collection.bots(only_ids=True)
    if len(collection_bots) > 0:
        return fail(
            HTTPStatus.BAD_REQUEST, f"Collection has related to bots {','.join(collection_bots)}, can not be deleted"
        )
    await delete_sync_documents_cron_job(collection.id)
    collection.status = db_models.Collection.Status.DELETED
    collection.gmt_deleted = timezone.now()
    await collection.asave()
    await delete_lightrag_holder(collection)
    delete_collection_task.delay(collection_id)
    return success(build_collection_response(collection))


async def create_search_test(
    user: str, collection_id: str, data: view_models.SearchTestRequest
) -> view_models.SearchTestResult:
    collection = await query_collection(user, collection_id)
    if not collection:
        return fail(404, "Collection not found")
    nodes = {}
    edges = []
    query = data.query
    if data.search_type == "vector":
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
        end_node = node_id
    elif data.search_type == "fulltext":
        node_id = "keyword_search"
        nodes[node_id] = NodeInstance(
            id=node_id,
            type="keyword_search",
            input_values={
                "query": query,
                "top_k": data.vector_search.topk if data.vector_search else 5,
                "collection_ids": [collection_id],
            },
        )
        end_node = node_id
    elif data.search_type == "hybrid":
        nodes["vector_search"] = NodeInstance(
            id="vector_search",
            type="vector_search",
            input_values={
                "query": query,
                "top_k": data.vector_search.topk if data.vector_search else 5,
                "similarity_threshold": data.vector_search.similarity if data.vector_search else 0.7,
                "collection_ids": [collection_id],
            },
        )
        nodes["keyword_search"] = NodeInstance(
            id="keyword_search",
            type="keyword_search",
            input_values={
                "query": query,
                "top_k": data.vector_search.topk if data.vector_search else 5,
                "collection_ids": [collection_id],
            },
        )
        nodes["merge"] = NodeInstance(
            id="merge",
            type="merge",
            input_values={
                "merge_strategy": "union",
                "deduplicate": True,
                "vector_search_docs": "{{ nodes.vector_search.output.docs }}",
                "keyword_search_docs": "{{ nodes.keyword_search.output.docs }}",
            },
        )
        edges = [
            Edge(source="vector_search", target="merge"),
            Edge(source="keyword_search", target="merge"),
        ]
        end_node = "merge"
    else:
        return fail(400, "Invalid search_type")
    flow = FlowInstance(
        name=f"search_test_{data.search_type}",
        title=f"Search Test {data.search_type}",
        nodes=nodes,
        edges=edges,
    )
    engine = FlowEngine()
    initial_data = {"query": query, "user": user}
    result, _ = await engine.execute_flow(flow, initial_data)
    if not result:
        return fail(400, "Failed to execute flow")
    end_nodes = engine.find_end_nodes(flow)
    if not end_nodes:
        return fail(400, "No output node found")
    end_node = end_nodes[0]
    docs = result.get(end_node, {}).docs
    items = []
    for idx, doc in enumerate(docs):
        items.append(
            SearchTestResultItem(rank=idx + 1, score=doc.score, content=doc.text, source=doc.metadata.get("source", ""))
        )
    record = await SearchTestHistory.objects.acreate(
        user=user,
        query=data.query,
        collection_id=collection_id,
        search_type=data.search_type,
        vector_search=data.vector_search.dict() if data.vector_search else None,
        fulltext_search=data.fulltext_search.dict() if data.fulltext_search else None,
        items=[item.dict() for item in items],
    )
    result = SearchTestResult(
        id=record.id,
        query=record.query,
        search_type=record.search_type,
        vector_search=record.vector_search,
        fulltext_search=record.fulltext_search,
        items=items,
        created=record.gmt_created.isoformat(),
    )
    return success(result)


async def list_search_tests(user: str, collection_id: str) -> view_models.SearchTestResultList:
    qs = SearchTestHistory.objects.filter(user=user, collection_id=collection_id, gmt_deleted__isnull=True).order_by(
        "-gmt_created"
    )[:50]
    resultList = []
    async for record in qs:
        items = []
        for item in record.items:
            items.append(
                SearchTestResultItem(
                    rank=item["rank"],
                    score=item["score"],
                    content=item["content"],
                    source=item["source"],
                )
            )
        result = SearchTestResult(
            id=record.id,
            query=record.query,
            search_type=record.search_type,
            vector_search=record.vector_search,
            fulltext_search=record.fulltext_search,
            items=items,
            created=record.gmt_created.isoformat(),
        )
        resultList.append(result)
    return success(SearchTestResultList(items=resultList))


async def delete_search_test(user: str, collection_id: str, search_test_id: str):
    await SearchTestHistory.objects.filter(
        user=user, id=search_test_id, collection_id=collection_id, gmt_deleted__isnull=True
    ).aupdate(gmt_deleted=timezone.now())
    return success({})
