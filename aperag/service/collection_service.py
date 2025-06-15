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

from http import HTTPStatus

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.config import settings
from aperag.db import models as db_models
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.flow.base.models import Edge, FlowInstance, NodeInstance
from aperag.flow.engine import FlowEngine
from aperag.schema import view_models
from aperag.schema.utils import dumpCollectionConfig, parseCollectionConfig
from aperag.schema.view_models import (
    Collection,
    CollectionList,
    SearchTestResult,
    SearchTestResultItem,
    SearchTestResultList,
)
from aperag.utils.constant import QuotaType
from aperag.views.utils import fail, success, validate_source_connect_config
from config.celery_tasks import collection_delete_task, collection_init_task


class CollectionService:
    """Collection service that handles business logic for collections"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    def build_collection_response(self, instance: db_models.Collection) -> view_models.Collection:
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

    async def create_collection(self, user: str, collection: view_models.CollectionCreate) -> view_models.Collection:
        collection_config = collection.config
        if collection.type == db_models.CollectionType.DOCUMENT:
            is_validate, error_msg = validate_source_connect_config(collection_config)
            if not is_validate:
                return fail(HTTPStatus.BAD_REQUEST, error_msg)

        # Check quota limit on collection
        if settings.max_collection_count:
            collection_limit = await self.db_ops.query_user_quota(user, QuotaType.MAX_COLLECTION_COUNT)
            if collection_limit is None:
                collection_limit = settings.max_collection_count
            if collection_limit and await self.db_ops.query_collections_count(user) >= collection_limit:
                return fail(HTTPStatus.FORBIDDEN, f"collection number has reached the limit of {collection_limit}")

        async def _create_operation(session):
            # Use DatabaseOps to create collection
            db_ops_session = AsyncDatabaseOps(session)
            config_str = dumpCollectionConfig(collection_config) if collection.config is not None else None

            instance = await db_ops_session.create_collection(
                user=user,
                title=collection.title,
                description=collection.description,
                collection_type=collection.type,
                config=config_str,
            )

            # Initialize collection based on type
            if instance.type == db_models.CollectionType.DOCUMENT:
                document_user_quota = await self.db_ops.query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
                collection_init_task.delay(instance.id, document_user_quota)
            else:
                raise ValueError("unknown collection type")

            return self.build_collection_response(instance)

        try:
            result = await self.db_ops.execute_with_transaction(_create_operation)
            return success(result)
        except ValueError as e:
            return fail(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to create collection: {str(e)}")

    async def list_collections(self, user: str) -> view_models.CollectionList:
        collections = await self.db_ops.query_collections([user])
        response = []
        for collection in collections:
            response.append(self.build_collection_response(collection))
        return success(CollectionList(items=response))

    async def get_collection(self, user: str, collection_id: str) -> view_models.Collection:
        collection = await self.db_ops.query_collection(user, collection_id)
        if collection is None:
            return fail(HTTPStatus.NOT_FOUND, "Collection not found")
        return success(self.build_collection_response(collection))

    async def update_collection(
        self, user: str, collection_id: str, collection: view_models.CollectionUpdate
    ) -> view_models.Collection:
        # First check if collection exists
        instance = await self.db_ops.query_collection(user, collection_id)
        if instance is None:
            return fail(HTTPStatus.NOT_FOUND, "Collection not found")

        async def _update_operation(session):
            # Use DatabaseOps to update collection
            db_ops_session = AsyncDatabaseOps(session)
            config_str = dumpCollectionConfig(collection.config)

            updated_instance = await db_ops_session.update_collection_by_id(
                user=user,
                collection_id=collection_id,
                title=collection.title,
                description=collection.description,
                config=config_str,
            )

            if not updated_instance:
                raise ValueError("Collection not found")

            return self.build_collection_response(updated_instance)

        try:
            result = await self.db_ops.execute_with_transaction(_update_operation)
            return success(result)
        except ValueError as e:
            return fail(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to update collection: {str(e)}")

    async def delete_collection(self, user: str, collection_id: str) -> view_models.Collection:
        # First check if collection exists
        collection = await self.db_ops.query_collection(user, collection_id)
        if collection is None:
            return fail(HTTPStatus.NOT_FOUND, "Collection not found")

        async def _delete_operation(session):
            # Use DatabaseOps to delete collection
            db_ops_session = AsyncDatabaseOps(session)
            deleted_instance = await db_ops_session.delete_collection_by_id(user, collection_id)

            if not deleted_instance:
                raise ValueError("Collection not found")

            # Clean up related resources
            collection_delete_task.delay(collection_id)

            return self.build_collection_response(deleted_instance)

        try:
            result = await self.db_ops.execute_with_transaction(_delete_operation)
            return success(result)
        except ValueError as e:
            return fail(HTTPStatus.BAD_REQUEST, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to delete collection: {str(e)}")

    async def create_search_test(
        self, user: str, collection_id: str, data: view_models.SearchTestRequest
    ) -> view_models.SearchTestResult:
        collection = await self.db_ops.query_collection(user, collection_id)
        if not collection:
            return fail(404, "Collection not found")

        # Build flow for search test execution
        nodes = {}
        edges = []
        end_node_id = "merge"
        end_node_values = {
            "merge_strategy": "union",
            "deduplicate": True,
        }
        query = data.query

        # Configure search nodes based on request
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

        # Execute search flow
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

        # Process search results
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

        async def _create_search_test_operation(session):
            # Use DatabaseOps to create search test record
            db_ops_session = AsyncDatabaseOps(session)
            record = await db_ops_session.create_search_test(
                user=user,
                collection_id=collection_id,
                query=data.query,
                vector_search=data.vector_search.dict() if data.vector_search else None,
                fulltext_search=data.fulltext_search.dict() if data.fulltext_search else None,
                graph_search=data.graph_search.dict() if data.graph_search else None,
                items=[item.dict() for item in items],
            )

            return SearchTestResult(
                id=record.id,
                query=record.query,
                vector_search=record.vector_search,
                fulltext_search=record.fulltext_search,
                graph_search=record.graph_search,
                items=items,
                created=record.gmt_created.isoformat(),
            )

        try:
            result = await self.db_ops.execute_with_transaction(_create_search_test_operation)
            return success(result)
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to create search test: {str(e)}")

    async def list_search_tests(self, user: str, collection_id: str) -> view_models.SearchTestResultList:
        collection = await self.db_ops.query_collection(user, collection_id)
        if not collection:
            return fail(404, "Collection not found")

        # Use DatabaseOps to query search tests
        search_tests = await self.db_ops.query_search_tests(user, collection_id)

        items = []
        for search_test in search_tests:
            test_result_items = []
            for item_data in search_test.items:
                test_result_items.append(SearchTestResultItem(**item_data))
            items.append(
                SearchTestResult(
                    id=search_test.id,
                    query=search_test.query,
                    vector_search=search_test.vector_search,
                    fulltext_search=search_test.fulltext_search,
                    graph_search=search_test.graph_search,
                    items=test_result_items,
                    created=search_test.gmt_created.isoformat(),
                )
            )
        return success(SearchTestResultList(items=items))

    async def delete_search_test(self, user: str, collection_id: str, search_test_id: str):
        collection = await self.db_ops.query_collection(user, collection_id)
        if not collection:
            return fail(404, "Collection not found")

        async def _delete_search_test_operation(session):
            # Use DatabaseOps to delete search test
            db_ops_session = AsyncDatabaseOps(session)
            deleted = await db_ops_session.delete_search_test(user, collection_id, search_test_id)

            if not deleted:
                raise ValueError("Search test not found")
            return {}

        try:
            await self.db_ops.execute_with_transaction(_delete_search_test_operation)
            return success({})
        except ValueError as e:
            return fail(404, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to delete search test: {str(e)}")


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
collection_service = CollectionService()
