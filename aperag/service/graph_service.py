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

import logging
from typing import Any, Dict, List

from aperag.db.models import MergeSuggestionStatus
from aperag.db.ops import async_db_ops
from aperag.exceptions import CollectionNotFoundException
from aperag.graph import lightrag_manager
from aperag.schema import view_models
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)


class GraphService:
    """Service for knowledge graph operations"""

    def __init__(self):
        from aperag.service.collection_service import collection_service

        self.collection_service = collection_service
        self.db_ops = async_db_ops

    async def get_graph_labels(self, user_id: str, collection_id: str) -> view_models.GraphLabelsResponse:
        """Get available node labels in the knowledge graph"""
        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        rag = await lightrag_manager.create_lightrag_instance(db_collection)
        try:
            labels = await rag.get_graph_labels()
            return view_models.GraphLabelsResponse(labels=labels)
        finally:
            await rag.finalize_storages()

    def _optimize_graph_for_visualization(self, nodes, edges, max_nodes):
        """Optimize graph by selecting well-connected nodes"""
        if len(nodes) <= max_nodes:
            return nodes, edges

        # Calculate node degrees
        degree_map = {node.id: 0 for node in nodes}
        for edge in edges:
            if edge.source in degree_map and edge.target in degree_map:
                degree_map[edge.source] += 1
                degree_map[edge.target] += 1

        # Select top nodes by degree
        sorted_nodes = sorted(nodes, key=lambda node: (-degree_map[node.id], node.id))
        selected_nodes = sorted_nodes[:max_nodes]
        selected_node_ids = {node.id for node in selected_nodes}

        # Filter edges between selected nodes
        optimized_edges = [
            edge for edge in edges if edge.source in selected_node_ids and edge.target in selected_node_ids
        ]

        return selected_nodes, optimized_edges

    async def get_knowledge_graph(
        self,
        user_id: str,
        collection_id: str,
        label: str = None,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> Dict[str, Any]:
        """Get knowledge graph with overview or subgraph mode"""
        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        rag = await lightrag_manager.create_lightrag_instance(db_collection)
        try:
            # Determine query parameters
            if not label or label == "*":
                node_label, query_max_nodes = "*", max_nodes * 2
                mode_description = "overview"
            else:
                node_label, query_max_nodes = label, max_nodes
                mode_description = f"subgraph from '{label}'"

            # Get knowledge graph
            kg = await rag.get_knowledge_graph(
                node_label=node_label,
                max_depth=max_depth,
                max_nodes=query_max_nodes,
            )

            # Optimize if needed
            if (not label or label == "*") and len(kg.nodes) > max_nodes:
                optimized_nodes, optimized_edges = self._optimize_graph_for_visualization(kg.nodes, kg.edges, max_nodes)
                is_truncated = True
            else:
                optimized_nodes, optimized_edges = kg.nodes, kg.edges
                is_truncated = getattr(kg, "is_truncated", False)

            result = self._convert_graph_to_dict(optimized_nodes, optimized_edges, is_truncated)

            logger.info(
                f"Retrieved {mode_description} graph for collection {collection_id}: "
                f"{len(result['nodes'])} nodes, {len(result['edges'])} edges"
            )
            return result
        finally:
            await rag.finalize_storages()

    def _convert_graph_to_dict(self, nodes, edges, is_truncated=False) -> Dict[str, Any]:
        """Convert LightRAG graph objects to dictionary format"""

        def extract_properties(obj, default_fields):
            if hasattr(obj, "properties") and obj.properties:
                return obj.properties
            return {field: getattr(obj, field, None) for field in default_fields if hasattr(obj, field)}

        return {
            "nodes": [
                {
                    "id": node.id,
                    "labels": [node.id] if hasattr(node, "id") else [],
                    "properties": extract_properties(
                        node, ["entity_id", "entity_type", "description", "source_id", "file_path"]
                    ),
                }
                for node in nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "type": getattr(edge, "type", "DIRECTED"),
                    "source": edge.source,
                    "target": edge.target,
                    "properties": extract_properties(
                        edge, ["weight", "description", "keywords", "source_id", "file_path"]
                    ),
                }
                for edge in edges
            ],
            "is_truncated": is_truncated,
        }

    async def get_or_generate_merge_suggestions(
        self,
        user_id: str,
        collection_id: str,
        max_suggestions: int = 10,
        max_concurrent_llm_calls: int = 4,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Get cached suggestions or generate new ones"""
        await self._get_and_validate_collection(user_id, collection_id)

        # Check cache first
        if not force_refresh:
            cached_suggestions = await self.db_ops.get_valid_suggestions(collection_id)
            if cached_suggestions:
                logger.info(f"Found {len(cached_suggestions)} cached suggestions for collection {collection_id}")
                return self._format_suggestions_response(cached_suggestions, from_cache=True)

        # Generate new suggestions
        logger.info(f"Generating new merge suggestions for collection {collection_id}")
        llm_result = await self.generate_merge_suggestions(
            user_id, collection_id, max_suggestions, max_concurrent_llm_calls
        )

        # Prepare suggestion data
        suggestion_data = [
            {
                "collection_id": collection_id,
                "entity_ids": [entity["entity_id"] for entity in suggestion["entities"]],
                "confidence_score": suggestion["confidence_score"],
                "merge_reason": suggestion["merge_reason"],
                "suggested_target_entity": suggestion["suggested_target_entity"],
            }
            for suggestion in llm_result.get("suggestions", [])
        ]

        if suggestion_data:
            # If force_refresh=true, delete all existing suggestions before storing new ones
            if force_refresh:
                deleted_count = await self.db_ops.delete_all_suggestions_for_collection(collection_id)
                logger.info(
                    f"Deleted {deleted_count} existing suggestions for collection {collection_id} (force_refresh=true)"
                )

            # Store new suggestions
            stored_suggestions = await self.db_ops.batch_create_suggestions(suggestion_data)
            logger.info(f"Stored {len(stored_suggestions)} new suggestions for collection {collection_id}")
            # Extract metadata from llm_result, excluding 'suggestions' to avoid parameter conflict
            llm_metadata = {k: v for k, v in llm_result.items() if k != "suggestions"}
            return self._format_suggestions_response(stored_suggestions, from_cache=False, **llm_metadata)

        # Extract metadata from llm_result, excluding 'suggestions' to avoid parameter conflict
        llm_metadata = {k: v for k, v in llm_result.items() if k != "suggestions"}
        return self._format_suggestions_response([], from_cache=False, **llm_metadata)

    def _format_suggestions_response(self, suggestions: List, from_cache: bool = False, **kwargs) -> dict[str, Any]:
        """Format suggestions response with statistics"""
        suggestion_items = [
            {
                "id": suggestion.id,
                "collection_id": suggestion.collection_id,
                "suggestion_batch_id": suggestion.suggestion_batch_id,
                "entity_ids": suggestion.entity_ids,
                "confidence_score": float(suggestion.confidence_score),
                "merge_reason": suggestion.merge_reason,
                "suggested_target_entity": suggestion.suggested_target_entity,
                "status": suggestion.status,
                "created": suggestion.gmt_created,
                "expires_at": suggestion.expires_at,
                "operated_at": suggestion.operated_at,
            }
            for suggestion in suggestions
        ]

        # Count by status
        status_counts = {"PENDING": 0, "ACCEPTED": 0, "REJECTED": 0, "EXPIRED": 0}
        for suggestion in suggestions:
            status_counts[suggestion.status] += 1

        return {
            "suggestions": suggestion_items,
            "total_analyzed_nodes": kwargs.get("total_analyzed_nodes", 0),
            "processing_time_seconds": kwargs.get("processing_time_seconds", 0.0),
            "from_cache": from_cache,
            "generated_at": utc_now(),
            "total_suggestions": len(suggestion_items),
            "pending_count": status_counts["PENDING"],
            "accepted_count": status_counts["ACCEPTED"],
            "rejected_count": status_counts["REJECTED"],
            "expired_count": status_counts["EXPIRED"],
        }

    async def generate_merge_suggestions(
        self,
        user_id: str,
        collection_id: str,
        max_suggestions: int = 10,
        max_concurrent_llm_calls: int = 4,
    ) -> dict[str, Any]:
        """Generate node merge suggestions using LLM analysis"""
        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        rag = await lightrag_manager.create_lightrag_instance(db_collection)
        try:
            return await rag.agenerate_merge_suggestions(
                max_suggestions=max_suggestions,
                entity_types=None,  # Default to None (consider all entity types)
                debug_mode=False,  # Default to False
                max_concurrent_llm_calls=max_concurrent_llm_calls,
            )
        finally:
            await rag.finalize_storages()

    async def merge_nodes(
        self,
        user_id: str,
        collection_id: str,
        entity_ids: list[str] | None = None,
        suggestion_id: str | None = None,
        target_entity_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge graph nodes with suggestion or entity IDs (suggestion_id takes precedence)"""
        if not suggestion_id and not entity_ids:
            raise ValueError("Must specify either entity_ids or suggestion_id")

        db_collection = await self._get_and_validate_collection(user_id, collection_id)

        # Prepare merge parameters
        if suggestion_id:
            suggestions = await self.db_ops.get_suggestions_by_ids([suggestion_id])
            if not suggestions:
                raise ValueError(f"Suggestion not found: {suggestion_id}")

            suggestion = suggestions[0]
            merge_entity_ids = suggestion.entity_ids
            merge_target_data = target_entity_data or suggestion.suggested_target_entity
        else:
            merge_entity_ids = entity_ids
            merge_target_data = target_entity_data

        # Execute merge
        rag = await lightrag_manager.create_lightrag_instance(db_collection)
        try:
            result = await rag.amerge_nodes(
                entity_ids=merge_entity_ids,
                target_entity_data=merge_target_data,
            )

            result.update(
                {
                    "suggestion_id": suggestion_id,
                    "entity_ids": merge_entity_ids,
                }
            )

            # Update suggestion status if applicable
            if suggestion_id:
                await self.db_ops.update_suggestion_status(suggestion_id, MergeSuggestionStatus.ACCEPTED, utc_now())
                await self.db_ops.expire_related_suggestions(collection_id, merge_entity_ids)

            logger.info(f"Successfully merged entities {merge_entity_ids} in collection {collection_id}")
            return result
        finally:
            await rag.finalize_storages()

    async def _get_and_validate_collection(self, user_id: str, collection_id: str):
        """Get collection and validate knowledge graph is enabled"""
        try:
            view_collection = await self.collection_service.get_collection(user_id, collection_id)
        except Exception:
            raise CollectionNotFoundException(collection_id)

        if not view_collection.config or not view_collection.config.enable_knowledge_graph:
            raise ValueError(f"Knowledge graph is not enabled for collection {collection_id}")

        db_collection = await self.collection_service.db_ops.query_collection(user_id, collection_id)
        if not db_collection:
            raise CollectionNotFoundException(collection_id)

        return db_collection


# Global service instance
graph_service = GraphService()
