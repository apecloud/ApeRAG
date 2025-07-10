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
from typing import Any, Dict

from aperag.exceptions import CollectionNotFoundException, GraphServiceError
from aperag.graph import lightrag_manager
from aperag.schema import view_models

logger = logging.getLogger(__name__)


class GraphService:
    """Service for handling knowledge graph operations and index management"""

    def __init__(self):
        # Import here to avoid circular imports
        from aperag.service.collection_service import collection_service

        self.collection_service = collection_service

    # ==================== Graph Query Operations ====================

    async def get_graph_labels(self, user_id: str, collection_id: str) -> view_models.GraphLabelsResponse:
        """
        Get all available node labels in the collection's knowledge graph

        Args:
            user_id: User ID
            collection_id: Collection ID

        Returns:
            GraphLabelsResponse: Response containing available labels

        Raises:
            CollectionNotFoundException: If collection is not found
            ValueError: If knowledge graph is not enabled for the collection
        """
        # Get and validate collection
        collection = await self._get_and_validate_collection(user_id, collection_id)

        try:
            # Create LightRAG instance
            rag = await lightrag_manager.create_lightrag_instance(collection)

            # Get all available labels
            labels = await rag.get_graph_labels()

            # Clean up
            await rag.finalize_storages()

            return view_models.GraphLabelsResponse(labels=labels)

        except Exception as e:
            logger.error(f"Failed to get graph labels for collection {collection_id}: {str(e)}")
            raise

    def _optimize_graph_for_visualization(self, nodes, edges, max_nodes):
        """
        Optimize graph for visualization by prioritizing well-connected nodes

        Strategy:
        1. Calculate degree for each node
        2. Prefer nodes with higher connectivity (not isolated)
        3. Keep edges only between selected nodes

        Args:
            nodes: List of KnowledgeGraphNode objects
            edges: List of KnowledgeGraphEdge objects
            max_nodes: Maximum number of nodes to keep

        Returns:
            Tuple of (optimized_nodes, optimized_edges)
        """
        if len(nodes) <= max_nodes:
            return nodes, edges

        # Build degree map: node_id -> degree count
        degree_map = {}
        edge_map = {}  # node_id -> set of connected node_ids

        # Initialize all nodes with degree 0
        for node in nodes:
            degree_map[node.id] = 0
            edge_map[node.id] = set()

        # Count degrees from edges
        for edge in edges:
            source_id = edge.source
            target_id = edge.target

            if source_id in degree_map and target_id in degree_map:
                degree_map[source_id] += 1
                degree_map[target_id] += 1
                edge_map[source_id].add(target_id)
                edge_map[target_id].add(source_id)

        # Sort nodes by degree (descending), then by id for deterministic ordering
        sorted_nodes = sorted(nodes, key=lambda node: (-degree_map[node.id], node.id))

        # Select top nodes by degree
        selected_nodes = sorted_nodes[:max_nodes]
        selected_node_ids = {node.id for node in selected_nodes}

        # Filter edges to only include those between selected nodes
        optimized_edges = [
            edge for edge in edges if edge.source in selected_node_ids and edge.target in selected_node_ids
        ]

        logger.debug(
            f"Graph optimization: {len(nodes)} -> {len(selected_nodes)} nodes, "
            f"{len(edges)} -> {len(optimized_edges)} edges. "
            f"Degree range: {degree_map[sorted_nodes[0].id]} -> {degree_map[sorted_nodes[-1].id] if sorted_nodes else 0}"
        )

        return selected_nodes, optimized_edges

    async def get_knowledge_graph(
        self,
        user_id: str,
        collection_id: str,
        label: str = None,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> Dict[str, Any]:
        """
        Get knowledge graph - supports both overview and subgraph modes

        Args:
            user_id: User ID
            collection_id: Collection ID
            label: Label of the starting node. If None/empty, uses overview mode ("*")
            max_depth: Maximum depth of the subgraph (default: 3)
            max_nodes: Maximum number of nodes to return (default: 1000)

        Returns:
            Dict containing knowledge graph data with nodes and edges

        Raises:
            CollectionNotFoundException: If collection is not found
            ValueError: If knowledge graph is not enabled for the collection
        """
        # Get and validate collection
        collection = await self._get_and_validate_collection(user_id, collection_id)

        try:
            # Create LightRAG instance
            rag = await lightrag_manager.create_lightrag_instance(collection)

            # Determine mode based on label parameter
            if not label or label == "*":
                # Overview mode: use "*" to get entire graph
                node_label = "*"
                # Get more nodes for optimization in overview mode
                query_max_nodes = max_nodes * 2
                mode_description = "overview (full graph)"
            else:
                # Subgraph mode: use specific label
                node_label = label
                query_max_nodes = max_nodes
                mode_description = f"subgraph from '{label}'"

            # Get knowledge graph
            kg = await rag.get_knowledge_graph(
                node_label=node_label,
                max_depth=max_depth,
                max_nodes=query_max_nodes,
            )

            # Clean up
            await rag.finalize_storages()

            # Optimize node selection for overview mode if needed
            if (not label or label == "*") and len(kg.nodes) > max_nodes:
                optimized_nodes, optimized_edges = self._optimize_graph_for_visualization(kg.nodes, kg.edges, max_nodes)
                is_truncated = True
            else:
                optimized_nodes = kg.nodes
                optimized_edges = kg.edges
                is_truncated = kg.is_truncated if hasattr(kg, "is_truncated") else False

            # Convert to dict format expected by frontend
            result = self._convert_graph_to_dict(optimized_nodes, optimized_edges, is_truncated=is_truncated)

            logger.info(
                f"Retrieved knowledge graph for collection {collection_id} ({mode_description}): "
                f"{len(result['nodes'])} nodes, {len(result['edges'])} edges "
                f"(truncated: {is_truncated})"
            )

            return result

        except Exception as e:
            logger.error(
                f"Failed to get knowledge graph for collection {collection_id}, mode '{mode_description}': {str(e)}"
            )
            raise

    def _convert_graph_to_dict(self, nodes, edges, is_truncated=False) -> Dict[str, Any]:
        """
        Convert LightRAG graph nodes and edges to dictionary format

        Args:
            nodes: List of LightRAG node objects
            edges: List of LightRAG edge objects
            is_truncated: Whether the graph was truncated

        Returns:
            Dict with 'nodes', 'edges', and 'is_truncated' keys
        """
        result = {
            "nodes": [],
            "edges": [],
            "is_truncated": is_truncated,
        }

        # Convert nodes to dict format
        for node in nodes:
            node_dict = {"id": node.id, "labels": [node.id] if hasattr(node, "id") else [], "properties": {}}

            # Add properties from the node
            if hasattr(node, "properties") and node.properties:
                node_dict["properties"] = node.properties
            else:
                # Fallback: build properties from individual fields
                properties = {}
                if hasattr(node, "entity_id"):
                    properties["entity_id"] = node.entity_id
                if hasattr(node, "entity_type"):
                    properties["entity_type"] = node.entity_type
                if hasattr(node, "description"):
                    properties["description"] = node.description
                if hasattr(node, "source_id"):
                    properties["source_id"] = node.source_id
                if hasattr(node, "file_path"):
                    properties["file_path"] = node.file_path
                node_dict["properties"] = properties

            result["nodes"].append(node_dict)

        # Convert edges to dict format
        for edge in edges:
            edge_dict = {
                "id": edge.id,
                "type": getattr(edge, "type", "DIRECTED"),
                "source": edge.source,
                "target": edge.target,
                "properties": {},
            }

            # Add properties from the edge
            if hasattr(edge, "properties") and edge.properties:
                edge_dict["properties"] = edge.properties
            else:
                # Fallback: build properties from individual fields
                properties = {}
                if hasattr(edge, "weight"):
                    properties["weight"] = edge.weight
                if hasattr(edge, "description"):
                    properties["description"] = edge.description
                if hasattr(edge, "keywords"):
                    properties["keywords"] = edge.keywords
                if hasattr(edge, "source_id"):
                    properties["source_id"] = edge.source_id
                if hasattr(edge, "file_path"):
                    properties["file_path"] = edge.file_path
                edge_dict["properties"] = properties

            result["edges"].append(edge_dict)

        return result

    # ==================== Graph Index Operations ====================

    async def generate_merge_suggestions(
        self,
        user_id: str,
        collection_id: str,
        max_suggestions: int = 10,
        entity_types: list[str] | None = None,
        debug_mode: bool = False,
        max_concurrent_llm_calls: int = 4,
    ) -> dict[str, Any]:
        """
        Generate node merge suggestions using LLM analysis.

        Args:
            user_id: User ID
            collection_id: Collection ID
            max_suggestions: Maximum number of suggestions to return (default: 10)
            entity_types: Optional filter for specific entity types
            debug_mode: Enable debug mode with lower confidence threshold and verbose logging
            max_concurrent_llm_calls: Maximum concurrent LLM calls for batch analysis (default: 4)

        Returns:
            Dict containing merge suggestions and processing statistics

        Raises:
            CollectionNotFoundException: If collection is not found
            ValueError: If knowledge graph is not enabled for the collection
            GraphServiceError: If suggestion generation fails
        """
        # Get and validate collection
        collection = await self._get_and_validate_collection(user_id, collection_id)

        try:
            # Create LightRAG instance
            rag = await lightrag_manager.create_lightrag_instance(collection)

            # Call LightRAG method to generate suggestions
            result = await rag.agenerate_merge_suggestions(
                max_suggestions=max_suggestions,
                entity_types=entity_types,
                debug_mode=debug_mode,
                max_concurrent_llm_calls=max_concurrent_llm_calls,
            )

            return result

        except Exception as e:
            logger.error(f"Failed to generate merge suggestions for collection {collection_id}: {str(e)}")
            raise GraphServiceError(f"Failed to generate merge suggestions: {str(e)}") from e
        finally:
            if "rag" in locals():
                await rag.finalize_storages()

    async def merge_nodes(
        self,
        user_id: str,
        collection_id: str,
        entity_ids: list[str],
        target_entity_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Merge multiple graph nodes using LightRAG.

        Args:
            user_id: User ID
            collection_id: Collection ID
            entity_ids: List of entity IDs to merge
            target_entity_data: Optional target entity configuration including entity_name and other properties

        Returns:
            Dict containing merge operation results

        Raises:
            CollectionNotFoundException: If collection is not found
            ValueError: If knowledge graph is not enabled for the collection
            GraphServiceError: If merge operation fails
        """
        # Get and validate collection
        collection = await self._get_and_validate_collection(user_id, collection_id)

        try:
            # Create LightRAG instance
            rag = await lightrag_manager.create_lightrag_instance(collection)

            # Call LightRAG merge method
            result = await rag.amerge_nodes(
                entity_ids=entity_ids,
                target_entity_data=target_entity_data,
            )

            return result

        except Exception as e:
            logger.error(f"Failed to merge nodes in collection {collection_id}: {str(e)}")
            raise GraphServiceError(f"Failed to merge nodes: {str(e)}") from e
        finally:
            if "rag" in locals():
                await rag.finalize_storages()

    # ==================== Common Helper Methods ====================

    async def _get_and_validate_collection(self, user_id: str, collection_id: str):
        """
        Get collection database model and validate that knowledge graph is enabled

        Args:
            user_id: User ID
            collection_id: Collection ID

        Returns:
            Collection database model (needed for lightrag_manager)

        Raises:
            CollectionNotFoundException: If collection is not found
            ValueError: If knowledge graph is not enabled
        """
        # Validate that user has access to the collection
        try:
            view_collection: view_models.Collection = await self.collection_service.get_collection(
                user_id, collection_id
            )
        except Exception:
            raise CollectionNotFoundException(collection_id)

        # Check if knowledge graph is enabled in the view model
        if view_collection.config:
            config = view_collection.config
            if not config.enable_knowledge_graph:
                raise ValueError(f"Knowledge graph is not enabled for collection {collection_id}")
        else:
            raise ValueError(f"Knowledge graph is not enabled for collection {collection_id}")

        # Get the database model (needed for lightrag_manager which expects config as JSON string)
        db_collection = await self.collection_service.db_ops.query_collection(user_id, collection_id)
        if not db_collection:
            raise CollectionNotFoundException(collection_id)

        return db_collection


# Global service instances (maintaining backward compatibility)
graph_service = GraphService()
