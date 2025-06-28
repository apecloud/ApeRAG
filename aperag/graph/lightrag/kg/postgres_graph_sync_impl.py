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

"""
LightRAG Module for ApeRAG

This module is based on the original LightRAG project with extensive modifications.

Original Project:
- Repository: https://github.com/HKUDS/LightRAG
- Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (arXiv:2410.05779)
- Authors: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- License: MIT License

Modifications by ApeRAG Team:
- Removed global state management for true concurrent processing
- Added stateless interfaces for Celery/Prefect integration
- Implemented instance-level locking mechanism
- Enhanced error handling and stability
- See changelog.md for detailed modifications
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import final

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger

# Import client manager
try:
    from aperag.db.postgres_graph_sync_manager import PostgreSQLGraphClientManager, PostgreSQLGraphDB
except ImportError:
    PostgreSQLGraphClientManager = None
    PostgreSQLGraphDB = None


@final
@dataclass
class PostgreSQLGraphSyncStorage(BaseGraphStorage):
    """
    PostgreSQL graph storage implementation using workspace-agnostic global connection pool.
    Uses native PostgreSQL tables with individual fields for optimal query performance.
    """

    def __init__(self, namespace, workspace, embedding_func=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=None,
        )

    async def initialize(self):
        """Initialize storage."""
        logger.debug(f"PostgreSQLGraphSyncStorage initialized for workspace '{self.workspace}'")

    async def finalize(self):
        """Clean up resources."""
        logger.debug(f"PostgreSQLGraphSyncStorage finalized for workspace '{self.workspace}'")

    async def _get_db(self):
        """Get database connection from global client manager."""
        if PostgreSQLGraphClientManager is None:
            raise RuntimeError("PostgreSQL Graph client manager is not available")
        return await PostgreSQLGraphClientManager.get_client()

    #################### upsert method ################
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database - using individual fields for optimal performance."""
        # Only set entity_name if it's explicitly provided and different from entity_id
        entity_name = node_data.get("entity_name") if node_data.get("entity_name") != node_id else None
        entity_type = node_data.get("entity_type") or None
        description = node_data.get("description") or None  
        source_id = node_data.get("source_id") or None
        file_path = node_data.get("file_path") or None
        
        logger.debug(f"Upserted node with entity_id '{node_id}', entity_type '{entity_type}'")
        
        sql = """
            INSERT INTO LIGHTRAG_GRAPH_NODES(entity_id, entity_name, entity_type, description, source_id, file_path, workspace, createtime, updatetime)
            VALUES(%(entity_id)s, %(entity_name)s, %(entity_type)s, %(description)s, %(source_id)s, %(file_path)s, %(workspace)s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (workspace, entity_id) DO UPDATE SET
                entity_name = EXCLUDED.entity_name,
                entity_type = EXCLUDED.entity_type,
                description = EXCLUDED.description,
                source_id = EXCLUDED.source_id,
                file_path = EXCLUDED.file_path,
                updatetime = CURRENT_TIMESTAMP
        """
        data = {
            "workspace": self.workspace,
            "entity_id": node_id,
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "source_id": source_id,
            "file_path": file_path,
        }
        
        db = await self._get_db()
        async with db.get_transaction() as tx:
            await tx.execute(sql, data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """Upsert an edge between two nodes - using individual fields for optimal performance."""
        weight = float(edge_data.get("weight", 0.0))
        keywords = edge_data.get("keywords") or None  # Keep None as None, don't convert to empty string
        description = edge_data.get("description") or None
        source_id = edge_data.get("source_id") or None
        file_path = edge_data.get("file_path") or None
        
        logger.debug(f"Upserted edge from '{source_node_id}' to '{target_node_id}'")
        
        sql = """
            INSERT INTO LIGHTRAG_GRAPH_EDGES(source_entity_id, target_entity_id, weight, keywords, description, source_id, file_path, workspace, createtime, updatetime)
            VALUES(%(source_entity_id)s, %(target_entity_id)s, %(weight)s, %(keywords)s, %(description)s, %(source_id)s, %(file_path)s, %(workspace)s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (workspace, source_entity_id, target_entity_id) DO UPDATE SET
                weight = EXCLUDED.weight,
                keywords = EXCLUDED.keywords,
                description = EXCLUDED.description,
                source_id = EXCLUDED.source_id,
                file_path = EXCLUDED.file_path,
                updatetime = CURRENT_TIMESTAMP
        """
        data = {
            "workspace": self.workspace,
            "source_entity_id": source_node_id,
            "target_entity_id": target_node_id,
            "weight": weight,
            "keywords": keywords,
            "description": description,
            "source_id": source_id,
            "file_path": file_path,
        }
        
        db = await self._get_db()
        async with db.get_transaction() as tx:
            await tx.execute(sql, data)

    # Query methods
    async def has_node(self, node_id: str) -> bool:
        sql = """
            SELECT COUNT(id) AS cnt FROM LIGHTRAG_GRAPH_NODES 
            WHERE workspace = %(workspace)s AND entity_id = %(entity_id)s
        """
        param = {"entity_id": node_id, "workspace": self.workspace}
        
        db = await self._get_db()
        async with db.get_transaction() as tx:
            result = await tx.query(sql, param)
            return result["cnt"] != 0 if result else False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        sql = """
            SELECT COUNT(id) AS cnt FROM LIGHTRAG_GRAPH_EDGES 
            WHERE workspace = %(workspace)s AND source_entity_id = %(source_entity_id)s AND target_entity_id = %(target_entity_id)s
        """
        param = {
            "source_entity_id": source_node_id,
            "target_entity_id": target_node_id,
            "workspace": self.workspace,
        }
        
        db = await self._get_db()
        async with db.get_transaction() as tx:
            result = await tx.query(sql, param)
            return result["cnt"] != 0 if result else False

    async def node_degree(self, node_id: str) -> int:
        # Use UNION ALL for better index utilization (workspace + source/target indexes)
        sql = """
            SELECT COUNT(*) AS cnt FROM (
                SELECT 1 FROM LIGHTRAG_GRAPH_EDGES 
                WHERE workspace = %(workspace)s AND source_entity_id = %(entity_id)s
                UNION ALL
                SELECT 1 FROM LIGHTRAG_GRAPH_EDGES 
                WHERE workspace = %(workspace)s AND target_entity_id = %(entity_id)s
            ) AS degree_count
        """
        param = {"entity_id": node_id, "workspace": self.workspace}
        
        db = await self._get_db()
        async with db.get_transaction() as tx:
            result = await tx.query(sql, param)
            return result["cnt"] if result else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        sql = """
            SELECT entity_id, entity_name, entity_type, description, source_id, file_path, 
                   EXTRACT(EPOCH FROM createtime)::INTEGER as created_at
            FROM LIGHTRAG_GRAPH_NODES WHERE workspace = %(workspace)s AND entity_id = %(entity_id)s
        """
        param = {"entity_id": node_id, "workspace": self.workspace}
        
        db = await self._get_db()
        async with db.get_transaction() as tx:
            result = await tx.query(sql, param)
            if result:
                # Return node data assembled from individual fields - similar to Neo4j format
                node_dict = {
                    "entity_id": result["entity_id"],
                    "entity_type": result["entity_type"],
                    "description": result["description"],
                    "source_id": result["source_id"],
                    "file_path": result["file_path"],
                    "created_at": result["created_at"],
                }
                # Only include entity_name if it's different from entity_id and not None
                if result["entity_name"] and result["entity_name"] != result["entity_id"]:
                    node_dict["entity_name"] = result["entity_name"]
                
                # Remove None values for cleaner output
                return {k: v for k, v in node_dict.items() if v is not None}
            return None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        sql = """
            SELECT source_entity_id, target_entity_id, weight, keywords, description, source_id, file_path
            FROM LIGHTRAG_GRAPH_EDGES WHERE workspace = %(workspace)s AND source_entity_id = %(source_entity_id)s AND target_entity_id = %(target_entity_id)s
        """
        param = {
            "source_entity_id": source_node_id,
            "target_entity_id": target_node_id,
            "workspace": self.workspace,
        }
        
        db = await self._get_db()
        async with db.get_transaction() as tx:
            result = await tx.query(sql, param)
            if result:
                # Return edge data assembled from individual fields - similar to Neo4j format
                # Ensure required keys exist with defaults (matching Neo4j behavior)
                edge_result = {
                    "weight": float(result["weight"]) if result["weight"] is not None else 0.0,
                    "keywords": result["keywords"],  # Keep None as None
                    "description": result["description"],
                    "source_id": result["source_id"],
                    "file_path": result["file_path"],
                }
                # Only remove None values for optional fields, keep required fields even if None
                filtered_result = {}
                required_fields = {"weight", "keywords", "description", "source_id"}
                for k, v in edge_result.items():
                    if k in required_fields or v is not None:
                        filtered_result[k] = v
                
                return filtered_result
            return None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        db = await self._get_db()
        
        async with db.get_transaction() as tx:
            sql = """
                SELECT source_entity_id, target_entity_id
                FROM LIGHTRAG_GRAPH_EDGES WHERE workspace = %(workspace)s AND source_entity_id = %(source_entity_id)s
            """
            param = {"source_entity_id": source_node_id, "workspace": self.workspace}
            results = await tx.query(sql, param, multirows=True)
            edges = []
            if results:
                edges.extend([(r["source_entity_id"], r["target_entity_id"]) for r in results])
            
            # Also get incoming edges
            sql_incoming = """
                SELECT source_entity_id, target_entity_id
                FROM LIGHTRAG_GRAPH_EDGES WHERE workspace = %(workspace)s AND target_entity_id = %(target_entity_id)s
            """
            incoming_results = await tx.query(sql_incoming, {"target_entity_id": source_node_id, "workspace": self.workspace}, multirows=True)
            if incoming_results:
                edges.extend([(r["source_entity_id"], r["target_entity_id"]) for r in incoming_results])
            
            return edges if edges else None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes in batch (simplified implementation)."""
        nodes = {}
        for node_id in node_ids:
            node_data = await self.get_node(node_id)
            if node_data:
                nodes[node_id] = node_data
        return nodes

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieve degrees for multiple nodes."""
        degrees = {}
        for node_id in node_ids:
            degrees[node_id] = await self.node_degree(node_id)
        return degrees

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """Calculate combined degrees for edges."""
        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = await self.edge_degree(src, tgt)
        return edge_degrees

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple pairs."""
        edges_dict = {}
        for pair in pairs:
            src, tgt = pair["src"], pair["tgt"]
            edge_data = await self.get_edge(src, tgt)
            if edge_data:
                edges_dict[(src, tgt)] = edge_data
            else:
                # Return default structure with required fields (matching Neo4j behavior)
                edges_dict[(src, tgt)] = {
                    "weight": 0.0,
                    "keywords": None,
                    "description": None,
                    "source_id": None,
                }
        return edges_dict

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes."""
        edges_dict = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            edges_dict[node_id] = edges or []
        return edges_dict

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its related edges in a single transaction"""
        db = await self._get_db()
        
        async with db.get_transaction() as tx:
            # First delete all edges related to this node
            delete_edges_sql = """
                DELETE FROM LIGHTRAG_GRAPH_EDGES
                WHERE workspace = %(workspace)s AND (source_entity_id = %(entity_id)s OR target_entity_id = %(entity_id)s)
            """
            await tx.execute(
                delete_edges_sql,
                {"entity_id": node_id, "workspace": self.workspace},
            )

            # Then delete the node itself
            delete_node_sql = """
                DELETE FROM LIGHTRAG_GRAPH_NODES
                WHERE workspace = %(workspace)s AND entity_id = %(entity_id)s
            """
            await tx.execute(
                delete_node_sql,
                {"entity_id": node_id, "workspace": self.workspace},
            )

        logger.debug(
            f"Node {node_id} and its related edges have been deleted from the graph"
        )

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes"""
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges in a single transaction"""
        db = await self._get_db()
        
        async with db.get_transaction() as tx:
            sql = """
                DELETE FROM LIGHTRAG_GRAPH_EDGES
                WHERE workspace = %(workspace)s AND source_entity_id = %(source_entity_id)s AND target_entity_id = %(target_entity_id)s
            """
            for source, target in edges:
                await tx.execute(
                    sql,
                    {"source_entity_id": source, "target_entity_id": target, "workspace": self.workspace},
                )

    async def get_all_labels(self) -> list[str]:
        """Get all entity names in the database"""
        sql = """
            SELECT DISTINCT entity_id as label
            FROM LIGHTRAG_GRAPH_NODES
            WHERE workspace = %(workspace)s
            ORDER BY entity_id
        """
        
        db = await self._get_db()
        async with db.get_transaction() as tx:
            result = await tx.query(
                sql,
                {"workspace": self.workspace},
                multirows=True,
            )

            if not result:
                return []

            # Extract all labels (entity names)
            return [item["label"] for item in result]

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int = 3, max_nodes: int = 1000
    ) -> KnowledgeGraph:
        """
        Get a connected subgraph of nodes matching the specified label
        """
        result = KnowledgeGraph()
        MAX_GRAPH_NODES = max_nodes

        db = await self._get_db()
        
        async with db.get_transaction() as tx:
            # Get matching nodes
            if node_label == "*":
                # Handle special case, get all nodes
                sql = """
                    SELECT entity_id, entity_name, entity_type, description, source_id, file_path FROM LIGHTRAG_GRAPH_NODES
                    WHERE workspace = %(workspace)s
                    ORDER BY entity_id
                    LIMIT %(max_nodes)s
                """
                node_results = await tx.query(
                    sql,
                    {"workspace": self.workspace, "max_nodes": MAX_GRAPH_NODES},
                    multirows=True,
                )
            else:
                # Get nodes matching the label
                label_pattern = f"%{node_label}%"
                sql = """
                    SELECT entity_id, entity_name, entity_type, description, source_id, file_path FROM LIGHTRAG_GRAPH_NODES
                    WHERE workspace = %(workspace)s AND entity_id LIKE %(label_pattern)s
                    ORDER BY entity_id
                """
                node_results = await tx.query(
                    sql,
                    {"workspace": self.workspace, "label_pattern": label_pattern},
                    multirows=True,
                )

        if not node_results:
            logger.warning(f"No nodes found matching label {node_label}")
            return result

        # Limit the number of returned nodes
        if len(node_results) > MAX_GRAPH_NODES:
            node_results = node_results[:MAX_GRAPH_NODES]

        # Extract node names for edge query
        node_names = [node["entity_id"] for node in node_results]

        # Add nodes to result
        for node in node_results:
            # Assemble properties from individual fields
            properties = {
                "entity_id": node["entity_id"],
                "entity_type": node["entity_type"],
                "description": node["description"],
                "source_id": node["source_id"],
                "file_path": node["file_path"],
            }
            # Only include entity_name if it's different from entity_id and not None
            if node["entity_name"] and node["entity_name"] != node["entity_id"]:
                properties["entity_name"] = node["entity_name"]
            
            # Remove None values for cleaner output
            properties = {k: v for k, v in properties.items() if v is not None}
            
            result.nodes.append(
                KnowledgeGraphNode(
                    id=node["entity_id"],
                    labels=[node.get("entity_type", node["entity_id"])],
                    properties=properties,
                )
            )

            # Get related edges
            if node_names:
                sql = """
                    SELECT source_entity_id, target_entity_id, weight, keywords, description, source_id, file_path FROM LIGHTRAG_GRAPH_EDGES
                    WHERE workspace = %(workspace)s 
                    AND (source_entity_id = ANY(%(node_names)s) OR target_entity_id = ANY(%(node_names)s))
                """
                edge_results = await tx.query(
                    sql,
                    {"workspace": self.workspace, "node_names": node_names},
                    multirows=True,
                )

            if edge_results:
                # Add edges to result
                for edge in edge_results:
                    # Only include edges related to selected nodes
                    if (
                        edge["source_entity_id"] in node_names
                        and edge["target_entity_id"] in node_names
                    ):
                        edge_id = f"{edge['source_entity_id']}-{edge['target_entity_id']}"
                        
                        # Assemble edge properties from individual fields
                        edge_properties = {
                            "weight": float(edge["weight"]) if edge["weight"] is not None else 0.0,
                            "keywords": edge["keywords"],
                            "description": edge["description"],
                            "source_id": edge["source_id"],
                            "file_path": edge["file_path"],
                        }
                        # Remove None values for cleaner output
                        edge_properties = {k: v for k, v in edge_properties.items() if v is not None}

                        result.edges.append(
                            KnowledgeGraphEdge(
                                id=edge_id,
                                type="DIRECTED",
                                source=edge["source_entity_id"],
                                target=edge["target_entity_id"],
                                properties=edge_properties,
                            )
                        )

        logger.info(
            f"Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def drop(self) -> dict[str, str]:
        """Drop the storage in a single transaction"""
        try:
            db = await self._get_db()
            
            async with db.get_transaction() as tx:
                # Delete all edges for this workspace
                await tx.execute(
                    "DELETE FROM LIGHTRAG_GRAPH_EDGES WHERE workspace = %(workspace)s",
                    {"workspace": self.workspace}
                )
                # Delete all nodes for this workspace
                await tx.execute(
                    "DELETE FROM LIGHTRAG_GRAPH_NODES WHERE workspace = %(workspace)s", 
                    {"workspace": self.workspace}
                )
            
            logger.info(f"Successfully dropped all data for workspace {self.workspace}")
            return {"status": "success", "message": "graph data dropped"}
        except Exception as e:
            logger.error(f"Error dropping graph for workspace {self.workspace}: {e}")
            return {"status": "error", "message": str(e)}



