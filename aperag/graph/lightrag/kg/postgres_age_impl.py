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
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, final

import psycopg
from psycopg.rows import namedtuple_row
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aperag.db.postgres_age_sync_manager import PostgreSQLAGESyncConnectionManager

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger

# Set psycopg logger level to ERROR to suppress warning logs
logging.getLogger("psycopg").setLevel(logging.WARNING)


class AGEQueryException(Exception):
    """Exception for AGE queries."""

    def __init__(self, exception: str) -> None:
        self.message = exception

    def get_message(self) -> str:
        return self.message


@final
@dataclass
class PostgreSQLAGEStorage(BaseGraphStorage):
    """
    PostgreSQL AGE storage implementation using sync driver with async interface.
    This avoids event loop issues while maintaining compatibility with async code.
    
    Performance Strategy:
    - Simple queries over complex batch operations (AGE limitations)
    - Fast fallback from batch to individual operations
    - Increased timeouts for AGE query processing
    - Minimal result parsing overhead
    """

    def __init__(self, namespace, workspace, embedding_func=None):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=None,
        )
        self._graph_name = None

    @staticmethod
    def _encode_graph_label(label: str) -> str:
        """
        Since AGE supports only alphanumerical labels, encode as HEX string.
        """
        return "x" + label.encode().hex()

    @staticmethod
    def _decode_graph_label(encoded_label: str) -> str:
        """
        Decode HEX string back to original label.
        """
        return bytes.fromhex(encoded_label.removeprefix("x")).decode()

    @staticmethod
    def _format_properties(properties: Dict[str, Any]) -> str:
        """
        Convert properties dictionary to AGE format string.
        """
        props = []
        for k, v in properties.items():
            prop = f"`{k}`: {json.dumps(v)}"
            props.append(prop)
        return "{" + ", ".join(props) + "}"

    def _wrap_cypher_query(self, cypher_query: str) -> str:
        """
        Wrap cypher query for AGE execution with minimal field specification.
        """
        # Count return fields to build proper AGE schema
        if "return" in cypher_query.lower():
            return_part = cypher_query.lower().split("return")[-1].split("order by")[0].split("limit")[0]
            field_count = len([f.strip() for f in return_part.split(",") if f.strip()])
            fields_str = ", ".join([f"field_{i} agtype" for i in range(field_count)])
        else:
            fields_str = "result agtype"

        return f"""
        SELECT * FROM ag_catalog.cypher('{self._graph_name}', $$
            {cypher_query}
        $$) AS ({fields_str})
        """

    async def _execute_cypher(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a cypher query with optimized parsing and error handling.
        """
        def _sync_execute():
            try:
                wrapped_query = self._wrap_cypher_query(cypher_query)
                
                with PostgreSQLAGESyncConnectionManager.get_cursor(row_factory=namedtuple_row) as (cur, conn):
                    # Conservative timeout for AGE to avoid deadlocks
                    cur.execute("SET statement_timeout = '30s'")
                    cur.execute("SET lock_timeout = '10s'")
                    cur.execute("SET idle_in_transaction_session_timeout = '30s'")
                    cur.execute(wrapped_query)
                    data = cur.fetchall()
                    conn.commit()
                    
                    if not data:
                        return []
                    
                    # Fast result parsing
                    results = []
                    for row in data:
                        row_dict = {}
                        for i, field_name in enumerate(row._fields):
                            value = row[i]
                            
                            # Fast path for simple values
                            if not isinstance(value, str) or "::" not in value:
                                row_dict[field_name] = value
                                continue
                            
                            # Parse AGE types
                            try:
                                value_part, dtype = value.split("::", 1)
                                if dtype == "vertex":
                                    vertex_data = json.loads(value_part)
                                    properties = vertex_data.get("properties", {})
                                                    # Remove label field addition - keep only original properties
                                    row_dict[field_name] = properties
                                elif dtype == "edge":
                                    edge_data = json.loads(value_part)
                                    row_dict[field_name] = edge_data.get("properties", {})
                                else:
                                    try:
                                        row_dict[field_name] = json.loads(value_part)
                                    except:
                                        row_dict[field_name] = value_part
                            except Exception:
                                row_dict[field_name] = value
                        
                        results.append(row_dict)
                    
                    return results
                    
            except psycopg.Error as e:
                raise AGEQueryException(f"AGE query failed: {str(e)}")

        return await asyncio.to_thread(_sync_execute)

    async def initialize(self):
        """Initialize storage and prepare graph."""
        if PostgreSQLAGESyncConnectionManager is None:
            raise RuntimeError("PostgreSQL AGE sync connection manager is not available")

        self._graph_name = await asyncio.to_thread(
            PostgreSQLAGESyncConnectionManager.prepare_graph, self.workspace
        )

        logger.debug(f"PostgreSQLAGEStorage initialized for workspace '{self.workspace}', graph '{self._graph_name}'")

    async def finalize(self):
        """Clean up resources."""
        logger.debug(f"PostgreSQLAGEStorage finalized for workspace '{self.workspace}'")

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists using encoded label."""
        encoded_label = self._encode_graph_label(node_id.strip('"'))
        cypher = f"MATCH (n:`{encoded_label}`) RETURN count(n) AS node_count"
        
        try:
            results = await self._execute_cypher(cypher)
            count = int(results[0]["field_0"]) if results else 0
            return count > 0
        except Exception as e:
            logger.warning(f"Failed to check node {node_id}: {e}")
            return False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes using property matching."""
        # Use entity_id property matching like Neo4j for more reliable results
        cypher = f"MATCH (a {{entity_id: '{source_node_id}'}})-[r]-(b {{entity_id: '{target_node_id}'}}) RETURN COUNT(r) AS edge_count"
        
        try:
            results = await self._execute_cypher(cypher)
            count = int(results[0]["field_0"]) if results else 0
            return count > 0
        except Exception as e:
            logger.warning(f"Failed to check edge {source_node_id} -> {target_node_id}: {e}")
            return False

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by encoded label."""
        encoded_label = self._encode_graph_label(node_id.strip('"'))
        cypher = f"MATCH (n:`{encoded_label}`) RETURN n"
        
        try:
            results = await self._execute_cypher(cypher)
            if results and results[0]["field_0"]:
                node_data = results[0]["field_0"]
                # Ensure entity_id is present
                if "entity_id" not in node_data:
                    node_data["entity_id"] = node_id
                return node_data
        except Exception as e:
            logger.warning(f"Failed to get node {node_id}: {e}")
        return None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Retrieve multiple nodes. Use simple individual queries for reliability.
        AGE's complex batch queries are unreliable, so we prioritize correctness.
        """
        if not node_ids:
            return {}

        nodes = {}
        
        # For AGE, individual queries are more reliable than complex batch operations
        for node_id in node_ids:
            try:
                node_data = await self.get_node(node_id)
                if node_data:
                    nodes[node_id] = node_data
            except Exception as e:
                logger.warning(f"Failed to get node {node_id}: {e}")
                continue
                
        return nodes

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node using OPTIONAL MATCH like Neo4j."""
        encoded_label = self._encode_graph_label(node_id.strip('"'))
        
        # Use OPTIONAL MATCH for reliable degree calculation (similar to Neo4j)
        cypher = f"MATCH (n:`{encoded_label}`) OPTIONAL MATCH (n)-[r]-() RETURN count(r) AS degree"
        
        try:
            results = await self._execute_cypher(cypher)
            return int(results[0]["field_0"]) if results else 0
        except Exception as e:
            logger.warning(f"Failed to get degree for {node_id}: {e}")
            return 0

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """
        Retrieve degrees for multiple nodes.
        Uses individual queries for AGE reliability.
        """
        if not node_ids:
            return {}
            
        degrees = {}
        
        # AGE performs better with individual simple queries than complex batch operations
        for node_id in node_ids:
            try:
                degree = await self.node_degree(node_id)
                degrees[node_id] = degree
            except Exception as e:
                logger.warning(f"Failed to get degree for {node_id}: {e}")
                degrees[node_id] = 0
                
        return degrees

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of two nodes."""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return int(src_degree) + int(tgt_degree)

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """Calculate combined degrees for edges."""
        # Collect unique node IDs
        unique_node_ids = set()
        for src, tgt in edge_pairs:
            unique_node_ids.add(src)
            unique_node_ids.add(tgt)

        # Get degrees for all unique nodes
        degrees = await self.node_degrees_batch(list(unique_node_ids))

        # Calculate edge degrees
        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get edge properties between two nodes using property matching."""
        # Use entity_id property matching for consistency with has_edge
        cypher = f"MATCH (a {{entity_id: '{source_node_id}'}})-[r]-(b {{entity_id: '{target_node_id}'}}) RETURN properties(r) AS edge_props LIMIT 1"
        
        try:
            results = await self._execute_cypher(cypher)
            if results and results[0]["field_0"]:
                edge_data = results[0]["field_0"]
                
                # Handle case where edge_data might be a string (AGE parsing issue)
                if isinstance(edge_data, str):
                    try:
                        edge_data = json.loads(edge_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse edge data as JSON: {edge_data}")
                        edge_data = {}
                
                # Ensure it's a dictionary
                if not isinstance(edge_data, dict):
                    edge_data = {}
                
                # Ensure required keys exist with defaults
                required_keys = {
                    "weight": 0.0,
                    "source_id": None,
                    "description": None,
                    "keywords": None,
                }
                for key, default_value in required_keys.items():
                    if key not in edge_data:
                        edge_data[key] = default_value
                return edge_data
        except Exception as e:
            logger.warning(f"Failed to get edge {source_node_id} -> {target_node_id}: {e}")
        
        return None

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple pairs using individual queries."""
        if not pairs:
            return {}
            
        edges_dict = {}
        
        # Initialize with default values
        for pair in pairs:
            src, tgt = pair["src"], pair["tgt"]
            edges_dict[(src, tgt)] = {
                "weight": 0.0,
                "source_id": None,
                "description": None,
                "keywords": None,
            }
        
        # Use individual queries for AGE reliability
        for pair in pairs:
            src, tgt = pair["src"], pair["tgt"]
            try:
                edge_data = await self.get_edge(src, tgt)
                if edge_data and isinstance(edge_data, dict):
                    edges_dict[(src, tgt)] = edge_data
            except Exception as e:
                logger.warning(f"Failed to get edge {src} -> {tgt}: {e}")
                continue
                
        return edges_dict

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges for a node using Neo4j-like pattern."""
        encoded_label = self._encode_graph_label(source_node_id.strip('"'))
        
        # Use pattern similar to Neo4j with source node, relationship, and connected node
        cypher = f"MATCH (n:`{encoded_label}`) OPTIONAL MATCH (n)-[r]-(connected) WHERE connected.entity_id IS NOT NULL RETURN n, r, connected"
        
        try:
            results = await self._execute_cypher(cypher)
            edges = []
            seen_targets = set()  # Deduplicate
            
            for result in results:
                source_node = result.get("field_0")
                relationship = result.get("field_1") 
                connected_node = result.get("field_2")
                
                # Skip if no connection
                if not source_node or not connected_node:
                    continue
                
                # Extract entity_id from both nodes
                source_entity_id = source_node.get("entity_id") if isinstance(source_node, dict) else None
                target_entity_id = connected_node.get("entity_id") if isinstance(connected_node, dict) else None
                
                if source_entity_id and target_entity_id and target_entity_id not in seen_targets:
                    edges.append((source_entity_id, target_entity_id))
                    seen_targets.add(target_entity_id)
                        
            return edges if edges else None
            
        except Exception as e:
            logger.warning(f"Failed to get edges for {source_node_id}: {e}")
            return None

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes using individual queries."""
        if not node_ids:
            return {}
            
        edges_dict = {node_id: [] for node_id in node_ids}
        
        # Use individual queries for AGE reliability
        for node_id in node_ids:
            try:
                edges = await self.get_node_edges(node_id)
                edges_dict[node_id] = edges or []
            except Exception as e:
                logger.warning(f"Failed to get edges for {node_id}: {e}")
                edges_dict[node_id] = []
        
        return edges_dict

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AGEQueryException, psycopg.Error)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database."""
        encoded_label = self._encode_graph_label(node_id.strip('"'))
        properties = node_data.copy()
        
        # Ensure entity_id is in properties
        if "entity_id" not in properties:
            properties["entity_id"] = node_id

        cypher = f"MERGE (n:`{encoded_label}`) SET n += {self._format_properties(properties)}"
        
        await self._execute_cypher(cypher)
        logger.debug(f"Upserted node with ID '{node_id}'")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AGEQueryException, psycopg.Error)),
    )
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Upsert an edge between two nodes using property matching."""
        # Use entity_id property matching for consistency
        cypher = f"""
        MATCH (source {{entity_id: '{source_node_id}'}})
        WITH source
        MATCH (target {{entity_id: '{target_node_id}'}})
        MERGE (source)-[r:DIRECTED]-(target)
        SET r += {self._format_properties(edge_data)}
        """
        
        await self._execute_cypher(cypher)
        logger.debug(f"Upserted edge from '{source_node_id}' to '{target_node_id}'")

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """Retrieve a connected subgraph."""
        result = KnowledgeGraph()
        
        if node_label == "*":
            # Get all nodes with degree ordering
            cypher = f"""
            MATCH (n)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) AS degree
            ORDER BY degree DESC
            LIMIT {max_nodes}
            RETURN n
            """
        else:
            # Get subgraph starting from specific node
            encoded_label = self._encode_graph_label(node_label.strip('"'))
            cypher = f"""
            MATCH (start:`{encoded_label}`)
            OPTIONAL MATCH path = (start)-[*0..{max_depth}]-(connected)
            WITH nodes(path) as path_nodes, relationships(path) as path_rels
            LIMIT {max_nodes}
            RETURN path_nodes, path_rels
            """
        
        try:
            results = await self._execute_cypher(cypher)
            seen_nodes = set()
            seen_edges = set()
            
            for record in results:
                # Process nodes and edges from results
                # Implementation simplified for AGE limitations
                if "field_0" in record and record["field_0"]:
                    # Handle nodes
                    nodes_data = record["field_0"]
                    if isinstance(nodes_data, list):
                        for node in nodes_data:
                            if node and isinstance(node, dict):
                                node_id = str(node.get("id", ""))
                                if node_id and node_id not in seen_nodes:
                                    result.nodes.append(
                                        KnowledgeGraphNode(
                                            id=node_id,
                                            labels=[node.get("label", "")],
                                            properties=node,
                                        )
                                    )
                                    seen_nodes.add(node_id)
                    elif isinstance(nodes_data, dict):
                        node_id = str(nodes_data.get("id", ""))
                        if node_id and node_id not in seen_nodes:
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id,
                                    labels=[nodes_data.get("label", "")],
                                    properties=nodes_data,
                                )
                            )
                            seen_nodes.add(node_id)
            
            logger.info(f"Retrieved subgraph with {len(result.nodes)} nodes and {len(result.edges)} edges")
            
        except Exception as e:
            logger.error(f"Failed to get knowledge graph: {e}")
        
        return result

    async def get_all_labels(self) -> list[str]:
        """Get all node labels by querying entity_id property."""
        cypher = "MATCH (n) WHERE n.entity_id IS NOT NULL RETURN DISTINCT n.entity_id AS label"
        
        try:
            results = await self._execute_cypher(cypher)
            all_labels = []
            
            for record in results:
                if "field_0" in record and record["field_0"]:
                    label = record["field_0"]
                    if label:
                        all_labels.append(str(label))
            
            return sorted(list(set(all_labels)))
            
        except Exception as e:
            logger.error(f"Failed to get labels: {e}")
            return []

    async def delete_node(self, node_id: str) -> None:
        """Delete a node using encoded label."""
        encoded_label = self._encode_graph_label(node_id.strip('"'))
        
        # Delete edges first
        cypher_edges = f"MATCH (n:`{encoded_label}`)-[r]-() DELETE r"
        try:
            await self._execute_cypher(cypher_edges)
            logger.debug(f"Deleted edges for node {node_id}")
        except Exception as e:
            logger.warning(f"Failed to delete edges for node {node_id}: {e}")
        
        # Delete the node itself
        cypher_node = f"MATCH (n:`{encoded_label}`) DELETE n"
        await self._execute_cypher(cypher_node)
        logger.debug(f"Deleted node with ID '{node_id}'")

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes."""
        for node in nodes:
            try:
                await self.delete_node(node)
            except Exception as e:
                logger.warning(f"Failed to delete node {node}: {e}")
                continue

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges using property matching."""
        for source, target in edges:
            try:
                # Use entity_id property matching for consistency
                cypher = f"MATCH (a {{entity_id: '{source}'}})-[r]-(b {{entity_id: '{target}'}}) DELETE r"
                await self._execute_cypher(cypher)
                logger.debug(f"Deleted edge from '{source}' to '{target}'")
            except Exception as e:
                logger.warning(f"Failed to delete edge {source} -> {target}: {e}")
                continue

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage."""
        def _sync_drop():
            try:
                with PostgreSQLAGESyncConnectionManager.get_cursor() as (cur, conn):
                    cur.execute(f"SELECT drop_graph('{self._graph_name}', true)")
                    conn.commit()
                    logger.info(f"Dropped graph {self._graph_name}")
                    return {"status": "success", "message": "graph data dropped"}
            except Exception as e:
                logger.error(f"Error dropping graph {self._graph_name}: {e}")
                return {"status": "error", "message": str(e)}

        return await asyncio.to_thread(_sync_drop)