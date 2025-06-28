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
from typing import Any, Dict, List, NamedTuple, Optional, final

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
        Since AGE supports only alphanumerical labels, we will encode generic label as HEX string.
        
        Args:
            label (str): the original label
            
        Returns:
            str: the encoded label
        """
        return "x" + label.encode().hex()

    @staticmethod
    def _decode_graph_label(encoded_label: str) -> str:
        """
        Decode HEX string back to original label.
        
        Args:
            encoded_label (str): the encoded label
            
        Returns:
            str: the decoded label
        """
        return bytes.fromhex(encoded_label.removeprefix("x")).decode()

    @staticmethod
    def _format_properties(properties: Dict[str, Any], _id: Optional[str] = None) -> str:
        """
        Convert a dictionary of properties to a string representation that
        can be used in a cypher query insert/merge statement.
        
        Args:
            properties (Dict[str,str]): a dictionary containing node/edge properties
            _id (Optional[str]): the id of the node or None if none exists
            
        Returns:
            str: the properties dictionary as a properly formatted string
        """
        props = []
        # wrap property key in backticks to escape
        for k, v in properties.items():
            prop = f"`{k}`: {json.dumps(v)}"
            props.append(prop)
        if _id is not None and "id" not in properties:
            props.append(
                f"id: {json.dumps(_id)}" if isinstance(_id, str) else f"id: {_id}"
            )
        return "{" + ", ".join(props) + "}"

    @staticmethod
    def _wrap_query(query: str, graph_name: str) -> str:
        """
        Convert a cypher query to an Apache AGE compatible SQL query.
        
        Args:
            query (str): a valid cypher query
            graph_name (str): the name of the graph to query
            
        Returns:
            str: an equivalent PostgreSQL query
        """
        # Handle RETURN statements
        if "return" in query.lower():
            # Parse return statement to identify returned fields
            fields = (
                query.lower()
                .split("return")[-1]
                .split("distinct")[-1]
                .split("order by")[0]
                .split("skip")[0]
                .split("limit")[0]
                .split(",")
            )

            # Raise exception if RETURN * is found as we can't resolve the fields
            if "*" in [x.strip() for x in fields]:
                raise ValueError(
                    "AGE graph does not support 'RETURN *' statements in Cypher queries"
                )

            # Build resulting PostgreSQL relation
            fields_str = ", ".join([f"field_{idx} agtype" for idx in range(len(fields))])
        else:
            fields_str = "result agtype"

        template = f"""
        SELECT * FROM ag_catalog.cypher('{graph_name}', $$
            {query}
        $$) AS ({fields_str})
        """
        
        return template

    async def _query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a cypher query against the AGE graph with optimized parsing.
        
        Args:
            query (str): a cypher query to be executed
            
        Returns:
            List[Dict[str, Any]]: a list of dictionaries containing the result set
        """
        def _sync_query():
            try:
                wrapped_query = self._wrap_query(query, self._graph_name)
                
                with PostgreSQLAGESyncConnectionManager.get_cursor(row_factory=namedtuple_row) as (cur, conn):
                    # Set some optimization parameters for AGE
                    cur.execute("SET statement_timeout = '30s'")  # Prevent runaway queries
                    cur.execute("SET enable_nestloop = off")  # AGE optimization
                    
                    cur.execute(wrapped_query)
                    data = cur.fetchall()
                    conn.commit()
                    
                    if not data:
                        return []
                    
                    # Optimized result parsing
                    results = []
                    for row in data:
                        row_dict = {}
                        for i, field_name in enumerate(row._fields):
                            value = row[i]
                            
                            # Fast path for simple values
                            if not isinstance(value, str) or "::" not in value:
                                row_dict[field_name] = value
                                continue
                            
                            # Parse AGE types efficiently
                            parts = value.split("::", 1)
                            if len(parts) != 2:
                                row_dict[field_name] = value
                                continue
                                
                            value_part, dtype = parts[0], parts[1]
                            
                            if dtype == "vertex":
                                try:
                                    vertex_data = json.loads(value_part)
                                    properties = vertex_data.get("properties", {})
                                    if "label" in vertex_data and vertex_data["label"].startswith("x"):
                                        try:
                                            properties["label"] = self._decode_graph_label(vertex_data["label"])
                                        except:
                                            properties["label"] = vertex_data["label"]
                                    row_dict[field_name] = properties
                                except json.JSONDecodeError:
                                    row_dict[field_name] = value_part
                            elif dtype == "edge":
                                try:
                                    edge_data = json.loads(value_part)
                                    row_dict[field_name] = edge_data.get("properties", {})
                                except json.JSONDecodeError:
                                    row_dict[field_name] = {}
                            else:
                                try:
                                    row_dict[field_name] = json.loads(value_part)
                                except json.JSONDecodeError:
                                    row_dict[field_name] = value_part
                        
                        results.append(row_dict)
                    
                    return results
                    
            except psycopg.Error as e:
                raise AGEQueryException(f"Error executing graph query: {query} - {str(e)}")

        return await asyncio.to_thread(_sync_query)

    async def initialize(self):
        """Initialize storage and prepare graph."""
        if PostgreSQLAGESyncConnectionManager is None:
            raise RuntimeError("PostgreSQL AGE sync connection manager is not available")

        # Prepare graph in thread to avoid blocking
        self._graph_name = await asyncio.to_thread(
            PostgreSQLAGESyncConnectionManager.prepare_graph, self.workspace
        )

        logger.debug(f"PostgreSQLAGEStorage initialized for workspace '{self.workspace}', graph '{self._graph_name}'")

    async def finalize(self):
        """Clean up resources."""
        # Nothing to clean up - connection managed at worker level
        logger.debug(f"PostgreSQLAGEStorage finalized for workspace '{self.workspace}'")

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        entity_name_label = node_id.strip('"')
        encoded_label = self._encode_graph_label(entity_name_label)
        
        query = f"""
        MATCH (n:`{encoded_label}`) 
        RETURN count(n) > 0 AS node_exists
        """
        
        results = await self._query(query)
        return results[0]["field_0"] if results else False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""
        src_encoded = self._encode_graph_label(source_node_id.strip('"'))
        tgt_encoded = self._encode_graph_label(target_node_id.strip('"'))
        
        query = f"""
        MATCH (a:`{src_encoded}`)-[r]-(b:`{tgt_encoded}`)
        RETURN COUNT(r) > 0 AS edge_exists
        """
        
        results = await self._query(query)
        return results[0]["field_0"] if results else False

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier."""
        entity_name_label = node_id.strip('"')
        encoded_label = self._encode_graph_label(entity_name_label)
        
        query = f"""
        MATCH (n:`{encoded_label}`) 
        RETURN n
        """
        
        results = await self._query(query)
        if results:
            node_data = results[0]["field_0"]
            if node_data:
                # Add entity_id which is the node ID itself
                node_data["entity_id"] = node_id
                return node_data
        return None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes using true batch query."""
        if not node_ids:
            return {}

        nodes = {}
        # Process in smaller batches to avoid AGE query complexity
        batch_size = 20
        for i in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[i:i + batch_size]
            
            # Build UNION query for batch processing
            union_parts = []
            for node_id in batch_ids:
                entity_name_label = node_id.strip('"')
                encoded_label = self._encode_graph_label(entity_name_label)
                union_parts.append(f"MATCH (n:`{encoded_label}`) RETURN n, '{node_id}' as original_id")
            
            if union_parts:
                query = " UNION ALL ".join(union_parts)
                try:
                    results = await self._query(query)
                    for result in results:
                        if result.get("field_0"):  # node data
                            original_id = result.get("field_1")  # original node ID
                            node_data = result["field_0"]
                            node_data["entity_id"] = original_id
                            nodes[original_id] = node_data
                except Exception as e:
                    logger.warning(f"Batch query failed, falling back to individual queries: {e}")
                    # Fallback to individual queries
                    for node_id in batch_ids:
                        node_data = await self.get_node(node_id)
                        if node_data:
                            nodes[node_id] = node_data
                    
        return nodes

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node (both incoming and outgoing relationships)."""
        entity_name_label = node_id.strip('"')
        encoded_label = self._encode_graph_label(entity_name_label)
        
        query = f"""
        MATCH (n:`{encoded_label}`)
        OPTIONAL MATCH (n)-[r]-(connected)
        RETURN count(r) AS degree
        """
        
        results = await self._query(query)
        return results[0]["field_0"] if results else 0

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieve degrees for multiple nodes using optimized batch query."""
        if not node_ids:
            return {}
            
        degrees = {}
        # Process in smaller batches for AGE
        batch_size = 15  # Smaller batch for complex degree queries
        
        for i in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[i:i + batch_size]
            
            # Build optimized batch degree query
            union_parts = []
            for node_id in batch_ids:
                entity_name_label = node_id.strip('"')
                encoded_label = self._encode_graph_label(entity_name_label)
                union_parts.append(f"""
                MATCH (n:`{encoded_label}`)
                OPTIONAL MATCH (n)-[r]-(connected)
                RETURN '{node_id}' as node_id, count(r) AS degree
                """)
            
            if union_parts:
                query = " UNION ALL ".join(union_parts)
                try:
                    results = await self._query(query)
                    for result in results:
                        node_id = result.get("field_0")
                        degree = result.get("field_1", 0)
                        if node_id:
                            degrees[node_id] = int(degree) if degree is not None else 0
                except Exception as e:
                    logger.warning(f"Batch degree query failed, falling back to individual queries: {e}")
                    # Fallback to individual queries
                    for node_id in batch_ids:
                        degree = await self.node_degree(node_id)
                        degrees[node_id] = degree
        
        # Ensure all requested nodes have degree entries
        for node_id in node_ids:
            if node_id not in degrees:
                degrees[node_id] = 0
                
        return degrees

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of two nodes."""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return int(src_degree) + int(tgt_degree)

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """Calculate combined degrees for edges."""
        edge_degrees = {}
        for src, tgt in edge_pairs:
            degree = await self.edge_degree(src, tgt)
            edge_degrees[(src, tgt)] = degree
        return edge_degrees

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get edge properties between two nodes."""
        src_encoded = self._encode_graph_label(source_node_id.strip('"'))
        tgt_encoded = self._encode_graph_label(target_node_id.strip('"'))
        
        query = f"""
        MATCH (a:`{src_encoded}`)-[r]->(b:`{tgt_encoded}`)
        RETURN properties(r) as edge_properties
        LIMIT 1
        """
        
        results = await self._query(query)
        if results and results[0]["field_0"]:
            edge_data = results[0]["field_0"]
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
        return None

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple pairs using batch query."""
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
        
        # Process in small batches for AGE
        batch_size = 10
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            # Build batch query for edges
            union_parts = []
            for pair in batch_pairs:
                src, tgt = pair["src"], pair["tgt"]
                src_encoded = self._encode_graph_label(src.strip('"'))
                tgt_encoded = self._encode_graph_label(tgt.strip('"'))
                union_parts.append(f"""
                MATCH (a:`{src_encoded}`)-[r]->(b:`{tgt_encoded}`)
                RETURN '{src}' as src_id, '{tgt}' as tgt_id, properties(r) as edge_props
                """)
            
            if union_parts:
                query = " UNION ALL ".join(union_parts)
                try:
                    results = await self._query(query)
                    
                    for result in results:
                        src_id = result.get("field_0")
                        tgt_id = result.get("field_1")
                        edge_props = result.get("field_2")
                        
                        if src_id and tgt_id and edge_props:
                            # Ensure required keys exist
                            for key, default in {
                                "weight": 0.0,
                                "source_id": None,
                                "description": None,
                                "keywords": None,
                            }.items():
                                if key not in edge_props:
                                    edge_props[key] = default
                            
                            edges_dict[(src_id, tgt_id)] = edge_props
                    
                except Exception as e:
                    logger.warning(f"Batch edge properties query failed, falling back to individual queries: {e}")
                    # Fallback to individual queries
                    for pair in batch_pairs:
                        src, tgt = pair["src"], pair["tgt"]
                        edge_data = await self.get_edge(src, tgt)
                        if edge_data:
                            edges_dict[(src, tgt)] = edge_data
                
        return edges_dict

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges for a node - optimized version."""
        node_label = source_node_id.strip('"')
        encoded_label = self._encode_graph_label(node_label)
        
        # Simplified query that directly returns the node IDs we need
        query = f"""
        MATCH (n:`{encoded_label}`)-[r]-(connected)
        RETURN '{source_node_id}' as source_id, connected
        """
        
        try:
            results = await self._query(query)
            edges = []
            seen_edges = set()  # Deduplicate edges
            
            for result in results:
                source_id = result.get("field_0")  # Always our source node
                connected_node = result.get("field_1")
                
                if source_id and connected_node:
                    # Extract entity_id from connected node or use label
                    target_id = None
                    if isinstance(connected_node, dict):
                        # Try to get entity_id first, then label
                        target_id = connected_node.get("entity_id") or connected_node.get("label")
                        # If still no target_id, try to decode from encoded labels
                        if not target_id and "label" in connected_node:
                            try:
                                target_id = self._decode_graph_label(connected_node["label"])
                            except:
                                continue
                    
                    if target_id and target_id != source_id:
                        # Create edge tuple and deduplicate
                        edge = (source_id, target_id)
                        reverse_edge = (target_id, source_id)
                        
                        # Only add if we haven't seen this edge in either direction
                        if edge not in seen_edges and reverse_edge not in seen_edges:
                            edges.append(edge)
                            seen_edges.add(edge)
                            
            return edges if edges else None
            
        except Exception as e:
            logger.warning(f"Optimized get_node_edges failed for {source_node_id}: {e}")
            # Fallback to simpler approach
            try:
                simple_query = f"""
                MATCH (n:`{encoded_label}`)
                MATCH (n)-[r]-(m)
                RETURN '{source_node_id}' as src, id(m) as tgt_id
                """
                results = await self._query(simple_query)
                edges = []
                for result in results:
                    src = result.get("field_0")
                    tgt = result.get("field_1")
                    if src and tgt:
                        edges.append((src, str(tgt)))
                return edges if edges else None
            except:
                return None

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes using optimized query."""
        if not node_ids:
            return {}
            
        edges_dict = {node_id: [] for node_id in node_ids}
        
        # Process in small batches for AGE
        batch_size = 10  # Very small batch for complex edge queries
        
        for i in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[i:i + batch_size]
            
            # Build batch query for node edges
            union_parts = []
            for node_id in batch_ids:
                entity_name_label = node_id.strip('"')
                encoded_label = self._encode_graph_label(entity_name_label)
                union_parts.append(f"""
                MATCH (n:`{encoded_label}`)-[r]-(connected)
                RETURN '{node_id}' as source_id, connected as target_node
                """)
            
            if union_parts:
                query = " UNION ALL ".join(union_parts)
                try:
                    results = await self._query(query)
                    
                    # Process results and group by source node
                    for result in results:
                        source_id = result.get("field_0")
                        connected_node = result.get("field_1")
                        
                        if source_id and connected_node and source_id in edges_dict:
                            # Extract target ID from connected node
                            target_id = None
                            if isinstance(connected_node, dict):
                                target_id = connected_node.get("entity_id") or connected_node.get("label")
                                if not target_id and "label" in connected_node:
                                    try:
                                        target_id = self._decode_graph_label(connected_node["label"])
                                    except:
                                        continue
                            
                            if target_id and target_id != source_id:
                                edge = (source_id, target_id)
                                # Simple deduplication
                                if edge not in edges_dict[source_id]:
                                    edges_dict[source_id].append(edge)
                    
                except Exception as e:
                    logger.warning(f"Batch edges query failed, falling back to individual queries: {e}")
                    # Fallback to individual queries
                    for node_id in batch_ids:
                        edges = await self.get_node_edges(node_id)
                        edges_dict[node_id] = edges or []
        
        return edges_dict

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AGEQueryException, psycopg.Error)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database."""
        label = node_id.strip('"')
        encoded_label = self._encode_graph_label(label)
        properties = node_data.copy()
        
        # Ensure entity_id is in properties
        if "entity_id" not in properties:
            properties["entity_id"] = node_id

        query = f"""
        MERGE (n:`{encoded_label}`)
        SET n += {self._format_properties(properties)}
        """
        
        await self._query(query)
        logger.debug(f"Upserted node with label '{label}'")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((AGEQueryException, psycopg.Error)),
    )
    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Upsert an edge between two nodes."""
        src_encoded = self._encode_graph_label(source_node_id.strip('"'))
        tgt_encoded = self._encode_graph_label(target_node_id.strip('"'))
        
        query = f"""
        MATCH (source:`{src_encoded}`)
        WITH source
        MATCH (target:`{tgt_encoded}`)
        MERGE (source)-[r:DIRECTED]->(target)
        SET r += {self._format_properties(edge_data)}
        RETURN r
        """
        
        await self._query(query)
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
            # Get all nodes ordered by degree
            query = f"""
            MATCH (n)
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) AS degree
            ORDER BY degree DESC
            LIMIT {max_nodes}
            RETURN n, degree
            """
        else:
            # Get subgraph starting from specific node
            encoded_label = self._encode_graph_label(node_label.strip('"'))
            query = f"""
            MATCH (start:`{encoded_label}`)
            OPTIONAL MATCH path = (start)-[*0..{max_depth}]-(connected)
            WITH nodes(path) as path_nodes, relationships(path) as path_rels
            LIMIT {max_nodes}
            RETURN path_nodes, path_rels
            """
        
        results = await self._query(query)
        seen_nodes = set()
        seen_edges = set()
        
        for record in results:
            # Process nodes
            if "field_0" in record:  # nodes
                if isinstance(record["field_0"], list):
                    # Path nodes
                    for node in record["field_0"]:
                        if node and "id" in node:
                            node_id = str(node["id"])
                            if node_id not in seen_nodes:
                                result.nodes.append(
                                    KnowledgeGraphNode(
                                        id=node_id,
                                        labels=[node.get("label", "")],
                                        properties=node,
                                    )
                                )
                                seen_nodes.add(node_id)
                else:
                    # Single node
                    node = record["field_0"]
                    if node and "id" in node:
                        node_id = str(node["id"])
                        if node_id not in seen_nodes:
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id,
                                    labels=[node.get("label", "")],
                                    properties=node,
                                )
                            )
                            seen_nodes.add(node_id)
            
            # Process edges
            if "field_1" in record:  # relationships
                if isinstance(record["field_1"], list):
                    # Path relationships
                    for rel in record["field_1"]:
                        if rel and "start_id" in rel and "end_id" in rel:
                            edge_id = f"{rel['start_id']}-{rel['end_id']}"
                            if edge_id not in seen_edges:
                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=edge_id,
                                        type="DIRECTED",
                                        source=str(rel["start_id"]),
                                        target=str(rel["end_id"]),
                                        properties=rel.get("properties", {}),
                                    )
                                )
                                seen_edges.add(edge_id)
        
        logger.info(f"Retrieved subgraph with {len(result.nodes)} nodes and {len(result.edges)} edges")
        return result

    async def get_all_labels(self) -> list[str]:
        """Get all node labels."""
        query = """
        MATCH (n)
        RETURN DISTINCT labels(n) AS node_labels
        """
        
        results = await self._query(query)
        all_labels = []
        
        for record in results:
            if "field_0" in record and record["field_0"]:
                for label in record["field_0"]:
                    if label:
                        decoded_label = self._decode_graph_label(label)
                        all_labels.append(decoded_label)
        
        return sorted(list(set(all_labels)))

    async def delete_node(self, node_id: str) -> None:
        """Delete a node."""
        entity_name_label = node_id.strip('"')
        encoded_label = self._encode_graph_label(entity_name_label)
        
        query = f"""
        MATCH (n:`{encoded_label}`)
        DETACH DELETE n
        """
        
        await self._query(query)
        logger.debug(f"Deleted node with label '{entity_name_label}'")

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes."""
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges."""
        for source, target in edges:
            src_encoded = self._encode_graph_label(source.strip('"'))
            tgt_encoded = self._encode_graph_label(target.strip('"'))
            
            query = f"""
            MATCH (source:`{src_encoded}`)-[r]->(target:`{tgt_encoded}`)
            DELETE r
            """
            
            await self._query(query)
            logger.debug(f"Deleted edge from '{source}' to '{target}'")

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage."""
        def _sync_drop():
            try:
                with PostgreSQLAGESyncConnectionManager.get_cursor() as (cur, conn):
                    # Drop the entire graph
                    cur.execute(f"SELECT drop_graph('{self._graph_name}', true)")
                    conn.commit()
                    logger.info(f"Dropped graph {self._graph_name}")
                    return {"status": "success", "message": "graph data dropped"}
            except Exception as e:
                logger.error(f"Error dropping graph {self._graph_name}: {e}")
                return {"status": "error", "message": str(e)}

        return await asyncio.to_thread(_sync_drop)