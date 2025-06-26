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
import time
from dataclasses import dataclass
from typing import final

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger

# Import sync connection manager
try:
    from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
except ImportError:
    NebulaSyncConnectionManager = None


@final
@dataclass
class NebulaSyncStorage(BaseGraphStorage):
    """
    NebulaGraph storage implementation using sync driver with async interface.
    This avoids event loop issues while maintaining compatibility with async code.
    """

    def __init__(self, namespace, workspace, embedding_func):
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            embedding_func=embedding_func,
        )
        self._space_name = None

    async def initialize(self):
        """Initialize storage and prepare space."""
        if NebulaSyncConnectionManager is None:
            raise RuntimeError("NebulaGraph sync connection manager is not available")

        # Prepare space in thread to avoid blocking
        self._space_name = await asyncio.to_thread(NebulaSyncConnectionManager.prepare_space, self.workspace)

        logger.debug(f"NebulaSyncStorage initialized for workspace '{self.workspace}', space '{self._space_name}'")

    async def finalize(self):
        """Clean up resources."""
        # Nothing to clean up - connection managed at worker level
        logger.debug(f"NebulaSyncStorage finalized for workspace '{self.workspace}'")

    def _escape_string(self, value: str) -> str:
        """Escape string for NebulaGraph queries."""
        if value is None:
            return "NULL"
        if not isinstance(value, str):
            value = str(value)
        
        # Escape special characters for NebulaGraph
        # Order matters: escape backslashes first, then other characters
        escaped = (value
                  .replace("\\", "\\\\")    # Escape backslashes first
                  .replace('"', '\\"')      # Escape double quotes
                  .replace("'", "\\'")      # Escape single quotes
                  .replace("\n", "\\n")     # Escape newlines
                  .replace("\r", "\\r")     # Escape carriage returns
                  .replace("\t", "\\t")     # Escape tabs
                  .replace("\b", "\\b")     # Escape backspace
                  .replace("\f", "\\f"))    # Escape form feed
        
        return escaped

    def _sanitize_tag_name(self, entity_type: str) -> str:
        """Sanitize entity type to be a valid NebulaGraph tag name."""
        if not entity_type:
            return "entity"
        
        # Replace invalid characters with underscores and ensure it starts with letter/underscore
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', entity_type)
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = f"t_{sanitized}"
        
        # Limit length and ensure it's not empty
        sanitized = sanitized[:64] if sanitized else "entity"
        
        return sanitized

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""

        def _sync_has_node():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # In NebulaGraph, we need to quote vertex IDs
                query = f'FETCH PROP ON base "{self._escape_string(node_id)}" YIELD vertex as v'
                result = session.execute(query)

                if result.is_succeeded() and result.row_size() > 0:
                    return True
                return False

        return await asyncio.to_thread(_sync_has_node)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists between two nodes."""

        def _sync_has_edge():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Check both directions since we treat edges as undirected
                query = f'''
                FETCH PROP ON DIRECTED "{self._escape_string(source_node_id)}" -> "{self._escape_string(target_node_id)}" YIELD edge as e
                UNION
                FETCH PROP ON DIRECTED "{self._escape_string(target_node_id)}" -> "{self._escape_string(source_node_id)}" YIELD edge as e
                '''
                result = session.execute(query)

                if result.is_succeeded() and result.row_size() > 0:
                    return True
                return False

        return await asyncio.to_thread(_sync_has_edge)

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier."""

        def _sync_get_node():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Fetch properties from both base and entity tags
                query = f'''
                FETCH PROP ON base, entity "{self._escape_string(node_id)}" 
                YIELD properties(vertex) as props
                '''
                result = session.execute(query)

                if result.is_succeeded() and result.row_size() > 0:
                    row = result.row_values(0)
                    props_value = row[0]

                    # Parse properties from NebulaGraph format
                    if props_value.is_map():
                        props_map = props_value.as_map()
                        node_dict = {}

                        for key, value in props_map.items():
                            key_str = key.decode() if isinstance(key, bytes) else str(key)

                            if value.is_string():
                                node_dict[key_str] = value.as_string()
                            elif value.is_int():
                                node_dict[key_str] = value.as_int()
                            elif value.is_double():
                                node_dict[key_str] = value.as_double()
                            elif value.is_null():
                                node_dict[key_str] = None
                            else:
                                # Handle other types as string
                                node_dict[key_str] = str(value)

                        # Neo4j compatibility: return node properties as-is without adding defaults
                        return node_dict

                return None

        return await asyncio.to_thread(_sync_get_node)

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """Retrieve multiple nodes in one query."""

        def _sync_get_nodes_batch():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                nodes = {}
                
                # Build a single query for all nodes using UNION
                if not node_ids:
                    return nodes
                
                # Create UNION query to fetch all nodes at once
                union_queries = []
                for node_id in node_ids:
                    union_queries.append(f'''
                    FETCH PROP ON base, entity "{self._escape_string(node_id)}" 
                    YIELD "{self._escape_string(node_id)}" AS node_id, properties(vertex) as props
                    ''')
                
                query = " UNION ".join(union_queries)
                result = session.execute(query)

                if result.is_succeeded():
                    for row in result:
                        node_id = row.values()[0].as_string()
                        props_value = row.values()[1]

                        if props_value.is_map():
                            props_map = props_value.as_map()
                            node_dict = {}

                            for key, value in props_map.items():
                                key_str = key.decode() if isinstance(key, bytes) else str(key)

                                if value.is_string():
                                    node_dict[key_str] = value.as_string()
                                elif value.is_int():
                                    node_dict[key_str] = value.as_int()
                                elif value.is_double():
                                    node_dict[key_str] = value.as_double()
                                elif value.is_null():
                                    node_dict[key_str] = None
                                else:
                                    node_dict[key_str] = str(value)

                            # Neo4j compatibility: return node properties as-is without adding defaults
                            nodes[node_id] = node_dict

                return nodes

        return await asyncio.to_thread(_sync_get_nodes_batch)

    async def node_degree(self, node_id: str) -> int:
        """Get the degree of a node."""

        def _sync_node_degree():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Count both incoming and outgoing edges
                query = f'''
                GO FROM "{self._escape_string(node_id)}" OVER * BIDIRECT 
                YIELD src(edge) as src, dst(edge) as dst
                | YIELD COUNT(*) as degree
                '''
                result = session.execute(query)

                if result.is_succeeded() and result.row_size() > 0:
                    row = result.row_values(0)
                    return row[0].as_int()

                return 0

        return await asyncio.to_thread(_sync_node_degree)

    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """Retrieve degrees for multiple nodes."""

        def _sync_node_degrees_batch():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                degrees = {}
                
                if not node_ids:
                    return degrees
                
                # Build a single query for all nodes using UNION
                union_queries = []
                for node_id in node_ids:
                    union_queries.append(f'''
                    GO FROM "{self._escape_string(node_id)}" OVER * BIDIRECT 
                    YIELD "{self._escape_string(node_id)}" AS node_id, src(edge) as src, dst(edge) as dst
                    ''')
                
                query = " UNION ".join(union_queries)
                result = session.execute(query)

                # Count degrees for each node
                node_edge_counts = {}
                if result.is_succeeded():
                    for row in result:
                        node_id = row.values()[0].as_string()
                        node_edge_counts[node_id] = node_edge_counts.get(node_id, 0) + 1

                # Set degrees for all requested nodes (including those with 0 degree)
                for node_id in node_ids:
                    degrees[node_id] = node_edge_counts.get(node_id, 0)

                return degrees

        return await asyncio.to_thread(_sync_node_degrees_batch)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree of two nodes."""
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)
        return int(src_degree) + int(trg_degree)

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
        """Calculate combined degrees for edges."""
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})

        degrees = await self.node_degrees_batch(list(unique_node_ids))

        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict[str, str] | None:
        """Get edge properties between two nodes."""

        def _sync_get_edge():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Try both directions
                query = f'''
                FETCH PROP ON DIRECTED "{self._escape_string(source_node_id)}" -> "{self._escape_string(target_node_id)}" 
                YIELD properties(edge) as props
                '''
                result = session.execute(query)

                if not result.is_succeeded() or result.row_size() == 0:
                    # Try reverse direction
                    query = f'''
                    FETCH PROP ON DIRECTED "{self._escape_string(target_node_id)}" -> "{self._escape_string(source_node_id)}" 
                    YIELD properties(edge) as props
                    '''
                    result = session.execute(query)

                if result.is_succeeded() and result.row_size() > 0:
                    row = result.row_values(0)
                    props_value = row[0]

                    # Parse properties
                    if props_value.is_map():
                        props_map = props_value.as_map()
                        edge_dict = {
                            "weight": 0.0,
                            "source_id": None,
                            "description": None,
                            "keywords": None,
                        }

                        for key, value in props_map.items():
                            key_str = key.decode() if isinstance(key, bytes) else str(key)

                            if value.is_string():
                                edge_dict[key_str] = value.as_string()
                            elif value.is_double():
                                edge_dict[key_str] = value.as_double()
                            elif value.is_int():
                                edge_dict[key_str] = value.as_int()
                            elif value.is_null():
                                edge_dict[key_str] = None
                            else:
                                edge_dict[key_str] = str(value)

                        return edge_dict

                return None

        return await asyncio.to_thread(_sync_get_edge)

    async def get_edges_batch(self, pairs: list[dict[str, str]]) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple pairs."""

        def _sync_get_edges_batch():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                edges_dict = {}
                
                if not pairs:
                    return edges_dict
                
                # Build a single query for all edges using UNION
                union_queries = []
                for pair in pairs:
                    src = pair["src"]
                    tgt = pair["tgt"]
                    
                    # Add both directions in a single UNION query
                    union_queries.append(f'''
                    FETCH PROP ON DIRECTED "{self._escape_string(src)}" -> "{self._escape_string(tgt)}" 
                    YIELD "{self._escape_string(src)}" AS src, "{self._escape_string(tgt)}" AS tgt, properties(edge) as props
                    ''')
                    union_queries.append(f'''
                    FETCH PROP ON DIRECTED "{self._escape_string(tgt)}" -> "{self._escape_string(src)}" 
                    YIELD "{self._escape_string(tgt)}" AS src, "{self._escape_string(src)}" AS tgt, properties(edge) as props
                    ''')
                
                query = " UNION ".join(union_queries)
                result = session.execute(query)

                # Track which edges we found
                found_edges = set()
                
                if result.is_succeeded():
                    for row in result:
                        src = row.values()[0].as_string()
                        tgt = row.values()[1].as_string()
                        props_value = row.values()[2]

                        # Use original pair order as key
                        edge_key = None
                        for pair in pairs:
                            if (pair["src"] == src and pair["tgt"] == tgt) or (pair["src"] == tgt and pair["tgt"] == src):
                                edge_key = (pair["src"], pair["tgt"])
                                break
                        
                        if edge_key and edge_key not in found_edges:
                            found_edges.add(edge_key)
                            
                            if props_value.is_map():
                                props_map = props_value.as_map()
                                edge_props = {}

                                for key, value in props_map.items():
                                    key_str = key.decode() if isinstance(key, bytes) else str(key)

                                    if value.is_string():
                                        edge_props[key_str] = value.as_string()
                                    elif value.is_double():
                                        edge_props[key_str] = value.as_double()
                                    elif value.is_int():
                                        edge_props[key_str] = value.as_int()
                                    elif value.is_null():
                                        edge_props[key_str] = None
                                    else:
                                        edge_props[key_str] = str(value)

                                # Ensure required keys exist
                                for key, default in {
                                    "weight": 0.0,
                                    "source_id": None,
                                    "description": None,
                                    "keywords": None,
                                }.items():
                                    if key not in edge_props:
                                        edge_props[key] = default

                                edges_dict[edge_key] = edge_props

                # Add default values for edges not found
                for pair in pairs:
                    edge_key = (pair["src"], pair["tgt"])
                    if edge_key not in edges_dict:
                        edges_dict[edge_key] = {
                            "weight": 0.0,
                            "source_id": None,
                            "description": None,
                            "keywords": None,
                        }

                return edges_dict

        return await asyncio.to_thread(_sync_get_edges_batch)

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Get all edges for a node."""

        def _sync_get_node_edges():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Get both incoming and outgoing edges
                query = f'''
                GO FROM "{self._escape_string(source_node_id)}" OVER * BIDIRECT 
                YIELD src(edge) as src, dst(edge) as dst
                '''
                result = session.execute(query)

                edges = []
                if result.is_succeeded():
                    for row in result:
                        src_val = row.values()[0].as_string()
                        dst_val = row.values()[1].as_string()
                        edges.append((src_val, dst_val))

                return edges if edges else None

        return await asyncio.to_thread(_sync_get_node_edges)

    async def get_nodes_edges_batch(self, node_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes."""

        def _sync_get_nodes_edges_batch():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                edges_dict = {node_id: [] for node_id in node_ids}

                if not node_ids:
                    return edges_dict

                # Build a single query for all nodes using UNION
                union_queries = []
                for node_id in node_ids:
                    union_queries.append(f'''
                    GO FROM "{self._escape_string(node_id)}" OVER * BIDIRECT 
                    YIELD "{self._escape_string(node_id)}" AS queried_node, src(edge) as src, dst(edge) as dst
                    ''')

                query = " UNION ".join(union_queries)
                result = session.execute(query)

                if result.is_succeeded():
                    for row in result:
                        queried_node = row.values()[0].as_string()
                        src_val = row.values()[1].as_string()
                        dst_val = row.values()[2].as_string()
                        
                        # Add edge in the correct direction based on the queried node
                        if src_val == queried_node:
                            edges_dict[queried_node].append((src_val, dst_val))
                        else:
                            edges_dict[queried_node].append((dst_val, src_val))

                return edges_dict

        return await asyncio.to_thread(_sync_get_nodes_edges_batch)

    def _safe_error_msg(self, result) -> str:
        """Safely decode error message, falling back to repr if decoding fails."""
        try:
            return result.error_msg()
        except UnicodeDecodeError:
            return result._resp.error_msg.decode('utf-8', errors='ignore')

    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """Upsert a node in the database."""

        def _sync_upsert_node():
            # Import inside the function to ensure it's available
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            
            properties = node_data.copy()
            entity_type = properties.get("entity_type", "entity")

            if "entity_id" not in properties:
                raise ValueError("NebulaGraph: node properties must contain an 'entity_id' field")

            # Sanitize entity_type for use as tag name (NebulaGraph tag names must be valid identifiers)
            sanitized_entity_type = self._sanitize_tag_name(entity_type)

            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Step 1: Ensure the vertex exists with the 'base' tag.
                insert_base_query = f'INSERT VERTEX IF NOT EXISTS base() VALUES "{self._escape_string(node_id)}":()'
                result = session.execute(insert_base_query)
                if not result.is_succeeded():
                    error_msg = self._safe_error_msg(result)
                    logger.error(f"Failed to ensure base vertex exists: {error_msg}")
                    raise RuntimeError(f"Failed to ensure base vertex exists: {error_msg}")

                # Step 2: Update the properties on the 'base' tag.
                # Use multiple single-property updates to avoid complex escaping issues
                for key, value in properties.items():
                    if value is None:
                        update_query = f'UPDATE VERTEX "{self._escape_string(node_id)}" SET base.`{key}` = NULL'
                    elif isinstance(value, str):
                        # For strings, escape for NebulaGraph and use double quotes
                        escaped_value = self._escape_string(value)
                        update_query = f'UPDATE VERTEX "{self._escape_string(node_id)}" SET base.`{key}` = "{escaped_value}"'
                    elif isinstance(value, (int, float)):
                        update_query = f'UPDATE VERTEX "{self._escape_string(node_id)}" SET base.`{key}` = {value}'
                    else:
                        escaped_value = self._escape_string(str(value))
                        update_query = f'UPDATE VERTEX "{self._escape_string(node_id)}" SET base.`{key}` = "{escaped_value}"'
                    
                    result = session.execute(update_query)
                    if not result.is_succeeded():
                        error_msg = self._safe_error_msg(result)
                        logger.error(f"Failed to update base property {key}: {error_msg}")
                        # Continue with other properties instead of failing completely
                        continue

                # Step 3: Add entity type-specific tag for better Schema visualization
                if sanitized_entity_type and sanitized_entity_type != "base":
                    # First ensure the tag exists (create if needed)
                    try:
                        NebulaSyncConnectionManager.ensure_tag_exists(self._space_name, sanitized_entity_type)
                    except Exception as e:
                        logger.debug(f"Failed to ensure tag {sanitized_entity_type} exists: {e}")

                    # Ensure the vertex has the entity type-specific tag
                    insert_type_query = f'INSERT VERTEX IF NOT EXISTS `{sanitized_entity_type}`() VALUES "{self._escape_string(node_id)}":()'
                    result = session.execute(insert_type_query)
                    if not result.is_succeeded():
                        error_msg = self._safe_error_msg(result)
                        logger.debug(f"Failed to ensure {sanitized_entity_type} tag exists: {error_msg}")

                    # Update properties on the entity type-specific tag
                    # Use multiple single-property updates to avoid complex escaping issues
                    for key, value in properties.items():
                        if value is None:
                            update_query = f'UPDATE VERTEX "{self._escape_string(node_id)}" SET `{sanitized_entity_type}`.`{key}` = NULL'
                        elif isinstance(value, str):
                            # For strings, escape for NebulaGraph and use double quotes
                            escaped_value = self._escape_string(value)
                            update_query = f'UPDATE VERTEX "{self._escape_string(node_id)}" SET `{sanitized_entity_type}`.`{key}` = "{escaped_value}"'
                        elif isinstance(value, (int, float)):
                            update_query = f'UPDATE VERTEX "{self._escape_string(node_id)}" SET `{sanitized_entity_type}`.`{key}` = {value}'
                        else:
                            escaped_value = self._escape_string(str(value))
                            update_query = f'UPDATE VERTEX "{self._escape_string(node_id)}" SET `{sanitized_entity_type}`.`{key}` = "{escaped_value}"'
                        
                        result = session.execute(update_query)
                        if not result.is_succeeded():
                            error_msg = self._safe_error_msg(result)
                            logger.debug(f"Failed to update {sanitized_entity_type} property {key}: {error_msg}")
                            # Continue with other properties instead of failing completely
                            continue

                logger.debug(f"Upserted node with entity_id '{node_id}' and type '{sanitized_entity_type}'")

        return await asyncio.to_thread(_sync_upsert_node)

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]) -> None:
        """Upsert an edge between two nodes."""

        def _sync_upsert_edge():
            # Import inside the function to ensure it's available
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            
            edge_properties = edge_data.copy()

            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # First ensure the edge exists
                insert_edge_query = f'''
                INSERT EDGE IF NOT EXISTS DIRECTED() VALUES "{self._escape_string(source_node_id)}" -> "{self._escape_string(target_node_id)}"@0:()
                '''
                result = session.execute(insert_edge_query)
                if not result.is_succeeded():
                    error_msg = self._safe_error_msg(result)
                    logger.debug(f"Failed to ensure edge exists: {error_msg}")

                # Update properties one by one to avoid complex escaping issues
                for key, value in edge_properties.items():
                    if value is None:
                        update_query = f'UPDATE EDGE ON DIRECTED "{self._escape_string(source_node_id)}" -> "{self._escape_string(target_node_id)}"@0 SET `{key}` = NULL'
                    elif isinstance(value, str):
                        # For strings, escape for NebulaGraph and use double quotes
                        escaped_value = self._escape_string(value)
                        update_query = f'UPDATE EDGE ON DIRECTED "{self._escape_string(source_node_id)}" -> "{self._escape_string(target_node_id)}"@0 SET `{key}` = "{escaped_value}"'
                    elif isinstance(value, (int, float)):
                        update_query = f'UPDATE EDGE ON DIRECTED "{self._escape_string(source_node_id)}" -> "{self._escape_string(target_node_id)}"@0 SET `{key}` = {value}'
                    else:
                        escaped_value = self._escape_string(str(value))
                        update_query = f'UPDATE EDGE ON DIRECTED "{self._escape_string(source_node_id)}" -> "{self._escape_string(target_node_id)}"@0 SET `{key}` = "{escaped_value}"'
                    
                    result = session.execute(update_query)
                    if not result.is_succeeded():
                        error_msg = self._safe_error_msg(result)
                        logger.debug(f"Failed to update edge property {key}: {error_msg}")
                        # Continue with other properties instead of failing completely
                        continue

                logger.debug(f"Upserted edge from '{source_node_id}' to '{target_node_id}'")

        return await asyncio.to_thread(_sync_upsert_edge)

    async def delete_node(self, node_id: str) -> None:
        """Delete a node."""

        def _sync_delete_node():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # DELETE VERTEX will automatically delete associated edges
                query = f'DELETE VERTEX "{self._escape_string(node_id)}" WITH EDGE'
                result = session.execute(query)

                if not result.is_succeeded():
                    error_msg = self._safe_error_msg(result)
                    logger.error(f"Failed to delete node: {error_msg}")
                    raise RuntimeError(f"Failed to delete node: {error_msg}")

                logger.debug(f"Deleted node with label '{node_id}'")

        return await asyncio.to_thread(_sync_delete_node)

    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes."""
        for node in nodes:
            await self.delete_node(node)

    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges."""

        def _sync_remove_edge(source: str, target: str):
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Delete edge in both directions to handle undirected nature
                query = f'''
                DELETE EDGE DIRECTED "{self._escape_string(source)}" -> "{self._escape_string(target)}"
                '''
                result = session.execute(query)

                if result.is_succeeded():
                    logger.debug(f"Deleted edge from '{source}' to '{target}'")

                # Also try reverse direction
                query = f'''
                DELETE EDGE DIRECTED "{self._escape_string(target)}" -> "{self._escape_string(source)}"
                '''
                result = session.execute(query)

                if result.is_succeeded():
                    logger.debug(f"Deleted edge from '{target}' to '{source}'")

        for source, target in edges:
            await asyncio.to_thread(_sync_remove_edge, source, target)

    async def get_all_labels(self) -> list[str]:
        """Get all node labels."""

        def _sync_get_all_labels():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Use NebulaGraph native syntax to get all entity_id values
                query = """
                FETCH PROP ON base * YIELD vertex as v, properties(vertex).entity_id as entity_id
                | WHERE $-.entity_id IS NOT NULL
                | YIELD DISTINCT $-.entity_id AS label
                | ORDER BY $-.label
                """
                result = session.execute(query)

                labels = []
                if result.is_succeeded():
                    for row in result:
                        if row.values()[0].is_string():
                            label = row.values()[0].as_string()
                            labels.append(label)

                return labels

        return await asyncio.to_thread(_sync_get_all_labels)

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """Retrieve a connected subgraph."""

        def _sync_get_knowledge_graph():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            result = KnowledgeGraph()
            seen_nodes = set()
            seen_edges = set()

            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                if node_label == "*":
                    # Get all nodes up to max_nodes using NebulaGraph syntax
                    query = f"""
                    FETCH PROP ON base * YIELD vertex as v, properties(vertex) as props
                    | LIMIT {max_nodes}
                    """
                else:
                    # BFS from specific node using GO statement
                    query = f'''
                    GO {max_depth} STEPS FROM "{self._escape_string(node_label)}" OVER * BIDIRECT 
                    YIELD DISTINCT dst(edge) as node_id
                    | FETCH PROP ON base, entity $-.node_id 
                    YIELD id(vertex) as node_id, properties(vertex) as props
                    | LIMIT {max_nodes}
                    '''

                result_nodes = session.execute(query)

                if result_nodes.is_succeeded():
                    node_ids = []

                    for row in result_nodes:
                        node_id = row.values()[0].as_string() if row.values()[0].is_string() else str(row.values()[0])

                        if node_id not in seen_nodes:
                            # Parse properties
                            props = {}
                            if len(row.values()) > 1 and row.values()[1].is_map():
                                props_map = row.values()[1].as_map()
                                for key, value in props_map.items():
                                    key_str = key.decode() if isinstance(key, bytes) else str(key)
                                    if value.is_string():
                                        props[key_str] = value.as_string()
                                    elif value.is_int():
                                        props[key_str] = value.as_int()
                                    elif value.is_double():
                                        props[key_str] = value.as_double()
                                    else:
                                        props[key_str] = str(value)

                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=node_id,
                                    labels=[props.get("entity_id", node_id)],
                                    properties=props,
                                )
                            )
                            seen_nodes.add(node_id)
                            node_ids.append(node_id)

                    # Get edges between collected nodes
                    if node_ids:
                        # Build a query to get all edges between the nodes
                        node_id_list = ", ".join([f'"{self._escape_string(nid)}"' for nid in node_ids])
                        edge_query = f"""
                        GO FROM {node_id_list} OVER * 
                        WHERE dst(edge) IN [{node_id_list}]
                        YIELD src(edge) as src, dst(edge) as dst, properties(edge) as props, type(edge) as edge_type
                        """

                        result_edges = session.execute(edge_query)

                        if result_edges.is_succeeded():
                            for row in result_edges:
                                src = row.values()[0].as_string()
                                dst = row.values()[1].as_string()
                                edge_type = row.values()[3].as_string()

                                edge_id = f"{src}-{edge_type}-{dst}"
                                if edge_id not in seen_edges:
                                    # Parse edge properties
                                    edge_props = {}
                                    if row.values()[2].is_map():
                                        props_map = row.values()[2].as_map()
                                        for key, value in props_map.items():
                                            key_str = key.decode() if isinstance(key, bytes) else str(key)
                                            if value.is_string():
                                                edge_props[key_str] = value.as_string()
                                            elif value.is_double():
                                                edge_props[key_str] = value.as_double()
                                            elif value.is_int():
                                                edge_props[key_str] = value.as_int()
                                            else:
                                                edge_props[key_str] = str(value)

                                    result.edges.append(
                                        KnowledgeGraphEdge(
                                            id=edge_id,
                                            type=edge_type,
                                            source=src,
                                            target=dst,
                                            properties=edge_props,
                                        )
                                    )
                                    seen_edges.add(edge_id)

            logger.info(f"Retrieved subgraph with {len(result.nodes)} nodes and {len(result.edges)} edges")
            return result

        return await asyncio.to_thread(_sync_get_knowledge_graph)

    async def drop(self) -> dict[str, str]:
        """Drop all data from storage."""

        def _sync_drop():
            from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager
            with NebulaSyncConnectionManager.get_session(space=self._space_name) as session:
                # Delete all vertices (which will also delete edges)
                query = "DELETE VERTEX * WITH EDGE"
                result = session.execute(query)

                if result.is_succeeded():
                    logger.info(f"Dropped all data from space {self._space_name}")
                    return {"status": "success", "message": "data dropped"}
                else:
                    error_msg = self._safe_error_msg(result)
                    logger.error(f"Failed to drop data: {error_msg}")
                    return {"status": "error", "message": error_msg}

        return await asyncio.to_thread(_sync_drop)
