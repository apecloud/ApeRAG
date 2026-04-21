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

from __future__ import annotations

import time
from typing import Any

from ...concurrent_control import LockProtocol
from .utils import compute_mdhash_id, logger


async def _get_nodes_by_id_batch(chunk_entity_relation_graph, node_ids: list[str]) -> dict[str, dict[str, Any]]:
    unique_node_ids = list(dict.fromkeys(node_id for node_id in node_ids if node_id))
    if not unique_node_ids:
        return {}
    return await chunk_entity_relation_graph.get_nodes_batch(unique_node_ids)


async def _get_edges_for_nodes(
    chunk_entity_relation_graph,
    node_ids: list[str],
) -> list[tuple[str, str, dict[str, Any]]]:
    unique_node_ids = list(dict.fromkeys(node_id for node_id in node_ids if node_id))
    if not unique_node_ids:
        return []

    node_edges_batch = await chunk_entity_relation_graph.get_incident_edges_with_data_batch(unique_node_ids)
    seen_edge_pairs: set[tuple[str, str]] = set()
    deduplicated_edges: list[tuple[str, str, dict[str, Any]]] = []
    for node_id in unique_node_ids:
        for source, target, edge_data in node_edges_batch.get(node_id, []) or []:
            edge_pair = (source, target)
            if edge_pair in seen_edge_pairs:
                continue
            seen_edge_pairs.add(edge_pair)
            deduplicated_edges.append((source, target, edge_data))
    return deduplicated_edges


def _build_canonical_merge_rag(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
):
    """Build a minimal LightRAG facade for legacy merge callers."""
    from .lightrag import LightRAG

    workspace = (
        getattr(entities_vdb, "workspace", None)
        or getattr(relationships_vdb, "workspace", None)
        or getattr(chunk_entity_relation_graph, "workspace", None)
    )
    if workspace is None:
        raise ValueError("Canonical merge shim requires storages with a workspace")

    rag = LightRAG.__new__(LightRAG)
    rag.workspace = workspace
    rag.chunk_entity_relation_graph = chunk_entity_relation_graph
    rag.entities_vdb = entities_vdb
    rag.relationships_vdb = relationships_vdb
    rag.lightrag_logger = logger
    return rag


async def adelete_by_entity(
    chunk_entity_relation_graph, entities_vdb, relationships_vdb, entity_name: str, graph_db_lock: LockProtocol = None
) -> None:
    """Asynchronously delete an entity and all its relationships.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        entity_name: Name of the entity to delete
    """

    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            await entities_vdb.delete_entity(entity_name)
            await relationships_vdb.delete_entity_relation(entity_name)
            await chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(f"Entity '{entity_name}' and its relationships have been deleted.")
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")


async def adelete_by_relation(
    chunk_entity_relation_graph,
    relationships_vdb,
    source_entity: str,
    target_entity: str,
    graph_db_lock: LockProtocol = None,
) -> None:
    """Asynchronously delete a relation between two entities.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        relationships_vdb: Vector database storage for relationships
        source_entity: Name of the source entity
        target_entity: Name of the target entity
        graph_db_lock: Optional lock for ensuring atomic graph and vector db operations
    """

    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            edge_data = await chunk_entity_relation_graph.get_edge(source_entity, target_entity)
            if edge_data is None:
                logger.warning(f"Relation from '{source_entity}' to '{target_entity}' does not exist")
                return

            # Delete relation from vector database
            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-", workspace=relationships_vdb.workspace
            )
            await relationships_vdb.delete([relation_id])

            # Delete relation from knowledge graph
            await chunk_entity_relation_graph.remove_edges([(source_entity, target_entity)])

            logger.info(f"Successfully deleted relation from '{source_entity}' to '{target_entity}'")
        except Exception as e:
            logger.error(f"Error while deleting relation from '{source_entity}' to '{target_entity}': {e}")


async def aedit_entity(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    entity_name: str,
    updated_data: dict[str, str],
    allow_rename: bool = True,
    graph_db_lock: LockProtocol = None,
) -> dict[str, Any]:
    """Asynchronously edit entity information.

    Updates entity information in the knowledge graph and re-embeds the entity in the vector database.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        entity_name: Name of the entity to edit
        updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}
        allow_rename: Whether to allow entity renaming, defaults to True
        graph_db_lock: Optional lock for ensuring atomic graph and vector db operations

    Returns:
        Dictionary containing updated entity information
    """

    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # 1. Get current entity information
            entity_ids_to_fetch = [entity_name]
            new_entity_name = updated_data.get("entity_name", entity_name)
            is_renaming = new_entity_name != entity_name
            if is_renaming:
                entity_ids_to_fetch.append(new_entity_name)

            existing_nodes = await _get_nodes_by_id_batch(chunk_entity_relation_graph, entity_ids_to_fetch)
            node_data = existing_nodes.get(entity_name)
            if node_data is None:
                raise ValueError(f"Entity '{entity_name}' does not exist")

            # Check if entity is being renamed
            # If renaming, check if new name already exists
            if is_renaming:
                if not allow_rename:
                    raise ValueError("Entity renaming is not allowed. Set allow_rename=True to enable this feature")

                if new_entity_name in existing_nodes:
                    raise ValueError(f"Entity name '{new_entity_name}' already exists, cannot rename")

            # 2. Update entity information in the graph
            new_node_data = {**node_data, **updated_data}
            new_node_data["entity_id"] = new_entity_name

            if "entity_name" in new_node_data:
                del new_node_data["entity_name"]  # Node data should not contain entity_name field

            # If renaming entity
            if is_renaming:
                logger.info(f"Renaming entity '{entity_name}' to '{new_entity_name}'")

                # Create new entity
                await chunk_entity_relation_graph.upsert_node(new_entity_name, new_node_data)

                # Store relationships that need to be updated
                relations_to_update = []
                relations_to_delete = []
                # Get all edges related to the original entity
                related_edges = await _get_edges_for_nodes(chunk_entity_relation_graph, [entity_name])
                for source, target, edge_data in related_edges:
                    relations_to_delete.append(
                        compute_mdhash_id(source + target, prefix="rel-", workspace=relationships_vdb.workspace)
                    )
                    relations_to_delete.append(
                        compute_mdhash_id(target + source, prefix="rel-", workspace=relationships_vdb.workspace)
                    )
                    if source == entity_name:
                        await chunk_entity_relation_graph.upsert_edge(new_entity_name, target, edge_data)
                        relations_to_update.append((new_entity_name, target, edge_data))
                    else:  # target == entity_name
                        await chunk_entity_relation_graph.upsert_edge(source, new_entity_name, edge_data)
                        relations_to_update.append((source, new_entity_name, edge_data))

                # Delete old entity
                await chunk_entity_relation_graph.delete_node(entity_name)

                # Delete old entity record from vector database
                old_entity_id = compute_mdhash_id(entity_name, prefix="ent-", workspace=entities_vdb.workspace)
                await entities_vdb.delete([old_entity_id])
                logger.info(f"Deleted old entity '{entity_name}' and its vector embedding from database")

                # Delete old relation records from vector database
                await relationships_vdb.delete(relations_to_delete)
                logger.info(
                    f"Deleted {len(relations_to_delete)} relation records for entity '{entity_name}' from vector database"
                )

                # Update relationship vector representations
                for src, tgt, edge_data in relations_to_update:
                    description = edge_data.get("description", "")
                    keywords = edge_data.get("keywords", "")
                    source_id = edge_data.get("source_id", "")
                    weight = float(edge_data.get("weight", 1.0))

                    # Create new content for embedding
                    content = f"{src}\t{tgt}\n{keywords}\n{description}"

                    # Calculate relationship ID
                    relation_id = compute_mdhash_id(src + tgt, prefix="rel-", workspace=relationships_vdb.workspace)

                    # Prepare data for vector database update
                    relation_data = {
                        relation_id: {
                            "content": content,
                            "src_id": src,
                            "tgt_id": tgt,
                            "source_id": source_id,
                            "description": description,
                            "keywords": keywords,
                            "weight": weight,
                        }
                    }

                    # Update vector database
                    await relationships_vdb.upsert(relation_data)

                # Update working entity name to new name
                entity_name = new_entity_name
            else:
                # If not renaming, directly update node data
                await chunk_entity_relation_graph.upsert_node(entity_name, new_node_data)

            # 3. Recalculate entity's vector representation and update vector database
            description = new_node_data.get("description", "")
            source_id = new_node_data.get("source_id", "")
            entity_type = new_node_data.get("entity_type", "")
            content = entity_name + "\n" + description

            # Calculate entity ID
            entity_id = compute_mdhash_id(entity_name, prefix="ent-", workspace=entities_vdb.workspace)

            # Prepare data for vector database update
            entity_data = {
                entity_id: {
                    "content": content,
                    "entity_name": entity_name,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                }
            }

            # Update vector database
            await entities_vdb.upsert(entity_data)

            logger.info(f"Entity '{entity_name}' successfully updated")
            return await get_entity_info(
                chunk_entity_relation_graph,
                entities_vdb,
                entity_name,
                include_vector_data=True,
            )
        except Exception as e:
            logger.error(f"Error while editing entity '{entity_name}': {e}")
            raise


async def aedit_relation(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    source_entity: str,
    target_entity: str,
    updated_data: dict[str, Any],
    graph_db_lock: LockProtocol = None,
) -> dict[str, Any]:
    """Asynchronously edit relation information.

    Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        source_entity: Name of the source entity
        target_entity: Name of the target entity
        updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "keywords": "new keywords"}
        graph_db_lock: Optional lock for ensuring atomic graph and vector db operations

    Returns:
        Dictionary containing updated relation information
    """

    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # 1. Get current relation information
            edge_data = await chunk_entity_relation_graph.get_edge(source_entity, target_entity)
            if edge_data is None:
                raise ValueError(f"Relation from '{source_entity}' to '{target_entity}' does not exist")
            # Important: First delete the old relation record from the vector database
            old_relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-", workspace=relationships_vdb.workspace
            )
            await relationships_vdb.delete([old_relation_id])
            logger.info(
                f"Deleted old relation record from vector database for relation {source_entity} -> {target_entity}"
            )

            # 2. Update relation information in the graph
            new_edge_data = {**edge_data, **updated_data}
            await chunk_entity_relation_graph.upsert_edge(source_entity, target_entity, new_edge_data)

            # 3. Recalculate relation's vector representation and update vector database
            description = new_edge_data.get("description", "")
            keywords = new_edge_data.get("keywords", "")
            source_id = new_edge_data.get("source_id", "")
            weight = float(new_edge_data.get("weight", 1.0))

            # Create content for embedding
            content = f"{source_entity}\t{target_entity}\n{keywords}\n{description}"

            # Calculate relation ID
            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-", workspace=relationships_vdb.workspace
            )

            # Prepare data for vector database update
            relation_data = {
                relation_id: {
                    "content": content,
                    "src_id": source_entity,
                    "tgt_id": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "keywords": keywords,
                    "weight": weight,
                }
            }

            # Update vector database
            await relationships_vdb.upsert(relation_data)

            logger.info(f"Relation from '{source_entity}' to '{target_entity}' successfully updated")
            return await get_relation_info(
                chunk_entity_relation_graph,
                relationships_vdb,
                source_entity,
                target_entity,
                include_vector_data=True,
            )
        except Exception as e:
            logger.error(f"Error while editing relation from '{source_entity}' to '{target_entity}': {e}")
            raise


async def acreate_entity(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    entity_name: str,
    entity_data: dict[str, Any],
    graph_db_lock: LockProtocol = None,
) -> dict[str, Any]:
    """Asynchronously create a new entity.

    Creates a new entity in the knowledge graph and adds it to the vector database.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        entity_name: Name of the new entity
        entity_data: Dictionary containing entity attributes, e.g. {"description": "description", "entity_type": "type"}
        graph_db_lock: Optional lock for ensuring atomic graph and vector db operations

    Returns:
        Dictionary containing created entity information
    """

    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            # Check if entity already exists
            existing_node = await chunk_entity_relation_graph.has_node(entity_name)
            if existing_node:
                raise ValueError(f"Entity '{entity_name}' already exists")

            # Prepare node data with defaults if missing
            node_data = {
                "entity_id": entity_name,
                "entity_type": entity_data.get("entity_type", "UNKNOWN"),
                "description": entity_data.get("description", ""),
                "source_id": entity_data.get("source_id", "manual_creation"),
                "file_path": entity_data.get("file_path", "manual_creation"),
                "created_at": int(time.time()),
            }

            # Add entity to knowledge graph
            await chunk_entity_relation_graph.upsert_node(entity_name, node_data)

            # Prepare content for entity
            description = node_data.get("description", "")
            source_id = node_data.get("source_id", "")
            entity_type = node_data.get("entity_type", "")
            content = entity_name + "\n" + description

            # Calculate entity ID
            entity_id = compute_mdhash_id(entity_name, prefix="ent-", workspace=entities_vdb.workspace)

            # Prepare data for vector database update
            entity_data_for_vdb = {
                entity_id: {
                    "content": content,
                    "entity_name": entity_name,
                    "source_id": source_id,
                    "description": description,
                    "entity_type": entity_type,
                    "file_path": entity_data.get("file_path", "manual_creation"),
                }
            }

            # Update vector database
            await entities_vdb.upsert(entity_data_for_vdb)

            logger.info(f"Entity '{entity_name}' successfully created")
            return await get_entity_info(
                chunk_entity_relation_graph,
                entities_vdb,
                entity_name,
                include_vector_data=True,
            )
        except Exception as e:
            logger.error(f"Error while creating entity '{entity_name}': {e}")
            raise


async def acreate_relation(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    source_entity: str,
    target_entity: str,
    relation_data: dict[str, Any],
    graph_db_lock: LockProtocol = None,
) -> dict[str, Any]:
    """Asynchronously create a new relation between entities.

    Creates a new relation (edge) in the knowledge graph and adds it to the vector database.

    Args:
        chunk_entity_relation_graph: Graph storage instance
        entities_vdb: Vector database storage for entities
        relationships_vdb: Vector database storage for relationships
        source_entity: Name of the source entity
        target_entity: Name of the target entity
        relation_data: Dictionary containing relation attributes, e.g. {"description": "description", "keywords": "keywords"}
        graph_db_lock: Optional lock for ensuring atomic graph and vector db operations

    Returns:
        Dictionary containing created relation information
    """

    # Use graph database lock to ensure atomic graph and vector db operations
    async with graph_db_lock:
        try:
            existing_nodes = await _get_nodes_by_id_batch(chunk_entity_relation_graph, [source_entity, target_entity])

            if source_entity not in existing_nodes:
                raise ValueError(f"Source entity '{source_entity}' does not exist")
            if target_entity not in existing_nodes:
                raise ValueError(f"Target entity '{target_entity}' does not exist")

            # Check if relation already exists
            existing_edge = await chunk_entity_relation_graph.get_edge(source_entity, target_entity)
            if existing_edge is not None:
                raise ValueError(f"Relation from '{source_entity}' to '{target_entity}' already exists")

            # Prepare edge data with defaults if missing
            edge_data = {
                "description": relation_data.get("description", ""),
                "keywords": relation_data.get("keywords", ""),
                "source_id": relation_data.get("source_id", "manual_creation"),
                "weight": float(relation_data.get("weight", 1.0)),
                "file_path": relation_data.get("file_path", "manual_creation"),
                "created_at": int(time.time()),
            }

            # Add relation to knowledge graph
            await chunk_entity_relation_graph.upsert_edge(source_entity, target_entity, edge_data)

            # Prepare content for embedding
            description = edge_data.get("description", "")
            keywords = edge_data.get("keywords", "")
            source_id = edge_data.get("source_id", "")
            weight = edge_data.get("weight", 1.0)

            # Create content for embedding
            content = f"{keywords}\t{source_entity}\n{target_entity}\n{description}"

            # Calculate relation ID
            relation_id = compute_mdhash_id(
                source_entity + target_entity, prefix="rel-", workspace=relationships_vdb.workspace
            )

            # Prepare data for vector database update
            relation_data_for_vdb = {
                relation_id: {
                    "content": content,
                    "src_id": source_entity,
                    "tgt_id": target_entity,
                    "source_id": source_id,
                    "description": description,
                    "keywords": keywords,
                    "weight": weight,
                    "file_path": relation_data.get("file_path", "manual_creation"),
                }
            }

            # Update vector database
            await relationships_vdb.upsert(relation_data_for_vdb)

            logger.info(f"Relation from '{source_entity}' to '{target_entity}' successfully created")
            return await get_relation_info(
                chunk_entity_relation_graph,
                relationships_vdb,
                source_entity,
                target_entity,
                include_vector_data=True,
            )
        except Exception as e:
            logger.error(f"Error while creating relation from '{source_entity}' to '{target_entity}': {e}")
            raise


async def amerge_entities(
    chunk_entity_relation_graph,
    entities_vdb,
    relationships_vdb,
    source_entities: list[str],
    target_entity: str,
    merge_strategy: dict[str, str] = None,
    target_entity_data: dict[str, Any] = None,
    graph_db_lock: LockProtocol = None,
) -> dict[str, Any]:
    """Legacy shim that routes all merges through LightRAG.amerge_nodes()."""

    if merge_strategy not in (None, {}):
        raise ValueError("Custom merge_strategy is no longer supported. Use LightRAG.amerge_nodes() instead")

    target_payload = {"entity_name": target_entity}
    if target_entity_data:
        target_payload.update(target_entity_data)

    logger.info("utils_graph.amerge_entities() is deprecated; routing merge through LightRAG.amerge_nodes()")

    async with graph_db_lock:
        rag = _build_canonical_merge_rag(
            chunk_entity_relation_graph,
            entities_vdb,
            relationships_vdb,
        )
        return await rag.amerge_nodes(
            entity_ids=source_entities,
            target_entity_data=target_payload,
        )


async def get_entity_info(
    chunk_entity_relation_graph,
    entities_vdb,
    entity_name: str,
    include_vector_data: bool = False,
) -> dict[str, str | None | dict[str, str]]:
    """Get detailed information of an entity"""

    # Get information from the graph
    node_data = await chunk_entity_relation_graph.get_node(entity_name)
    source_id = node_data.get("source_id") if node_data else None

    result: dict[str, str | None | dict[str, str]] = {
        "entity_name": entity_name,
        "source_id": source_id,
        "graph_data": node_data,
    }

    # Optional: Get vector database information
    if include_vector_data:
        entity_id = compute_mdhash_id(entity_name, prefix="ent-", workspace=entities_vdb.workspace)
        vector_data = await entities_vdb.get_by_id(entity_id)
        result["vector_data"] = vector_data

    return result


async def get_relation_info(
    chunk_entity_relation_graph,
    relationships_vdb,
    src_entity: str,
    tgt_entity: str,
    include_vector_data: bool = False,
) -> dict[str, str | None | dict[str, str]]:
    """Get detailed information of a relationship"""

    # Get information from the graph
    edge_data = await chunk_entity_relation_graph.get_edge(src_entity, tgt_entity)
    source_id = edge_data.get("source_id") if edge_data else None

    result: dict[str, str | None | dict[str, str]] = {
        "src_entity": src_entity,
        "tgt_entity": tgt_entity,
        "source_id": source_id,
        "graph_data": edge_data,
    }

    # Optional: Get vector database information
    if include_vector_data:
        rel_id = compute_mdhash_id(src_entity + tgt_entity, prefix="rel-", workspace=relationships_vdb.workspace)
        vector_data = await relationships_vdb.get_by_id(rel_id)
        result["vector_data"] = vector_data

    return result
