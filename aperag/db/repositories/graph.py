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
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, delete, func, or_, select, text
from sqlalchemy.dialects.postgresql import insert

from aperag.db.models import LightRAGGraphEdge, LightRAGGraphNode

logger = logging.getLogger(__name__)


class GraphRepositoryMixin:
    """Graph Repository Mixin for LightRAG Graph operations using SQLAlchemy"""

    # Node operations
    def upsert_graph_node(self, workspace: str, node_id: str, node_data: Dict[str, Any]) -> None:
        """Upsert a graph node"""
        def _upsert_node(session):
            # Prepare node data
            entity_name = node_data.get("entity_name") if node_data.get("entity_name") != node_id else None
            entity_type = node_data.get("entity_type")
            description = node_data.get("description")
            source_id = node_data.get("source_id")
            file_path = node_data.get("file_path")

            # Use PostgreSQL ON CONFLICT for true upsert
            stmt = insert(LightRAGGraphNode).values(
                workspace=workspace,
                entity_id=node_id,
                entity_name=entity_name,
                entity_type=entity_type,
                description=description,
                source_id=source_id,
                file_path=file_path,
            )
            
            # ON CONFLICT DO UPDATE
            stmt = stmt.on_conflict_do_update(
                index_elements=['workspace', 'entity_id'],
                set_=dict(
                    entity_name=stmt.excluded.entity_name,
                    entity_type=stmt.excluded.entity_type,
                    description=stmt.excluded.description,
                    source_id=stmt.excluded.source_id,
                    file_path=stmt.excluded.file_path,
                    updatetime=func.now()
                )
            )
            
            session.execute(stmt)
            session.flush()
            logger.debug(f"Upserted graph node: {node_id} in workspace {workspace}")

        return self._execute_transaction(_upsert_node)

    def upsert_graph_edge(self, workspace: str, source_node_id: str, target_node_id: str, 
                         edge_data: Dict[str, Any]) -> None:
        """Upsert a graph edge"""
        def _upsert_edge(session):
            # Prepare edge data
            weight = float(edge_data.get("weight", 0.0))
            keywords = edge_data.get("keywords")
            description = edge_data.get("description")
            source_id = edge_data.get("source_id")
            file_path = edge_data.get("file_path")

            # Use PostgreSQL ON CONFLICT for true upsert
            stmt = insert(LightRAGGraphEdge).values(
                workspace=workspace,
                source_entity_id=source_node_id,
                target_entity_id=target_node_id,
                weight=weight,
                keywords=keywords,
                description=description,
                source_id=source_id,
                file_path=file_path,
            )
            
            # ON CONFLICT DO UPDATE
            stmt = stmt.on_conflict_do_update(
                index_elements=['workspace', 'source_entity_id', 'target_entity_id'],
                set_=dict(
                    weight=stmt.excluded.weight,
                    keywords=stmt.excluded.keywords,
                    description=stmt.excluded.description,
                    source_id=stmt.excluded.source_id,
                    file_path=stmt.excluded.file_path,
                    updatetime=func.now()
                )
            )
            
            session.execute(stmt)
            session.flush()
            logger.debug(f"Upserted graph edge: {source_node_id} -> {target_node_id} in workspace {workspace}")

        return self._execute_transaction(_upsert_edge)

    def has_graph_node(self, workspace: str, node_id: str) -> bool:
        """Check if a graph node exists"""
        def _has_node(session):
            stmt = select(func.count(LightRAGGraphNode.id)).where(
                and_(
                    LightRAGGraphNode.workspace == workspace,
                    LightRAGGraphNode.entity_id == node_id
                )
            )
            result = session.execute(stmt)
            return result.scalar() > 0

        return self._execute_query(_has_node)

    def has_graph_edge(self, workspace: str, source_node_id: str, target_node_id: str) -> bool:
        """Check if a graph edge exists"""
        def _has_edge(session):
            stmt = select(func.count(LightRAGGraphEdge.id)).where(
                and_(
                    LightRAGGraphEdge.workspace == workspace,
                    LightRAGGraphEdge.source_entity_id == source_node_id,
                    LightRAGGraphEdge.target_entity_id == target_node_id
                )
            )
            result = session.execute(stmt)
            return result.scalar() > 0

        return self._execute_query(_has_edge)

    def get_graph_node(self, workspace: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a graph node by ID"""
        def _get_node(session):
            stmt = select(LightRAGGraphNode).where(
                and_(
                    LightRAGGraphNode.workspace == workspace,
                    LightRAGGraphNode.entity_id == node_id
                )
            )
            result = session.execute(stmt)
            node = result.scalar_one_or_none()
            
            if not node:
                return None
                
            # Convert to dict format matching the original interface
            node_dict = {
                "entity_id": node.entity_id,
                "entity_type": node.entity_type,
                "description": node.description,
                "source_id": node.source_id,
                "file_path": node.file_path,
                "created_at": int(node.createtime.timestamp()) if node.createtime else None,
            }
            
            # Only include entity_name if it's different from entity_id and not None
            if node.entity_name and node.entity_name != node.entity_id:
                node_dict["entity_name"] = node.entity_name
                
            # Remove None values for cleaner output
            return {k: v for k, v in node_dict.items() if v is not None}

        return self._execute_query(_get_node)

    def get_graph_edge(self, workspace: str, source_node_id: str, target_node_id: str) -> Optional[Dict[str, Any]]:
        """Get a graph edge"""
        def _get_edge(session):
            stmt = select(LightRAGGraphEdge).where(
                and_(
                    LightRAGGraphEdge.workspace == workspace,
                    LightRAGGraphEdge.source_entity_id == source_node_id,
                    LightRAGGraphEdge.target_entity_id == target_node_id
                )
            )
            result = session.execute(stmt)
            edge = result.scalar_one_or_none()
            
            if not edge:
                return None
                
            # Convert to dict format matching the original interface
            edge_dict = {
                "weight": float(edge.weight) if edge.weight is not None else 0.0,
                "keywords": edge.keywords,
                "description": edge.description,
                "source_id": edge.source_id,
                "file_path": edge.file_path,
            }
            
            # Keep required fields even if None, remove optional fields if None
            required_fields = {"weight", "keywords", "description", "source_id"}
            filtered_result = {}
            for k, v in edge_dict.items():
                if k in required_fields or v is not None:
                    filtered_result[k] = v
                    
            return filtered_result

        return self._execute_query(_get_edge)

    def get_graph_node_degree(self, workspace: str, node_id: str) -> int:
        """Get the degree of a graph node"""
        def _get_degree(session):
            # Count edges where node is either source or target
            outgoing_stmt = select(func.count(LightRAGGraphEdge.id)).where(
                and_(
                    LightRAGGraphEdge.workspace == workspace,
                    LightRAGGraphEdge.source_entity_id == node_id
                )
            )
            incoming_stmt = select(func.count(LightRAGGraphEdge.id)).where(
                and_(
                    LightRAGGraphEdge.workspace == workspace,
                    LightRAGGraphEdge.target_entity_id == node_id
                )
            )
            
            outgoing_count = session.execute(outgoing_stmt).scalar()
            incoming_count = session.execute(incoming_stmt).scalar()
            
            return outgoing_count + incoming_count

        return self._execute_query(_get_degree)

    def get_graph_node_edges(self, workspace: str, source_node_id: str) -> List[Tuple[str, str]]:
        """Get all edges for a node"""
        def _get_node_edges(session):
            # Get outgoing edges
            outgoing_stmt = select(
                LightRAGGraphEdge.source_entity_id,
                LightRAGGraphEdge.target_entity_id
            ).where(
                and_(
                    LightRAGGraphEdge.workspace == workspace,
                    LightRAGGraphEdge.source_entity_id == source_node_id
                )
            )
            
            # Get incoming edges  
            incoming_stmt = select(
                LightRAGGraphEdge.source_entity_id,
                LightRAGGraphEdge.target_entity_id
            ).where(
                and_(
                    LightRAGGraphEdge.workspace == workspace,
                    LightRAGGraphEdge.target_entity_id == source_node_id
                )
            )
            
            edges = []
            
            # Process outgoing edges
            outgoing_result = session.execute(outgoing_stmt)
            edges.extend([(row[0], row[1]) for row in outgoing_result])
            
            # Process incoming edges
            incoming_result = session.execute(incoming_stmt)
            edges.extend([(row[0], row[1]) for row in incoming_result])
            
            return edges if edges else []

        return self._execute_query(_get_node_edges)

    def delete_graph_node(self, workspace: str, node_id: str) -> None:
        """Delete a graph node and all its edges"""
        def _delete_node(session):
            # First delete all edges related to this node
            edge_delete_stmt = delete(LightRAGGraphEdge).where(
                and_(
                    LightRAGGraphEdge.workspace == workspace,
                    or_(
                        LightRAGGraphEdge.source_entity_id == node_id,
                        LightRAGGraphEdge.target_entity_id == node_id
                    )
                )
            )
            session.execute(edge_delete_stmt)
            
            # Then delete the node itself
            node_delete_stmt = delete(LightRAGGraphNode).where(
                and_(
                    LightRAGGraphNode.workspace == workspace,
                    LightRAGGraphNode.entity_id == node_id
                )
            )
            session.execute(node_delete_stmt)
            session.flush()
            logger.debug(f"Deleted graph node: {node_id} and its edges in workspace {workspace}")

        return self._execute_transaction(_delete_node)

    def delete_graph_edges(self, workspace: str, edges: List[Tuple[str, str]]) -> None:
        """Delete multiple graph edges"""
        if not edges:
            return
            
        def _delete_edges(session):
            for source, target in edges:
                delete_stmt = delete(LightRAGGraphEdge).where(
                    and_(
                        LightRAGGraphEdge.workspace == workspace,
                        LightRAGGraphEdge.source_entity_id == source,
                        LightRAGGraphEdge.target_entity_id == target
                    )
                )
                session.execute(delete_stmt)
            session.flush()
            logger.debug(f"Deleted {len(edges)} graph edges in workspace {workspace}")

        return self._execute_transaction(_delete_edges)

    def get_all_graph_labels(self, workspace: str) -> List[str]:
        """Get all entity labels in the graph"""
        def _get_labels(session):
            stmt = select(LightRAGGraphNode.entity_id).where(
                LightRAGGraphNode.workspace == workspace
            ).order_by(LightRAGGraphNode.entity_id)
            
            result = session.execute(stmt)
            return [row[0] for row in result]

        return self._execute_query(_get_labels)

    def drop_graph_workspace(self, workspace: str) -> Dict[str, str]:
        """Drop all graph data for a workspace"""
        def _drop_workspace(session):
            try:
                # Delete all edges for this workspace
                edges_delete_stmt = delete(LightRAGGraphEdge).where(
                    LightRAGGraphEdge.workspace == workspace
                )
                session.execute(edges_delete_stmt)
                
                # Delete all nodes for this workspace  
                nodes_delete_stmt = delete(LightRAGGraphNode).where(
                    LightRAGGraphNode.workspace == workspace
                )
                session.execute(nodes_delete_stmt)
                
                session.flush()
                logger.info(f"Successfully dropped all graph data for workspace {workspace}")
                return {"status": "success", "message": "graph data dropped"}
            except Exception as e:
                logger.error(f"Error dropping graph for workspace {workspace}: {e}")
                return {"status": "error", "message": str(e)}

        return self._execute_transaction(_drop_workspace)




 