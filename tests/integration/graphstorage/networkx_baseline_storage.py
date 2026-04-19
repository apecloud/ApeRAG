"""
NetworkX-based in-memory graph storage implementation for testing baseline.

This implementation serves as the "ground truth" for comparing other graph storage backends.
It implements the BaseGraphStorage interface using NetworkX for accurate and predictable results.
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from aperag.graph.lightrag.base import BaseGraphStorage
from aperag.graph.lightrag.types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode


class NetworkXBaselineStorage(BaseGraphStorage):
    """
    NetworkX-based baseline implementation for testing graph storage consistency.

    This serves as the "ground truth" for comparing other storage implementations.
    All operations are performed in-memory with NetworkX for reliable and predictable results.
    """

    def __init__(self, namespace: str, workspace: str, embedding_func=None):
        super().__init__(namespace=namespace, workspace=workspace, embedding_func=embedding_func)
        self.graph = nx.Graph()
        self._node_data = {}  # Store complete node data
        self._edge_data = {}  # Store complete edge data
        self._initialized = False

    async def initialize(self):
        """Initialize the storage."""
        self._initialized = True

    async def finalize(self):
        """Clean up the storage."""
        self.graph.clear()
        self._node_data.clear()
        self._edge_data.clear()
        self._initialized = False

    # ===== Node Operations =====

    async def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        return node_id in self.graph.nodes

    async def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data."""
        if node_id not in self.graph.nodes:
            return None
        return self._node_data.get(node_id, {}).copy()

    async def get_nodes_batch(self, node_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get multiple nodes in batch."""
        result = {}
        for node_id in node_ids:
            if node_id in self.graph.nodes:
                result[node_id] = self._node_data.get(node_id, {}).copy()
        return result

    async def node_degree(self, node_id: str) -> int:
        """Get node degree."""
        if node_id not in self.graph.nodes:
            return 0
        return self.graph.degree(node_id)

    async def node_degrees_batch(self, node_ids: List[str]) -> Dict[str, int]:
        """Get degrees for multiple nodes."""
        result = {}
        for node_id in node_ids:
            result[node_id] = self.graph.degree(node_id) if node_id in self.graph.nodes else 0
        return result

    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """Insert or update a node."""
        if "entity_id" not in node_data:
            raise ValueError("Node data must contain 'entity_id' field")

        # Add node to graph
        self.graph.add_node(node_id)

        # Store complete node data
        self._node_data[node_id] = node_data.copy()

    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its edges."""
        if node_id in self.graph.nodes:
            self.graph.remove_node(node_id)
            self._node_data.pop(node_id, None)

            # Remove related edge data
            edges_to_remove = []
            for edge_key in self._edge_data.keys():
                if isinstance(edge_key, tuple) and (edge_key[0] == node_id or edge_key[1] == node_id):
                    edges_to_remove.append(edge_key)

            for edge_key in edges_to_remove:
                self._edge_data.pop(edge_key, None)

    async def remove_nodes(self, node_ids: List[str]) -> None:
        """Batch delete nodes."""
        for node_id in node_ids:
            await self.delete_node(node_id)

    # ===== Edge Operations =====

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if an edge exists."""
        return self.graph.has_edge(source_node_id, target_node_id)

    async def get_edge(self, source_node_id: str, target_node_id: str) -> Optional[Dict[str, Any]]:
        """Get edge data."""
        if not self.graph.has_edge(source_node_id, target_node_id):
            return None

        # Try both directions for edge key
        edge_key = (source_node_id, target_node_id)
        reverse_key = (target_node_id, source_node_id)

        edge_data = self._edge_data.get(edge_key) or self._edge_data.get(reverse_key)
        if edge_data:
            edge_result = edge_data.copy()
            # Ensure required keys exist with defaults (consistent with Neo4j implementation)
            required_keys = {
                "weight": 0.0,
                "source_id": None,
                "description": None,
                "keywords": None,
            }
            for key, default_value in required_keys.items():
                if key not in edge_result:
                    edge_result[key] = default_value
            return edge_result
        return None

    async def get_edges_batch(self, pairs: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Get multiple edges in batch."""
        result = {}
        for pair in pairs:
            src, tgt = pair["src"], pair["tgt"]
            edge_data = await self.get_edge(src, tgt)
            if edge_data:
                result[(src, tgt)] = edge_data
        return result

    async def get_node_edges(self, source_node_id: str) -> Optional[List[Tuple[str, str]]]:
        """Get all edges for a node."""
        if source_node_id not in self.graph.nodes:
            return None

        edges = []
        for neighbor in self.graph.neighbors(source_node_id):
            edges.append((source_node_id, neighbor))

        return edges if edges else None

    async def get_nodes_edges_batch(self, node_ids: List[str]) -> Dict[str, List[Tuple[str, str]]]:
        """Get edges for multiple nodes."""
        result = {}
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            result[node_id] = edges if edges else []
        return result

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get combined degree of two nodes."""
        src_degree = await self.node_degree(src_id)
        tgt_degree = await self.node_degree(tgt_id)
        return src_degree + tgt_degree

    async def edge_degrees_batch(self, edge_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
        """Get combined degrees for multiple edges."""
        result = {}
        for src, tgt in edge_pairs:
            result[(src, tgt)] = await self.edge_degree(src, tgt)
        return result

    async def upsert_edge(self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]) -> None:
        """Insert or update an edge."""
        # Add nodes if they don't exist
        if source_node_id not in self.graph.nodes:
            await self.upsert_node(
                source_node_id,
                {
                    "entity_id": source_node_id,
                    "entity_type": "UNKNOWN",
                    "description": "Auto-created node",
                    "source_id": "auto",
                },
            )

        if target_node_id not in self.graph.nodes:
            await self.upsert_node(
                target_node_id,
                {
                    "entity_id": target_node_id,
                    "entity_type": "UNKNOWN",
                    "description": "Auto-created node",
                    "source_id": "auto",
                },
            )

        # Add edge to graph
        self.graph.add_edge(source_node_id, target_node_id)

        # Store edge data (use consistent key ordering)
        edge_key = tuple(sorted([source_node_id, target_node_id]))
        self._edge_data[edge_key] = edge_data.copy()

    async def remove_edges(self, edges: List[Tuple[str, str]]) -> None:
        """Batch delete edges."""
        for source_node_id, target_node_id in edges:
            if self.graph.has_edge(source_node_id, target_node_id):
                self.graph.remove_edge(source_node_id, target_node_id)

                # Remove edge data (try both key orders)
                edge_key = (source_node_id, target_node_id)
                reverse_key = (target_node_id, source_node_id)
                self._edge_data.pop(edge_key, None)
                self._edge_data.pop(reverse_key, None)

    # ===== Graph Operations =====

    async def get_all_labels(self) -> List[str]:
        """Get all node labels (node IDs in this case)."""
        return list(self.graph.nodes())

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """Get subgraph around a node or all nodes if label is '*'."""
        if node_label == "*":
            # Return entire graph
            nodes_to_include = list(self.graph.nodes())[:max_nodes]
        else:
            # BFS from the specified node
            if node_label not in self.graph.nodes:
                return KnowledgeGraph(nodes=[], edges=[])

            nodes_to_include = set()
            current_level = {node_label}
            nodes_to_include.add(node_label)

            for depth in range(max_depth):
                if len(nodes_to_include) >= max_nodes:
                    break

                next_level = set()
                for node in current_level:
                    for neighbor in self.graph.neighbors(node):
                        if neighbor not in nodes_to_include:
                            next_level.add(neighbor)
                            nodes_to_include.add(neighbor)

                            if len(nodes_to_include) >= max_nodes:
                                break

                    if len(nodes_to_include) >= max_nodes:
                        break

                current_level = next_level
                if not current_level:
                    break

            nodes_to_include = list(nodes_to_include)

        # Build knowledge graph
        kg_nodes = []
        for node_id in nodes_to_include:
            node_data = self._node_data.get(node_id, {})
            kg_node = KnowledgeGraphNode(
                id=node_id,
                entity_type=node_data.get("entity_type", "UNKNOWN"),
                description=node_data.get("description", ""),
                source_id=node_data.get("source_id", ""),
            )
            kg_nodes.append(kg_node)

        kg_edges = []
        for edge in self.graph.edges():
            src, tgt = edge
            if src in nodes_to_include and tgt in nodes_to_include:
                edge_data = await self.get_edge(src, tgt)
                if edge_data:
                    kg_edge = KnowledgeGraphEdge(
                        source_id=src,
                        target_id=tgt,
                        description=edge_data.get("description", ""),
                        keywords=edge_data.get("keywords", ""),
                        weight=edge_data.get("weight", 1.0),
                    )
                    kg_edges.append(kg_edge)

        return KnowledgeGraph(nodes=kg_nodes, edges=kg_edges)

    async def drop(self) -> Dict[str, str]:
        """Drop all data."""
        self.graph.clear()
        self._node_data.clear()
        self._edge_data.clear()
        return {"status": "success", "message": "All data dropped"}

    # ===== Utility Methods =====

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "is_connected": nx.is_connected(self.graph) if self.graph.number_of_nodes() > 0 else False,
            "density": nx.density(self.graph),
            "average_clustering": nx.average_clustering(self.graph) if self.graph.number_of_nodes() > 2 else 0.0,
        }

    def get_connected_components(self) -> List[List[str]]:
        """Get connected components."""
        return [list(component) for component in nx.connected_components(self.graph)]

    async def compare_with_other_storage(
        self, other_storage: BaseGraphStorage, node_ids: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare this baseline storage with another storage implementation.
        This is useful for validating that other storage backends behave consistently.
        """
        if node_ids is None:
            node_ids = list(self.graph.nodes())[:100]  # Sample for performance

        comparison_result = {
            "nodes_compared": len(node_ids),
            "nodes_match": 0,
            "nodes_mismatch": 0,
            "edges_compared": 0,
            "edges_match": 0,
            "edges_mismatch": 0,
            "mismatches": [],
        }

        # Compare nodes
        for node_id in node_ids:
            baseline_node = await self.get_node(node_id)
            other_node = await other_storage.get_node(node_id)

            if baseline_node is None and other_node is None:
                comparison_result["nodes_match"] += 1
            elif baseline_node is None or other_node is None:
                comparison_result["nodes_mismatch"] += 1
                comparison_result["mismatches"].append(
                    {
                        "type": "node_existence",
                        "node_id": node_id,
                        "baseline": baseline_node is not None,
                        "other": other_node is not None,
                    }
                )
            else:
                # Compare key fields
                key_fields = ["entity_id", "entity_type", "description"]
                node_match = True
                for field in key_fields:
                    if baseline_node.get(field) != other_node.get(field):
                        node_match = False
                        comparison_result["mismatches"].append(
                            {
                                "type": "node_field",
                                "node_id": node_id,
                                "field": field,
                                "baseline": baseline_node.get(field),
                                "other": other_node.get(field),
                            }
                        )

                if node_match:
                    comparison_result["nodes_match"] += 1
                else:
                    comparison_result["nodes_mismatch"] += 1

        # Compare edges
        edge_pairs = []
        for node_id in node_ids:
            edges = await self.get_node_edges(node_id)
            if edges:
                edge_pairs.extend(edges)

        # Deduplicate edges
        unique_edges = list(set(edge_pairs))
        comparison_result["edges_compared"] = len(unique_edges)

        for src, tgt in unique_edges:
            baseline_edge = await self.get_edge(src, tgt)
            other_edge = await other_storage.get_edge(src, tgt)

            if baseline_edge is None and other_edge is None:
                comparison_result["edges_match"] += 1
            elif baseline_edge is None or other_edge is None:
                comparison_result["edges_mismatch"] += 1
                comparison_result["mismatches"].append(
                    {
                        "type": "edge_existence",
                        "edge": (src, tgt),
                        "baseline": baseline_edge is not None,
                        "other": other_edge is not None,
                    }
                )
            else:
                # Compare key fields
                key_fields = ["weight", "description"]
                edge_match = True
                for field in key_fields:
                    baseline_val = baseline_edge.get(field)
                    other_val = other_edge.get(field)

                    # Handle numeric comparison with tolerance
                    if field == "weight" and baseline_val is not None and other_val is not None:
                        if abs(float(baseline_val) - float(other_val)) > 1e-6:
                            edge_match = False
                    elif baseline_val != other_val:
                        edge_match = False

                    if not edge_match:
                        comparison_result["mismatches"].append(
                            {
                                "type": "edge_field",
                                "edge": (src, tgt),
                                "field": field,
                                "baseline": baseline_val,
                                "other": other_val,
                            }
                        )
                        break

                if edge_match:
                    comparison_result["edges_match"] += 1
                else:
                    comparison_result["edges_mismatch"] += 1

        return comparison_result
