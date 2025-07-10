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

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: list[str]
    low_level_keywords: list[str]


class KnowledgeGraphNode(BaseModel):
    id: str
    labels: list[str]
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraphEdge(BaseModel):
    id: str
    type: Optional[str]
    source: str  # id of source node
    target: str  # id of target node
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraph(BaseModel):
    nodes: list[KnowledgeGraphNode] = []
    edges: list[KnowledgeGraphEdge] = []
    is_truncated: bool = False


# ============= Graph Data Types =============


class GraphNodeData(BaseModel):
    """Represents complete data of a graph node"""

    entity_id: str
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    description: Optional[str] = None
    source_id: Optional[str] = None
    file_path: Optional[str] = None
    created_at: Optional[int] = None
    degree: Optional[int] = None  # Node degree (optional, only when queried)

    class Config:
        extra = "allow"  # Allow additional fields


class GraphEdgeData(BaseModel):
    """Represents complete data of a graph edge"""

    source: str
    target: str
    weight: Optional[float] = None
    description: Optional[str] = None
    keywords: Optional[str] = None
    source_id: Optional[str] = None
    file_path: Optional[str] = None
    created_at: Optional[int] = None

    class Config:
        extra = "allow"  # Allow additional fields


# ============= Graph Collections Types =============


class GraphNodeDataDict(BaseModel):
    """Dictionary of graph nodes with utility methods"""

    nodes_by_id: Dict[str, GraphNodeData]

    def get_node(self, node_id: str) -> Optional[GraphNodeData]:
        """Get node by ID"""
        return self.nodes_by_id.get(node_id)

    def get_node_degree(self, node_id: str) -> int:
        """Get degree for a specific node"""
        node = self.nodes_by_id.get(node_id)
        return node.degree if node and node.degree is not None else 0

    def get_high_degree_nodes(self, limit: int = None) -> List[GraphNodeData]:
        """Get nodes sorted by degree (highest first)"""
        sorted_nodes = sorted(self.nodes_by_id.values(), key=lambda x: x.degree or 0, reverse=True)
        return sorted_nodes[:limit] if limit else sorted_nodes

    @property
    def labels(self) -> List[str]:
        """Get all node IDs for backward compatibility"""
        return list(self.nodes_by_id.keys())

    @property
    def degrees_map(self) -> Dict[str, int]:
        """Get degrees mapping for backward compatibility"""
        return {node_id: node.degree or 0 for node_id, node in self.nodes_by_id.items()}

    @property
    def nodes_map(self) -> Dict[str, GraphNodeData]:
        """Get nodes mapping for backward compatibility"""
        return self.nodes_by_id


# ============= Merge Suggestions Types =============


class MergeSuggestion(BaseModel):
    """Single merge suggestion with simplified structure"""

    entities: List[GraphNodeData]  # Entities to be merged
    confidence_score: float
    merge_reason: str
    suggested_target_entity: GraphNodeData  # Suggested target after merge


class MergeSuggestionsResult(BaseModel):
    """Complete result of merge suggestions analysis"""

    suggestions: List[MergeSuggestion]
    total_analyzed_nodes: int
    processing_time_seconds: float
