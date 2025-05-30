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

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from aperag.chat.history.base import BaseChatMessageHistory
from aperag.flow.base.exceptions import CycleError


@dataclass
class NodeInstance:
    """Instance of a node in the flow"""

    id: str
    type: str  # NodeDefinition.type
    input_schema: dict = field(default_factory=dict)
    input_values: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    title: Optional[str] = None


@dataclass
class Edge:
    """Connection between nodes in the flow"""

    source: str
    target: str


@dataclass
class FlowInstance:
    """Instance of a flow with nodes and edges"""

    name: str
    title: str
    nodes: Dict[str, NodeInstance]
    edges: List[Edge]

    def validate(self) -> None:
        """Validate the flow configuration"""
        self._topological_sort()

    def _topological_sort(self) -> List[str]:
        """Perform topological sort to detect cycles"""
        # Build dependency graph
        in_degree = {node_id: 0 for node_id in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] += 1

        # Topological sort
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        if len(queue) == 0:
            raise CycleError("Flow contains cycles")

        sorted_nodes = []

        while queue:
            node_id = queue.popleft()
            sorted_nodes.append(node_id)

            # Update in-degree of successor nodes
            for edge in self.edges:
                if edge.source == node_id:
                    in_degree[edge.target] -= 1
                    if in_degree[edge.target] == 0:
                        queue.append(edge.target)

        if len(sorted_nodes) != len(self.nodes):
            raise CycleError("Flow contains cycles")

        return sorted_nodes


@dataclass
class ExecutionContext:
    """Context for flow execution, storing outputs and global state"""

    outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    system_outputs: Dict[str, Any] = field(default_factory=dict)
    global_variables: Dict[str, Any] = field(default_factory=dict)

    def get_input(self, node_id: str, field: str) -> Any:
        """Get input value for a node field"""
        return self.outputs.get(node_id, {}).get(field)

    def set_output(self, node_id: str, outputs: Dict[str, Any]) -> None:
        """Set output values for a node"""
        self.outputs[node_id] = outputs

    def get_global(self, name: str) -> Any:
        """Get global variable value"""
        return self.global_variables.get(name)

    def set_global(self, name: str, value: Any) -> None:
        """Set global variable value"""
        self.global_variables[name] = value

    def set_system_output(self, node_id: str, system_output: Any) -> None:
        """Set system output for a node"""
        self.system_outputs[node_id] = system_output

    def get_system_output(self, node_id: str) -> Any:
        """Get system output for a node"""
        return self.system_outputs.get(node_id)


NODE_RUNNER_REGISTRY = {}


class BaseNodeRunner(ABC):
    @abstractmethod
    async def run(self, ui: Any, si: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError


def register_node_runner(
    node_type: str,
    input_model,
    output_model,
):
    def decorator(cls):
        NODE_RUNNER_REGISTRY[node_type] = {
            "runner": cls(),
            "input_model": input_model,
            "output_model": output_model,
        }
        return cls

    return decorator


class SystemInput:
    query: str
    user: str
    history: Optional[BaseChatMessageHistory] = None
    message_id: Optional[str] = None

    def __init__(
        self, query: str, user: str, history: Optional[BaseChatMessageHistory] = None, message_id: Optional[str] = None
    ):
        self.query = query
        self.user = user
        self.history = history
        self.message_id = message_id
