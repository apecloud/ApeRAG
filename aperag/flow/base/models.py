from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from collections import deque
from aperag.flow.base.exceptions import CycleError

@dataclass
class NodeInstance:
    """Instance of a node in the flow """
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

NODE_RUNNER_REGISTRY = {}

class BaseNodeRunner(ABC):

    @abstractmethod
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        raise NotImplementedError

def register_node_runner(node_type: str):
    def decorator(cls):
        NODE_RUNNER_REGISTRY[node_type] = cls()
        return cls
    return decorator
