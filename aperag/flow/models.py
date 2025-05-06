from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from enum import Enum
from collections import deque
from aperag.flow.exceptions import ValidationError, CycleError

class NodeType(str, Enum):
    """Node types in the flow"""
    INPUT = "input"
    PROCESS = "process"
    OUTPUT = "output"

class FieldType(str, Enum):
    """Field types for node inputs and outputs"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"

class InputSourceType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    GLOBAL = "global"

@dataclass
class FieldDefinition:
    """Definition of a field in node input/output schema"""
    name: str
    type: FieldType
    description: Optional[str] = None
    required: bool = False
    default: Any = None

@dataclass
class NodeDefinition:
    """Definition of a node type"""
    id: str
    name: str
    type: NodeType
    config_schema: List[FieldDefinition]  # Static configuration parameters
    input_schema: List[FieldDefinition]   # Dynamic input parameters
    output_schema: List[FieldDefinition]
    description: Optional[str] = None

class NodeRegistry:
    """Registry for node type definitions"""
    _nodes: Dict[str, NodeDefinition] = {}

    @classmethod
    def register(cls, node_def: NodeDefinition) -> None:
        """Register a new node type"""
        cls._nodes[node_def.id] = node_def

    @classmethod
    def get(cls, node_id: str) -> NodeDefinition:
        """Get a node type definition by ID"""
        if node_id not in cls._nodes:
            raise KeyError(f"Node definition not found: {node_id}")
        return cls._nodes[node_id]

    @classmethod
    def list_nodes(cls) -> List[NodeDefinition]:
        """List all registered node types"""
        return list(cls._nodes.values())

# Register predefined nodes
NodeRegistry.register(NodeDefinition(
    id="vector_search",
    name="Vector Search",
    type=NodeType.PROCESS,
    config_schema=[
        FieldDefinition("top_k", FieldType.INTEGER, default=5),
        FieldDefinition("similarity_threshold", FieldType.FLOAT, default=0.7)
    ],
    input_schema=[
        FieldDefinition("query", FieldType.STRING)
    ],
    output_schema=[
        FieldDefinition("docs", FieldType.ARRAY)
    ]
))

NodeRegistry.register(NodeDefinition(
    id="keyword_search",
    name="Keyword Search",
    type=NodeType.PROCESS,
    config_schema=[
    ],
    input_schema=[
        FieldDefinition("query", FieldType.STRING)
    ],
    output_schema=[
        FieldDefinition("docs", FieldType.ARRAY)
    ]
))

NodeRegistry.register(NodeDefinition(
    id="merge",
    name="Merge Results",
    type=NodeType.PROCESS,
    config_schema=[
        FieldDefinition("merge_strategy", FieldType.STRING, default="union"),
        FieldDefinition("deduplicate", FieldType.BOOLEAN, default=True)
    ],
    input_schema=[
        FieldDefinition("docs_a", FieldType.ARRAY),
        FieldDefinition("docs_b", FieldType.ARRAY)
    ],
    output_schema=[
        FieldDefinition("merged_docs", FieldType.ARRAY)
    ]
))

NodeRegistry.register(NodeDefinition(
    id="rerank",
    name="Rerank",
    type=NodeType.PROCESS,
    config_schema=[
        FieldDefinition("model", FieldType.STRING, default="gpt-4o"),
    ],
    input_schema=[
        FieldDefinition("docs_a", FieldType.ARRAY),
        FieldDefinition("docs_b", FieldType.ARRAY)
    ],
    output_schema=[
        FieldDefinition("ranked_docs", FieldType.ARRAY)
    ]
))

NodeRegistry.register(NodeDefinition(
    id="llm",
    name="LLM",
    type=NodeType.PROCESS,
    config_schema=[
        FieldDefinition("model", FieldType.STRING, default="gpt-4o"),
        FieldDefinition("temperature", FieldType.FLOAT, default=0.7),
        FieldDefinition("max_tokens", FieldType.INTEGER, default=1000)
    ],
    input_schema=[
        FieldDefinition("query", FieldType.STRING),
        FieldDefinition("context", FieldType.ARRAY)
    ],
    output_schema=[
        FieldDefinition("answer", FieldType.STRING)
    ]
))

@dataclass
class InputBinding:
    """Binding of an input field to its source"""
    name: str  # Name of the input field
    source_type: InputSourceType  
    value: Any = None  # for static
    ref_node: Optional[str] = None  # for dynamic
    ref_field: Optional[str] = None  # for dynamic
    global_var: Optional[str] = None  # for global

@dataclass
class NodeInstance:
    """Instance of a node in the flow"""
    id: str
    type: str  # NodeDefinition.id
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: List[InputBinding] = field(default_factory=list)
    depends_on: Set[str] = field(default_factory=set)

@dataclass
class Edge:
    """Connection between nodes in the flow"""
    source: str
    target: str
    source_field: Optional[str] = None
    target_field: Optional[str] = None

@dataclass
class GlobalVariable:
    """Global variable that can be accessed by any node"""
    name: str
    description: str
    type: FieldType
    value: Any = None

@dataclass
class FlowInstance:
    """Instance of a flow with nodes and edges"""
    id: str
    name: str
    nodes: Dict[str, NodeInstance]
    edges: List[Edge]
    global_variables: Dict[str, GlobalVariable] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def validate(self) -> None:
        """Validate the flow configuration"""
        # 1. Validate node dependencies (check for cycles)
        sorted_nodes = self._topological_sort()
        
        # 2. Validate each node's input dependencies
        for node_id in sorted_nodes:
            node = self.nodes[node_id]
            current_index = sorted_nodes.index(node_id)
            preceding_nodes = set(sorted_nodes[:current_index])
            
            for input_binding in node.inputs:
                if input_binding.source_type == InputSourceType.STATIC:
                    continue  # Static config doesn't need validation
                elif input_binding.source_type == InputSourceType.GLOBAL:
                    # Validate global variable exists
                    if input_binding.global_var not in self.global_variables:
                        raise ValidationError(
                            f"Node {node.id} references non-existent global variable: {input_binding.global_var}")
                elif input_binding.source_type == InputSourceType.DYNAMIC:
                    # Validate referenced node exists
                    if input_binding.ref_node not in self.nodes:
                        raise ValidationError(
                            f"Node {node.id} references non-existent node: {input_binding.ref_node}")
                    # Validate referenced node is a preceding node
                    if input_binding.ref_node not in preceding_nodes:
                        raise ValidationError(
                            f"Node {node.id} references non-preceding node: {input_binding.ref_node}")
                    # Validate referenced field exists
                    ref_node = self.nodes[input_binding.ref_node]
                    ref_node_def = NodeRegistry.get(ref_node.type)
                    if not any(field.name == input_binding.ref_field for field in ref_node_def.output_schema):
                        raise ValidationError(
                            f"Node {node.id} references non-existent field {input_binding.ref_field} "
                            f"in node {input_binding.ref_node}")

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
    """Context for flow execution, storing variables and global state"""
    variables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    global_variables: Dict[str, Any] = field(default_factory=dict)

    def get_input(self, node_id: str, field: str) -> Any:
        """Get input value for a node field"""
        return self.variables.get(node_id, {}).get(field)

    def set_output(self, node_id: str, outputs: Dict[str, Any]) -> None:
        """Set output values for a node"""
        self.variables[node_id] = outputs

    def get_global(self, name: str) -> Any:
        """Get global variable value"""
        return self.global_variables.get(name)

    def set_global(self, name: str, value: Any) -> None:
        """Set global variable value"""
        self.global_variables[name] = value