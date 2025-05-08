from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from enum import Enum
from collections import deque
from aperag.flow.base.exceptions import ValidationError, CycleError

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
    type: str  # e.g. "vector_search"
    vars_schema: List[FieldDefinition]
    output_schema: List[FieldDefinition]
    description: Optional[str] = None

class NodeRegistry:
    """Registry for node type definitions"""
    _nodes: Dict[str, NodeDefinition] = {}

    @classmethod
    def register(cls, node_def: NodeDefinition) -> None:
        """Register a new node type"""
        cls._nodes[node_def.type] = node_def

    @classmethod
    def get(cls, node_type: str) -> NodeDefinition:
        """Get a node type definition by type (string)"""
        if node_type not in cls._nodes:
            raise KeyError(f"Node definition not found: {node_type}")
        return cls._nodes[node_type]

    @classmethod
    def list_nodes(cls) -> List[NodeDefinition]:
        """List all registered node types"""
        return list(cls._nodes.values())

NODE_DEFINITION_REGISTRY = {}

def register_node_definition(node_def: NodeDefinition):
    NodeRegistry.register(node_def)
    NODE_DEFINITION_REGISTRY[node_def.type] = node_def
    return node_def

register_node_definition(NodeDefinition(
    type="vector_search",
    vars_schema=[
        FieldDefinition("top_k", FieldType.INTEGER, default=5),
        FieldDefinition("similarity_threshold", FieldType.FLOAT, default=0.7),
        FieldDefinition("query", FieldType.STRING)
    ],
    output_schema=[
        FieldDefinition("vector_search_docs", FieldType.ARRAY)
    ]
))

register_node_definition(NodeDefinition(
    type="keyword_search",
    vars_schema=[
        FieldDefinition("query", FieldType.STRING)
    ],
    output_schema=[
        FieldDefinition("keyword_search_docs", FieldType.ARRAY)
    ]
))

register_node_definition(NodeDefinition(
    type="merge",
    vars_schema=[
        FieldDefinition("merge_strategy", FieldType.STRING, default="union"),
        FieldDefinition("deduplicate", FieldType.BOOLEAN, default=True),
        FieldDefinition("vector_search_docs", FieldType.ARRAY),
        FieldDefinition("keyword_search_docs", FieldType.ARRAY)
    ],
    output_schema=[
        FieldDefinition("docs", FieldType.ARRAY)
    ]
))

register_node_definition(NodeDefinition(
    type="rerank",
    vars_schema=[
        FieldDefinition("model", FieldType.STRING, default="bge-reranker"),
        FieldDefinition("docs", FieldType.ARRAY)
    ],
    output_schema=[
        FieldDefinition("docs", FieldType.ARRAY)
    ]
))

register_node_definition(NodeDefinition(
    type="llm",
    vars_schema=[
        FieldDefinition("model", FieldType.STRING, default="gpt-4o"),
        FieldDefinition("temperature", FieldType.FLOAT, default=0.7),
        FieldDefinition("max_tokens", FieldType.INTEGER, default=1000),
        FieldDefinition("query", FieldType.STRING),
        FieldDefinition("docs", FieldType.ARRAY)
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
    type: str  # NodeDefinition.type
    vars: List[InputBinding] = field(default_factory=list)
    depends_on: Set[str] = field(default_factory=set)
    name: Optional[str] = None

@dataclass
class Edge:
    """Connection between nodes in the flow"""
    source: str
    target: str

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
        sorted_nodes = self._topological_sort()
        for node_id in sorted_nodes:
            node = self.nodes[node_id]
            node_def = NodeRegistry.get(node.type)
            self._validate_node_vars_schema(node, node_def, sorted_nodes)

    def _validate_node_vars_schema(self, node: NodeInstance, node_def: NodeDefinition, sorted_nodes: List[str]):
        schema_fields = {f.name: f for f in node_def.vars_schema}
        for var in node.vars:
            if var.name not in schema_fields:
                raise ValidationError(f"Unknown var '{var.name}' for node type {node.type}")
            expected_type = schema_fields[var.name].type
            if var.source_type == InputSourceType.STATIC:
                self._validate_node_var_static(var, expected_type, node, schema_fields)
            elif var.source_type == InputSourceType.GLOBAL:
                self._validate_node_var_global(var, node)
            elif var.source_type == InputSourceType.DYNAMIC:
                self._validate_node_var_dynamic(var, node, sorted_nodes)
        for field in node_def.vars_schema:
            if field.required and not any(var.name == field.name for var in node.vars):
                raise ValidationError(f"Missing required var '{field.name}' for node {node.type}")

    def _validate_node_var_static(self, var: InputBinding, expected_type: FieldType, node: NodeInstance, schema_fields: Dict[str, FieldDefinition]):
        if var.value is None and schema_fields[var.name].required:
            raise ValidationError(f"Static var '{var.name}' is required for node {node.id}")
        if var.value is not None:
            if expected_type == FieldType.STRING and not isinstance(var.value, str):
                raise ValidationError(f"Var '{var.name}' should be string for node {node.id}")
            if expected_type == FieldType.INTEGER and not isinstance(var.value, int):
                raise ValidationError(f"Var '{var.name}' should be integer for node {node.id}")
            if expected_type == FieldType.FLOAT and not isinstance(var.value, float):
                raise ValidationError(f"Var '{var.name}' should be float for node {node.id}")
            if expected_type == FieldType.BOOLEAN and not isinstance(var.value, bool):
                raise ValidationError(f"Var '{var.name}' should be boolean for node {node.id}")
            if expected_type == FieldType.ARRAY and not isinstance(var.value, list):
                raise ValidationError(f"Var '{var.name}' should be array for node {node.id}")
            if expected_type == FieldType.OBJECT and not isinstance(var.value, dict):
                raise ValidationError(f"Var '{var.name}' should be object for node {node.id}")

    def _validate_node_var_global(self, var: InputBinding, node: NodeInstance):
        if not var.global_var:
            raise ValidationError(f"Global var '{var.name}' must specify global_var for node {node.id}")
        if var.global_var not in self.global_variables:
            raise ValidationError(f"Node {node.id} references non-existent global variable: {var.global_var}")

    def _validate_node_var_dynamic(self, var: InputBinding, node: NodeInstance, sorted_nodes: List[str]):
        if not var.ref_node or not var.ref_field:
            raise ValidationError(f"Dynamic var '{var.name}' must specify ref_node and ref_field for node {node.id}")
        if var.ref_node not in self.nodes:
            raise ValidationError(f"Node {node.id} references non-existent node: {var.ref_node}")
        current_index = sorted_nodes.index(node.id)
        preceding_nodes = set(sorted_nodes[:current_index])
        if var.ref_node not in preceding_nodes:
            raise ValidationError(f"Node {node.id} references non-preceding node: {var.ref_node}")
        ref_node = self.nodes[var.ref_node]
        ref_node_def = NodeRegistry.get(ref_node.type)
        if not any(field.name == var.ref_field for field in ref_node_def.output_schema):
            raise ValidationError(f"Node {node.id} references non-existent field {var.ref_field} in node {var.ref_node}")

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
