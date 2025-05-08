import yaml
from typing import Dict, Any
from aperag.flow.base.models import (
    FlowInstance, InputSourceType, NodeInstance, Edge, InputBinding,
    GlobalVariable, FieldType
)
from .base.exceptions import ValidationError

class FlowParser:
    """Parser for flow configuration in YAML format"""

    @staticmethod
    def parse_yaml(yaml_content: str) -> FlowInstance:
        """Parse YAML content into a FlowInstance"""
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML format: {str(e)}")

        # Parse global variables
        global_variables = {}
        for var_data in data.get("global_variables", []):
            var = FlowParser._parse_global_variable(var_data)
            global_variables[var.name] = var

        # Parse nodes
        nodes = {}
        for node_data in data.get("nodes", []):
            node = FlowParser._parse_node(node_data)
            nodes[node.id] = node

        # Parse edges and update node dependencies
        edges = []
        for edge_data in data.get("edges", []):
            edge = FlowParser._parse_edge(edge_data)
            edges.append(edge)
            # Add edge-based dependency
            nodes[edge.target].depends_on.add(edge.source)

        # Create FlowInstance
        flow = FlowInstance(
            id=data.get("name", "unnamed_flow"),
            name=data.get("name", "Unnamed Flow"),
            nodes=nodes,
            edges=edges,
            global_variables=global_variables
        )

        # Validate flow configuration
        flow.validate()

        return flow

    @staticmethod
    def _parse_global_variable(var_data: Dict[str, Any]) -> GlobalVariable:
        """Parse a global variable definition"""
        return GlobalVariable(
            name=var_data["name"],
            description=var_data.get("description", ""),
            type=FieldType(var_data["type"]),
            value=var_data.get("value")
        )

    @staticmethod
    def _parse_node(node_data: Dict[str, Any]) -> NodeInstance:
        """Parse a node definition"""
        vars = []
        depends_on = set()
        input_list = node_data["data"]["vars"]
        for input_data in input_list:
            input_binding = InputBinding(
                name=input_data["name"],
                source_type=InputSourceType(input_data.get("source_type", "static")),
                value=input_data.get("value"),
                ref_node=input_data.get("ref_node"),
                ref_field=input_data.get("ref_field"),
                global_var=input_data.get("global_var")
            )
            vars.append(input_binding)
            if input_binding.source_type == InputSourceType.DYNAMIC and input_binding.ref_node:
                depends_on.add(input_binding.ref_node)
        node = NodeInstance(
            id=node_data["id"],
            type=node_data["type"],
            vars=vars,
            depends_on=depends_on
        )
        if "name" in node_data:
            node.name = node_data["name"]
        return node

    @staticmethod
    def _parse_edge(edge_data: Dict[str, Any]) -> Edge:
        """Parse an edge definition"""
        return Edge(
            source=edge_data["source"],
            target=edge_data["target"]
        )

    @staticmethod
    def load_from_file(file_path: str) -> FlowInstance:
        """Load flow configuration from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            return FlowParser.parse_yaml(yaml_content)
        except FileNotFoundError:
            raise ValidationError(f"Flow configuration file not found: {file_path}")
        except Exception as e:
            raise ValidationError(f"Error loading flow configuration: {str(e)}") 