from typing import Any, Dict

import jsonref
import yaml

from aperag.flow.base.models import Edge, FlowInstance, NodeInstance

from .base.exceptions import ValidationError


class FlowParser:
    """Parser for flow configuration in YAML format"""

    @staticmethod
    def parse(data: str | dict[str, Any]) -> FlowInstance:
        """Parse YAML content into a FlowInstance"""
        try:
            if isinstance(data, str):
                data = yaml.safe_load(data)
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML format: {str(e)}")

        # Dereference $ref in flow configuration
        data = jsonref.replace_refs(data)

        # Parse nodes
        nodes = {}
        for node_data in data.get("nodes", []):
            node = FlowParser._parse_node(node_data)
            nodes[node.id] = node

        # Parse edges
        edges = []
        for edge_data in data.get("edges", []):
            edge = FlowParser._parse_edge(edge_data)
            edges.append(edge)

        # Create FlowInstance
        flow = FlowInstance(
            name=data.get("name", "Unnamed Flow"),
            title=data.get("title", "Unnamed Flow"),
            nodes=nodes,
            edges=edges,
        )

        # Validate flow configuration
        flow.validate()

        return flow

    @staticmethod
    def _parse_node(node_data: Dict[str, Any]) -> NodeInstance:
        """Parse a node definition"""
        data = node_data.get("data", {})
        input_schema = data.get("input", {}).get("schema", {})
        output_schema = data.get("output", {}).get("schema", {})
        node = NodeInstance(
            id=node_data["id"],
            type=node_data["type"],
            input_schema=input_schema,
            input_values=data.get("input", {}).get("values", {}),
            output_schema=output_schema,
        )
        if "title" in node_data:
            node.title = node_data["title"]
        return node

    @staticmethod
    def _parse_edge(edge_data: Dict[str, Any]) -> Edge:
        """Parse an edge definition"""
        return Edge(source=edge_data["source"], target=edge_data["target"])

    @staticmethod
    def load_from_file(file_path: str) -> FlowInstance:
        """Load flow configuration from a file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()
            return FlowParser.parse(yaml_content)
        except FileNotFoundError:
            raise ValidationError(f"Flow configuration file not found: {file_path}")
        except Exception as e:
            raise ValidationError(f"Error loading flow configuration: {str(e)}")
