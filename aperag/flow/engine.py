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

import asyncio
import logging
import uuid
from collections import deque
from typing import Any, AsyncGenerator, Dict, List, Set

from jinja2 import Environment, StrictUndefined

import aperag.flow.runners  # noqa: F401
from aperag.flow.base.exceptions import CycleError, ValidationError
from aperag.flow.base.models import NODE_RUNNER_REGISTRY, ExecutionContext, FlowInstance, NodeInstance, SystemInput
from aperag.utils.utils import utc_now

# Configure logging
logger = logging.getLogger(__name__)


class FlowEvent:
    """Event emitted during flow execution"""

    def __init__(self, event_type: str, node_id: str, node_type: str, execution_id: str, data: Dict[str, Any] = None):
        self.event_type = event_type
        self.node_id = node_id
        self.node_type = node_type
        self.execution_id = execution_id
        self.timestamp = utc_now().isoformat()
        self.data = data or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "node_id": self.node_id,
            "node_type": self.node_type or "",
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class FlowEventType:
    """Event types for flow execution"""

    NODE_START = "node_start"
    NODE_END = "node_end"
    NODE_ERROR = "node_error"
    FLOW_START = "flow_start"
    FLOW_END = "flow_end"
    FLOW_ERROR = "flow_error"


# FlowEngine is responsible for executing a FlowInstance (a flow definition with nodes and edges).
# Each FlowEngine instance maintains its own execution context (self.context) and execution_id.
# Usage notes:
# - Do NOT reuse the same FlowEngine instance for multiple or concurrent flow executions.
#   Each execution should use a new FlowEngine instance to avoid context and execution_id conflicts.
# - The context stores all global variables and node outputs for the current execution.
# - The execution_id is a unique identifier for the current execution, mainly for logging and tracing.
# - Reusing the same FlowEngine instance for multiple executions will result in data corruption or unexpected behavior.
class FlowEngine:
    """Engine for executing flow instances"""

    def __init__(self):
        self.context = ExecutionContext()
        self.execution_id = None
        self._event_queue = asyncio.Queue()
        self.jinja_env = Environment(undefined=StrictUndefined)

    async def emit_event(self, event: FlowEvent):
        """Emit an event to all consumers"""
        await self._event_queue.put(event)
        # Also log the event
        logger.info(
            f"Flow event: {event.event_type} for {event.node_type} node {event.node_id}",
            extra={"execution_id": self.execution_id},
        )

    async def get_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Get events as an async generator"""
        try:
            while True:
                event = await self._event_queue.get()
                yield event.to_dict()
                self._event_queue.task_done()
        except asyncio.CancelledError:
            pass

    async def execute_flow(self, flow: FlowInstance, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a flow instance with optional initial data

        Args:
            flow: The flow instance to execute
            initial_data: Optional dictionary of initial global variable values

        Returns:
            Dictionary of final output values from the flow execution
        """
        # Generate execution ID
        self.execution_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        logger.info(
            f"Starting flow execution {self.execution_id} for flow {flow.name}",
            extra={"execution_id": self.execution_id},
        )

        try:
            # Emit flow start event
            await self.emit_event(
                FlowEvent(
                    event_type=FlowEventType.FLOW_START,
                    execution_id=self.execution_id,
                    node_id=None,
                    node_type=None,
                    data={"flow_name": flow.name},
                )
            )

            # Initialize global variables
            if initial_data:
                for var_name, var_value in initial_data.items():
                    self.context.set_global(var_name, var_value)

            # Build dependency graph and perform topological sort
            sorted_nodes = self._topological_sort(flow)

            # Execute nodes
            for node_group in self._find_parallel_groups(flow, sorted_nodes):
                await self._execute_node_group(flow, node_group)

            # Emit flow end event
            await self.emit_event(
                FlowEvent(
                    event_type=FlowEventType.FLOW_END,
                    execution_id=self.execution_id,
                    node_id=None,
                    node_type=None,
                    data={"flow_name": flow.name},
                )
            )

            logger.info(f"Completed flow execution {self.execution_id}", extra={"execution_id": self.execution_id})
            return self.context.outputs, self.context.system_outputs

        except Exception as e:
            # Emit flow error event
            await self.emit_event(
                FlowEvent(
                    event_type=FlowEventType.FLOW_ERROR,
                    execution_id=self.execution_id,
                    node_id=None,
                    node_type=None,
                    data={"flow_name": flow.name, "error": str(e)},
                )
            )
            raise e

    def _topological_sort(self, flow: FlowInstance) -> List[str]:
        """Perform topological sort to detect cycles

        Args:
            flow: The flow instance

        Returns:
            Topologically sorted list of node IDs

        Raises:
            CycleError: If the flow contains cycles
        """
        # Build dependency graph from edges
        in_degree = {node_id: 0 for node_id in flow.nodes}
        for edge in flow.edges:
            in_degree[edge.target] += 1

        # Start with nodes that have no dependencies
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        if len(queue) == 0:
            raise CycleError("Flow contains cycles")

        sorted_nodes = []

        while queue:
            node_id = queue.popleft()
            sorted_nodes.append(node_id)

            # Update in-degree of successor nodes
            for edge in flow.edges:
                if edge.source == node_id:
                    in_degree[edge.target] -= 1
                    if in_degree[edge.target] == 0:
                        queue.append(edge.target)

        if len(sorted_nodes) != len(flow.nodes):
            raise CycleError("Flow contains cycles")

        return sorted_nodes

    def _find_parallel_groups(self, flow: FlowInstance, sorted_nodes: List[str]) -> List[Set[str]]:
        """Find groups of nodes that can be executed in parallel (level by level)

        Args:
            flow: The flow instance
            sorted_nodes: Topologically sorted list of node IDs

        Returns:
            List of node groups, where each group can be executed in parallel
        """
        # Build in-degree map
        in_degree = {node_id: 0 for node_id in flow.nodes}
        for edge in flow.edges:
            in_degree[edge.target] += 1

        # Track processed nodes
        processed = set()
        groups = []

        while len(processed) < len(sorted_nodes):
            # Find all nodes with in-degree 0 and not processed
            current_group = set(
                node_id for node_id in sorted_nodes if in_degree[node_id] == 0 and node_id not in processed
            )
            if not current_group:
                break  # Should not happen if topological sort is correct
            groups.append(current_group)
            # Mark nodes as processed and update in-degree for successors
            for node_id in current_group:
                processed.add(node_id)
                for edge in flow.edges:
                    if edge.source == node_id:
                        in_degree[edge.target] -= 1
        return groups

    async def _execute_node_group(self, flow: FlowInstance, node_group: Set[str]):
        """Execute a group of nodes (possibly in parallel)"""
        logger.info(f"Executing node group: {node_group}", extra={"execution_id": self.execution_id})
        if len(node_group) == 1:
            node_id = next(iter(node_group))
            node = flow.nodes[node_id]
            await self._execute_node(node)
        else:
            tasks = []
            for node_id in node_group:
                node = flow.nodes[node_id]
                tasks.append(self._execute_node(node))
            await asyncio.gather(*tasks)

    def _resolve_variable(self, expr: str, nodes_ctx: dict):
        """
        Resolve variable path like 'nodes.start.output.query' from nodes_ctx.
        """
        parts = expr.strip().split(".")
        if not parts:
            return None
        if parts[0] == "nodes":
            if len(parts) < 4 or parts[2] != "output":
                raise ValidationError(f"Invalid variable reference: ${{{{ {expr} }}}}")
            node_id = parts[1]
            field_path = parts[3:]
            node_outputs = self.context.outputs.get(node_id, {})
            value = node_outputs
            for key in field_path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                elif isinstance(value, object) and hasattr(value, key):
                    value = getattr(value, key)
                else:
                    raise ValidationError(f"Cannot resolve variable: ${{{{ {expr} }}}}")
            return value
        else:
            raise ValidationError(f"Unknown variable scope: ${{{{ {expr} }}}}")

    def resolve_expression(self, value, node_id=None, nodes_ctx=None):
        """
        Recursively resolve input values.
        1. If value is a string and starts with ${{ ... }}, resolve as variable path.
        2. Otherwise, use jinja2 template rendering with nodes_ctx as context.
        3. Recursively handle dict/list.
        """
        if nodes_ctx is None:
            nodes_ctx = {nid: {"output": outputs} for nid, outputs in self.context.outputs.items()}
        if isinstance(value, dict):
            return {k: self.resolve_expression(v, node_id, nodes_ctx) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve_expression(v, node_id, nodes_ctx) for v in value]
        if not isinstance(value, str):
            return value

        value_strip = value.strip()
        # Only handle variable reference like {{ ... }}
        # This is a workaround for the fact that the rendered output of jinja2 is a string, but we want to get the original value of the variable
        if value_strip.startswith("{{") and value_strip.endswith("}}"):
            expr = value_strip[2:-2].strip()
            return self._resolve_variable(expr, nodes_ctx)

        # Otherwise, use jinja2 template rendering
        try:
            template = self.jinja_env.from_string(value)
            rendered = template.render(nodes=nodes_ctx)
        except Exception as e:
            raise ValidationError(f"Jinja2 render error in node '{node_id}': {e}")
        return rendered

    def convert_type_by_schema(self, value, field_schema):
        """Convert value to the type declared in field_schema (jsonschema property)."""
        if value is None:
            return None
        typ = field_schema.get("type")
        if typ == "string":
            return str(value)
        if typ == "integer":
            try:
                return int(value)
            except Exception:
                raise ValueError(f"Cannot convert '{value}' to integer")
        if typ == "number":
            try:
                return float(value)
            except Exception:
                raise ValueError(f"Cannot convert '{value}' to float")
        if typ == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                if value.lower() in ["true", "1", "yes"]:
                    return True
                if value.lower() in ["false", "0", "no"]:
                    return False
            if isinstance(value, int):
                return bool(value)
            raise ValueError(f"Cannot convert '{value}' to boolean")
        if typ == "array":
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                import json

                try:
                    arr = json.loads(value)
                    if isinstance(arr, list):
                        return arr
                except Exception as e:
                    raise ValidationError(f"Cannot convert '{value}' to array: {e}")
                # Try comma split
                return [v.strip() for v in value.split(",") if v.strip()]
            raise ValueError(f"Cannot convert '{value}' to array")
        if typ == "object":
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                import json

                try:
                    obj = json.loads(value)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass
            raise ValueError(f"Cannot convert '{value}' to object")
        return value

    def _bind_node_inputs(self, node: NodeInstance, runner_info: dict) -> tuple:
        """
        Bind input variables for a node using Pydantic model from runner_info.
        Returns (user_input, sys_input)
        """
        raw_inputs = getattr(node, "input_values", {})
        resolved_inputs = self.resolve_expression(raw_inputs, node.id)
        input_model = runner_info["input_model"]
        try:
            user_input = input_model.model_validate(resolved_inputs)
        except Exception as e:
            raise ValidationError(f"Input validation error for node {node.id}: {e}")
        sys_input = SystemInput(**self.context.global_variables)
        return user_input, sys_input

    async def _execute_node(self, node: NodeInstance) -> None:
        """
        Execute a single node using the provided context, using runner_info from registry.
        """
        runner_info = NODE_RUNNER_REGISTRY.get(node.type)
        if not runner_info:
            raise ValidationError(f"Unknown node type: {node.type}")
        runner = runner_info["runner"]
        try:
            user_input, sys_input = self._bind_node_inputs(node, runner_info)
            await self.emit_event(
                FlowEvent(
                    FlowEventType.NODE_START,
                    node.id,
                    node.type,
                    self.execution_id,
                    {"node_type": node.type, "inputs": user_input.model_dump()},
                )
            )
            outputs = await runner.run(user_input, sys_input)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                output_data, system_output = outputs
            else:
                output_data, system_output = outputs, None
            self.context.set_output(node.id, output_data)
            if system_output is not None:
                self.context.set_system_output(node.id, system_output)
            await self.emit_event(
                FlowEvent(
                    FlowEventType.NODE_END,
                    node.id,
                    node.type,
                    self.execution_id,
                    {"node_type": node.type, "outputs": output_data},
                )
            )
        except Exception as e:
            await self.emit_event(
                FlowEvent(
                    FlowEventType.NODE_ERROR,
                    node.id,
                    node.type,
                    self.execution_id,
                    {"node_type": node.type, "error": str(e)},
                )
            )
            raise e

    def update_node_input(self, flow: FlowInstance, node_id: str, value: Any):
        """Update the input values for a node"""
        flow.nodes[node_id].input_values.update(value)

    def find_start_nodes(self, flow: FlowInstance) -> str:
        """Find all start nodes (nodes with in-degree == 0) in the flow"""
        in_degree = {node_id: 0 for node_id in flow.nodes}
        for edge in flow.edges:
            in_degree[edge.target] += 1
        start_nodes = [node_id for node_id in flow.nodes if in_degree[node_id] == 0]
        if len(start_nodes) != 1:
            raise ValidationError("Flow must have exactly one start node")
        return start_nodes[0]

    def find_end_nodes(self, flow: FlowInstance) -> List[str]:
        """Find all output nodes (nodes with in-degree > 0 and out-degree 0) in the flow"""
        out_degree = {node_id: 0 for node_id in flow.nodes}
        for edge in flow.edges:
            out_degree[edge.source] += 1
        output_nodes = [node_id for node_id in flow.nodes if out_degree[node_id] == 0]
        return output_nodes
