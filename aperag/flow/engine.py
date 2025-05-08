from collections import deque
from typing import List, Set, Dict, Any, Optional
from aperag.flow.base.models import FlowInstance, NodeInstance, ExecutionContext, NodeRegistry
from aperag.flow.base.exceptions import CycleError, ValidationError
from aperag.flow.base.models import InputSourceType, NODE_RUNNER_REGISTRY
import logging
import uuid
from .runners import *
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(execution_id)s] - %(message)s'
)
logger = logging.getLogger(__name__)

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
        logger.info(f"Starting flow execution {self.execution_id} for flow {flow.id}", extra={'execution_id': self.execution_id})

        # Initialize global variables
        if initial_data:
            for var_name, var_value in initial_data.items():
                self.context.set_global(var_name, var_value)

        # Build dependency graph and perform topological sort
        sorted_nodes = self._topological_sort(flow)
        
        # Execute nodes
        for node_group in self._find_parallel_groups(flow, sorted_nodes):
            await self._execute_node_group(flow, node_group)
            
        logger.info(f"Completed flow execution {self.execution_id}", extra={'execution_id': self.execution_id})
        return self.context.variables

    async def execute_node(self, node: NodeInstance, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a single node with manual input binding
        
        Args:
            node: The node instance to execute
            inputs: Optional dictionary of input values to bind manually
                   If not provided, will use the node's configured input bindings
        
        Returns:
            Dictionary of output values from the node execution
        """
        # Create a temporary context for this node
        temp_context = ExecutionContext()
        
        # If manual inputs provided, bind them
        if inputs:
            temp_context.variables[node.id] = inputs
        else:
            # Use node's input bindings (vars)
            for var in node.vars:
                if var.source_type == InputSourceType.STATIC:
                    if node.id not in temp_context.variables:
                        temp_context.variables[node.id] = {}
                    temp_context.variables[node.id][var.name] = var.value
                elif var.source_type == InputSourceType.GLOBAL:
                    if node.id not in temp_context.variables:
                        temp_context.variables[node.id] = {}
                    temp_context.variables[node.id][var.name] = self.context.get_global(var.global_var)
                elif var.source_type == InputSourceType.DYNAMIC:
                    ref_value = self.context.get_input(var.ref_node, var.ref_field)
                    if node.id not in temp_context.variables:
                        temp_context.variables[node.id] = {}
                    temp_context.variables[node.id][var.name] = ref_value
        
        # Execute the node
        await self._execute_node(node, temp_context)
        
        # Return the node's outputs
        return temp_context.variables.get(node.id, {})

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
                node_id for node_id in sorted_nodes
                if in_degree[node_id] == 0 and node_id not in processed
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
        logger.info(f"Executing node group: {node_group}", extra={'execution_id': self.execution_id})
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

    def _bind_node_inputs(self, node: NodeInstance) -> dict:
        """Bind input variables for a node according to its input schema and bindings
        
        Args:
            node: The node instance
        Returns:
            Dictionary of input values for the node
        Raises:
            ValidationError: If required input is missing
        """
        node_def = NodeRegistry.get(node.type)
        inputs = self.context.global_variables.copy()
        for field in node_def.vars_schema:
            value = None
            for var in node.vars:
                if var.name == field.name:
                    if var.source_type == InputSourceType.STATIC:
                        value = var.value
                    elif var.source_type == InputSourceType.GLOBAL:
                        value = self.context.get_global(var.global_var)
                    elif var.source_type == InputSourceType.DYNAMIC:
                        value = self.context.get_input(var.ref_node, var.ref_field)
                    break
            if field.required and value is None:
                raise ValidationError(f"Required input '{field.name}' not provided for node {node.id}")
            inputs[field.name] = value
        return inputs

    async def _execute_node(self, node: NodeInstance) -> None:
        """Execute a single node using the provided context
        
        Args:
            node: The node instance to execute
        """
        # Bind inputs using the helper method
        inputs = self._bind_node_inputs(node)

        # Execute node logic
        outputs = await self._execute_node_logic(node, inputs)

        # Validate outputs
        node_def = NodeRegistry.get(node.type)
        for field in node_def.output_schema:
            if field.required and field.name not in outputs:
                raise ValidationError(f"Required output '{field.name}' not produced by node {node.id}")
        # Store outputs in context
        self.context.set_output(node.id, outputs)

    async def _execute_node_logic(self, node: NodeInstance, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch node execution to the registered NodeRunner class"""
        runner = NODE_RUNNER_REGISTRY.get(node.type)
        if not runner:
            raise ValidationError(f"Unknown node type: {node.type}")
        return await runner.run(node, inputs)

    def find_output_nodes(self, flow: FlowInstance) -> List[str]:
        """Find all output nodes (nodes with in-degree > 0 and out-degree 0) in the flow"""
        out_degree = {node_id: 0 for node_id in flow.nodes}
        for edge in flow.edges:
            out_degree[edge.source] += 1
        output_nodes = [node_id for node_id in flow.nodes if out_degree[node_id] == 0]
        return output_nodes
