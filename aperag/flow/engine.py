from collections import deque
from typing import List, Set, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, wait
from aperag.flow.models import FlowInstance, NodeInstance, ExecutionContext, NodeRegistry
from aperag.flow.exceptions import CycleError, NodeNotFoundError, ValidationError
from aperag.flow.models import InputSourceType
import logging
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(execution_id)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class FlowEngine:
    """Engine for executing flow instances"""
    
    def __init__(self):
        self.context = ExecutionContext()
        self.execution_id = None

    def execute_flow(self, flow: FlowInstance, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
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
                if var_name in flow.global_variables:
                    self.context.set_global(var_name, var_value)

        # Build dependency graph and perform topological sort
        sorted_nodes = self._topological_sort(flow)
        
        # Execute nodes
        for node_group in self._find_parallel_groups(flow, sorted_nodes):
            self._execute_node_group(flow, node_group)
            
        logger.info(f"Completed flow execution {self.execution_id}", extra={'execution_id': self.execution_id})
        return self.context.variables

    def execute_node(self, node: NodeInstance, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            # Use node's configured input bindings
            for input_binding in node.inputs:
                if input_binding.source_type == InputSourceType.STATIC:
                    temp_context.variables[node.id][input_binding.name] = input_binding.value
                elif input_binding.source_type == InputSourceType.GLOBAL:
                    temp_context.variables[node.id][input_binding.name] = self.context.get_global(input_binding.global_var)
                elif input_binding.source_type == InputSourceType.DYNAMIC:
                    ref_value = self.context.get_input(input_binding.ref_node, input_binding.ref_field)
                    temp_context.variables[node.id][input_binding.name] = ref_value
        
        # Execute the node
        self._execute_node(node, temp_context)
        
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

    def _execute_node_group(self, flow: FlowInstance, node_group: Set[str]):
        """Execute a group of nodes (possibly in parallel)"""
        logger.info(f"Executing node group: {node_group}", extra={'execution_id': self.execution_id})
        if len(node_group) == 1:
            # Serial execution
            node_id = next(iter(node_group))
            node = flow.nodes[node_id]
            self._execute_node(node)
        else:
            # Parallel execution
            with ThreadPoolExecutor() as executor:
                futures = []
                for node_id in node_group:
                    node = flow.nodes[node_id]
                    futures.append(executor.submit(self._execute_node, node))
                wait(futures)

    def _bind_node_inputs(self, node: NodeInstance, ctx: ExecutionContext) -> dict:
        """Bind input variables for a node according to its input schema and bindings
        
        Args:
            node: The node instance
            ctx: The execution context to use
        Returns:
            Dictionary of input values for the node
        Raises:
            ValidationError: If required input is missing
        """
        node_def = NodeRegistry.get(node.type)
        inputs = {}
        for field in node_def.input_schema:
            value = None
            for input_binding in node.inputs:
                if input_binding.name == field.name:
                    if input_binding.source_type == InputSourceType.STATIC:
                        value = input_binding.value
                    elif input_binding.source_type == InputSourceType.GLOBAL:
                        value = ctx.get_global(input_binding.global_var)
                    elif input_binding.source_type == InputSourceType.DYNAMIC:
                        value = ctx.get_input(input_binding.ref_node, input_binding.ref_field)
                    break
            if value is None:
                value = ctx.get_input(node.id, field.name)
            if field.required and value is None:
                raise ValidationError(f"Required input '{field.name}' not provided for node {node.id}")
            inputs[field.name] = value
        return inputs

    def _execute_node(self, node: NodeInstance, context: Optional[ExecutionContext] = None) -> None:
        """Execute a single node using the provided context
        
        Args:
            node: The node instance to execute
            context: Optional execution context to use. If not provided, uses the engine's context
        """
        # Use provided context or default to engine's context
        ctx = context or self.context
        # Bind inputs using the helper method
        inputs = self._bind_node_inputs(node, ctx)
        # Get node definition
        node_def = NodeRegistry.get(node.type)
        # Execute node logic
        outputs = self._execute_node_logic(node, inputs)
        # Validate outputs
        for field in node_def.output_schema:
            if field.required and field.name not in outputs:
                raise ValidationError(f"Required output '{field.name}' not produced by node {node.id}")
        # Store outputs in context
        ctx.set_output(node.id, outputs)

    def _execute_node_logic(self, node: NodeInstance, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the actual logic of a node
        
        Args:
            node: The node instance to execute
            inputs: Dictionary of input values
            
        Returns:
            Dictionary of output values
        """
        logger.info(f"Executing node logic: {node.type}, inputs: {inputs}", extra={'execution_id': self.execution_id})
        # Get node definition
        node_def = NodeRegistry.get(node.type)
        
        # Execute based on node type
        if node.type == "vector_search":
            # For now, just echo the inputs
            return {"docs": [{"content": "doc1", "score": 0.8}, {"content": "doc2", "score": 0.7}]}
        elif node.type == "keyword_search":
            # For now, just echo the inputs
            return {"docs": [{"content": "doc1", "score": 0.7}, {"content": "doc3", "score": 0.6}]}
        elif node.type == "merge":
            # Merge docs from both sources
            docs_a = inputs["docs_a"]
            docs_b = inputs["docs_b"]
            merge_strategy = node.config.get("merge_strategy", "union")
            deduplicate = node.config.get("deduplicate", True)
            
            if merge_strategy == "union":
                # Simple union of docs
                all_docs = docs_a + docs_b
                if deduplicate:
                    # Remove duplicates based on content
                    seen = set()
                    unique_docs = []
                    for doc in all_docs:
                        if doc["content"] not in seen:
                            seen.add(doc["content"])
                            unique_docs.append(doc)
                    return {"merged_docs": unique_docs}
                return {"merged_docs": all_docs}
            else:
                raise ValidationError(f"Unknown merge strategy: {merge_strategy}")
        elif node.type == "rerank":
            # Combine and sort docs from both sources
            docs_a = inputs["docs_a"]
            docs_b = inputs["docs_b"]
            all_docs = docs_a + docs_b
            # Sort by score
            ranked_docs = sorted(all_docs, key=lambda x: x["score"], reverse=True)
            return {"ranked_docs": ranked_docs}
        elif node.type == "llm":
            # For now, just echo the inputs
            return {"answer": f"Based on the context: {inputs['context']}"}
        else:
            raise ValidationError(f"Unknown node type: {node.type}")