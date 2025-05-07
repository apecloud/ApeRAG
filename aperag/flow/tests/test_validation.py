import pytest
from aperag.flow.base.models import FlowInstance, InputSourceType, NodeInstance, Edge, InputBinding, GlobalVariable, FieldType
from aperag.flow.base.exceptions import ValidationError, CycleError

def test_valid_flow():
    """Test a valid flow configuration"""
    # Create global variables
    global_vars = {
        "query": GlobalVariable(
            name="query",
            description="User query",
            type=FieldType.STRING
        )
    }

    # Create nodes
    nodes = {
        "vector_search": NodeInstance(
            id="vector_search",
            type="vector_search",
            vars=[
                InputBinding(
                    name="query",
                    source_type=InputSourceType.GLOBAL,
                    global_var="query"
                ),
                InputBinding(
                    name="top_k",
                    source_type=InputSourceType.STATIC,
                    value=5
                )
            ]
        ),
        "keyword_search": NodeInstance(
            id="keyword_search",
            type="keyword_search",
            vars=[
                InputBinding(
                    name="query",
                    source_type=InputSourceType.GLOBAL,
                    global_var="query"
                )
            ]
        ),
        "rerank": NodeInstance(
            id="rerank",
            type="rerank",
            vars=[
                InputBinding(
                    name="docs",
                    source_type=InputSourceType.DYNAMIC,
                    ref_node="vector_search",
                    ref_field="docs"
                ),
                InputBinding(
                    name="docs",
                    source_type=InputSourceType.DYNAMIC,
                    ref_node="keyword_search",
                    ref_field="docs"
                )
            ]
        )
    }

    # Create edges
    edges = [
        Edge(source="vector_search", target="rerank"),
        Edge(source="keyword_search", target="rerank")
    ]

    # Create flow instance
    flow = FlowInstance(
        id="test_flow",
        name="Test Flow",
        nodes=nodes,
        edges=edges,
        global_variables=global_vars
    )

    # Validate flow
    flow.validate()  # Should not raise any exception

def test_cyclic_dependency():
    """Test cyclic dependency detection"""
    nodes = {
        "node1": NodeInstance(id="node1", type="type1"),
        "node2": NodeInstance(id="node2", type="type2"),
        "node3": NodeInstance(id="node3", type="type3")
    }

    edges = [
        Edge(source="node1", target="node2"),
        Edge(source="node2", target="node3"),
        Edge(source="node3", target="node1")  # Create a cycle
    ]

    flow = FlowInstance(
        id="cyclic_flow",
        name="Cyclic Flow",
        nodes=nodes,
        edges=edges
    )

    with pytest.raises(CycleError):
        flow.validate()

def test_invalid_global_variable():
    """Test invalid global variable reference"""
    nodes = {
        "node1": NodeInstance(
            id="node1",
            type="type1",
            vars=[
                InputBinding(
                    source_type=InputSourceType.GLOBAL,
                    global_var="non_existent_var"
                )
            ]
        )
    }

    flow = FlowInstance(
        id="invalid_global_flow",
        name="Invalid Global Flow",
        nodes=nodes,
        edges=[]
    )

    with pytest.raises(ValidationError) as exc_info:
        flow.validate()
    assert "non-existent global variable" in str(exc_info.value)

def test_invalid_node_reference():
    """Test invalid node reference"""
    nodes = {
        "node1": NodeInstance(
            id="node1",
            type="type1",
            vars=[
                InputBinding(
                    source_type=InputSourceType.DYNAMIC,
                    ref_node="non_existent_node",
                    ref_field="output"
                )
            ]
        )
    }

    flow = FlowInstance(
        id="invalid_node_flow",
        name="Invalid Node Flow",
        nodes=nodes,
        edges=[]
    )

    with pytest.raises(ValidationError) as exc_info:
        flow.validate()
    assert "non-existent node" in str(exc_info.value)

def test_invalid_field_reference():
    """Test invalid field reference"""
    nodes = {
        "node1": NodeInstance(
            id="node1",
            type="type1",
            vars=[
                InputBinding(
                    source_type=InputSourceType.DYNAMIC,
                    ref_node="node2",
                    ref_field="non_existent_field"
                )
            ]
        ),
        "node2": NodeInstance(
            id="node2",
            type="type2"
        )
    }

    edges = [
        Edge(source="node2", target="node1")
    ]

    flow = FlowInstance(
        id="invalid_field_flow",
        name="Invalid Field Flow",
        nodes=nodes,
        edges=edges
    )

    with pytest.raises(ValidationError) as exc_info:
        flow.validate()
    assert "non-existent field" in str(exc_info.value)

def test_non_preceding_node_reference():
    """Test reference to non-preceding node"""
    nodes = {
        "node1": NodeInstance(
            id="node1",
            type="type1",
            vars=[
                InputBinding(
                    source_type=InputSourceType.DYNAMIC,
                    ref_node="node2",
                    ref_field="output"
                )
            ]
        ),
        "node2": NodeInstance(
            id="node2",
            type="type2"
        )
    }

    edges = [
        Edge(source="node1", target="node2")  # node1 depends on node2, but node2 comes after node1
    ]

    flow = FlowInstance(
        id="invalid_order_flow",
        name="Invalid Order Flow",
        nodes=nodes,
        edges=edges
    )

    with pytest.raises(ValidationError) as exc_info:
        flow.validate()
    assert "non-preceding node" in str(exc_info.value) 