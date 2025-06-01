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

import pytest

from aperag.flow.base.exceptions import CycleError, ValidationError
from aperag.flow.base.models import Edge, FlowInstance, NodeInstance


def test_valid_flow():
    """Test a valid flow configuration"""
    # Create nodes
    nodes = {
        "start": NodeInstance(
            id="start", type="start", input_values={"query": "hello"}, output_values={"query": "hello"}
        ),
        "vector_search": NodeInstance(
            id="vector_search",
            type="vector_search",
            input_values={
                "query": "{{ .nodes.start.output.query }}",
                "top_k": 5,
            },
            output_values={"docs": "docs"},
        ),
        "fulltext_search": NodeInstance(
            id="fulltext_search",
            type="fulltext_search",
            input_values={
                "query": "{{ .nodes.start.output.query }}",
                "top_k": 5,
            },
            output_values={"docs": "docs"},
        ),
        "merge": NodeInstance(
            id="merge",
            type="merge",
            input_values={
                "fulltext_search_docs": "{{ .nodes.fulltext_search.output.docs }}",
                "vector_search_docs": "{{ .nodes.vector_search.output.docs }}",
            },
            output_values={"docs": "docs"},
        ),
        "rerank": NodeInstance(
            id="rerank",
            type="rerank",
            input_values={
                "docs": "{{ .nodes.merge.output.docs }}",
            },
        ),
    }

    # Create edges
    edges = [Edge(source="vector_search", target="rerank"), Edge(source="fulltext_search", target="rerank")]

    # Create flow instance
    flow = FlowInstance(
        name="Test Flow",
        nodes=nodes,
        edges=edges,
    )

    # Validate flow
    flow.validate()  # Should not raise any exception


def test_cyclic_dependency():
    """Test cyclic dependency detection"""
    nodes = {
        "node1": NodeInstance(id="node1", type="type1"),
        "node2": NodeInstance(id="node2", type="type2"),
        "node3": NodeInstance(id="node3", type="type3"),
    }

    edges = [
        Edge(source="node1", target="node2"),
        Edge(source="node2", target="node3"),
        Edge(source="node3", target="node1"),  # Create a cycle
    ]

    flow = FlowInstance(name="Cyclic Flow", nodes=nodes, edges=edges)

    with pytest.raises(CycleError):
        flow.validate()


def test_invalid_node_reference():
    """Test invalid node reference"""
    nodes = {
        "node1": NodeInstance(
            id="node1", type="type1", input_values={"non_existent_node": "{{ .nodes.non_existent_node.output.docs }}"}
        )
    }

    flow = FlowInstance(name="Invalid Node Flow", nodes=nodes, edges=[])

    with pytest.raises(ValidationError) as exc_info:
        flow.validate()
    assert "non-existent node" in str(exc_info.value)


def test_invalid_field_reference():
    """Test invalid field reference"""
    nodes = {
        "node1": NodeInstance(
            id="node1",
            type="type1",
            input_values={"non_existent_field": "{{ .nodes.node2.output.non_existent_field }}"},
        ),
        "node2": NodeInstance(id="node2", type="type2"),
    }

    edges = [Edge(source="node2", target="node1")]

    flow = FlowInstance(name="Invalid Field Flow", nodes=nodes, edges=edges)

    with pytest.raises(ValidationError) as exc_info:
        flow.validate()
    assert "non-existent field" in str(exc_info.value)


def test_non_preceding_node_reference():
    """Test reference to non-preceding node"""
    nodes = {
        "node1": NodeInstance(
            id="node1", type="type1", input_values={"non_existent_node": "{{ .nodes.node2.output.docs }}"}
        ),
        "node2": NodeInstance(id="node2", type="type2"),
    }

    edges = [
        Edge(source="node1", target="node2")  # node1 depends on node2, but node2 comes after node1
    ]

    flow = FlowInstance(name="Invalid Order Flow", nodes=nodes, edges=edges)

    with pytest.raises(ValidationError) as exc_info:
        flow.validate()
    assert "non-preceding node" in str(exc_info.value)
