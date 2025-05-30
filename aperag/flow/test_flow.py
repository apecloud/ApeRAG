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

import os

from aperag.flow.engine import FlowEngine
from aperag.flow.parser import FlowParser


def test_rag_flow():
    """Test the RAG flow execution"""
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "examples", "rag_flow.yaml")

    # Load flow configuration
    flow = FlowParser.load_from_file(yaml_path)

    # Create execution engine
    engine = FlowEngine()

    # Execute flow with initial data
    initial_data = {"query": "What is the capital of France?"}

    try:
        engine.execute_flow(flow, initial_data)
        print("Flow executed successfully!")
    except Exception as e:
        print(f"Error executing flow: {str(e)}")


if __name__ == "__main__":
    test_rag_flow()
