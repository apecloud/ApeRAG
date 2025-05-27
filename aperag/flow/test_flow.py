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
