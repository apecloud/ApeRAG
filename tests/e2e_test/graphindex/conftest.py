"""
Shared fixtures and utilities for graph storage testing.
Provides common test data and helper functions for all graph storage test suites.
"""

import pytest


@pytest.fixture(scope="session")
def graph_data():
    """Load test graph data once per session"""
    """Load graph test data from JSON file"""
    import json
    import os

    # Get absolute path to test data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "graph_storage_test_data.json")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"📚 Loaded graph data: {len(data.get('nodes', {}))} nodes, {len(data.get('edges', []))} edges")
            return data
    except FileNotFoundError:
        print(f"⚠️  Test data file not found: {data_file}")
        return {"nodes": {}, "edges": []}
    except Exception as e:
        print(f"⚠️  Error loading test data: {e}")
        return {"nodes": {}, "edges": []}
