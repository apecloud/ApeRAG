# Graph Storage Tests

This directory contains end-to-end tests for graph storage implementations in ApeRAG.

## Architecture Overview

The test architecture follows the principle of DRY (Don't Repeat Yourself) and uses the BaseGraphStorage interface to provide storage-agnostic testing.

### Files Structure

```
tests/e2e_test/graphindex/
├── test_graph_storage.py      # Universal test suite (storage-agnostic)
├── test_neo4j_storage.py      # Neo4j-specific test runner
├── test_nebula_storage.py     # Nebula-specific test runner
├── graph_storage_test_data.json  # Real test data (4117 nodes, 6795 edges)
└── README.md                  # This file
```

### Components

#### 1. `test_graph_storage.py` - Universal Test Suite

**Purpose**: Contains storage-agnostic test logic that works with any BaseGraphStorage implementation.

**Key Classes**:
- `TestDataLoader` - Loads and validates test data from JSON
- `GraphStorageTestSuite` - Static methods containing all test logic
- `GraphStorageTestRunner` - Test runner base class for inheritance

**Test Coverage**: All BaseGraphStorage interface methods:
- Node operations: `has_node`, `get_node`, `get_nodes_batch`, `node_degree`, `node_degrees_batch`, `upsert_node`, `delete_node`, `remove_nodes`
- Edge operations: `has_edge`, `get_edge`, `get_edges_batch`, `get_node_edges`, `get_nodes_edges_batch`, `edge_degree`, `edge_degrees_batch`, `upsert_edge`, `remove_edges`
- Graph operations: `get_all_labels`, `get_knowledge_graph`
- Data integrity and interface coverage validation

#### 2. `test_neo4j_storage.py` - Neo4j Test Runner

**Purpose**: Provides Neo4j storage instances and inherits all tests from `GraphStorageTestRunner`.

**Key Features**:
- Initializes `Neo4JSyncStorage` with test data
- Implements caching to avoid repeated data loading
- Automatically runs all BaseGraphStorage tests

#### 3. `test_nebula_storage.py` - Nebula Test Runner

**Purpose**: Provides Nebula storage instances and inherits all tests from `GraphStorageTestRunner`.

**Key Features**:
- Initializes `NebulaSyncStorage` with test data
- Handles Nebula-specific schema requirements
- Note: May skip tests if Nebula setup issues occur

## Usage

### Run All Neo4j Tests
```bash
NEO4J_DATABASE=neo4j pytest tests/e2e_test/graphindex/test_neo4j_storage.py -v
```

### Run All Nebula Tests
```bash
pytest tests/e2e_test/graphindex/test_nebula_storage.py -v
```

### Run Specific Test for Neo4j
```bash
NEO4J_DATABASE=neo4j pytest tests/e2e_test/graphindex/test_neo4j_storage.py::TestNeo4jStorage::test_has_node -v
```

### Run Both Storage Tests
```bash
NEO4J_DATABASE=neo4j pytest tests/e2e_test/graphindex/test_neo4j_storage.py tests/e2e_test/graphindex/test_nebula_storage.py -v
```

## Test Data

The tests use real graph data from `graph_storage_test_data.json` containing:
- **4,117 nodes** from "三国演义" (Romance of the Three Kingdoms)
- **6,795 edges** representing relationships between entities
- Rich properties including descriptions, entity types, and weights

## Adding New Storage Implementations

To add tests for a new graph storage implementation:

1. **Create a new test file** (e.g., `test_mystorage_storage.py`)
2. **Import the base classes**:
   ```python
   from tests.e2e_test.graphindex.test_graph_storage import GraphStorageTestRunner, graph_data, mock_embedding_func
   ```
3. **Create storage fixture**:
   ```python
   @pytest_asyncio.fixture(scope="function")
   async def storage_with_data(mock_embedding_func, graph_data):
       # Initialize your storage implementation
       storage = MyStorage(...)
       # Populate with test data
       # Return (storage, graph_data)
   ```
4. **Create test class**:
   ```python
   @pytest.mark.asyncio
   class TestMyStorage(GraphStorageTestRunner):
       pass  # All tests inherited automatically
   ```

## Benefits of This Architecture

### ✅ **DRY Principle**
- Test logic written once, reused across all storage implementations
- Consistent test coverage for all storage backends

### ✅ **Interface Compliance**
- Ensures all implementations correctly follow BaseGraphStorage interface
- Catches interface violations early

### ✅ **Maintainability**
- Adding new tests requires updating only `GraphStorageTestSuite`
- All storage implementations automatically get new tests

### ✅ **Performance**
- Single data import per storage type
- Cached storage instances reduce initialization overhead

### ✅ **Real Data Testing**
- Uses actual production-like data from "三国演义"
- Tests with realistic graph structure and content

## Current Status

### ✅ Neo4j Storage
- **Status**: All 21 tests passing
- **Performance**: ~40 seconds (including data import)
- **Coverage**: Complete BaseGraphStorage interface

### ⚠️ Nebula Storage
- **Status**: Schema synchronization issues
- **Known Issues**: `SemanticError: No schema found for 'base'`
- **Solution**: Requires Nebula Graph configuration or existing space usage

## Future Enhancements

1. **Parallel Testing**: Run multiple storage tests in parallel
2. **Performance Benchmarks**: Compare storage implementations
3. **Memory Testing**: Add memory usage and leak detection
4. **Stress Testing**: Large dataset and concurrent operation testing 