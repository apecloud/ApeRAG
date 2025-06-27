# Graph Storage Tests

This directory contains end-to-end tests for graph storage implementations in ApeRAG.

## Test Overview

The `test_graph_storage.py` file provides comprehensive testing for:
- **Neo4j Sync Storage** (`Neo4JSyncStorage`)
- **Nebula Sync Storage** (`NebulaSyncStorage`)

## Current Test Status

### ✅ Working Tests (Neo4j)
The following Neo4j tests are currently passing:
- Basic node operations (create, read, update)
- Basic edge operations 
- Degree calculations
- Batch operations
- Knowledge graph retrieval
- Deletion operations
- Concurrent operations
- Performance comparisons

### ⚠️ Partial Issues (Neo4j)
- **Error handling test**: Minor assertion issue with error message checking
- **Workspace isolation test**: Using unique workspace names for better isolation

### ❌ Known Issues (Nebula)
Nebula tests are currently failing due to schema synchronization issues:
- **Root Cause**: Nebula Graph's asynchronous space creation and schema propagation
- **Error**: `SemanticError: No schema found for 'base'` or `Storage Error: Tag not found`
- **Impact**: All Nebula storage operations fail

## How to Run Tests

### Prerequisites

1. **Environment Setup**:
   ```bash
   # Copy and configure environment
   cp envs/env.template .env
   # Edit .env with your database credentials
   ```

2. **Database Services**:
   ```bash
   # Neo4j should be running on default port 7687
   # Nebula should be running on default port 9669
   ```

### Running Tests

```bash
# Run all graph storage tests
NEO4J_DATABASE=neo4j pytest tests/e2e_test/graphindex/test_graph_storage.py -v

# Run only Neo4j tests (recommended)
NEO4J_DATABASE=neo4j pytest tests/e2e_test/graphindex/test_graph_storage.py -k "neo4j" -v

# Run only Nebula tests (currently failing)
NEO4J_DATABASE=neo4j pytest tests/e2e_test/graphindex/test_graph_storage.py -k "nebula" -v

# Run specific test
NEO4J_DATABASE=neo4j pytest tests/e2e_test/graphindex/test_graph_storage.py::TestGraphStorage::test_node_operations_neo4j -v
```

### Environment Variables

- `NEO4J_DATABASE=neo4j`: Use default Neo4j database (required for Community Edition)
- `NEO4J_URI`: Neo4j connection string (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME`: Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD`: Neo4j password
- `NEBULA_HOST`: Nebula host (default: `127.0.0.1`)
- `NEBULA_PORT`: Nebula port (default: `9669`)
- `NEBULA_USER`: Nebula username (default: `root`)
- `NEBULA_PASSWORD`: Nebula password (default: `nebula`)

## Test Architecture

### Test Structure
```
test_graph_storage.py
├── TestGraphStorage (Main test class)
│   ├── Fixtures (mock_embedding_func, test_workspace, etc.)
│   ├── Storage Fixtures (neo4j_storage, nebula_storage)
│   ├── Sample Data (sample_nodes, sample_edges)
│   └── Test Methods (grouped by functionality)
└── TestGraphStorageCompatibility (Cross-storage tests)
```

### Test Categories

1. **Basic Operations**: Node and edge CRUD operations
2. **Batch Operations**: Efficient bulk operations testing
3. **Degree Operations**: Node/edge degree calculations
4. **Graph Retrieval**: Knowledge graph construction and querying
5. **Deletion Operations**: Data cleanup and removal
6. **Error Handling**: Edge cases and error conditions
7. **Concurrency**: Multi-threaded operation safety
8. **Isolation**: Workspace/tenant separation
9. **Performance**: Batch vs individual operation benchmarks
10. **Compatibility**: Cross-storage data format validation

### Common Test Logic

Tests use shared implementation methods (`_test_*`) to ensure consistent validation across different storage backends:

```python
# Example: Node operations test
async def _test_node_operations(self, storage, sample_nodes):
    # Test node creation, retrieval, and updates
    # Used by both Neo4j and Nebula specific test methods
```

## Development Guidelines

### Adding New Tests

1. **Create test method**: Follow naming pattern `test_*_neo4j` and `test_*_nebula`
2. **Implement common logic**: Use `_test_*` helper methods for shared validation
3. **Handle errors gracefully**: Use try-catch for storage-specific issues
4. **Clean up resources**: Ensure proper cleanup in test fixtures

### Debugging Failed Tests

1. **Check database connectivity**:
   ```bash
   # Test Neo4j connection
   echo "MATCH (n) RETURN count(n)" | cypher-shell -u neo4j -p password
   
   # Test Nebula connection  
   # Use Nebula console to verify connectivity
   ```

2. **Verify environment variables**: Ensure all required database credentials are set

3. **Run individual tests**: Isolate specific failing tests for detailed analysis

4. **Check logs**: Look for detailed error messages in test output

## Known Limitations

1. **Nebula Schema Sync**: Space creation and schema propagation timing issues
2. **Neo4j Community**: Limited multi-database support requires using default database
3. **Test Isolation**: Some tests may interfere with each other if run concurrently
4. **Resource Cleanup**: Manual cleanup may be needed if tests are interrupted

## Future Improvements

1. **Nebula Schema Fix**: Implement robust schema verification and retry logic
2. **Test Parallelization**: Enable safe concurrent test execution
3. **Mock Backends**: Add in-memory storage implementations for faster testing
4. **Performance Benchmarks**: Add comprehensive performance comparison framework
5. **Integration Tests**: Add tests with real LightRAG workflows

## Support

For issues with graph storage tests:
1. Check this README for known issues
2. Verify database setup and connectivity
3. Run tests individually to isolate problems
4. Check database logs for detailed error information

## Contributing

When modifying graph storage tests:
1. Follow existing patterns for consistency
2. Add both Neo4j and Nebula variants for new tests
3. Update this README with any new limitations or requirements
4. Ensure proper resource cleanup in all test paths 