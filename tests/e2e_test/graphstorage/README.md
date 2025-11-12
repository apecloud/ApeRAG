# GraphIndex E2E Test Suite

This directory contains the end-to-end test suite for the ApeRAG GraphIndex module, used to verify the correctness and consistency of various graph storage implementations.

## 🌟 Highlights

### Test Oracle Testing Mode ⚖️

This test suite adopts the **Test Oracle Mode**, an elegant and powerful testing methodology:

  - **Dual Verification**: Each graph storage implementation is compared against a validated NetworkX baseline implementation.
  - **Automatic Synchronization**: The Oracle automatically synchronizes all write operations to both the real storage and the baseline storage.
  - **Instant Verification**: Every query operation compares the results from both storages in real-time, ensuring complete consistency.
  - **Zero Error Tolerance**: Any inconsistency immediately raises an exception, ensuring strict correctness of the implementation.
  - **Flexible Comparison**: Supports intelligent comparison strategies such as floating-point tolerance, field order independence, and edge direction normalization.

This approach ensures that every graph storage implementation undergoes **rigorous behavioral verification**, avoiding edge cases that might be missed by simply relying on expected results.

## 📁 File Structure

```
tests/e2e_test/graphindex/
├── conftest.py                     # pytest configuration
├── graph_storage_oracle.py         # Test Oracle implementation ⚖️
├── networkx_baseline_storage.py    # NetworkX baseline implementation
├── test_graph_storage.py           # Generic test suite
├── test_neo4j_storage.py           # Neo4j storage tests
└── graph_storage_test_data.json    # Test data
```

## 🎯 Test Case Coverage

### Node Operation Tests

  - `test_has_node` - Node existence check
  - `test_get_node` - Single node data retrieval
  - `test_get_nodes_batch` - Batch node retrieval
  - `test_node_degree` - Node degree calculation
  - `test_node_degrees_batch` - Batch node degree calculation
  - `test_upsert_node` - Node creation/update
  - `test_delete_node` - Single node deletion
  - `test_remove_nodes` - Batch node deletion

### Edge Operation Tests

  - `test_has_edge` - Edge existence check
  - `test_get_edge` - Single edge data retrieval
  - `test_get_edges_batch` - Batch edge retrieval
  - `test_get_node_edges` - Node associated edge retrieval
  - `test_get_nodes_edges_batch` - Batch node associated edge retrieval
  - `test_edge_degree` - Edge degree calculation
  - `test_edge_degrees_batch` - Batch edge degree calculation
  - `test_upsert_edge` - Edge creation/update
  - `test_remove_edges` - Batch edge deletion

### Complex Operation Tests

  - `test_data_integrity` - Data integrity verification
  - `test_large_batch_operations` - Large batch operation performance
  - `test_data_consistency_after_operations` - Consistency check after operations
  - `test_get_all_labels` - All label retrieval
  - `test_interface_coverage_summary` - Interface coverage summary

## 🛠️ Environment Configuration

### Required Environment Variables

#### Neo4j Configuration

```bash
NEO4J_HOST=127.0.0.1
NEO4J_PORT=7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### How to Configure Environment Variables

1.  **Using a .env file** (Recommended)

    ```bash
    # Create a .env file in the project root
    cp envs/env.template .env
    # Edit the .env file to add the configurations above
    ```

2.  **Using Environment Variables**

    ```bash
    export NEO4J_HOST=127.0.0.1
    export NEO4J_PORT=7687
    # ... other variables
    ```

## 🚀 Running Tests

### Run All GraphIndex Tests

```bash
# Execute from the project root
uv run pytest tests/e2e_test/graphindex/ -v
```

### Run Specific Database Tests

#### Neo4j Storage Tests

```bash
uv run pytest tests/e2e_test/graphindex/test_neo4j_storage.py::TestNeo4jStorage -v
```


### Run Specific Test Cases

```bash
# Test Neo4j node operations
uv run pytest tests/e2e_test/graphindex/test_neo4j_storage.py::TestNeo4jStorage::test_has_node -v
```

## 📊 Test Data

  - **Data File**: `graph_storage_test_data.json`
  - **Data Format**: Each line is a JSON object (node or edge)
  - **Data Scale**: Contains a large amount of real graph data to ensure comprehensive testing
  - **Data Source**: Graph structures exported from actual LightRAG runs

## ⚠️ Notes

### Environment Dependencies

  - If corresponding database environment variables are missing, relevant tests will be **automatically skipped**.
  - Tests will automatically detect database connection availability.
  - It is recommended to run in an isolated environment to avoid affecting production data.

### Test Data Management

  - Each test class uses a unique workspace to avoid conflicts.
  - Test data will be automatically cleaned up after tests complete.
  - Use `DROP SPACE/DATABASE` to ensure thorough cleanup.

### Performance Considerations

  - The full test suite may take a significant amount of time (involving large data imports).
  - It is recommended to use SSD storage to improve I/O performance.
  - You can run a subset of tests using the `-k` parameter.

## 🔧 Troubleshooting

### Common Issues

1.  **Database Connection Failure**

    ```bash
    # Check database service status
    docker ps | grep neo4j
    ```

2.  **Environment Variables Not Set**

    ```bash
    # Verify environment variables
    echo $NEO4J_HOST
    ```

3.  **Missing Test Data File**

    ```bash
    # Check for test data file
    ls -la tests/e2e_test/graphindex/graph_storage_test_data.json
    ```

### Debug Mode

```bash
# Enable verbose logging
uv run pytest tests/e2e_test/graphindex/ -v -s --log-cli-level=DEBUG
```

## 🎉 Extending with New Storage Implementations

To add tests for a new graph storage implementation:

1.  **Implement Storage Interface**: Inherit from `BaseGraphStorage`.
2.  **Create Test File**: Refer to the structure of `test_neo4j_storage.py`.
3.  **Configure Oracle**: Wrap your implementation with `GraphStorageOracle`.
4.  **Call Test Suite**: Directly use the static methods of `GraphStorageTestSuite`.

This design ensures that **all storage implementations use the same testing standards**, guaranteeing API consistency and reliability.

-----

**Test Oracle Mode gives us full confidence in the correctness of our graph storage implementations\!** ⚖️✨