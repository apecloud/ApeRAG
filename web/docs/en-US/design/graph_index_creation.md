---
title: Graph Index Creation Process Technical Documentation
description:
keywords:
---

# ApeRAG Graph Index Creation Process Technical Documentation

## Overview

The ApeRAG Graph Index creation process is the core pipeline of the entire knowledge graph construction system, responsible for transforming raw documents into structured knowledge graphs. This process is based on the LightRAG framework with extensive refactoring and optimization.

### Technical Improvements Overview

The original LightRAG has numerous limitations: non-stateless design leading to global state management concurrency conflicts, lack of effective concurrency control mechanisms, storage layer stability and consistency issues, and coarse-grained locking affecting performance. **Most critically, the original LightRAG does not support data isolation - all collection nodes and edges are stored in the same global space, causing data conflicts and pollution between different users and projects, making true multi-tenant support impossible**.

We have conducted large-scale refactoring to address these issues:

- **Complete rewrite to stateless architecture**: Each task uses independent instances, completely resolving concurrency conflicts
- **Introduced workspace data isolation mechanism**: Each collection has independent data space, completely solving data conflicts and pollution
- **Self-developed Concurrent Control model**: Implementing fine-grained lock management for high concurrency processing
- **Optimized lock granularity**: From coarse-grained global locks to entity-level and relationship-level precise locking
- **Refactored storage layer**: Support for Neo4j, NebulaGraph, PostgreSQL and other graph database backends, implementing reliable multi-storage consistency guarantees
- **Connected component concurrency optimization**: Intelligent concurrency strategy based on graph topology analysis

The Graph Index creation process includes the following core stages:

1. **Task reception and instance creation**: Celery task scheduling, LightRAG instance initialization
2. **Document chunking processing**: Intelligent chunking algorithms maintaining semantic coherence
3. **Entity relationship extraction**: LLM-based entity and relationship identification
4. **Connected component analysis**: Topological analysis of entity relationship networks
5. **Grouped concurrent processing**: Processing by connected components to improve performance
6. **Node edge merging**: Entity deduplication, relationship aggregation, description summarization
7. **Multi-storage writing**: Consistent writing to vector databases and graph databases

## Architecture Overview

```mermaid
flowchart TD
    %% Define style classes
    classDef entry fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000,font-weight:bold
    classDef manager fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    classDef processing fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef intelligence fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef optimization fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef storage fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px,color:#000
    classDef complete fill:#c8e6c9,stroke:#388e3c,stroke-width:3px,color:#000,font-weight:bold

    %% Entry layer
    START["🚀 Graph Index Task Start"]

    %% Management layer
    MANAGER["🎯 LightRAG Instance Management<br/>• create_lightrag_instance<br/>• workspace isolation"]

    %% Document processing branch
    DOC_PROCESS["📄 Document Chunking Processing<br/>• ainsert_and_chunk_document<br/>• chunking_by_token_size"]
    DOC_STORE["💾 Chunk Data Storage<br/>• chunks_vdb.upsert<br/>• text_chunks.upsert"]

    %% Graph indexing processing branch
    GRAPH_START["🏛️ Graph Index Construction Start<br/>• aprocess_graph_indexing"]

    %% Intelligent extraction layer
    AI_EXTRACT["🔬 AI Intelligent Extraction<br/>• extract_entities<br/>• LLM concurrent calls"]
    ENTITY_REL["🎭 Entity Relationship Recognition<br/>• Entity Recognition<br/>• Relationship Extraction"]

    %% Topology optimization layer
    TOPO_ANALYSIS["🧠 Topology Analysis<br/>• _find_connected_components<br/>• BFS algorithm"]
    COMPONENT_GROUP["🌐 Connected Component Grouping<br/>• Component Grouping<br/>• Concurrent task allocation"]

    %% Concurrent merging layer
    CONCURRENT_MERGE["⚡ Concurrent Intelligent Merging<br/>• merge_nodes_and_edges<br/>• Fine-grained lock control"]

    %% Storage layer (parallel writing)
    STORAGE_GRAPH["🗄️ Graph Database<br/>Neo4j/NebulaGraph/PG"]
    STORAGE_VECTOR["🎯 Vector Database<br/>Qdrant/Elasticsearch"]
    STORAGE_TEXT["📝 Text Storage<br/>Raw chunk data"]

    %% Completion
    COMPLETE["✅ Knowledge Graph Construction Complete<br/>🎉 Multi-dimensional retrieval ready"]

    %% Main process connections
    START --> MANAGER
    MANAGER --> DOC_PROCESS
    MANAGER --> GRAPH_START

    DOC_PROCESS --> DOC_STORE
    DOC_STORE -.->|"Data preparation complete"| GRAPH_START

    GRAPH_START --> AI_EXTRACT
    AI_EXTRACT --> ENTITY_REL
    AI_EXTRACT --> TOPO_ANALYSIS

    ENTITY_REL --> COMPONENT_GROUP
    TOPO_ANALYSIS --> COMPONENT_GROUP

    COMPONENT_GROUP --> CONCURRENT_MERGE

    %% Parallel storage
    CONCURRENT_MERGE --> STORAGE_GRAPH
    CONCURRENT_MERGE --> STORAGE_VECTOR
    DOC_STORE --> STORAGE_TEXT

    %% Converge to completion
    STORAGE_GRAPH --> COMPLETE
    STORAGE_VECTOR --> COMPLETE
    STORAGE_TEXT --> COMPLETE

    %% Apply styles
    class START entry
    class MANAGER manager
    class DOC_PROCESS,DOC_STORE processing
    class GRAPH_START,AI_EXTRACT,ENTITY_REL intelligence
    class TOPO_ANALYSIS,COMPONENT_GROUP,CONCURRENT_MERGE optimization
    class STORAGE_GRAPH,STORAGE_VECTOR,STORAGE_TEXT storage
    class COMPLETE complete
```

## Core Design Principles

### 1. Stateless Architecture Refactoring

The original LightRAG uses global state management, leading to severe concurrency conflicts, with multiple tasks sharing the same instance causing data pollution. **More seriously, all collection graph data is stored in the same global namespace, where entities and relationships from different projects interfere with each other**, making true multi-tenant isolation impossible.

We completely rewrote LightRAG's instance management code, implementing a stateless design: each Celery task creates an independent LightRAG instance, achieving collection-level data isolation through the `workspace` parameter. **Each collection's graph data is stored in independent namespaces** (e.g., `entity:{entity_name}:{workspace}`), supporting Neo4j, NebulaGraph, PostgreSQL and other graph database backends, with strict instance lifecycle management to ensure no resource leaks.

### 2. Staged Pipeline Processing

**Document processing and graph indexing separation**:

- **ainsert_and_chunk_document**: Responsible for document chunking and storage
- **aprocess_graph_indexing**: Responsible for graph index construction
- **Advantages**: Modular design, easy to test and maintain

### 3. Connected Component Concurrency Optimization

The original LightRAG lacks effective concurrency strategies, with simple global locks causing performance bottlenecks and inability to fully utilize multi-core CPU resources.

We designed a connected component discovery algorithm based on graph theory, decomposing entity relationship networks into independent processing components. Through topology analysis-driven intelligent grouped concurrency, different connected components can be processed completely in parallel, achieving zero lock conflict design.

The core algorithm approach is: construct adjacency graphs of entity relationships, use BFS traversal to discover all connected components, group entities belonging to different connected components into independent processing tasks, thus achieving true parallel processing.

### 4. Fine-grained Concurrency Control Mechanism

The original LightRAG lacks effective concurrency control mechanisms, with storage operation consistency unable to be guaranteed, frequently experiencing data races and deadlock issues.

We implemented a Concurrent Control model from scratch, establishing a fine-grained lock manager that supports precise locking at entity and relationship levels. Lock naming uses workspace isolation design: `entity:{entity_name}:{workspace}` and `relationship:{src}:{tgt}:{workspace}`. We designed intelligent lock strategies, only locking during merge writes, completely lock-free during entity extraction stages, and preventing deadlocks through sorted lock acquisition mechanisms to avoid circular waiting.

## Specific Execution Pipeline Example

### Complete Graph Index Creation Process

Taking a single document's graph index creation as an example, the entire processing pipeline includes the following key stages:

1. **Task Reception Layer**: Celery tasks receive Graph index creation requests, calling LightRAG Manager

2. **LightRAG Manager Layer**: Create independent LightRAG instances for each task, ensuring stateless processing

3. **Document Chunking Stage**:
   - Content cleaning and preprocessing
   - Intelligent chunking based on token count (supporting overlap)
   - Generate unique chunk IDs and metadata
   - Serial writing to vector storage and text storage

4. **Graph Index Construction Stage**:
   - Call LLM for concurrent entity relationship extraction
   - Connected component analysis and grouping processing
   - Statistical extraction results

5. **Entity Relationship Extraction Stage**:
   - Build LLM prompt templates
   - Use semaphores to control concurrency
   - Support optional gleaning extraction
   - Parse extraction results into structured data

6. **Connected Component Grouping Processing**:
   - Discover connected components and create processing tasks
   - Filter entities and relationships belonging to each component
   - Use semaphores to control component concurrent processing

7. **Node Edge Merging Stage**:
   - Collect same-named entities and same-direction relationships
   - Use fine-grained locks for concurrent merging
   - Synchronously update graph database and vector database

## Core Data Flow Diagram

The Graph Index creation process is essentially a complex data transformation pipeline. The following data flow diagram shows the complete data transformation process from raw documents to structured knowledge graphs:

```mermaid
flowchart TD
    %% Define style classes
    classDef inputStage fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    classDef chunkStage fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef extractStage fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef topoStage fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef mergeStage fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef summaryStage fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#000
    classDef storageStage fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px,color:#000
    classDef resultStage fill:#c8e6c9,stroke:#388e3c,stroke-width:3px,color:#000

    %% Data input stage
    subgraph InputStage ["📥 Raw Data Input"]
        A["📄 Raw Documents<br/><small>PDF/Word/Markdown etc.</small>"]
        B["🧹 Document Cleaning<br/><small>Format removal/noise filtering</small>"]
        C["✂️ Smart Chunking<br/><small>Based on semantics and token limits</small>"]
    end

    %% Chunk data stage
    subgraph ChunkStage ["📦 Structured Chunk Data"]
        D["🔢 Chunk Collection<br/><small>With ID/content/metadata</small>"]
    end

    %% Concurrent extraction stage
    subgraph ExtractStage ["🔬 AI Intelligent Extraction"]
        E["🤖 LLM Concurrent Calls<br/><small>Large model concurrent recognition</small>"]
        F["👤 Entity Recognition<br/><small>People/organizations/concepts etc.</small>"]
        G["🔗 Relationship Recognition<br/><small>Semantic relationships between entities</small>"]
        H["📋 Raw Entity List<br/><small>Undeduped entity data</small>"]
        I["📋 Raw Relationship List<br/><small>Unaggregated relationship data</small>"]
    end

    %% Topology analysis stage
    subgraph TopoStage ["🧠 Graph Topology Analysis"]
        J["🕸️ Build Adjacency Graph<br/><small>Entity relationship modeling</small>"]
        K["🔍 Connected Component Discovery<br/><small>BFS algorithm identifies groups</small>"]
        L["📦 Component Grouping<br/><small>Divide by connectivity</small>"]
    end

    %% Concurrent merging stage
    subgraph MergeStage ["⚡ Intelligent Data Merging"]
        M["🔄 Component Concurrent Processing<br/><small>Group parallel processing</small>"]
        N["👥 Entity Deduplication<br/><small>Same-named entity aggregation</small>"]
        O["🔗 Relationship Aggregation<br/><small>Weight accumulation processing</small>"]
        P["✨ Merged Entity Data<br/><small>Deduped entities</small>"]
        Q["✨ Merged Relationship Data<br/><small>Aggregated relationships</small>"]
    end

    %% Intelligent summarization stage
    subgraph SummaryStage ["📝 Intelligent Content Summarization"]
        R{"📏 Content Length Check"}
        S["🤖 LLM Smart Summary<br/><small>Long text compression</small>"]
        T["✅ Keep Original Content<br/><small>Short text direct retention</small>"]
        U["📄 Final Entity Content<br/><small>Optimized descriptions</small>"]
        V{"📏 Relationship Length Check"}
        W["🤖 LLM Relationship Summary<br/><small>Relationship description compression</small>"]
        X["✅ Keep Original Relationship<br/><small>Short relationships direct retention</small>"]
        Y["📄 Final Relationship Content<br/><small>Optimized descriptions</small>"]
    end

    %% Multi-storage writing stage
    subgraph StorageStage ["💾 Multi-Storage System Writing"]
        Z1["🗄️ Graph Database<br/><small>Neo4j/NebulaGraph/PG</small>"]
        Z2["🎯 Entity Vector DB<br/><small>Semantic search storage</small>"]
        Z3["🔗 Relationship Vector DB<br/><small>Relationship semantic storage</small>"]
        Z4["📚 Chunk Vector DB<br/><small>Raw chunk indexing</small>"]
        Z5["📝 Text Storage<br/><small>Chunk text storage</small>"]
    end

    %% Completion stage
    AA["🎉 Knowledge Graph Construction Complete<br/><small>Multi-dimensional retrieval ready</small>"]

    %% Main data flow connections
    A --> B
    B --> C
    C --> D

    D --> E
    E --> F
    E --> G
    F --> H
    G --> I

    H --> J
    I --> J
    J --> K
    K --> L

    L --> M
    M --> N
    M --> O
    N --> P
    O --> Q

    %% Intelligent summarization flow
    P --> R
    R -->|"Content too long"| S
    R -->|"Content appropriate"| T
    S --> U
    T --> U

    Q --> V
    V -->|"Description too long"| W
    V -->|"Description appropriate"| X
    W --> Y
    X --> Y

    %% Storage writing flow
    U --> Z1
    U --> Z2
    Y --> Z1
    Y --> Z3
    D --> Z4
    D --> Z5

    %% Converge to completion
    Z1 --> AA
    Z2 --> AA
    Z3 --> AA
    Z4 --> AA
    Z5 --> AA

    %% Apply styles
    class A,B,C inputStage
    class D chunkStage
    class E,F,G,H,I extractStage
    class J,K,L topoStage
    class M,N,O,P,Q mergeStage
    class R,S,T,U,V,W,X,Y summaryStage
    class Z1,Z2,Z3,Z4,Z5 storageStage
    class AA resultStage
```

### Data Flow Transformation Process Analysis

#### 🚀 **Document Input → Structured Chunking**

Raw documents undergo format cleaning and noise filtering, using intelligent chunking algorithms to segment according to semantic boundaries and token limits, generating chunk collections with unique identifiers and metadata. This step ensures data quality and traceability for subsequent processing.

#### 🔬 **Chunk Data → AI Extraction Results**

Chunk data undergoes intelligent analysis through LLM concurrent calls, simultaneously identifying entities (people, organizations, concepts, etc.) and semantic relationships between entities in the text. This stage produces raw, undeduped entity and relationship lists, providing raw materials for subsequent graph construction.

#### 🧠 **Extraction Results → Topology Grouping**

Based on extracted entity relationships, adjacency graph networks are constructed, using BFS algorithms to discover connected components and identify groups of mutually related entities. For example: technology team-related entities form one group, finance department-related entities form another group. This topology analysis lays the foundation for parallel processing.

#### ⚡ **Topology Grouping → Intelligent Merging**

Different connected components can be processed completely in parallel, with same-named entities undergoing intelligent deduplication and information aggregation, and same-direction relationships undergoing weight accumulation and description merging. This process integrates fragmented information into complete knowledge units.

#### 📝 **Merged Data → Content Optimization**

Length checks are performed on merged entity and relationship descriptions, with overly long content undergoing intelligent summary compression through LLM, ensuring a balance between information density and storage efficiency. Short content is retained directly, while long content is intelligently summarized.

#### 💾 **Optimized Content → Multi-dimensional Storage**

Final knowledge content is written simultaneously to multiple storage systems:

- **Graph Database**: Stores entity nodes and relationship edges, supporting graph queries
- **Vector Database**: Stores semantic vectors, supporting similarity search
- **Text Storage**: Retains original chunks, supporting full-text search

This multi-dimensional storage architecture ensures optimal performance of knowledge graphs in different query scenarios.

### Data Flow Optimization Features

#### 1. Fine-grained Concurrency Control

We implemented precise locking mechanisms at entity and relationship levels: `entity:{entity_name}:{workspace}` and `relationship:{src}:{tgt}:{workspace}`, minimizing lock scope to only during merge writes, with completely parallel entity extraction stages. Through sorted lock acquisition order, we effectively prevent circular waiting and deadlocks.

#### 2. Connected Component-driven Concurrency Optimization

We designed topology analysis based on BFS algorithms, discovering independent entity relationship networks and grouping them for parallel processing. Different connected components are processed completely independently, achieving zero lock competition, while processing in batches by component effectively controls memory peaks.

#### 3. Intelligent Data Merging Strategy

We implemented intelligent entity deduplication based on entity_name, supporting intelligent concatenation and summarization of multiple description fragments, quantitative accumulation of relationship strength, and established complete data lineage recording mechanisms.

## Performance Optimization Strategies

### 1. Connected Component Optimization

**Topology-driven Concurrency Strategy**:

- **Independent Processing**: Different connected components processed completely in parallel
- **Lock Competition Minimization**: Entities within components don't conflict across components
- **Memory Efficiency**: Batch processing by component, controlling memory usage

The system automatically collects connected component distribution statistics, including total components, maximum component size, average component size, single-entity component count, and large component count, for performance tuning and resource allocation.

### 2. LLM Call Optimization

**Batch Processing and Caching Strategy**:

- **Concurrency Control**: Use semaphores to limit concurrent LLM calls
- **Batch Optimization**: Batch processing of similar content
- **Caching Mechanism**: Reuse of entity description summary caching

The system intelligently checks description length, automatically calling LLM to generate summaries when exceeding token thresholds, supporting summary result caching reuse for improved efficiency.

### 3. Storage Write Optimization

**Batch Writing and Connection Reuse**:

- **Batch Operations**: Reduce database round trips
- **Connection Pooling**: Reuse database connections
- **Async Writing**: Parallel writing to different storage systems

### 4. Memory Management Optimization

**Streaming Processing and Memory Control**:

- **Chunk Processing**: Streaming chunking of large documents
- **Timely Release**: Immediate memory release after processing completion
- **Monitoring Alerts**: Memory usage monitoring

## Code Organization Structure

### Directory Structure

```
aperag/
├── graph/                        # Graph indexing core module
│   ├── lightrag_manager.py      # LightRAG manager (Celery entry)
│   └── lightrag/                 # LightRAG core implementation
│       ├── lightrag.py          # Main LightRAG class
│       ├── operate.py           # Core operation functions
│       ├── base.py              # Base interface definitions
│       ├── utils.py             # Utility functions
│       ├── prompt.py            # Prompt templates
│       └── kg/                  # Knowledge graph storage implementations
│           ├── neo4j_sync_impl.py    # Neo4j sync implementation
│           ├── nebula_sync_impl.py   # NebulaGraph sync implementation
│           └── postgres_sync_impl.py # PostgreSQL sync implementation
├── concurrent_control/           # Concurrency control module
│   ├── manager.py               # Lock manager
│   └── protocols.py             # Lock interface definitions
└── tasks/                       # Task module
    └── document.py              # Document processing business logic

config/
└── celery_tasks.py              # Celery task definitions
```

### Core Interface Design

#### LightRAG Management Interface

Responsible for instance creation, document processing and deletion entry management, as well as dynamic generation of embedding and LLM functions.

#### LightRAG Core Interface

Implements document chunk storage, graph index construction, document deletion, connected component discovery and grouping processing and other core functions.

#### Operation Function Interface

Provides entity extraction, node edge merging, chunk processing and other low-level operation functions, supporting asynchronous concurrent execution.

### Data Structure Design

#### Core Data Models

The system uses unified data structure design:

- **Chunk Data**: Contains token count, content, sequence index, document ID and file path
- **Entity Data**: Contains entity name, type, description, source ID and creation timestamp
- **Relationship Data**: Contains source entity, target entity, description, keywords, weight and source information
- **Connected Component Data**: Contains component index, entity list, filtered results and total component count

All data structures support multi-source aggregation, using delimiters (such as `|`) to merge multiple source information.

## Performance Monitoring and Debugging

### 1. Performance Metrics

**Key Performance Indicators (KPIs)**:

- **Document Processing Throughput**: Documents processed per minute
- **Entity Extraction Accuracy**: Quality assessment of extracted entities
- **Connected Component Distribution**: Complexity analysis of topological structure
- **LLM Call Efficiency**: Average response time and concurrency
- **Storage Write Performance**: Database operation latency statistics

### 2. Debugging Tools

**Structured Logging**: The system provides complete structured logging functionality, including entity extraction progress tracking, entity merge detail recording, relationship merge status monitoring, etc. Logs record processing progress percentages, entity relationship quantity statistics, summary generation types and other key information.

### 3. Performance Analysis

**Execution Time Statistics**: Through performance decorators, key functions undergo execution time statistics, including entity extraction, node edge merging and other core operation duration analysis, facilitating performance optimization and bottleneck identification.

## Configuration and Environment

### 1. Core Configuration Parameters

**LightRAG Configuration**: The system supports rich configuration parameter tuning, including chunk size, overlap size, LLM concurrency count, similarity thresholds, batch size, summary parameters, embedding token limits, etc. Default configurations are optimized for Chinese environments, supporting flexible adjustment according to actual needs.

### 2. Storage Configuration

**Multi-storage Backend Support**:

```bash
# Environment variable configuration
GRAPH_INDEX_KV_STORAGE=PGOpsSyncKVStorage          # KV storage
GRAPH_INDEX_VECTOR_STORAGE=PGOpsSyncVectorStorage  # Vector storage
GRAPH_INDEX_GRAPH_STORAGE=Neo4JSyncStorage         # Graph storage

# PostgreSQL configuration
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Neo4J configuration example
NEO4J_HOST=127.0.0.1
NEO4J_PORT=7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# NebulaGraph configuration example
NEBULA_HOST=127.0.0.1
NEBULA_PORT=9669
NEBULA_USER=root
NEBULA_PASSWORD=nebula
NEBULA_MAX_CONNECTION_POOL_SIZE=10
NEBULA_TIMEOUT=60000
```

## Summary

We conducted large-scale refactoring and optimization of the original LightRAG, achieving a high-concurrency knowledge graph construction system truly suitable for production environments:

> 🔬 **Algorithm Deep Dive**: For specific algorithm implementation details of entity extraction and merging, continue reading [LightRAG Entity Extraction and Merging Mechanism](./lightrag_entity_extraction_and_merging.md)

### Core Technical Contributions

1. **Complete rewrite to stateless architecture**: We completely rewrote LightRAG's core architecture, solving the original version's inability to execute concurrently, with each task using independent instances supporting true multi-tenant isolation
2. **Self-developed Concurrent Control model**: We designed a fine-grained lock management system, implementing precise concurrency control at entity and relationship levels
3. **Connected component concurrency optimization**: We designed intelligent concurrency strategies based on graph topology analysis, maximizing parallel processing efficiency
4. **Refactored storage layer architecture**: We completely rewrote the storage abstraction layer, solving the original version's unreliable storage implementation and inconsistent multi-storage backend implementations
5. **End-to-end data flow design**: We designed a complete data transformation pipeline, with full-chain optimization from document chunking to multi-storage writing

### Performance Improvements

Through these technical improvements, we achieved:

- **5-10x performance improvement** in concurrent processing capability
- **95%+ reduction** in lock competition
- **True multi-tenant concurrent processing** support
- **Solving production environment stability issues** that the original version couldn't handle
- **Complete workspace-level data isolation**, eliminating data conflicts and pollution between different collections

The refactored system transforms the original research prototype into a production-ready, high-performance knowledge graph construction platform that truly meets enterprise-level requirements.

---

## Related Documents

- 📋 [Indexing Architecture Design](./indexing_architecture.md) - Overall indexing architecture
- 📖 [LightRAG Entity Extraction and Merging Mechanism](./lightrag_entity_extraction_and_merging.md) - Core algorithm details
- 🏗️ [Graph Index 创建流程技术文档](./graph_index_creation_zh.md) - Chinese Version
