# LightRAG替换方案：高并发知识图谱构建系统设计指南

## 1. LightRAG核心问题分析

### 1.1 架构设计缺陷

**全局单例状态管理**
```python
# lightrag/kg/shared_storage.py - 全局变量导致多实例冲突
_is_multiprocess = None
_manager = None 
_shared_dicts: Optional[Dict[str, Any]] = None
_pipeline_status_lock: Optional[LockType] = None
```

**问题影响**：
- 无法在同一进程中创建多个LightRAG实例
- 多个collection之间会共享管道状态
- 全局锁导致严重的串行化瓶颈

### 1.2 并发处理限制

**管道状态冲突**
```python
# 所有实例共享同一个pipeline_status，导致状态覆盖
pipeline_status = await get_namespace_data("pipeline_status") 
async with pipeline_status_lock:
    if not pipeline_status.get("busy", False):
        # 只有一个实例能执行，其他被阻塞
```

**事件循环管理混乱**
```python
# always_get_an_event_loop()在多实例场景下不可靠
loop = always_get_an_event_loop()
loop.run_until_complete(self.ainsert(...))
```

### 1.3 扩展性问题

- **单working_dir限制**：每个实例绑定单一存储目录
- **内存共享冲突**：multiprocessing.Manager在K8s环境下不可靠
- **资源竞争**：全局锁机制在高并发下性能急剧下降

### 1.4 SaaS场景不适配

- **租户隔离困难**：缺乏多租户设计
- **动态扩容限制**：无法水平扩展处理能力
- **监控可观测性差**：缺乏细粒度的任务状态追踪

## 2. 自实现架构设计

### 2.1 整体架构原则

**无状态设计**：所有组件均为无状态，支持水平扩展
**任务分离**：文档处理、实体提取、存储分别独立
**异步优先**：全流程异步处理，避免阻塞
**租户隔离**：基于collection_id实现多租户隔离

### 2.2 技术栈选择

```yaml
# 技术栈清单
orchestration: Prefect 3.0  # 工作流编排
databases:
  - PostgreSQL 16+          # 文档、实体、关系元数据存储  
  - Neo4j 5.x              # 知识图谱存储
  - Redis                  # 缓存和会话存储
vector_db: Qdrant          # 向量搜索
compute: Kubernetes        # 容器编排
language: Python 3.11+    # 开发语言
```

### 2.3 数据库设计

**PostgreSQL Schema**
```sql
-- 文档表
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    collection_id VARCHAR(100) NOT NULL,
    file_path VARCHAR(500),
    content TEXT,
    content_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_collection_id (collection_id)
);

-- 文档块表  
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    collection_id VARCHAR(100) NOT NULL,
    content TEXT,
    chunk_order INTEGER,
    tokens INTEGER,
    embedding VECTOR(1536), -- OpenAI embedding dimension
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 实体表
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    collection_id VARCHAR(100) NOT NULL,
    name VARCHAR(200) NOT NULL,
    entity_type VARCHAR(100),
    description TEXT,
    source_chunk_ids UUID[],
    embedding VECTOR(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(collection_id, name)
);

-- 关系表
CREATE TABLE relationships (
    id UUID PRIMARY KEY,
    collection_id VARCHAR(100) NOT NULL,
    source_entity_id UUID REFERENCES entities(id),
    target_entity_id UUID REFERENCES entities(id),
    relationship_type VARCHAR(100),
    description TEXT,
    keywords TEXT[],
    weight FLOAT DEFAULT 1.0,
    source_chunk_ids UUID[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Neo4j设计**
```cypher
// 节点约束
CREATE CONSTRAINT entity_collection IF NOT EXISTS 
FOR (e:Entity) REQUIRE (e.collection_id, e.name) IS UNIQUE;

// 实体节点
CREATE (e:Entity {
    id: $entity_id,
    collection_id: $collection_id,
    name: $name,
    type: $entity_type,
    description: $description,
    source_chunks: $source_chunk_ids
})

// 关系边
CREATE (source)-[r:RELATES {
    id: $relationship_id,
    collection_id: $collection_id,
    type: $relationship_type,
    description: $description,
    weight: $weight,
    keywords: $keywords
}]->(target)
```

## 3. 核心组件实现

### 3.1 项目结构

```
knowledge_graph_service/
├── src/
│   ├── core/
│   │   ├── models.py          # 数据模型定义
│   │   ├── chunker.py         # 文档分块器
│   │   ├── extractor.py       # 实体关系提取器
│   │   └── storage/
│   │       ├── postgres.py    # PostgreSQL操作
│   │       ├── neo4j.py       # Neo4j操作
│   │       └── vector.py      # 向量数据库操作
│   ├── workflows/
│   │   ├── ingestion.py       # 数据摄入工作流
│   │   ├── processing.py      # 处理工作流
│   │   └── querying.py        # 查询工作流  
│   ├── api/
│   │   ├── routers/
│   │   │   ├── collections.py # Collection管理API
│   │   │   ├── documents.py   # 文档管理API
│   │   │   └── queries.py     # 查询API
│   │   └── main.py            # FastAPI应用
│   └── config/
│       ├── settings.py        # 配置管理
│       └── database.py        # 数据库连接
├── k8s/
│   ├── prefect/              # Prefect部署配置
│   ├── databases/            # 数据库部署配置
│   └── api/                  # API服务部署配置
├── requirements.txt
└── README.md
```

### 3.2 数据模型定义

**src/core/models.py**
```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class DocumentChunk:
    id: UUID
    document_id: UUID
    collection_id: str
    content: str
    chunk_order: int
    tokens: int
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass  
class Entity:
    id: UUID
    collection_id: str
    name: str
    entity_type: str
    description: str
    source_chunk_ids: List[UUID]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class Relationship:
    id: UUID
    collection_id: str
    source_entity_id: UUID
    target_entity_id: UUID
    relationship_type: str
    description: str
    keywords: List[str]
    weight: float
    source_chunk_ids: List[UUID]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class Document:
    id: UUID
    collection_id: str
    file_path: str
    content: str
    content_hash: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()
        if self.created_at is None:
            self.created_at = datetime.utcnow()
```

### 3.3 无状态文档分块器

**src/core/chunker.py**
```python
import tiktoken
from typing import List
from .models import DocumentChunk, Document

class StatelessChunker:
    """无状态文档分块器，支持并发处理"""
    
    def __init__(
        self, 
        tokenizer_name: str = "cl100k_base",
        chunk_size: int = 1200,
        overlap_size: int = 100
    ):
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        将文档分块
        
        Args:
            document: 文档对象
            
        Returns:
            List[DocumentChunk]: 文档块列表
        """
        tokens = self.tokenizer.encode(document.content)
        chunks = []
        
        for index, start in enumerate(
            range(0, len(tokens), self.chunk_size - self.overlap_size)
        ):
            chunk_tokens = tokens[start:start + self.chunk_size]
            chunk_content = self.tokenizer.decode(chunk_tokens)
            
            chunk = DocumentChunk(
                id=None,  # 自动生成
                document_id=document.id,
                collection_id=document.collection_id,
                content=chunk_content.strip(),
                chunk_order=index,
                tokens=len(chunk_tokens)
            )
            chunks.append(chunk)
        
        return chunks
    
    def estimate_chunks_count(self, content: str) -> int:
        """估算文档会产生多少个块"""
        tokens = self.tokenizer.encode(content)
        return max(1, len(tokens) // (self.chunk_size - self.overlap_size) + 1)
```

### 3.4 实体关系提取器

**src/core/extractor.py**
```python
import asyncio
import re
from typing import List, Tuple, Dict, Any, Callable
from .models import DocumentChunk, Entity, Relationship

class StatelessEntityExtractor:
    """无状态实体关系提取器"""
    
    def __init__(
        self, 
        llm_func: Callable,
        max_concurrent: int = 8,
        max_gleaning: int = 1
    ):
        self.llm_func = llm_func
        self.max_concurrent = max_concurrent
        self.max_gleaning = max_gleaning
    
    async def extract_from_chunks(
        self, 
        chunks: List[DocumentChunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        从文档块批量提取实体和关系
        
        Args:
            chunks: 文档块列表
            
        Returns:
            Tuple[List[Entity], List[Relationship]]: 实体列表和关系列表
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_chunk_with_semaphore(chunk):
            async with semaphore:
                return await self._extract_from_single_chunk(chunk)
        
        # 并行处理所有chunks
        tasks = [process_chunk_with_semaphore(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集所有成功的结果
        all_entities = []
        all_relationships = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing chunk {i}: {result}")
                continue
                
            entities, relationships = result
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        # 合并同名实体和关系
        merged_entities = self._merge_entities(all_entities)
        merged_relationships = self._merge_relationships(all_relationships)
        
        return merged_entities, merged_relationships
    
    async def _extract_from_single_chunk(
        self, 
        chunk: DocumentChunk
    ) -> Tuple[List[Entity], List[Relationship]]:
        """从单个chunk提取实体和关系"""
        
        # 构建提取prompt
        prompt = self._build_extraction_prompt(chunk.content)
        
        # 调用LLM进行提取
        response = await self.llm_func(prompt)
        
        # 解析LLM响应
        entities, relationships = self._parse_llm_response(
            response, chunk.collection_id, [chunk.id]
        )
        
        # 可选：进行gleaning (多轮提取)
        if self.max_gleaning > 0:
            additional_entities, additional_relationships = await self._gleaning_extraction(
                chunk, entities, relationships
            )
            entities.extend(additional_entities)
            relationships.extend(additional_relationships)
        
        return entities, relationships
    
    def _build_extraction_prompt(self, content: str) -> str:
        """构建实体关系提取的prompt"""
        return f"""
You are an expert in extracting entities and relationships from text.

Extract all entities and relationships from the following text:

Text: {content}

Format your response as follows:

ENTITIES:
(entity_name|entity_type|entity_description)
...

RELATIONSHIPS:  
(source_entity|target_entity|relationship_type|relationship_description|keywords|weight)
...

Guidelines:
- Extract only factual entities and relationships
- Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, etc.
- Relationship weight should be between 0.1 and 1.0
- Keywords should be comma-separated relevant terms
- Use exact entity names from the ENTITIES section in RELATIONSHIPS

Text: {content}
"""

    def _parse_llm_response(
        self, 
        response: str, 
        collection_id: str,
        source_chunk_ids: List[str]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """解析LLM响应，提取实体和关系"""
        entities = []
        relationships = []
        
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.upper().startswith('ENTITIES:'):
                current_section = 'entities'
                continue
            elif line.upper().startswith('RELATIONSHIPS:'):
                current_section = 'relationships'
                continue
            
            # 解析实体
            if current_section == 'entities' and line.startswith('(') and line.endswith(')'):
                entity = self._parse_entity_line(line, collection_id, source_chunk_ids)
                if entity:
                    entities.append(entity)
            
            # 解析关系  
            elif current_section == 'relationships' and line.startswith('(') and line.endswith(')'):
                relationship = self._parse_relationship_line(line, collection_id, source_chunk_ids)
                if relationship:
                    relationships.append(relationship)
        
        return entities, relationships
    
    def _parse_entity_line(
        self, 
        line: str, 
        collection_id: str,
        source_chunk_ids: List[str]
    ) -> Optional[Entity]:
        """解析单行实体数据"""
        try:
            # 移除括号并分割
            content = line[1:-1]  # 移除()
            parts = [part.strip() for part in content.split('|')]
            
            if len(parts) < 3:
                return None
                
            return Entity(
                id=None,  # 自动生成
                collection_id=collection_id,
                name=parts[0],
                entity_type=parts[1],
                description=parts[2],
                source_chunk_ids=source_chunk_ids
            )
        except Exception:
            return None
    
    def _parse_relationship_line(
        self, 
        line: str, 
        collection_id: str,
        source_chunk_ids: List[str]
    ) -> Optional[Relationship]:
        """解析单行关系数据"""
        try:
            content = line[1:-1]  # 移除()
            parts = [part.strip() for part in content.split('|')]
            
            if len(parts) < 6:
                return None
                
            # 这里需要根据实体名称查找实体ID，简化处理
            # 在实际实现中需要更复杂的实体匹配逻辑
            return Relationship(
                id=None,  # 自动生成
                collection_id=collection_id,
                source_entity_id=None,  # 需要后续解析
                target_entity_id=None,  # 需要后续解析  
                relationship_type=parts[2],
                description=parts[3],
                keywords=parts[4].split(',') if parts[4] else [],
                weight=float(parts[5]) if parts[5].replace('.','').isdigit() else 1.0,
                source_chunk_ids=source_chunk_ids,
                # 临时存储实体名称，后续处理时转换为ID
                _source_entity_name=parts[0],
                _target_entity_name=parts[1]
            )
        except Exception:
            return None
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """合并同名实体"""
        entity_groups = {}
        
        for entity in entities:
            key = (entity.collection_id, entity.name.lower())
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        merged_entities = []
        for group in entity_groups.values():
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                # 合并多个同名实体
                merged = self._merge_entity_group(group)
                merged_entities.append(merged)
        
        return merged_entities
    
    def _merge_entity_group(self, entities: List[Entity]) -> Entity:
        """合并同名实体组"""
        base_entity = entities[0]
        
        # 合并描述
        descriptions = [e.description for e in entities if e.description]
        merged_description = '. '.join(descriptions)
        
        # 合并source_chunk_ids
        all_chunk_ids = []
        for entity in entities:
            all_chunk_ids.extend(entity.source_chunk_ids)
        unique_chunk_ids = list(set(all_chunk_ids))
        
        return Entity(
            id=base_entity.id,
            collection_id=base_entity.collection_id,
            name=base_entity.name,
            entity_type=base_entity.entity_type,
            description=merged_description,
            source_chunk_ids=unique_chunk_ids
        )
    
    def _merge_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """合并重复关系"""
        # 简化实现：基于source和target实体名称去重
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            key = (
                rel.collection_id,
                getattr(rel, '_source_entity_name', ''),
                getattr(rel, '_target_entity_name', ''),
                rel.relationship_type
            )
            
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        return unique_relationships

    async def _gleaning_extraction(
        self,
        chunk: DocumentChunk,
        existing_entities: List[Entity], 
        existing_relationships: List[Relationship]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """补充提取 (gleaning)"""
        # 简化实现，实际可以基于已有实体进行更深入的提取
        # 这里只是示例框架
        return [], []
```

### 3.5 Prefect工作流设计

**src/workflows/processing.py**
```python
from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
from typing import List, Dict, Any
import asyncio
from ..core.models import Document, DocumentChunk, Entity, Relationship
from ..core.chunker import StatelessChunker
from ..core.extractor import StatelessEntityExtractor
from ..core.storage.postgres import PostgresStorage
from ..core.storage.neo4j import Neo4jStorage
from ..core.storage.vector import VectorStorage

@task(retries=3, retry_delay_seconds=10)
async def chunk_documents_task(
    documents: List[Document],
    chunker_config: Dict[str, Any]
) -> List[DocumentChunk]:
    """文档分块任务"""
    chunker = StatelessChunker(**chunker_config)
    all_chunks = []
    
    for document in documents:
        chunks = chunker.chunk_document(document)
        all_chunks.extend(chunks)
    
    return all_chunks

@task(retries=3, retry_delay_seconds=10)
async def extract_entities_task(
    chunks: List[DocumentChunk],
    llm_func: callable,
    extractor_config: Dict[str, Any]
) -> tuple[List[Entity], List[Relationship]]:
    """实体关系提取任务"""
    extractor = StatelessEntityExtractor(llm_func, **extractor_config)
    return await extractor.extract_from_chunks(chunks)

@task(retries=3, retry_delay_seconds=10)
async def store_postgres_task(
    chunks: List[DocumentChunk],
    entities: List[Entity],
    relationships: List[Relationship],
    postgres_config: Dict[str, Any]
):
    """PostgreSQL存储任务"""
    storage = PostgresStorage(**postgres_config)
    
    await asyncio.gather(
        storage.store_chunks(chunks),
        storage.store_entities(entities),
        storage.store_relationships(relationships)
    )

@task(retries=3, retry_delay_seconds=10) 
async def store_neo4j_task(
    entities: List[Entity],
    relationships: List[Relationship],
    neo4j_config: Dict[str, Any]
):
    """Neo4j存储任务"""
    storage = Neo4jStorage(**neo4j_config)
    await storage.store_graph(entities, relationships)

@task(retries=3, retry_delay_seconds=10)
async def store_vectors_task(
    chunks: List[DocumentChunk],
    entities: List[Entity], 
    embedding_func: callable,
    vector_config: Dict[str, Any]
):
    """向量存储任务"""
    storage = VectorStorage(embedding_func, **vector_config)
    
    await asyncio.gather(
        storage.store_chunk_embeddings(chunks),
        storage.store_entity_embeddings(entities)
    )

@flow(
    name="process-single-collection",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
async def process_single_collection_flow(
    collection_id: str,
    documents: List[Document],
    config: Dict[str, Any]
):
    """处理单个collection的完整流程"""
    
    # 1. 文档分块
    chunks = await chunk_documents_task(documents, config["chunker"])
    
    # 2. 实体关系提取
    entities, relationships = await extract_entities_task(
        chunks, 
        config["llm_func"],
        config["extractor"]
    )
    
    # 3. 并行存储到各个数据库
    storage_tasks = [
        store_postgres_task(chunks, entities, relationships, config["postgres"]),
        store_neo4j_task(entities, relationships, config["neo4j"]),
        store_vectors_task(chunks, entities, config["embedding_func"], config["vector"])
    ]
    
    await asyncio.gather(*storage_tasks)
    
    return {
        "collection_id": collection_id,
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "entities_count": len(entities),
        "relationships_count": len(relationships)
    }

@flow(
    name="process-multiple-collections",
    log_prints=True
)
async def process_multiple_collections_flow(
    collections_data: Dict[str, List[Document]],
    config: Dict[str, Any],
    max_concurrent_collections: int = 10
):
    """并行处理多个collections"""
    
    semaphore = asyncio.Semaphore(max_concurrent_collections)
    
    async def process_with_semaphore(collection_id: str, documents: List[Document]):
        async with semaphore:
            return await process_single_collection_flow(
                collection_id, documents, config
            )
    
    # 创建所有collection处理任务
    tasks = [
        process_with_semaphore(collection_id, documents)
        for collection_id, documents in collections_data.items()
    ]
    
    # 并行执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 统计结果
    successful_results = [r for r in results if not isinstance(r, Exception)]
    failed_results = [r for r in results if isinstance(r, Exception)]
    
    return {
        "total_collections": len(collections_data),
        "successful_collections": len(successful_results),
        "failed_collections": len(failed_results),
        "results": successful_results,
        "errors": failed_results
    }
```

### 3.6 查询系统实现

**src/workflows/querying.py**
```python
from prefect import flow, task
from typing import List, Dict, Any, Optional
from ..core.storage.postgres import PostgresStorage
from ..core.storage.neo4j import Neo4jStorage  
from ..core.storage.vector import VectorStorage

@task
async def vector_search_task(
    query: str,
    collection_id: str,
    top_k: int,
    vector_config: Dict[str, Any],
    embedding_func: callable
) -> List[Dict[str, Any]]:
    """向量搜索任务"""
    storage = VectorStorage(embedding_func, **vector_config)
    return await storage.search_chunks(query, collection_id, top_k)

@task
async def graph_traversal_task(
    query: str,
    collection_id: str,
    max_depth: int,
    neo4j_config: Dict[str, Any]
) -> Dict[str, Any]:
    """图遍历任务"""
    storage = Neo4jStorage(**neo4j_config)
    return await storage.traverse_graph(query, collection_id, max_depth)

@task
async def generate_answer_task(
    query: str,
    context: Dict[str, Any],
    llm_func: callable
) -> str:
    """答案生成任务"""
    
    # 构建上下文
    context_text = ""
    
    # 添加相关文档块
    if "chunks" in context:
        context_text += "Relevant Documents:\n"
        for i, chunk in enumerate(context["chunks"][:5]):  # 限制前5个
            context_text += f"{i+1}. {chunk['content']}\n"
    
    # 添加相关实体
    if "entities" in context:
        context_text += "\nRelevant Entities:\n"
        for entity in context["entities"][:10]:  # 限制前10个
            context_text += f"- {entity['name']} ({entity['type']}): {entity['description']}\n"
    
    # 添加相关关系
    if "relationships" in context:
        context_text += "\nRelevant Relationships:\n"
        for rel in context["relationships"][:10]:  # 限制前10个
            context_text += f"- {rel['source']} → {rel['target']}: {rel['description']}\n"
    
    # 构建最终prompt
    prompt = f"""
Based on the following context, answer the question accurately and comprehensively.

Context:
{context_text}

Question: {query}

Answer:
"""
    
    return await llm_func(prompt)

@flow(name="query-collection")
async def query_collection_flow(
    query: str,
    collection_id: str, 
    config: Dict[str, Any],
    top_k: int = 10,
    max_depth: int = 2
) -> Dict[str, Any]:
    """查询单个collection"""
    
    # 并行执行向量搜索和图遍历
    vector_results, graph_results = await asyncio.gather(
        vector_search_task(
            query, collection_id, top_k, 
            config["vector"], config["embedding_func"]
        ),
        graph_traversal_task(
            query, collection_id, max_depth,
            config["neo4j"]
        )
    )
    
    # 组合上下文
    context = {
        "chunks": vector_results,
        "entities": graph_results.get("entities", []),
        "relationships": graph_results.get("relationships", [])
    }
    
    # 生成答案
    answer = await generate_answer_task(
        query, context, config["llm_func"]
    )
    
    return {
        "query": query,
        "collection_id": collection_id,
        "answer": answer,
        "context": context,
        "sources": [chunk.get("file_path", "unknown") for chunk in vector_results]
    }

@flow(name="multi-collection-query")
async def multi_collection_query_flow(
    query: str,
    collection_ids: List[str],
    config: Dict[str, Any],
    top_k_per_collection: int = 5
) -> Dict[str, Any]:
    """跨多个collection查询"""
    
    # 并行查询所有collections
    query_tasks = [
        query_collection_flow(
            query, collection_id, config, top_k_per_collection
        )
        for collection_id in collection_ids
    ]
    
    results = await asyncio.gather(*query_tasks)
    
    # 合并和排序结果
    all_contexts = []
    all_sources = []
    
    for result in results:
        all_contexts.append(result["context"])
        all_sources.extend(result["sources"])
    
    # 生成综合答案
    combined_context = {
        "chunks": [],
        "entities": [],
        "relationships": []
    }
    
    for ctx in all_contexts:
        combined_context["chunks"].extend(ctx["chunks"])
        combined_context["entities"].extend(ctx["entities"]) 
        combined_context["relationships"].extend(ctx["relationships"])
    
    final_answer = await generate_answer_task(
        query, combined_context, config["llm_func"]
    )
    
    return {
        "query": query,
        "collection_ids": collection_ids,
        "answer": final_answer,
        "individual_results": results,
        "sources": list(set(all_sources))
    }
```

## 4. K8s部署架构

### 4.1 整体部署架构

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: knowledge-graph
---
# k8s/prefect/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prefect-server
  namespace: knowledge-graph
spec:
  replicas: 2
  selector:
    matchLabels:
      app: prefect-server
  template:
    metadata:
      labels:
        app: prefect-server
    spec:
      containers:
      - name: prefect-server
        image: prefecthq/prefect:3.0
        ports:
        - containerPort: 4200
        env:
        - name: PREFECT_SERVER_API_HOST
          value: "0.0.0.0"
        - name: PREFECT_API_DATABASE_CONNECTION_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: prefect-db-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi" 
            cpu: "500m"
---
# k8s/prefect/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prefect-worker
  namespace: knowledge-graph
spec:
  replicas: 10  # 根据处理需求调整
  selector:
    matchLabels:
      app: prefect-worker
  template:
    metadata:
      labels:
        app: prefect-worker
    spec:
      containers:
      - name: prefect-worker
        image: your-registry/kg-worker:latest
        env:
        - name: PREFECT_API_URL
          value: "http://prefect-server:4200/api"
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: postgres-url
        - name: NEO4J_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: neo4j-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: temp-storage
          mountPath: /tmp
      volumes:
      - name: temp-storage
        emptyDir: {}
---
# k8s/api/deployment.yaml  
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kg-api
  namespace: knowledge-graph
spec:
  replicas: 5
  selector:
    matchLabels:
      app: kg-api
  template:
    metadata:
      labels:
        app: kg-api
    spec:
      containers:
      - name: kg-api
        image: your-registry/kg-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: PREFECT_API_URL
          value: "http://prefect-server:4200/api"
        - name: POSTGRES_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: postgres-url
        - name: NEO4J_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: neo4j-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 4.2 数据库部署

```yaml
# k8s/databases/postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: knowledge-graph
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: pgvector/pgvector:pg16
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: knowledge_graph
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: postgres-user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: postgres-password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
---
# k8s/databases/neo4j-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet  
metadata:
  name: neo4j
  namespace: knowledge-graph
spec:
  serviceName: neo4j
  replicas: 1
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.15-community
        ports:
        - containerPort: 7474
        - containerPort: 7687
        env:
        - name: NEO4J_AUTH
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: neo4j-auth
        - name: NEO4J_PLUGINS
          value: '["apoc","graph-data-science"]'
        - name: NEO4J_dbms_memory_heap_initial__size
          value: "2G"
        - name: NEO4J_dbms_memory_heap_max__size
          value: "4G"
        - name: NEO4J_dbms_memory_pagecache_size
          value: "2G"
        volumeMounts:
        - name: neo4j-data
          mountPath: /data
        - name: neo4j-logs
          mountPath: /logs
        resources:
          requests:
            memory: "6Gi"
            cpu: "2" 
          limits:
            memory: "12Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: neo4j-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
  - metadata:
      name: neo4j-logs
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

## 5. 实施路线图

### 5.1 开发阶段

**第一阶段：核心组件开发 (2-3周)**
1. 实现数据模型和数据库Schema
2. 开发无状态分块器和提取器
3. 实现PostgreSQL和Neo4j存储层
4. 基础单元测试

**第二阶段：Prefect工作流开发 (2周)**
1. 实现文档处理工作流
2. 实现查询工作流
3. 错误处理和重试机制
4. 工作流集成测试

**第三阶段：API和部署 (2周)**
1. FastAPI接口开发
2. K8s配置文件编写
3. Docker镜像构建
4. 端到端测试

### 5.2 性能优化

**批处理优化**
- 文档批量分块：每批100-500个文档
- 实体提取并发：每个worker 8-16个并发任务
- 数据库批量写入：每批1000条记录

**缓存策略**
- LLM响应缓存：Redis存储提取结果
- 向量嵌入缓存：避免重复计算
- 查询结果缓存：常见查询结果缓存

**监控指标**
- 处理吞吐量：文档/分钟、实体/分钟
- 系统资源：CPU、内存、磁盘IO
- 任务状态：成功率、失败率、重试次数
- 查询性能：响应时间、准确率

### 5.3 扩展策略

**水平扩展**
- Prefect Worker自动扩缩容
- API服务弹性伸缩
- 数据库读副本扩展

**垂直优化**
- GPU加速嵌入计算
- 分布式向量数据库
- 图数据库集群模式

## 6. 关键优势总结

| 特性 | LightRAG | 自实现方案 |
|------|----------|------------|
| **并发处理** | ❌ 全局锁限制 | ✅ 真正无状态并发 |
| **多租户** | ❌ 不支持 | ✅ Collection级隔离 |
| **可扩展性** | ❌ 单实例限制 | ✅ K8s水平扩展 |
| **可观测性** | ⚠️ 有限监控 | ✅ Prefect完整监控 |
| **错误恢复** | ⚠️ 全局状态风险 | ✅ 任务级重试隔离 |
| **SaaS就绪** | ❌ 不适合 | ✅ 原生SaaS架构 |

这个自实现方案完全解决了LightRAG的架构限制，提供了生产级的高并发知识图谱构建能力，完全满足您的SaaS服务需求。