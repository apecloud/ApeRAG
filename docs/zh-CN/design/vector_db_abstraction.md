# 向量数据库抽象层设计分析（ApeRAG）

> Status: **M2 已落地**（本文的分析已经变成了现网代码）。后续 pgvector 的
> M3 工作仍在 roadmap 里，会按本文的分层和 DSL 约束来实现。

## 变更记录

| 日期 | 内容 |
|---|---|
| 2026-04-20 | 初稿：分层、DSL、三后端草图、路线图 |
| 2026-04-21 | M2 落地：`VectorFilter` DSL、Qdrant translator、`VectorPoint`、client pool、`retrieve()` 入基类 |

## 1. 背景与目标

当前 ApeRAG 只有一个向量后端：Qdrant。在
[`qdrant_memory_optimization.md`](./qdrant_memory_optimization.md) 的改造后，
Qdrant 连接器已经承担了三件相互耦合的事情：

1. **物理布局路由**：按 `(vector_size, distance)` 选全局 collection。
2. **多租户语义**：payload `collection_id` + `is_tenant` 索引。
3. **存储与索引调优**：INT8 量化、HNSW on_disk、segment 数、memmap 阈值等。

未来我们希望：

- 线上大集群保留 Qdrant（性能/隔离/量化成熟）；
- 中小规模或需要单一 PostgreSQL 技术栈的部署可以走 **pgvector**（少一个组件）；
- 未来潜在需求（超大规模、GPU 索引）可切换 **Milvus**；
- 对于"快速验证"或 CI/本地开发，期望一个纯内存实现（`:memory:` 或 in-proc）。

所以需要一个**向上/向下双向稳定**的抽象层。本文给出选型、接口草案与迁移策略。

---

## 2. 当前代码实际的抽象现状（事实核查）

| 层 | 文件 | 角色 | 健康度 |
|---|---|---|---|
| 抽象基类 | `aperag/vectorstore/base.py`：`VectorStoreConnector` | `search / delete / create_collection / delete_collection` | 不完整：`retrieve()` 只在 Qdrant 实现里有 |
| 适配器 | `aperag/vectorstore/connector.py`：`VectorStoreConnectorAdaptor` | `match vector_store_type` 分发 | 只支持 qdrant 分支 |
| 具体实现 | `aperag/vectorstore/qdrant_connector.py` | Qdrant | 承载多租户 + 优化所有细节 |
| 上层过滤器 | `aperag/context/context.py`：`ContextManager._create_combined_filter` | 业务过滤 | **直接 import `qdrant_client.models`**，抽象破口 |
| 上层入口 | 索引写入 `aperag/index/*.py`、检索 `aperag/service/search_pipeline_service.py` | 每次按需构造 `VectorStoreConnectorAdaptor`、`ContextManager` | 每次请求重建连接 |
| 分片路由 | `aperag/config.py`：`build_vector_db_context`、`get_vector_db_connector` | 注入 `multitenant/quantization/...` ctx | Qdrant 专属字段混在 Config 里 |

**当前抽象的 4 个真实缺口**（2026-04-20 分析；M2 之后 1/2/4 已关闭，3 仍保留）：

| # | 缺口 | M2 状态 |
|---|---|---|
| 1 | `retrieve(ids)` 只在 Qdrant 连接器上存在 | ✅ 已进基类，返回 `VectorPoint` |
| 2 | `ContextManager` 硬编码 `qdrant_client.models` | ✅ 已迁到 `VectorFilter` DSL |
| 3 | `LlamaIndex.QdrantVectorStore` 被直接暴露给业务层 | ⏸ 保留，等 M3 |
| 4 | 每次 search 重建 `QdrantClient` | ✅ 已进程级复用 |

缺口 3 的完整细节（保留给 M3）：

- `vector_store_adaptor.connector.store.add(nodes)` 仍在 vision_index.py
  两处、embedding_utils.py 使用；
- 写入 payload 依旧会被 LlamaIndex 注入 `_node_content` / `doc_id` /
  `document_id` / `ref_doc_id`；
- 这些字段是 Qdrant 后端的**实现细节**，在 M3 引入 pgvector 时，我们会
  提供一个新的 `connector.upsert(points)` 接口替代 `store.add(nodes)`，
  届时再一次性把 LlamaIndex 从业务写路径上拆掉。

---

## 3. 三个候选后端的能力矩阵

| 能力 | Qdrant ≥1.11 | pgvector (pg 16 + pgvector 0.7+) | Milvus 2.4+ |
|---|---|---|---|
| 原生 payload 过滤（keyword match/IN/OR） | ✅ | ✅ (`WHERE`) | ✅ (expr) |
| 向量量化（int8 / binary） | ✅ 内置 scalar+binary | `halfvec`(fp16)、`bit`、需手工 | ✅ IVF/HNSW+PQ/SQ |
| HNSW on_disk / mmap | ✅ | HNSW 常驻内存，不支持 on_disk；表/toast 走 OS page cache | ✅（DiskANN / MMAP） |
| 多租户 defragmentation | ✅ `is_tenant` payload | 需手工（按租户建 partitioned table） | ✅ partition/collection-per-tenant |
| 单机 in-proc 测试 | ✅ `:memory:` | 需起 pg | ✅ `milvus-lite` |
| Hybrid search（BM25 + 向量） | ✅（sparse vector + fusion） | `tsvector + vector` 手工 | ✅（sparse vector） |
| 操作学习成本 | 中（一套独立 API） | 低（就是 SQL） | 高（大量配置项） |
| 与现有技术栈耦合 | 独立服务 | ApeRAG 已有 PG，**省一个组件** | 独立服务 |

**结论**：
- Qdrant 仍然是**生产默认**（已对量化 + 多租户做过深度优化）。
- pgvector 适合作为**"少组件"部署模式**的一等公民（比如 ApeRAG-Lite、CI、自托管用户）。
- Milvus 作为**长期备选**（大规模 / GPU 加速场景），短期不投入实现。

---

## 4. 抽象层分层设计

建议抽象分成三层：

### 4.1 Transport / 连接管理层（L0）

- 责任：客户端生命周期、连接池、池化单例、健康探测。
- **不涉及任何数据语义**。
- 代码位置建议：`aperag/vectorstore/_client.py`（新增）。

```python
class VectorClientFactory:
    def get(self, backend: str, dsn_ctx: dict) -> Any:
        """Return a cached, process-level client. Thread-safe, idempotent."""
```

目的：修掉第 2.4 条"每次 search 重建 client"的性能缺口。

### 4.2 能力抽象层（L1）

- 责任：**业务侧能理解的向量操作**，与具体后端无关。
- 代码位置：`aperag/vectorstore/base.py`（扩展现有接口）。

建议的接口草案：

```python
class VectorStoreConnector(ABC):
    # ---- 集合 / 租户管理 ----
    @abstractmethod
    def ensure_tenant(self, tenant: TenantRef, shape: VectorShape) -> None: ...
    @abstractmethod
    def drop_tenant(self, tenant: TenantRef, *, purge_all_shapes: bool = False) -> None: ...

    # ---- 写入 ----
    @abstractmethod
    def upsert(self, tenant: TenantRef, points: Iterable[VectorPoint]) -> list[str]: ...
    @abstractmethod
    def delete_by_ids(self, tenant: TenantRef, ids: Sequence[str]) -> None: ...
    @abstractmethod
    def delete_by_filter(self, tenant: TenantRef, flt: VectorFilter) -> None: ...

    # ---- 查询 ----
    @abstractmethod
    def search(
        self, tenant: TenantRef, query: QueryWithEmbedding,
        *, flt: Optional[VectorFilter] = None,
        score_threshold: float | None = None,
        search_opts: SearchOptions | None = None,
    ) -> QueryResult: ...

    @abstractmethod
    def retrieve(self, tenant: TenantRef, ids: Sequence[str], *,
                 with_vectors: bool = False) -> list[VectorPoint]: ...
```

四个配套 DTO（统一跨后端的数据模型）：

```python
@dataclass(frozen=True)
class TenantRef:
    id: str  # ApeRAG collection id (or None if embed-only mode)
    shape: VectorShape | None = None  # optional hint

@dataclass(frozen=True)
class VectorShape:
    size: int
    distance: Literal["cosine", "dot", "euclid"]

@dataclass
class VectorPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]
    # on read: score optional
    score: float | None = None

class VectorFilter:
    """Backend-neutral filter tree. See §4.3."""
```

这一层要**严格禁止**暴露 LlamaIndex / Qdrant / psycopg 类型。

### 4.3 业务过滤器 DSL（L1 子模块）

**最痛的一处**：当前 `ContextManager._create_combined_filter` 直接构造
`qdrant_client.models.Filter`。pgvector/Milvus 无法复用。

建议一个极小的、**面向 RAG 使用场景**的过滤 DSL：

```python
@dataclass(frozen=True)
class Eq:      key: str;           value: str | int | float
@dataclass(frozen=True)
class In:      key: str;           values: Sequence[str | int | float]
@dataclass(frozen=True)
class IsEmpty: key: str
@dataclass(frozen=True)
class And:     parts: Sequence["VectorFilter"]
@dataclass(frozen=True)
class Or:      parts: Sequence["VectorFilter"]
@dataclass(frozen=True)
class Not:     inner: "VectorFilter"

VectorFilter = Union[Eq, In, IsEmpty, And, Or, Not]
```

每个后端有自己的 **translator**：

| DSL 节点 | Qdrant | pgvector | Milvus |
|---|---|---|---|
| `Eq(k, v)` | `FieldCondition(key=k, match=MatchValue(v))` | `payload->>'k' = :v` | `k == v` |
| `In(k, V)` | `FieldCondition(key=k, match=MatchAny(V))` | `payload->>'k' = ANY(:V)` | `k in [...]` |
| `IsEmpty(k)` | `IsEmptyCondition(PayloadField(k))` | `NOT (payload ? 'k')` | `not (exists k)` |
| `And` | `Filter(must=[...])` | `AND` | `&&` |
| `Or` | `Filter(should=[...])` | `OR` | `\|\|` |
| `Not` | `Filter(must_not=[...])` | `NOT` | `!` |

**多租户守卫自然内建**：Connector 在 `search / delete_by_filter` 时自动 `And(Eq("collection_id", tenant.id), user_filter)`；业务层完全无感。

### 4.4 存储/索引调优层（L2）

每个后端独有。不对外暴露。

- Qdrant: 当前的 `_hnsw_config / _optimizers_config / _quantization_config` 保留在该后端的 `*_connector.py` 中，通过 `SearchOptions.hints` 透传。
- pgvector: `CREATE INDEX ... USING hnsw WITH (m=16, ef_construction=64)`，量化用 `halfvec`。
- Milvus: `IVF_FLAT/HNSW/DISKANN` + PQ/SQ 参数。

L2 的配置**不**暴露给 L1 调用方；只暴露一个能力类的 `SearchOptions`：

```python
@dataclass
class SearchOptions:
    top_k: int
    consistency: Literal["eventual", "majority", "strong"] = "majority"
    hints: dict[str, Any] = field(default_factory=dict)   # 后端白名单参数
```

---

## 5. 三种后端的实现草图

### 5.1 Qdrant（现有 → 重构成 L1）

- `ensure_tenant` → 现有 `_ensure_collection` + `_ensure_tenant_payload_index`。
- `upsert` → 包装 `self.client.upsert`（不再经由 LlamaIndex `store.add`，从而解除 `_node_content` 依赖；或 keep-as-is 并在读路径做兼容）。
- 过滤器 translator：`_to_qdrant_filter(vf: VectorFilter) -> rest.Filter`。
- SearchOptions.hints 支持 `hnsw_ef`、`exact`。

### 5.2 pgvector（新增，新增包 `aperag/vectorstore/pgvector_connector.py`）

**Schema 设计**：

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- 每个 (size, distance) 一张表，类比 Qdrant 的多租户 collection
CREATE TABLE aperag_vectors_1024_cosine (
    id          UUID PRIMARY KEY,
    tenant_id   TEXT NOT NULL,               -- = ApeRAG collection id
    embedding   vector(1024) NOT NULL,
    payload     JSONB NOT NULL
);

-- 租户过滤 + HNSW 共用。分区表可选（pg 14+）。
CREATE INDEX ON aperag_vectors_1024_cosine (tenant_id);
CREATE INDEX ON aperag_vectors_1024_cosine USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);
-- JSONB 字段上的 GIN 索引供 payload 过滤
CREATE INDEX ON aperag_vectors_1024_cosine USING GIN (payload);
```

**搜索**：

```sql
SELECT id, payload, embedding <=> :q AS score
  FROM aperag_vectors_1024_cosine
 WHERE tenant_id = :tenant
   AND payload @> :payload_filter  -- 由 VectorFilter translator 生成
 ORDER BY embedding <=> :q
 LIMIT :k;
```

**量化**：用 `halfvec(1024)` 列替代 `vector(1024)`，磁盘/内存减半；搜索时
`HALFVEC_L2_OPS` / `HALFVEC_COSINE_OPS`。`bit(dim)` 也可用作更激进量化。

**多租户 defrag**：pg 14+ 可按 `tenant_id` 做 **list partitioning**，每个租户一张物理表，HNSW 索引天然按分区切分。初期不做，租户数量到万级再评估。

### 5.3 Milvus（未来）

- `DataType.FLOAT_VECTOR` + `IVF_SQ8 / HNSW`；
- payload 字段走 scalar field（`VarChar(tenant_id)`）；
- 每个 `(vector_size, distance)` 一个 collection，与 Qdrant 对齐；
- partition key = `tenant_id`（近似 Qdrant 的 `is_tenant`）。

只在需要时实现，目前不动。

---

## 6. 路线图（建议）

分成 4 个 milestone，每个都能**独立上线、独立回滚**：

### M1（本 PR 已落地的改造不变）

- Qdrant 多租户 + 量化 + embedding 锁定。
- 本文档纳入 repo，作为后续决策的依据。

### M2：抽象层最小可行（✅ **已落地**）

实际落地的范围（比最初草案更克制，符合 "零答疑" 原则）：

- ✅ `VectorFilter` DSL（`aperag/vectorstore/filters.py`）：`Eq / In /
  IsEmpty / And / Or / Not` + `all_of / any_of` 短路 helper。
- ✅ Qdrant translator（`_translate_filter` / `_normalize_filter_input`
  在 `qdrant_connector.py`）。兼容旧的 `rest.Filter` 直传（迁移脚本在用）。
- ✅ `ContextManager._create_combined_filter` 改为产出 DSL，**不再** import
  `qdrant_client.models`。
- ✅ `VectorStoreConnector` 基类补 `retrieve()` 抽象方法。
- ✅ 只引入一个最小 DTO `VectorPoint(id, payload, vector?)` 给 `retrieve()`
  用；没有引入 `TenantRef / VectorShape / QueryResult / SearchOptions`，
  避免"只有一个实现时多 DTO 的人生思考题"。
- ✅ `QdrantClient` 按 endpoint 进程级复用
  （`_get_or_create_client` + 双检锁）；`:memory:` 显式绕开缓存以保护测试隔离。
- **没做** 的事：解耦 LlamaIndex（`store.add(nodes)` 和
  `node_to_metadata_dict` 在当前是 Qdrant 后端的实现细节，等真做
  pgvector 的时候再一起换）；没引入统一的 `upsert(tenant, points)`
  写接口——目前的 `store.add` + `delete(ids)` 已经够用，M3 再说。

实际代码规模：5 个新文件 + 3 个现有文件重构；测试新增 `test_filters.py` /
`test_qdrant_filter_translation.py` / `test_qdrant_client_cache.py` /
`test_context_manager_filter.py`。

### M3：pgvector 实现（3~4 周）

- 新增 `pgvector_connector.py`，复用 `aperag/db/ops` 的 session 池。
- 新增 `VECTOR_DB_TYPE=pgvector` 路径；migration 模板（`alembic` 脚本建立
  每个需要的 `aperag_vectors_{size}_{distance}` 表）。
- Reuse embedding lock 的前端逻辑（和 Qdrant 完全一致）。
- 性能 benchmark：10 万 / 100 万 vectors 的 latency 对比报告。

### M4：生产切换策略（按需）

- 配置：`VECTOR_DB_TYPE=qdrant|pgvector`。
- 混合部署期间支持"读双写单"的数据迁移模式（类似本 PR 的
  `scripts/migrate_qdrant_multitenancy.py`）。
- Milvus 留作后续调研，不进本 roadmap。

---

## 7. 风险与取舍

| 风险 | 说明 | 缓解 |
|---|---|---|
| 抽象稀释 Qdrant 独有能力 | 如 sparse vector / hybrid search / HNSW ef_construct | `SearchOptions.hints` 留后门；新能力进抽象需要 ≥2 后端支持 |
| pgvector 高 QPS 表现不如 Qdrant | pg 进程级锁、HNSW 全量常驻内存 | M3 benchmark 门槛；低 QPS 场景优先 |
| 过滤 DSL 表达不完备 | 早期只支持 Eq/In/IsEmpty/And/Or/Not，没有 range/geo | 按需扩展，保持单一职责 |
| LlamaIndex 直接耦合 | `node_to_metadata_dict` 在 payload 里塞大量 `_node_content` 等字段 | M2 中 Qdrant 改为直接走 `client.upsert`，不再经 `LlamaIndex.QdrantVectorStore.add`；读路径做兼容 |
| embedding 模型切换 vs 抽象层 | 不同 backend 对重建索引成本不同 | 本次 embedding 锁定（见 §8）规避了该问题，抽象层之后也无需处理 |

---

## 8. 与"embedding 模型锁定"的关系

本 PR 引入的 embedding 锁定（`CollectionService._reject_embedding_change`）
是抽象层设计的**前置条件**：

- 没锁定：每个后端都要实现"向量维度变化时的数据迁移"——pgvector 要 alter
  column、Milvus 要 drop+recreate collection、Qdrant 要 scroll+upsert。
  每种都是大手术。
- 锁定后：向量维度 = collection 生命周期常量，后端只要专注"同维度高效检索"。

因此抽象层假定 `VectorShape` 在 `ensure_tenant` 之后不可变。

---

## 9. 开放问题

1. **是否把 fulltext/knowledge graph 也抽象到同一层？** 目前 ApeRAG 有
   - vector: Qdrant
   - fulltext: Elasticsearch
   - graph: LightRAG (存在 Postgres/Neo4j)
   三者 indexing 流程在 `aperag/index/` 已有 `BaseIndexer`，各自独立——**不建议**
   把它们塞进向量抽象层，保持单一职责。

2. **是否与 LlamaIndex 解耦？** LlamaIndex 带来了：chunking、node schema、
   retrieval pipeline。短期我们只想解 `QdrantVectorStore` 这一处。长期如果
   想支持多种 chunker（比如 unstructured、langchain），才值得把 LlamaIndex
   也抽象掉。

3. **是否支持"多向量字段"？** 如 dense + sparse 混合检索。当前架构
   `VectorShape(size, distance)` 只支持单向量；未来要扩展成
   `VectorShape(fields: Mapping[str, VectorField])`。

---

## 10. 附：何时**不**应该做这个抽象

如果短期内只跑 Qdrant，不会切换，做这层抽象就是**过度设计**。

要启动 M2+ 的触发条件（满足任一即可）：

- 用户明确要求某个部署不能依赖 Qdrant。
- 我们要做 "ApeRAG-Lite"（只依赖 PG + Redis 的极简部署包）。
- Qdrant 成本/运维成为增长瓶颈。

当前三者都还没触发——这份文档的作用是**提前想清楚路径**，等触发发生时
不用从零开始设计。
