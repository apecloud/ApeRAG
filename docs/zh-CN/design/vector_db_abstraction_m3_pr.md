# 向量数据库抽象层 M3（pgvector + 抽象层补完）— PR 说明

本 PR 在 M2 的基础上一次性做完了两件事：

1. **M3：pgvector 后端实现**。ApeRAG 现在支持 `VECTOR_DB_TYPE=qdrant |
   pgvector` 两种后端，默认 Qdrant。pgvector 版本与 Qdrant 功能对等，
   共用同一套 `VectorFilter` DSL、租户隔离语义、embedding 锁定。
2. **M2 的补完**：原本推到 M3 的 "DTO 全套" 和 "去 LlamaIndex 写依赖" 两
   项尾巴，这次一起做掉了。不留代码债、不留下次返工。

**合并即可用。** 没有新增 env、也没有任何必要的运维动作（除非你想启用
pgvector —— 那就翻一个 env flag 就行）。

## 为什么不留尾巴

上一轮我故意保守——"只有 Qdrant 一个后端时多 DTO 是答疑负担"。这次不再
成立：

- pgvector 真的要上了，两个后端的存在让 DTO 全套**立刻**有收益（避免
  per-callsite 分支）；
- LlamaIndex 的 `QdrantVectorStore.add(nodes)` 在引入 pgvector 时是
  硬阻塞（pgvector 没有对应 adapter）；
- 等下次 PR 再拆这两件事，就意味着本次 pgvector 落地时要么**复制**
  LlamaIndex 的 `_node_content` 序列化约定到 pgvector，要么引入一个
  LlamaIndex `PGVectorStore` 作为新依赖——都是"现在不痛、将来痛"的决策。
  趁还没分叉先拆掉，代价最低。

所以本 PR 的核心口号是："一次到位"。

## 抽象层现状（合并后）

```text
aperag/vectorstore/
├── base.py                    # VectorStoreConnector 抽象；签名全部 DTO 化
├── dto.py                     # TenantRef, VectorShape, VectorPoint,
│                              # QueryRequest, SearchHit, flatten_node_payload
├── filters.py                 # VectorFilter DSL: Eq/In/IsEmpty/And/Or/Not
├── connector.py               # 适配器：match vector_store_type
├── qdrant_connector.py        # Qdrant 实现 + DSL translator
├── pgvector_connector.py      # pgvector 实现 + SQL translator (新)
└── llama_index_adapter.py     # BaseNode -> VectorPoint (新)
```

契约（`base.py`）现在是：

```python
class VectorStoreConnector(ABC):
    @property
    @abstractmethod
    def tenant(self) -> TenantRef: ...
    @property
    @abstractmethod
    def shape(self) -> VectorShape: ...

    @abstractmethod
    def ensure_collection(self) -> None: ...
    @abstractmethod
    def drop_tenant(self, *, purge_all_shards: bool = False) -> None: ...

    @abstractmethod
    def upsert(self, points: Sequence[VectorPoint]) -> list[str]: ...
    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None: ...
    @abstractmethod
    def delete_by_filter(self, flt: VectorFilter) -> None: ...

    @abstractmethod
    def search(self, request: QueryRequest) -> list[SearchHit]: ...
    @abstractmethod
    def retrieve(self, ids: Sequence[str], *, with_vectors: bool = False) -> list[VectorPoint]: ...
```

九个方法，全部 DTO 化、全部不携带任何后端特征。

## pgvector 关键设计决策

### 部署形态：默认复用主 Postgres

- 默认 `PGVECTOR_DATABASE_URL=` 留空 → 使用 ApeRAG 主 DB (`DATABASE_URL`)。
- "私有化交付 / ApeRAG-Lite" 场景少一个组件，**零部署负担**就是抽象层
  的主要卖点。
- 规模上去了？设 `PGVECTOR_DATABASE_URL=postgresql://...` 指到独立 PG，
  一个 env 搞定，应用代码不动。

### Schema：动态建表、对齐 Qdrant 命名

每个 `(vector_size, distance)` 对一张 `aperag_vectors_<size>_<distance>`
表，与 Qdrant 的物理分片命名完全一致。这不是偶然——`purge_all_shards`
这类运维逻辑现在**两后端走同一条代码路径**：按 `aperag_vectors_*` 前缀
扫分片，按 `tenant_id` 删。

表结构：

```sql
CREATE TABLE aperag_vectors_1024_cosine (
    id          UUID PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    embedding   vector(1024) NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ... ON ... (tenant_id);
CREATE INDEX ... ON ... USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=64);
CREATE INDEX ... ON ... USING GIN (payload);
```

三个索引各司其职：tenant_id 用 B-tree 撑住"按租户 DELETE / SELECT"；
HNSW 用于近邻；GIN(payload) 让 DSL filter 能被 JSONB 索引加速。

**动态建表 vs migration 文件**：我选前者。原因：
- `(size, distance)` 组合是用户选 embedding model 时决定的，**不是**
  部署时间常量。主 alembic 迁移如果要包含所有可能组合，意味着要么
  硬编码 (1024, 1536, 3072) × (cosine, dot, euclid)，要么 migration
  失控。
- 连接器的 `ensure_collection` 幂等 + 进程级缓存，首次写时 ~20ms 延迟
  一次，之后零开销；这和 Qdrant 的 `collection_exists ? no-op :
  create_collection` 完全对称。
- 运维视角统一："哪里来的表？" → `aperag_vectors_*`；"按什么删？" →
  `tenant_id + id / filter`；"怎么清空某个 Collection？" → `DELETE
  FROM ... WHERE tenant_id = :t`。这些都是**查看即懂**的操作。

### SQL filter translator

`VectorFilter` DSL → `(where_sql, bind_params)`：

| DSL | SQL |
|---|---|
| `Eq(k, v)` | `payload->>'k' = :f0` |
| `In(k, [a, b])` | `payload->>'k' IN (:f0, :f1)` |
| `IsEmpty(k)` | `NOT (payload ? 'k') OR payload->'k' = 'null'::jsonb` |
| `And(...)` | `(...parts join ' AND '...)` |
| `Or(...)` | `(...parts join ' OR '...)` |
| `Not(inner)` | `NOT (...)` |

**所有值走 bind 参数**；JSON key 走白名单正则校验（见
`_escape_json_key`）。SQL 注入面彻底关死。

### 距离语义：统一"分数越高越好"

| 距离 | HNSW opclass | PG 操作 | 评分表达式 |
|---|---|---|---|
| cosine | `vector_cosine_ops` | `<=>` | `1 - (embedding <=> :q)` |
| euclid | `vector_l2_ops` | `<->` | `-(embedding <-> :q)` |
| dot | `vector_ip_ops` | `<#>` | `-(embedding <#> :q)` |

这让 `QueryRequest.score_threshold` 对三种距离都是"higher-is-better"
的一致语义，调用方无需分支。

### `CAST(:q AS vector)` 小坑

SQLAlchemy `text()` 的 `:name::vector` 写法在 psycopg2 下会被误解析
（部分参数被替换、部分原样保留，SQL 报错）。正确写法是 `CAST(:q AS
vector)`——在 `_DISTANCE_SPEC` 和 upsert 的 VALUES 里都已改用 CAST。
留了详细注释，后来人不会再踩。

## Qdrant 连接器的同步改造

为保持契约对称，Qdrant 连接器也一起改了：

- **原生 `upsert`**：`client.upsert(...)` 直接写，不再走 LlamaIndex
  `QdrantVectorStore.add(nodes)`。
- **`delete_by_filter`**：新方法，自动 AND tenant 守卫；空过滤器被显式
  rejected（否则在多租户下会 silently 等价于 `drop_tenant`）。
- **`retrieve` 返回 `List[VectorPoint]`**：`id` 归一化为 `str`、vector
  归一化为 `list[float]`。
- **`drop_tenant` 替代 `delete_collection`**：命名更准确。
- **去掉 `self.store`**：LlamaIndex `QdrantVectorStore` 不再在连接器里
  存在。`vector_store_adaptor.connector.store.add(...)` 这种写法彻底
  失效。

## 业务层改造

三个写入点 + 两个读取点：

| 位置 | 改造 |
|---|---|
| `embedding_utils.py::create_embeddings_and_store` | `store.add(nodes)` → `nodes_to_vector_points(nodes, tenant_id=...)` → `connector.upsert(points)` |
| `vision_index.py`（纯视觉 + 视觉转文本两处） | 同上 |
| `tasks/collection.py::_initialize_vector_databases` | `create_collection(vector_size=...)` → `ensure_collection()` |
| `tasks/collection.py::_delete_vector_databases` | `delete_collection(...)` → `drop_tenant(...)` |
| `document_service.py::get_document_chunks / _vision_chunks` | 手写 `_node_content` 解析 → `flatten_node_payload(point.payload)` |

`ContextManager.query()` 内部从 `connector.search(...).results`（老的
`QueryResult` 包装）切换到 `list[SearchHit]` + 自己适配成
`DocumentWithScore`，对外行为完全不变。

## 向后兼容：旧数据仍可读

已经写入的 Qdrant 数据（由 LlamaIndex `QdrantVectorStore.add` 产生，
payload 带 `_node_content` 字符串）**不需要任何迁移**。读路径统一走
`flatten_node_payload()`：

1. 有 `text` / `metadata` 顶层字段（新写入）→ 直接用。
2. 只有 `_node_content`（老数据）→ 解析 JSON 取字段。
3. 两者都有（过渡态）→ 优先用 flat 版本（最新写入意图）。
4. `metadata.source` 缺失但 `_node_content.relationships['1'].metadata.source`
   存在 → 派生 basename 作为 source（老文档预览页依赖这条路径）。

这个 helper 有专门单元测试（`test_dto.py`）盯住上面四条语义。

## 配置项

### 新增 env（都非必填）

```bash
# 向量后端选择，默认 qdrant。切 pgvector 只需改这一行。
VECTOR_DB_TYPE=qdrant

# pgvector 独立 DB URL（可选）。留空 → 复用主 PG。
PGVECTOR_DATABASE_URL=

# HNSW 调优
PGVECTOR_HNSW_M=16
PGVECTOR_HNSW_EF_CONSTRUCTION=64
PGVECTOR_HNSW_EF_SEARCH=40
```

### deploy

- `deploy/aperag/values.yaml` 补齐 `PGVECTOR_*`，默认与 env.template 一致。
- docker-compose 的 `aperag-postgres` 已经是 `apecloud/pgvector:pg16`，
  `vector` 扩展已安装。**本地直接开箱可用**：

```bash
# 切到 pgvector 后端（默认复用主 DB）
echo "VECTOR_DB_TYPE=pgvector" >> envs/.env
make run  # 应用自动建表、建索引、建 extension
```

## 测试

| 文件 | 范围 |
|---|---|
| `test_dto.py`（新） | `VectorShape` 归一化、`TenantRef` 非空、`VectorPoint` 类型检查、`QueryRequest` 字段默认、`flatten_node_payload` 四条语义路径 |
| `test_llama_index_adapter.py`（新） | `BaseNode` → `VectorPoint` 扁平转换、tenant 自动注入、顺序保持 |
| `test_pgvector_translator.py`（新） | 表名构造边界、DSL 每节点 → SQL 片段结构快照、SQL 注入面（参数 bind 而非插值） |
| `test_pgvector_end_to_end.py`（新，gated by `APERAG_TEST_PGVECTOR_URL`） | 10 个用例覆盖 upsert / search / retrieve / delete / delete_by_filter / drop_tenant / tenant 隔离 / 组合过滤 |
| `test_qdrant_multitenancy_integration.py`（更新） | 全量迁移到新 DTO API；新增 `delete_by_filter` 和 `upsert` 覆盖 |
| `test_qdrant_filter_translation.py`、`test_qdrant_client_cache.py`、`test_context_manager_filter.py`、`test_filters.py`（更新/保留） | M2 的测试集全部兼容新 API |

**结果**：121 tests in `tests/unit_test/vectorstore + service`（含 10 个
pgvector 端到端、gated），全部通过。更广的 455 测试（排除 pre-existing
`pydantic_ai` / MCP docstring 失败，那两组与本 PR 无关）也通过。

## 上线指南（两种场景）

### 场景 A：继续用 Qdrant（默认）

- 什么都不用改。合并即可。
- 新的 `connector.upsert` 写入格式不再包含 `_node_content`；已有老
  数据的读路径通过 `flatten_node_payload` 完全兼容，不需要重写老数据。

### 场景 B：切换到 pgvector

1. 确认 PostgreSQL 安装了 `pgvector` 扩展（`CREATE EXTENSION vector` 能
   跑通）。`apecloud/pgvector:pg16` 镜像已内置。
2. 设 `VECTOR_DB_TYPE=pgvector`。默认复用 `DATABASE_URL` 指向的主
   Postgres；如要独立 PG，设 `PGVECTOR_DATABASE_URL=...`。
3. 重启应用。首次写入时连接器自动 `CREATE TABLE IF NOT EXISTS
   aperag_vectors_<size>_<distance>` + 三个索引。
4. **Qdrant 和 pgvector 的数据目前不互通**——切换相当于新启用一个空的
   后端。已有 Qdrant 数据需要重新 ingest（或保留 Qdrant 继续读、同时
   pgvector 只收新数据：这种双写窗口设计不在本 PR 范围）。

## 下一步（非本 PR）

这些都不是 blocker，记录在这里供后续评估：

- **数据迁移脚本**（Qdrant → pgvector）：现在没做，需要时按
  `scripts/migrate_qdrant_multitenancy.py` 的思路扩一个 "scroll Qdrant →
  upsert pgvector" 的脚本，大概 300 行。
- **pgvector `halfvec` / `bit` 量化**：维度过大时开这个能省 50~90% 磁
  盘。目前 `pgvector_hnsw_*` 下留了 hook，实际启用需要给 `VectorShape`
  加一个 `storage_type` 字段 + connector 建表时用 `halfvec(dim)`。
  **按需再做**，不提前占位。
- **Milvus 后端**：架构已经齐了，新加一个 `milvus_connector.py` 实现
  9 个抽象方法 + 在 `connector.py::match` 加一条 case 就能接进来。预计
  1~2 周工作量。
