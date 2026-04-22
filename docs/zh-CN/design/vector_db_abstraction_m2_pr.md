# 向量数据库抽象层 M2 实现（PR 说明）

本 PR 把 [`vector_db_abstraction.md`](./vector_db_abstraction.md) 中 M2
的所有条目落地到代码。这是纯内部重构，**不改变任何用户可见行为**；但它
堵上了向量数据库抽象的四个历史破口中的三个，为将来可能的 pgvector/Milvus
切换铺好了路。

## 结论先行

- 上线后**不需要任何运维操作**：没新增 env、没新增 secret、没新增 CR、
  没新增存储 migration。
- 所有现有单元/集成测试通过；本 PR 新增 4 个测试文件共约 30 个用例。
- `docs/zh-CN/design/vector_db_abstraction.md` 已更新，M2 段落标注
  "✅ 已落地"。

## 做了什么

### 1. `VectorFilter` DSL（新文件 `aperag/vectorstore/filters.py`）

- 六个节点：`Eq / In / IsEmpty / And / Or / Not`，全部 frozen dataclass
  (天然可 hash / 可比较 / 不可变)。
- 两个人类友好 helper：`all_of(*parts)` / `any_of(*parts)`，自动跳过
  `None`、单参数退化、零参数返回 `None`，让 `ContextManager` 不用写
  `if len(parts) == 0 else ...` 之类的胶水。
- 文件顶部写清楚了 "加节点前读这里" 的设计约束：**最小化、仅标量值、
  不许 import 任何后端库**。未来新加节点要同步改所有 backend translator，
  成本心里有数。

### 2. Qdrant translator（`qdrant_connector.py` 内新增）

- `_translate_filter(flt)` 把 DSL 树转成 `qdrant_client.models.Filter`。
- `_normalize_filter_input(x)` 同时接受 DSL 节点、`rest.Filter` 直传、
  或 `None`；把 `rest.Filter` 直传留在这里**只是给迁移脚本用**——`ContextManager`
  已经完全走 DSL。
- 空 `In` 直接抛 `ValueError`——在单后端时代，这种静默"匹配空集"最容易
  生成"线上没数据"的疑云，现在崩溃比疑云更便宜。

### 3. `VectorStoreConnector` 抽象升级（`aperag/vectorstore/base.py`）

- 新增抽象方法 `retrieve(ids, *, with_payload, with_vectors)`。
  之前只有 Qdrant 实现，`document_service.py` 直接调用，切后端会
  `AttributeError`。
- 新增最小 DTO `VectorPoint(id: str, payload: dict, vector: list | None)`。
  `retrieve()` 返回它，彻底切断 `qdrant_client.http.models.Record` 对业务
  层的泄露。
- `search(query, *, filter, score_threshold, **kwargs)`：`filter` 从
  `**kwargs` 升级为显式关键字，类型注释 `Optional[VectorFilter]`。

### 4. `ContextManager` 去 Qdrant 化（`aperag/context/context.py`）

- 整个文件不再 import `qdrant_client` 任何东西。
- `_create_index_types_filter` / `_create_combined_filter` 产出 DSL。
- 历史上 `Filter(should=[FieldCondition(indexer IN [...]),
  IsEmptyCondition(indexer)])` 的 **"兼容老数据没有 indexer 字段"**
  语义被原样保留（在 `_create_index_types_filter` 的 `any_of(In, IsEmpty)`
  里）。这是我在 double-think 时反复确认的点——悄悄丢掉这个 branch 会让
  迁移前的数据在检索中消失一半。

### 5. `QdrantClient` 进程级复用（`qdrant_connector.py` 顶部）

- `_get_or_create_client(url, port, grpc_port, prefer_grpc, https, api_key, timeout)`
  按 endpoint 特征做双检锁缓存。
- `:memory:` URL **显式绕过缓存**，否则集成测试里各测试会共享一份内存
  store，隐性破坏隔离——这种"在 prod 用没事但在 test 里炸"的陷阱是
  长期最耗答疑时间的那种 bug，一次把它封死。
- `_reset_client_cache()` 只给测试用。

### 6. `QdrantVectorStoreConnector.retrieve` 返回类型切换

- 从"Qdrant `Record` 原样返回"变为"转成 `VectorPoint` 返回"。
- `id` 统一 `str(x)`——上游 `Chunk.id: Optional[str]`，Pydantic 本来就会
  强转，实际语义无变化，只是把隐式强转变成显式。
- 多向量字段的 collection 在 Qdrant 里是 `dict[str, list[float]]`，我们
  从来不用多向量，这里加了一个防御性的 `dict -> first.values()` 兜底，
  以后即便有人误配置了也不会 AttributeError。

### 7. `envs/env.template` 追加 `QDRANT_*` 多租户 / 优化开关

- 之前这些 env 只在 `deploy/aperag/values.yaml` 里（线上路径），本地
  `.env` 不知道它们的存在。
- 现在 `env.template` 里列齐了全部 10 个 `QDRANT_*` 开关，默认值与生产
  一致；开发者 copy 一份 `.env` 就直接能跑。
- `docker.env.overrides` 无需改动（它只 override host/port 类字段）。

## 没做什么（以及为什么）

| 没做 | 为什么 |
|---|---|
| 解耦 LlamaIndex `QdrantVectorStore.add(nodes)` | 两处 vision 写路径 + embedding_utils 深度依赖 LlamaIndex 的 node schema；拆解工作量 ≥ 本 PR。等 M3 做 pgvector 时必须拆，那时一起做 |
| 引入 `TenantRef / VectorShape / QueryResult / SearchOptions` 一堆 DTO | 只有 Qdrant 时这些 DTO 都只是 "存在感"，用户读到会问 "为啥 5 个类做的事情能用 3 个做"。等真加 pgvector 时，那时的工程师对 pgvector 的约束了解精确，一次定型好过现在拍脑袋 |
| 把 graph DB 一起抽象 | 你提到了 graph 抽象也要做；但 graph 的模式（Cypher/GQL）和向量完全不同，共用层只会是最低公共分母。**本 PR 只把向量 DSL 放在 `aperag/vectorstore/filters.py` 而不是 `aperag/filters.py`**，给未来 graph DSL 留了独立命名空间 |
| Milvus 实现 | 你说可以不做，暂不做 |
| 客户端连接池的更复杂策略（健康探测 / 驱逐 / 自动重连） | Qdrant Python 客户端本身自带 keep-alive 和重连；更复杂的池管理会在第一次遇到具体问题时再加。**"没问题时的复杂度 = 负价值"** |

## 接口兼容性

- `VectorStoreConnector.search(filter=X)` 的 `X`：
  - **推荐**：DSL 节点（`Eq`、`In`、`all_of(...)` 等）。
  - **仍兼容**：`qdrant_client.models.Filter`（迁移脚本在用）。
  - 其他类型：记 WARNING 并丢弃（与之前行为一致）。
- `VectorStoreConnector.retrieve(ids, *, with_payload, with_vectors)` 返回
  `List[VectorPoint]`。之前的 Qdrant `Record` 都被访问的是 `.id` / `.payload`，
  `VectorPoint` 完全兼容这两个属性；pydantic 下游 `Chunk(id=point.id)` 正常。
- 所有 `connector.delete(ids=...)` / `connector.create_collection(...)` /
  `connector.delete_collection(...)` 签名不变。
- `connector.store.add(nodes)`（LlamaIndex 写路径）不变。

## 测试矩阵

| 文件 | 覆盖点 |
|---|---|
| `test_filters.py` | DSL 构造规则、frozen 语义、`all_of/any_of` 短路、嵌套 |
| `test_qdrant_filter_translation.py` | 每个 DSL 节点 → Qdrant Filter 的结构快照；rest.Filter 直传短路；未知类型拒绝；空 In 报错 |
| `test_qdrant_client_cache.py` | 同 endpoint 复用、不同 endpoint 各开一个、`:memory:` 绝不缓存、并发首连接不雪崩 |
| `test_context_manager_filter.py` | 无过滤 → None、只 index_types → `Or(In, IsEmpty)`（保留 backward-compat）、只 chat_id → Eq、全开 → And(Or, Eq)、与 `vectordb_type` 字符串解耦 |

共 30 个新用例，全部通过。原有 39 个测试（包括 embedding lock、多租户
delete、purge_all_shards）全部继续通过。

## 上线核对清单

- [ ] 镜像构建通过。
- [ ] CI 所有测试绿（pytest + ruff）。
- [ ] 生产 Qdrant 是 1.10+。本 PR 对 Qdrant 版本要求与上一 PR 相同。
- [ ] `deploy/aperag/values.yaml` 的 `QDRANT_*` 字段与上一 PR 保持一致，
  无需运维动作。
- [ ] 合并后观测：
  - `qdrant-cluster-qdrant-0` RSS 曲线不应有突变（本 PR 不动内存策略）。
  - 首次搜索延迟可能小幅下降（client 复用省了 TCP/TLS 握手）。
  - 无新增 ERROR 级别日志。

## 后续

本 PR 合并后，[`vector_db_abstraction.md`](./vector_db_abstraction.md)
的 M3（pgvector 实现）就是"填空题" —— 只需要：

1. 新增 `aperag/vectorstore/pgvector_connector.py`：实现 `search /
   delete / retrieve / create_collection / delete_collection`；
2. 在 `VectorStoreConnectorAdaptor.match vector_store_type` 里加一条
   `case "pgvector"` 分支；
3. 在 `envs/env.template` 里加 `VECTOR_DB_TYPE=pgvector` 的例子；
4. 给 pgvector 单独一份 schema migration 脚本。

整个 M3 预计不需要再动 `ContextManager`、`base.py`、`filters.py` 中的
任何一行。这就是本 PR 要达到的 "M3 只是换后端、不是重写系统" 的目标。
