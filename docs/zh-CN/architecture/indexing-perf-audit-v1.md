# Indexing 链路性能审计 v1

> 审计范围：parse / index 4-lane / DB / 私有化部署 / 大量+长文档场景。
> 审计基线：`main` HEAD = `eb4c4f3d` (2026-04-30 18:46)。
> 协作：架构师 @符炫炜（本文）+ @ziang（读路径 / cleanup / 长文档端到端）。
> earayu2 directive (msg=718c79ba)：thread 内闭合讨论 + 详细方案报告 + 仅 2 人参与 + @不穷 跟踪。

---

## 0. 结论速览（按优先级）

| # | 类别 | 主要瓶颈 | 推荐动作 | 优先级 | 收益 | 风险 |
|---|---|---|---|---|---|---|
| **P0-1** | Index | Vector lane 每 chunk 一次 Qdrant `upsert` 调用（`wait=True` 同步往返） | 改成 backend 协议 `upsert_points(batch)`；按 256-512 / batch 提交 | P0 | 长文 5-30× 提速；万级文档总吞吐 3-10× | 低（仅写路径，已有 idempotent 删除） |
| **P0-2** | LLM | `EMBEDDING_MAX_WORKERS=1` + `EMBEDDING_MAX_CHUNKS_IN_BATCH=10` Helm 默认 | Helm 默认提到 4 / 32；按 provider 限速回退 | P0 | 5000-chunk 长文 embedding 时间 90% 缩减 | 中（provider 限速触发；已有 inline retry budget） |
| **P0-3** | Parse | `parse_document` 整文一次性 `bytes` 入内存 + 临时文件 + 全 `chunks` list 构造 | (a) 暂时上调 worker 内存预算 (b) 增加 ``MAX_DOCUMENT_SIZE`` 校验在 upload 路径前置 (c) Vision images stream-write，不全量 list | P0 | 长文 OOM 风险消除 | 低（结构化重构） |
| **P1-4** | Scheduler | 7 modality 并发上限硬编码（`vector=16/fulltext=32/graph_*=4/summary=4/vision=4/parse=8`），`reconciler interval=30s/batch=100` 也硬编码 | 全部抽到 settings + Helm values + per-modality；保留当前默认 | P1 | 私有化按硬件细调；不再需要改代码 | 低（默认值不变） |
| **P1-5** | DB | `RedisQuotaBackend` 已构造但 `_ = quota_backend` 没真正 acquire（task #24 占位）；DB pool 公式仍要操作员手算 | 推进 task #24 quota 接线；提供 `helm` 自动算 pool 的 helper / 校验 | P1 | 跨副本资源压制收敛；私有化 ops 摩擦下降 | 中（task #24 设计未完） |
| **P2-6** | Reconciler | 每行 `queue.push` 单独 RTT，无 Redis pipelining；6 个 scan 串行 | 加 Redis `pipeline()`；scan 间允许并发（独立表） | P2 | 大量 PENDING 时 reconcile 周期缩 50-70% | 低 |
| **P2-7** | Graph | extractor 主 pass `_DEFAULT_EXTRACTOR_LLM_CONCURRENCY=4` 硬编码 + `_MAIN_PASS_BATCH_SIZE=50` 硬编码 | 抽到 collection-level config；保留 default | P2 | 长文图谱抽取按 LLM 限速调；ETD 缩短 2-3× | 中（要重新跑长文回归） |
| **P2-8** | Helm | 单 replica 默认 + 无 HPA + `resources: {}`（裸调度） | values.yaml 提供 production preset（HPA + resources requests） | P2 | 私有化 onboarding 1 步到位 | 低 |
| **P3-9** | Cleanup | `CLEANUP_INTERVAL_SECONDS=300/BATCH=200` 硬编码 | 抽 settings | P3 | 大量删除场景 throughput 提升 | 低 |
| **P3-10** | Parser | 整文 `_split_chunks` 单线程纯 Python 字符循环（多 MB markdown 慢） | 改 segmented split + numpy str_view（可选） | P3 | 长 markdown chunk 计算时间 -50% | 低 |

> **判定准则（earayu2 simple-stable 4 guardrail）**：以下方案均不无限扩范围、不引新组件、不破坏私有化免维护，且都能在不大改业务的前提下尽快上线。所有 P0/P1 项可独立切片成 4-6 个 PR，每个 100-300 LOC，带 boundary test。

---

## 1. Parse 层

### 1.1 现况（main HEAD `eb4c4f3d`）

`aperag/cli/indexing_worker.py` 起 1 个 `q:parse` 后台 task（`run_parse_worker`，`DEFAULT_PARSE_CONCURRENCY=8`）。`process_one_parse_task` 全流程：

```
parse_orchestrator.process_one_parse_task
  → _read_source_bytes_sync(store, object_path)        # 1) 全文一次性 bytes 入内存
  → parse_document(...)                                # 2) DocParser 路径：bytes → tempfile → DocParser → 整 markdown
       _docparser_extract_markdown:
         tempfile.mkstemp + write source_bytes        # 内存 + 磁盘双份
         parts = parser.parse_file(tmp_path)
         markdown_parts = [p.markdown for p in parts] # 整 markdown list 入内存
         image_assets: list[_VisionImageAsset]         # 所有 PDF page bytes 入内存
       _build_outline(markdown)                         # 整字符串 regex 扫描
       _split_chunks(markdown, chunking)                # 单线程 char-by-char 循环
       write_atomic(chunks.jsonl)                       # 整 chunk list join 后一次写
  → dispatch_indexing(modalities=...)                   # 5) 5 lane 入队
```

### 1.2 已确认的瓶颈

| 项 | 代码 | 问题 |
|---|---|---|
| **B1** 全文 bytes 入内存 | `parse_orchestrator.py:_read_source_bytes_sync L156-166` | 100 MB 文档 → ~200-300 MB 进程峰值（bytes + tempfile write buffer + parser 内部 buffer）|
| **B2** Vision 图全量 list | `parser.py:_docparser_extract_markdown L504-530` | 1000-页彩色 PDF (asset 50 MB) → 几 GB 内存（list 持有全部 `data: bytes`），AssetBinPart 全量 fan-in |
| **B3** chunks list 整体构造 | `parser.py:_split_chunks L295-361` + `parser.py:L687-691` (write_atomic) | 5000 chunks × 800 char = ~4 MB markdown，但 join 多份序列化导致 8-16 MB 峰值 |
| **B4** 单线程 chunk 计算 | `parser.py:_split_chunks` Python loop | CPython 1 MB markdown ≈ 1-2s 纯 Python，10 MB ≈ 15-30s |
| **B5** parse 失败无重试 | `parse_orchestrator.process_one_parse_task` returns `failed_*` 后日志 + drop | 已有 `reconcile_stuck_documents_for_parse_reenqueue` 5min cooldown 兜底，但 doc-level 错误不计 retry_count，操作员只能重传 |

### 1.3 推荐方案

**P0-3a** Hard cap：upload 路径前置 `MAX_DOCUMENT_SIZE` 校验拒绝过大文件（已有 settings 默认 100 MB，但实际 enforcement 缺失需要 grep 确认）。短期内即不允许"一份 1 GB 的 PDF 直接走 parse worker"。

**P0-3b** Vision asset 边解析边写盘：`_docparser_extract_markdown` 把 `image_assets.append(...)` 改成立刻 `write_atomic(image_path, asset.data)`，descriptor 行只持有 metadata（`image_id / mime / page_idx / bbox`），返回 `descriptor_lines`，不返回 `data: bytes`。peak memory 降到 `O(单 image)`。

**P0-3c**（中期）DocParser 改 streaming：现在 DocParser 接 path，整 PDF 一次性 OCR。Wave 7+ 可以追求 page-level streaming（每解析完一页就 emit MarkdownPart），让 chunks 增量写盘。**先不做**，靠 B1 / B2 先把 OOM 风险消除。

**P3-10** `_split_chunks` 改 segmented：先按 `\n\n`/`\n` 大切，每段独立 split，避免一次性扫描整文。10 MB markdown 测试可达到 -50% 时间。

### 1.4 监控建议
- 在 worker 进程加 RSS gauge（已有 OTLP MeterProvider）；`process_resident_memory_bytes` p95 报警 > 2 GB。
- `parse_worker_duration_seconds` 按 extension 分位；> 5min 触发 ops alert。

---

## 2. Index 4-lane 调度

### 2.1 当前 11 lane 现况

```
indexing-worker 进程 (asyncio 单进程)
├── q:indexing:vector        Semaphore(16)
├── q:indexing:fulltext      Semaphore(32)
├── q:indexing:graph         Semaphore(4)   # legacy 兼容期
├── q:indexing:graph_facts   Semaphore(4)
├── q:indexing:graph_vectors Semaphore(4)
├── q:indexing:summary       Semaphore(4)
├── q:indexing:vision        Semaphore(4)
├── q:parse                  Semaphore(8)
├── reconcile loop (30s)     6 scan + asyncio.create_task fan-out
├── cleanup loop (300s)      batch 200
└── q:graph_curation_run     1 lane 独立队列
```

全部跑在单 asyncio process 内（CLI `aperag.cli.indexing_worker`），`replicaCount: 1` 默认。

### 2.2 已确认的瓶颈

| 项 | 代码 | 问题 |
|---|---|---|
| **C1** 并发上限硬编码 | `orchestrator.py:_entrypoint L838-876` | 7 modality × 1 个 hardcoded `concurrency=N`，私有化要按 LLM 限速 / VRAM 调要改代码 |
| **C2** Vector lane 单 chunk upsert | `vector.py:VectorModality.sync L229-247` 调 `backend.upsert_point(single)`，`worker_factory._QdrantPointBackend.upsert_point L150-158` 包成 `connector.upsert([single])` `wait=True` | 5000-chunk 长文 = 5000 RTT × Qdrant 同步落盘 ≈ 5-15 min（对比批量 wait=False ≈ 30-90s） |
| **C3** Embedding 串行 | `embedding_service.py:embed_documents L100-110` 用 `ThreadPoolExecutor(max_workers=settings.embedding_max_workers=1)` | Helm 默认下 5000 chunks / batch 10 = 500 sequential API call。openai/voyage 限 60 req/min → 8 min＋ |
| **C4** Reconciler 串行 push | `reconciler.py:reconcile_pending_dispatch L114-143` 每行 `await queue.push(...)` 单独 Redis RTT | 100 PENDING / cycle = 100 RTT；Redis pipelining 可 1 RTT |
| **C5** 单 replica + 单进程 | `values.yaml:indexingWorker.replicaCount=1` + `aperag/cli/indexing_worker.py:_amain` 全 11 lane 1 进程 | 单点瓶颈；GIL；任何 hang lane 影响 metrics + heartbeat |
| **C6** Graph extractor 硬编码 | `graph_extractor.py:_DEFAULT_EXTRACTOR_LLM_CONCURRENCY=4` + `_MAIN_PASS_BATCH_SIZE=50` | 长文图谱抽取吞吐死锁在 LLM concurrency=4 |

### 2.3 推荐方案

**P0-1（最高 ROI）** Vector lane 改 batch upsert：

```python
# aperag/indexing/vector.py — 新协议
class VectorBackend(Protocol):
    def upsert_points(self, *, points: Sequence[VectorPoint]) -> None: ...

# aperag/indexing/vector.py:sync — 改 batch 路径
embeddings = self._batch_embedder(texts) or [self._embedder(t) for t in texts]
batch_size = 256
for i in range(0, len(chunks), batch_size):
    batch = [build_point(c, e) for c, e in zip(chunks[i:i+batch_size], embeddings[i:i+batch_size])]
    self._backend.upsert_points(points=batch)
```

`_QdrantPointBackend.upsert_points` 直接 `connector.upsert(structs)` 一次提交。同样 Summary / Vision lane 单 modality 单 point 不需要批，但 Vector 必须。

**P0-2** Helm 默认调整：

```yaml
EMBEDDING_MAX_CHUNKS_IN_BATCH: 32   # 10 → 32（OpenAI/Voyage 都支持 ≥100）
EMBEDDING_MAX_WORKERS: 4            # 1 → 4（不冲掉 provider 限速）
```

加 settings 校验：`max_workers * max_chunks_in_batch <= provider_concurrency_budget`。

**P1-4** 抽 settings：

```python
# aperag/config.py 新增
indexing_vector_concurrency: int = Field(16, alias="INDEXING_VECTOR_CONCURRENCY")
indexing_fulltext_concurrency: int = Field(32, alias="INDEXING_FULLTEXT_CONCURRENCY")
indexing_graph_facts_concurrency: int = Field(4, alias="INDEXING_GRAPH_FACTS_CONCURRENCY")
indexing_graph_vectors_concurrency: int = Field(4, alias="INDEXING_GRAPH_VECTORS_CONCURRENCY")
indexing_summary_concurrency: int = Field(4, alias="INDEXING_SUMMARY_CONCURRENCY")
indexing_vision_concurrency: int = Field(4, alias="INDEXING_VISION_CONCURRENCY")
indexing_parse_concurrency: int = Field(8, alias="INDEXING_PARSE_CONCURRENCY")
indexing_reconcile_interval_seconds: int = Field(30, alias="INDEXING_RECONCILE_INTERVAL_SECONDS")
indexing_reconcile_batch_size: int = Field(100, alias="INDEXING_RECONCILE_BATCH_SIZE")
indexing_cleanup_interval_seconds: int = Field(300, alias="INDEXING_CLEANUP_INTERVAL_SECONDS")
indexing_cleanup_batch_size: int = Field(200, alias="INDEXING_CLEANUP_BATCH_SIZE")
```

`run_*_worker` 不再 hardcoded `_entrypoint(modality, concurrency=N)`，而是工厂方法接 settings。

**P2-6** Reconciler 加 Redis pipelining：`RedisWorkQueue.push_batch(payloads_by_modality)` 一次 `pipeline.rpush(...)`，6 scan 拼成 1 个 pipeline tx。100 PENDING / 30s cycle 时 Redis 网络往返从 100 降到 1。

**P2-7** Graph extractor 抽到 collection config：`graph_extraction_llm_concurrency` 默认 4，私有化 LLM 充足时调 16-32 显著加速长文图谱抽取。

**多副本路径**（P3）：当前 hardcut 后 indexing-worker 已经独立 deployment。多副本 BLPOP 互斥已经免费（Redis），所以 `replicaCount: N` 直接横向扩。不过 reconciler / cleanup loop 在每 pod 都跑会重复扫表 → 需要 leader-election (Redis Lua SETNX + lease)。**短期**先靠 ApeRAG 集群规模 ≤ 中型，单 replica 已够；**中期**当 doc 量级到 10K+ 主动加 leader-elect 启动开关。

### 2.4 长文（5000 chunks 单文档）期望提速

P0-1 + P0-2 完成后：

| 阶段 | 当前 | 改造后 | 加速 |
|---|---|---|---|
| Embedding | 500 batch × ~1s seq = ~500s | 500 batch / 4 worker = ~125s | 4× |
| Vector upsert | 5000 RTT × 30ms = 150s | 5000 / 256 = 20 batch × 100ms = 2s | 75× |
| Fulltext bulk | 已 batch ≈ 5s | 不变 | 1× |
| Graph extract | LLM concurrency=4 × ~3s/window = 60-120s | 8 × ~3s = 30-60s（P2-7） | 2× |
| **总计** | ~720s | ~190s | **3.7×** |

万级文档批量上传：embedding 是长尾，改造后 GPU/provider 上限决定吞吐，整个 fleet 接近 linear scale。

---

## 3. DB 层

### 3.1 现况

- PostgreSQL：DocumentIndex 表 + 5 索引（`uq_document_index_triple` UNIQUE 三元组 + 4 个查询索引 + 1 个 partial unique `uniq_document_index_serving WHERE is_serving=TRUE`）
- DB pool：API `dbPoolSize=5/dbMaxOverflow=5`；indexingWorker `dbPoolSize=10/dbMaxOverflow=10`；公式手算（values.yaml L324-333 注释里给了）
- pgvector：`HNSW_M=16, EF_CONSTRUCTION=64, EF_SEARCH=40`，piggyback 主 PG（`PGVECTOR_DATABASE_URL=""`）
- Qdrant：`vectors_on_disk=True, payload_on_disk=True, int8 quantization, mmap_threshold=20MB`，已经做了 memory-tuning

### 3.2 已确认的瓶颈

| 项 | 代码 | 问题 |
|---|---|---|
| **D1** 写入未批量 | `vector.py` + `_QdrantPointBackend` 单 point upsert（已在 P0-1 提到） | Qdrant `wait=True` 单 RTT 30-50ms |
| **D2** Pool 公式手算 | `values.yaml L324-333` | 操作员要先算 `replicas * (pool+overflow) + surge + reserved < max_connections * 0.7` 才敢 scale；摩擦点 |
| **D3** quota 未真正 wire | `cli/indexing_worker.py:L160 _ = quota_backend` + `runtime.py:IndexingRuntime.quota_backend` 仍 None-injected for plain workers | task #24 占位，跨副本 token bucket 没启用，理论上副本数 ↑ 时 LLM provider 可能瞬时打爆 |
| **D4** Cutover 3-stmt TX 长 | `orchestrator.py:_finalize_active_with_cutover L536-598` 一次 BEGIN 三个 UPDATE | 单文档无问题；万级并发 cutover 会增加 partial unique index 锁竞争 |
| **D5** Reconciler 6 scan 单 worker | `reconciler.py:run_reconcile_loop L818-874` | 30s 周期 6 个串行 scan + `asyncio.create_task` 派发 — collection_descriptions_hook 会在 worker 进程跑 regen task（task #17 hard cut 完后还混在 worker process）|
| **D6** ES `refresh=True` | `worker_factory._ElasticsearchFulltextBackend.bulk_index L216` | 每个文档 bulk 强制 refresh，万级文档批量入库时 ES merge pressure 飙升 |
| **D7** PG 主键 sequence 冲突 | DocumentIndex.id `Integer autoincrement` + 上传时 5 行 INSERT RETURNING id | 高并发上传时 sequence allocate 不是问题，但 `_insert_rows_sync` 用 Session.begin() 拿 5 个 RETURNING 一次提交，未发现明显 hot spot |

### 3.3 推荐方案

**P1-5a** quota 真接线（task #24 推进）：`RedisQuotaBackend.acquire(...)` 在 `process_one_task` 调 worker.derive 之前 acquire 1 token，`release` 在 finally。token bucket key = `quota:vector:user:<uid>` / `quota:graph_facts:org:<orgid>`。私有化部署 + 共享 LLM 时这是必须的。

**P1-5b** Helm 自动算 pool：

```yaml
# values.yaml
postgresqlMaxConnections: 100  # 提示用户从 PG 抓
dbPoolSafetyRatio: 0.7

# templates/_helpers.tpl 新增
{{- define "aperag.workerDbPoolSize" -}}
{{- $budget := mul .Values.postgresqlMaxConnections .Values.dbPoolSafetyRatio -}}
{{- $apiUsage := mul .Values.api.replicaCount (add (atoi .Values.api.dbPoolSize) (atoi .Values.api.dbMaxOverflow)) -}}
...
{{- end -}}
```

或者更简单：在 `aperag.cli.indexing_worker._amain` 启动时 query `SHOW max_connections` + 自检 pool 占比，超阈值打警告日志，不让进程启动失败。

**P2-6** ES `refresh=False` + 周期性 refresh interval（默认 ES 1s 已经够）：减少 forced refresh 压力。

**P3** Reconciler scan 间并发：6 scan 都是独立表，可以 `asyncio.gather`。但小心 DB pool — 6 个并发 session 要 6 槽。当前 worker pool 10+10 够用。

### 3.4 监控建议
- `pg_stat_activity` 监控 worker DB 连接占用 vs pool budget
- `document_index` 表 size growth + dead tuple ratio
- pgvector / Qdrant write latency p99
- ES bulk_index latency + refresh_total

---

## 4. 私有化部署（Helm + docker-compose）

### 4.1 现况

- `values.yaml` 提供完整 7-database 矩阵（postgres / redis / es / qdrant / neo4j / nebula / minio）默认 enabled
- `indexingWorker.replicaCount=1`, `api.replicaCount=1`
- 默认 `resources: {}`（无 requests/limits，Kubernetes 当 BestEffort QoS 调度）
- 存储：`OBJECT_STORE_TYPE=local /data/objects`，单机部署默认本地盘
- 没有 HPA / VPA / PodDisruptionBudget
- 没有 leader-election（reconciler / cleanup 默认单 replica 安全）
- ES `shards=1, replicas=0`：低规格 OK，单点风险
- Qdrant 已 memory-optimized

### 4.2 私有化场景特化建议

#### Tier 1（单机 / 测试 / SOHO）— 1-100 docs/day, ≤1 GB 总量
当前默认基本够用。建议：
- `INDEXING_MODE=inline`（已有，dispatcher 支持）— 跳过 Redis queue，HTTP 路径同步索引。**确认这条路径在 Wave 5 之后是否还测试过**（`dispatcher.py:IndexingMode.INLINE` 代码还在）。
- 单 docker-compose 全栈起，关掉 graph + vision（`enable_knowledge_graph=false, enable_vision=false`），减少 LLM 依赖。

#### Tier 2（中型私有化）— 1K-10K docs, 多用户, ≤1 TB
- `indexingWorker.replicaCount=2-4`（横向扩，BLPOP 自动 demux）
- 给 `resources.requests` 写实数：`api: cpu=500m mem=1Gi`, `worker: cpu=2 mem=4Gi`（embedding + parse 是大头）
- 上 PgBouncer（values.yaml 注释里已经埋了 future option）
- ES `replicas=1`（同 cluster 高可用）
- HPA on `q:parse` + `q:indexing:*` queue depth（需要新 metrics + KEDA）

#### Tier 3（大型企业，10K+ docs / 长文档）— 现在还不到这层
- 多 worker pod + leader-election 给 reconciler / cleanup
- 独立 vector DB（pgvector → Qdrant cluster）
- LLM provider quota 真接线（task #24）
- Object store 切 S3/MinIO（对象 GC + 跨可用区）

### 4.3 推荐方案

**P2-8** values.yaml 增加 production preset：

```yaml
# values.production.yaml （或 values.yaml 加 profile 维度）
indexingWorker:
  replicaCount: 2
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
api:
  resources:
    requests:
      cpu: "500m"
      memory: "1Gi"
hpa:
  enabled: false  # default — 用户自己决定
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
podDisruptionBudget:
  enabled: false
  minAvailable: 1
```

**P2-8b** docker-compose.prod.yml 增加资源 limits 提示。

**P2-8c** values.yaml 写明 tier 1/2/3 配方文档（profile 切换 onboarding 1 步到位，符合 simple-stable + zero-maintenance）。

---

## 5. 大量文档 + 长文档 端到端瓶颈排序

### 5.1 大量文档（10K docs 批量上传）

按瓶颈 ranking（最长尾在前）：

1. **Embedding API 限速**（P0-2 后）：4 worker × 32 batch × 1.5s/batch = 384 docs/min（小文档场景）。OpenAI 限 3000 tokens/min × 多账户聚合可线性扩。
2. **Vector upsert RTT**（P0-1 后）：批量 256/batch，Qdrant `wait=True` ≈ 100ms/batch，单 lane 16 并发 = ~40 docs/sec。
3. **Parse + DocParser**：MarkItDown for office docs ≈ 1s/MB；MinerU OCR 30s/PDF。parse worker concurrency=8 × ~5s/doc = ~96 docs/min。
4. **Graph extractor LLM**：collection 开了 KG 时，单文档抽取 ~30-90s（chunks × LLM call）。这是 KG 场景下的最长尾。
5. **DB Insert + cutover**：每文档 5 modality × 3-stmt cutover TX = 15 TX/doc，10K docs = 150K TX，PG 单实例 ≥1000 TPS 完全没问题。

**横向扩配方**：`replicaCount=4` indexing-worker + `dbPoolSize=10/dbMaxOverflow=10` × 4 + PgBouncer = 10K docs / 30 min（KG off）。

### 5.2 长文档（单文档 100 MB / 5000 chunks）

按瓶颈 ranking：

1. **Vector upsert**（当前架构最大瓶颈）：5000 × 30ms = 150s。**P0-1 后 → 2-5s**。
2. **Embedding**：5000 / 10 = 500 batch sequential ≈ 500-1000s。**P0-2 后 → 30-100s**。
3. **Parse 内存峰值**：100 MB 源 → 200-300 MB 进程 RSS（DocParser internal buffer）；千页 PDF 抽 vision asset 时再 +500 MB-1 GB。**P0-3 后峰值降到 200-300 MB**。
4. **Graph 抽取**：5000 chunks → window=4 默认 → 1250 LLM call × 3s × concurrency=4 = ~960s = 16 min。**P2-7 后 (concurrency=16) → 4 min**。
5. **chunks list 整体序列化**：5000 chunks × 800 char = 4 MB markdown，json.dumps + join → 8-12 MB peak（可接受）。

**长文档目标 SLO**：100 MB PDF 含 KG → < 5 min（当前 ~30 min）。P0-1 + P0-2 + P0-3 + P2-7 完成后可达。

---

## 6. 实施切片建议（不无限扩范围）

**Wave 1（1 周）— P0 三件套**
- PR-A: Vector lane batch upsert（`vector.py` + `worker_factory._QdrantPointBackend` + 单元测试）
- PR-B: Helm `EMBEDDING_MAX_*` 默认提升 + settings 校验
- PR-C: Vision asset stream-write（`parser.py` 改 image asset 边解析边落盘）

**Wave 2（1 周）— P1 调度抽象 + quota wire-in**
- PR-D: 11 lane 并发上限抽 settings + Helm values
- PR-E: task #24 quota 真接线（worker.derive 前 acquire / finally release）
- PR-F: Helm pool 校验（启动时 `SHOW max_connections` self-check）

**Wave 3（1 周）— P2 长尾 & 部署体验**
- PR-G: Reconciler Redis pipelining
- PR-H: Graph extractor concurrency 抽到 collection config
- PR-I: Helm production preset values + docs（Tier 1/2/3 配方）
- PR-J: ES `refresh=False`

**Wave 4（按需）— P3 长尾**
- PR-K: Cleanup interval/batch 抽 settings
- PR-L: `_split_chunks` segmented 优化

---

## 7. 验证方式（每个 PR 必须带）

1. **Boundary test**：
   - Vector batch upsert：单测 `_QdrantPointBackend.upsert_points` 接收 N 个 point + `connector.upsert(struct_list)` 一次调用（不是 N 次）
   - Concurrency settings：单测 `OrchestratorConfig.concurrency == settings.indexing_vector_concurrency`
   - Vision stream-write：跑 1000 page PDF 测试 RSS 峰值 ≤ 1 GB

2. **回归压测**：
   - 10 docs / 5 并发（task #17 base）：保持 < 200s
   - 100 MB PDF 单文档（含 KG）：从当前 ~30 min 降到 < 5 min
   - 10K docs 批量上传（KG off）：完成时间 < 30 min

3. **CI gate**：
   - `pytest tests/indexing/test_vector_batch_upsert.py`
   - `pytest tests/boundaries/test_concurrency_settings.py`
   - `pytest -m perf`（新建 perf marker，只在 nightly 跑）

---

## 8. 依赖 + 风险

- **P0-1 依赖**：无（仅写路径，已有 idempotent 删除前置）
- **P0-2 依赖**：provider rate limit 需要监控；inline retry budget=3 已有；建议在 settings 加 `EMBEDDING_PROVIDER_RPM_BUDGET` 显式 cap
- **P0-3 依赖**：DocParser 接口不变；只是 caller 改流式落盘
- **P1-5a 依赖**：task #24 spec lock（quota policy registry 已经有 `RedisQuotaBackend(quota_redis, quota_registry)` 构造，需要确认 token bucket key schema）
- **P2-7 依赖**：长文档 KG 回归测试（Harry Potter txt 跑通）

**关键风险**：
- P0-1 改 batch upsert 时，确保 partial failure 处理（一个 batch 里 1 个 point 失败 → 整 batch 回滚还是 best-effort？）
- P0-2 调高后 provider 限速触发概率上升，依赖 `_EMBED_INLINE_MAX_ATTEMPTS=3` + 指数 backoff 兜底
- P0-3 改 image asset stream-write 时，要保证 `vision/source.jsonl` 描述子顺序与原 list 一致（`image_id` 是 md5，幂等性 OK）

---

## 9. 不在本审计范围（明确排除）

- **检索读路径性能**：@ziang 在另外章节负责
- **graph store backend 选型**（PG vs Neo4j vs Nebula）：task #61 在做
- **MCP audit**：task #32 单独 spec
- **Rerank 删除**：task #35 + sub-task 已 ship
- **Workflow / agent runtime 优化**：跟 indexing 无关
- **Frontend 性能**：跟 indexing 无关

---

## 9. K8s 生产部署参数（per earayu2 msg=caf5c760 / msg=4e9c909c — K8s 走 prod, docker-compose 仅 e2e）

### 9.1 ApeRAG 部署形态确认

| 部署目标 | 用途 | 当前 default |
|---|---|---|
| **Kubernetes (KubeBlocks for DB)** | **生产** | Helm chart `deploy/aperag/` |
| docker-compose | e2e / 本地开发 / SOHO 单机 | `docker-compose.yml` |

K8s prod 的所有 perf-relevant 参数都在 `deploy/aperag/values.yaml`，本节按层列出 default + tier 1/2/3 推荐。

### 9.2 资源 requests/limits

| 组件 | 当前 | Tier 2 (中型 1K-10K docs) | Tier 3 (大型 10K+ / 长文) |
|---|---|---|---|
| `api` | `resources: {}` (Best-Effort QoS) | `requests: cpu=500m mem=1Gi` `limits: cpu=2 mem=4Gi` | `requests: cpu=1 mem=2Gi` `limits: cpu=4 mem=8Gi` |
| `indexingWorker` | `resources: {}` | `requests: cpu=2 mem=4Gi` `limits: cpu=4 mem=8Gi` | `requests: cpu=4 mem=8Gi` `limits: cpu=8 mem=16Gi` |
| `frontend` | `resources: {}` | `requests: cpu=100m mem=256Mi` `limits: cpu=500m mem=512Mi` | 同 Tier 2 |

**P0-Helm-1**：values.yaml 增加 `resources.requests` 默认（Tier 2 sane defaults）。理由：
- `resources: {}` 在 K8s 是 BestEffort QoS，OOM 时第一个被 kill，且没有 CPU 保留；生产环境直接死。
- 默认值不写死成 limits（避免 throttle 引起神奇 latency），只写 requests 保证调度命中合适节点。

### 9.3 副本数 + HPA

**当前**：`api.replicaCount=1`, `indexingWorker.replicaCount=1`，没有 HPA。

**P1-Helm-2** HPA 模板（per `q:parse` + `q:indexing:*` queue depth via KEDA）：

```yaml
# values.yaml 新增
hpa:
  api:
    enabled: false   # default off — 用户决定是否打开
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
  indexingWorker:
    enabled: false
    minReplicas: 2
    maxReplicas: 8
    # KEDA-based scaling on Redis queue depth
    keda:
      enabled: false
      triggers:
        - type: redis
          metadata:
            address: "{redis-host}:6379"
            listName: "q:parse"
            listLength: "10"  # scale up if >10 items pending
        - type: redis
          metadata:
            listName: "q:indexing:vector"
            listLength: "100"
```

**leader-election 边界**（开多 replica 时必须确认）：
- `run_reconcile_loop` 在每 pod 都跑会重复扫表（每 30s 都扫 PENDING + FAILED + RUNNING + stuck + collection_regen + graph_vectors_enqueue）
- `run_cleanup_loop` 在每 pod 都跑会重复扫 orphan
- 短期：`indexingWorker.replicaCount=1` 时无问题；扩 replica 前必须加 leader-election（Redis Lua SETNX + lease），否则 reconciler 会重复 push（虽然 idempotent，但浪费 Redis IOPS + DB 扫描成本）

**P1-Helm-3** Leader-election 简易实现：
- 启动时每 pod 用 `redis SETNX indexing:leader:<lane> <pod_name> EX 60`，赢的 pod 跑 reconciler / cleanup
- 每 20s renew lease（`SET ... XX EX 60`）；丢失 lease 立即停 reconciler / cleanup loop
- worker lane（vector/fulltext/graph_*/summary/vision/parse/graph_curation）多 replica 安全（Redis BLPOP 互斥）— 不需要 leader-election

### 9.4 PVC / 持久化

| 卷 | 用途 | Tier 2 | Tier 3 |
|---|---|---|---|
| `api-data` (`/data/aperag`) | API 临时缓存 | 10 GiB | 50 GiB |
| `indexingWorker-objects` (`/data/objects`) | parser 派生 artifact + 用户上传源 | 100 GiB | 1 TiB（或切 S3/MinIO）|
| postgres-data (KubeBlocks 管) | 主 DB + DocumentIndex + pgvector | 50 GiB | 200 GiB |
| qdrant-data (KubeBlocks 管) | 向量库 | 50 GiB | 500 GiB |
| es-data (KubeBlocks 管) | 全文索引 | 30 GiB | 200 GiB |
| redis-data (KubeBlocks 管) | 队列 + 缓存 + quota | 5 GiB | 20 GiB |

**P1-Helm-4**：`OBJECT_STORE_TYPE=local` 在 multi-replica 部署下不能用（每 pod 独立盘看不到对方写的 artifact）。Tier 2+ 必须切 `OBJECT_STORE_TYPE=s3`（含 MinIO）。values.yaml 加 enforcement：`indexingWorker.replicaCount > 1 && OBJECT_STORE_TYPE == "local"` → 启动 Helm template error。

### 9.5 PodDisruptionBudget + 滚动升级

**P2-Helm-5**：
```yaml
podDisruptionBudget:
  api:
    enabled: true
    minAvailable: 1
  indexingWorker:
    enabled: true
    minAvailable: 1
```

`indexing-worker-deployment.yaml` 已经用 `livenessProbe.exec: pgrep -f aperag.cli.indexing_worker` + 25s graceful drain（per task #17 cuiwenbo msg=f7868d2c），rolling update 时不会丢 in-flight task。

### 9.6 监控 + 告警（K8s prod 必备）

- `process_resident_memory_bytes` per-pod，p95 报警 > 80% requests
- `q:parse` / `q:indexing:*` queue depth（KEDA / Prometheus）
- `document_index` 表 size + dead tuple ratio
- `pg_stat_activity` 监控 worker DB 连接占用 vs pool budget
- pgvector / Qdrant write latency p99
- 每个 modality `derive` + `sync` duration p50/p95/p99（已有 OTLP 但默认 emitter=noop，prod 必须切 otlp）

---

## 10. PG + KubeBlocks + pgbouncer（per earayu2 msg=e6e4d366 / msg=2f9b062f）

### 10.1 当前现况

- `deploy/databases/postgresql/values.yaml` 走 KubeBlocks 部署 PG cluster（`pg-cluster-postgresql-postgresql` SVC）
- 没有 pgbouncer（ApeRAG 直连 PG）
- API pod `dbPoolSize=5/dbMaxOverflow=5`, indexingWorker `dbPoolSize=10/dbMaxOverflow=10`
- pool 公式手算（values.yaml L324-333）：`sum(replicas * (pool+overflow)) + surge + reserved < max_connections * 0.7`

### 10.2 引入 pgbouncer 的收益

1. **副本扩展不再卡 max_connections**：pgbouncer 把 `client_conn` 复用到固定 `pool_size` 个 server connection，PG max_connections 100 时 ApeRAG 端可以挂 200-500 client。
2. **冷启动 / 拥塞场景更稳**：rolling update 期间多 replica 同时建连不会瞬时打爆 PG。
3. **PG max_connections 不再频繁调高**（避免 PG memory overhead — 每连接 ~10 MB）。

### 10.3 关键决策：pooling mode

| 模式 | 兼容性 | 说明 |
|---|---|---|
| `session pooling` | 100% 兼容 ApeRAG | 每个 client 独占 1 个 server 连接直到 disconnect — 跟没装 pgbouncer 差不多，**不推荐** |
| `transaction pooling`（earayu2 directive） | **需要逐项验证** | 每个 transaction 独占 server 连接 — pool_size 可以远小于 client_conn |
| `statement pooling` | ApeRAG 不兼容 | 跨 statement 不保证同一个 server — 破坏 session 状态 |

**transaction pooling 兼容性 audit checklist**（必须在 PR 前 ✅）：

- [ ] **prepared statements**：transaction pooling 不支持跨 transaction 复用 prepared statement。SQLAlchemy 默认不用 server-side prepared，但 asyncpg dialect 有时会启用 → 必须 grep 确认 `prepare_threshold` / `statement_cache_size=0`
- [ ] **`SET LOCAL`**：transaction-scoped，安全。Grep `session.execute("SET LOCAL ...")` 确认所有调用都包在 begin block。
- [ ] **`SET` (session-scoped)**：会跨 transaction 漏到下个 client，必须用 `SET LOCAL`。Grep `session.execute("SET ...")` 看有无 non-LOCAL 用法。
- [ ] **temporary tables**：跨 transaction 不可见，ApeRAG 没用，✅
- [ ] **listen/notify**：跨 transaction 不可见，ApeRAG 没用，✅
- [ ] **advisory locks**：`pg_advisory_lock(...)` 是 session-scoped，会泄漏；必须用 `pg_advisory_xact_lock(...)`（transaction-scoped）。Grep ApeRAG 代码没找到 advisory lock 用法，✅
- [ ] **`reset` 行为**：`pgbouncer.ini` 设 `server_reset_query = DISCARD ALL` 兜底（默认就是这个）

### 10.4 推荐 pgbouncer 参数

```ini
# pgbouncer.ini — 中型私有化（Tier 2）
[databases]
aperag = host=pg-cluster-postgresql-postgresql port=5432 dbname=postgres

[pgbouncer]
pool_mode = transaction
listen_port = 6432
max_client_conn = 500           # ApeRAG 端可以挂 500 client
default_pool_size = 25          # 每 db 默认 25 server connection
reserve_pool_size = 5           # 拥塞时额外 5 个应急
reserve_pool_timeout = 3        # 等 3s 拿不到才走 reserve
server_idle_timeout = 600       # 10 min 空闲就关，让 PG 内存稳定
server_reset_query = DISCARD ALL
ignore_startup_parameters = extra_float_digits,application_name
log_connections = 0             # prod 关掉减压
log_disconnections = 0
```

PG 端配套：

```yaml
# KubeBlocks PG cluster values
postgresql:
  parameters:
    max_connections: "100"      # 内 reserve 给 pgbouncer (25*N + 10 maintenance)
    shared_buffers: "2GB"       # 25% of 8GB request
    work_mem: "16MB"            # vector / graph 复杂查询
    maintenance_work_mem: "256MB"
    max_wal_size: "2GB"
```

### 10.5 ApeRAG 侧改造

```yaml
# values.yaml
api:
  dbPoolSize: "20"          # 5 → 20（pgbouncer 拿一个，PG 端不感）
  dbMaxOverflow: "10"
indexingWorker:
  dbPoolSize: "30"          # 10 → 30
  dbMaxOverflow: "10"
postgres:
  POSTGRES_HOST: "pgbouncer-svc"  # 走 pgbouncer，不直连 pg-cluster
  POSTGRES_PORT: "6432"           # pgbouncer 端口
```

ApeRAG 应用层 `pool_size=20+10=30`，4 个 replica = 120 client 连 pgbouncer，pgbouncer pool_size=25 server connection 接 PG。PG max_connections=100，剩余 75 给 KubeBlocks 维护 + 备份 + 监控 + 应急。

**P1-Helm-6** Helm 模板：

1. 增加 `deploy/databases/pgbouncer/values.yaml`（KubeBlocks pgbouncer addon）
2. `deploy/aperag/values.yaml` 加 `postgres.via_pgbouncer: true` 默认 enable
3. `aperag-secret.yaml` 改用 pgbouncer SVC 注入 `DATABASE_URL`
4. 启动 self-check：`SHOW pool_mode` if 走 pgbouncer，必须 == `transaction`

### 10.6 验证

- 单元测试：mock pgbouncer `pool_mode=transaction`，跑全部 `tests/db/` 用例（特别是 `test_collection_regen_lease.py` advisory lock 类）
- 压测：4 replica × `dbPoolSize=30` 同时启动 → pgbouncer 不应该 client_conn 撑爆
- 回归：task #61 P2-S2 N-seed PG connection saturation 复测（已经 ship 的 fix），确认 pgbouncer 引入后还是稳定

---

## 11. 可配化清单（admin UI 接入建议，per earayu2 msg=99c1d23a）

### 11.1 现况

`/admin/configuration` page (`web/src/app/admin/configuration/page.tsx`) 已有：
- `ParserSettings`：`use_markitdown` / `use_mineru` / `mineru_api_token`
- `QuotaSettings`：每用户 / 每 collection 配额

settings 持久化走 `aperag/domains/governance/service/setting_service.py`（DB key-value 表），单条 update 用 `update_setting(key, value)`。

### 11.2 审计找到的新参数 — 建议接入 admin UI

#### 类 A：runtime perf 类（**强烈建议接入** — 非 ops，运维 + 开发都关心）

| 参数 | 当前 default | 建议 admin UI 范围 | 卡片归属 |
|---|---|---|---|
| `embedding_max_chunks_in_batch` | 10 | 8-128 | **新建 IndexingSettings 卡片** |
| `embedding_max_workers` | 1 | 1-8 | IndexingSettings |
| `indexing_vector_concurrency` | 16 | 4-64 | IndexingSettings |
| `indexing_fulltext_concurrency` | 32 | 8-128 | IndexingSettings |
| `indexing_graph_facts_concurrency` | 4 | 1-16 | IndexingSettings |
| `indexing_graph_vectors_concurrency` | 4 | 1-16 | IndexingSettings |
| `indexing_summary_concurrency` | 4 | 1-16 | IndexingSettings |
| `indexing_vision_concurrency` | 4 | 1-16 | IndexingSettings |
| `indexing_parse_concurrency` | 8 | 1-32 | IndexingSettings |
| `indexing_reconcile_interval_seconds` | 30 | 10-300 | IndexingSettings（高级）|
| `indexing_reconcile_batch_size` | 100 | 10-1000 | IndexingSettings（高级）|
| `indexing_cleanup_interval_seconds` | 300 | 60-3600 | IndexingSettings（高级）|
| `chunk_size` | 400 | 100-2000 | ParserSettings（已有卡，加新字段）|
| `chunk_overlap_size` | 20 | 0-200 | ParserSettings |

#### 类 B：collection-level（**已经在 collection config**，admin UI 是修改 default）

| 参数 | 用途 | admin UI 卡片 |
|---|---|---|
| `enable_vector` / `enable_fulltext` / `enable_knowledge_graph` / `enable_summary` / `enable_vision` | 5 lane on/off | 已在 collection 创建页；admin 设全局 default |
| `graph_extraction_window_size` | graph chunk window | 已在 collection.config；admin UI 没必要重复 |
| `graph_extraction_llm_concurrency` | graph LLM 并发（建议 P2-7 抽出） | 同上 |

#### 类 C：infra ops 类（**不接入 admin UI** — 走 Helm/env，避免运行时改动 K8s 配置）

| 参数 | 理由 |
|---|---|
| `db_pool_size` / `db_max_overflow` / `db_pool_timeout` | 改了要重启进程，是部署期参数，不是运行期参数 |
| `indexing_queue_redis_url` / `indexing_quota_redis_url` | 部署期参数 |
| `pgbouncer pool_size / max_client_conn / server_idle_timeout` | pgbouncer 自己的配置，跟 ApeRAG 无关 |
| `K8s resources.requests/limits` | Helm 改完滚动升级 |
| `HPA min/max replicas` | Helm 改完滚动升级 |
| `qdrant_quantization_*` / `qdrant_hnsw_on_disk` | Qdrant collection 创建参数，改了要重建集合 |
| `pgvector_hnsw_m / ef_construction` | 同上 |
| `MAX_DOCUMENT_SIZE` | 上传限制，admin UI 已有 quota 卡，应放 quota 卡而非 indexing 卡 |

### 11.3 推荐 UI 落地

**P2-Admin-1** 新建 `IndexingSettings` 卡片（`web/src/app/admin/configuration/indexing-settings.tsx`）：

```
┌─ Indexing Settings ───────────────────────────────────┐
│  [Embedding]                                          │
│    Max chunks per batch: [____10__] (8-128)           │
│    Max parallel workers: [_____1__] (1-8)             │
│                                                       │
│  [Worker concurrency] (per modality, asyncio Semaphore) │
│    Vector:        [____16__] (4-64)                   │
│    Fulltext:      [____32__] (8-128)                  │
│    Graph facts:   [_____4__] (1-16)                   │
│    Graph vectors: [_____4__] (1-16)                   │
│    Summary:       [_____4__] (1-16)                   │
│    Vision:        [_____4__] (1-16)                   │
│    Parse:         [_____8__] (1-32)                   │
│                                                       │
│  [Advanced: reconciler / cleanup]                     │
│    Reconcile interval (s):  [____30__] (10-300)       │
│    Reconcile batch size:    [___100__] (10-1000)      │
│    Cleanup interval (s):    [___300__] (60-3600)      │
│    Cleanup batch size:      [___200__] (10-1000)      │
│                                                       │
│  [⚠️ 改动需要重启 indexing-worker pod 才能生效]         │
│                                                       │
│  [Save]   [Reset to defaults]                         │
└───────────────────────────────────────────────────────┘
```

**关键 UX 决策**：改动后只入 DB（settings 表），不立即生效。worker 启动时读 settings → 覆盖 env default。这样：
- ops 不需要 kubectl edit values.yaml 滚动升级
- admin UI 改动 → kubectl rollout restart deployment/indexing-worker → 30s 内新值生效
- 兼容当前 env-based 部署：env 优先级 > DB settings（避免 admin UI 误改影响紧急止损）

**P2-Admin-2** Backend changes：
- `aperag/config.py`：每个 indexing_* 参数加一个 helper `get_indexing_setting(key, default)`，启动时优先读 env，env 没有则读 DB settings 表
- `aperag/cli/indexing_worker.py:_amain()` 启动时把 settings 注入 `OrchestratorConfig` / `ParseOrchestratorConfig`
- `aperag/domains/governance/service/setting_service.py` 增加 `get_indexing_settings()` / `update_indexing_settings(...)` helper
- 新建 OpenAPI route `GET/PUT /api/v1/admin/configuration/indexing`（参考已有 `/admin/configuration/parser`）

### 11.4 hook 给前端 (@dongdong)

- 新建 `web/src/features/admin/indexing-settings/`：i18n key `admin_config.indexing.*`，schema validation（zod）
- 加 sidebar 菜单项 `Indexing` 单独卡片或 ParserSettings 同卡分组
- e2e：`web-e2e/admin/configuration-indexing.spec.ts`

---

## 12. （待 @ziang 补充）读路径 + GraphVectors/chunks.jsonl 复用边界
## 13. （待 @ziang 补充）cleanup / deletion / reconciler 批删 SQL
## 14. （待 @ziang 补充）端到端瓶颈归因（你的视角，跟 §5 互补）
## 15. （待 @ziang 补充）KubeBlocks 研究（kubeblocks-skills 仓） → 折进 §10 PG/pgbouncer 章节
## 16. （待联合）验收 checklist + Wave 1-4 排期收口

---

> 文档版本：v1（§1-§11 by 符炫炜）
> 作者：@符炫炜（架构师）
> 待补：@ziang §12-§16
> 验收：@earayu2（msg=718c79ba directive）
> 跟踪：@不穷（PM）
> main HEAD pin: `eb4c4f3d` (2026-04-30 18:46)
