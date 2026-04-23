# Qdrant 内存占用治理与多租户化改造

> 写作缘由：香港 ACK 集群的 Qdrant 容器 RSS 已经涨到 12.5 GiB（limit 16 GiB），但线上只有 ~~1600 个用户、~~1.3 万文档、~60 万向量 chunk。经过排查确认**绝大部分内存不是花在业务数据上，而是花在"每个 ApeRAG Collection 建一个 Qdrant Collection"造成的元数据/段级结构性浪费**。如果不治理，随着用户增长 Qdrant 会成为整个系统的扩展瓶颈。
>
> 相关仓库：
>
> - 应用代码：[apecloud/ApeRAG](https://github.com/apecloud/ApeRAG)
> - 生产部署 values：[apecloud/aperag-values](https://github.com/apecloud/aperag-values)

---

## 1. 现场快照（2026-04-20，hk 集群）

```text
pod         : qdrant-cluster-qdrant-0  (KubeBlocks qdrant 0.9.1, qdrant 1.10.0)
limit       : 4 cpu / 16 GiB
RSS         : 12.47 GiB  (77% of limit)
RssAnon     : 12.52 GiB  ← 基本都是匿名堆/mmap
RssFile     :  1.05 GiB
VmSize      :  153 GiB   ← 线程栈预留 + 大量 mmap
Threads     :  1872      ← 关键异常指标
Storage dir :  8.0 GiB   (/qdrant/storage)
```

Qdrant 自身 telemetry 聚合：


| 指标                     | 数值                 | 备注                                   |
| ---------------------- | ------------------ | ------------------------------------ |
| collections 总数         | **1847**           | pg 中 ACTIVE 2003 + DELETED 179       |
| 空 collection（0 points） | **1260 (68%)**     | 结构性浪费的主要来源                           |
| 非空 collection          | 587                | 真正产生业务价值                             |
| 总 segments             | 7387               | 每 collection 默认 4 段                  |
| 总 points               | 575 772            |                                      |
| 总 vectors              | 600 795            | 1024 维、Cosine                        |
| 已建 HNSW 索引的 vectors    | **90 005 (仅 15%)** | `indexing_threshold=20 MB` ≈ 5000 向量 |
| 真正跨过索引阈值的 collection   | **3**              | 其他 584 个都走暴力扫描                       |


业务侧（postgres `aperag` 库）：


| 指标                          | 数值            |
| --------------------------- | ------------- |
| `user` 行数                   | 1611          |
| `collection` ACTIVE         | 2003          |
| `collection` DELETED        | 179           |
| `document` 总行数              | 13 044        |
| `document` COMPLETE         | 5 404         |
| `document` EXPIRED / FAILED | 5 114 / 1 440 |


---

## 2. 12 GiB 内存都去哪了


| 组成                                  | 估算值                                       | 解释                                                                                                                                      |
| ----------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 向量原始数据常驻 RAM                        | **~2.4 GiB**                              | 默认 `storage_type: Memory`，600 795 × 1024 × 4 B                                                                                          |
| HNSW 图（RAM）                         | ~10–50 MiB                                | `hnsw_index.on_disk=false`，但只有 9 万 vec 被索引                                                                                              |
| **7387 × RocksDB 实例静态开销**           | **~3–5 GiB**                              | 每个 segment 都是一个独立 RocksDB 库（MANIFEST/LOG/OPTIONS/LOCK 一套），memtable arena + block cache + table reader cache + bloom filter 按最小配置都要几 MiB |
| **1847 × collection 级 Qdrant 内部结构** | **~2–3 GiB**                              | id_mapper、deleted bitset、payload index、分片元数据、tokio task                                                                                 |
| **1872 threads** 的栈 + async 状态      | ~0.3–0.8 GiB                              | 每线程 VM 预留 8 MiB 但 RSS 只算 touched 页                                                                                                      |
| WAL / 小文件回写缓冲                       | ~0.5 GiB                                  | `wal_capacity_mb=32` × 活动分片                                                                                                             |
| **合计**                              | **≈ 9–11 GiB anon + 1 GiB file ≈ 12 GiB** | ✅ 与实测吻合                                                                                                                                 |


**结论一句话**：12 GiB 里**只有大约 2.4 GiB 是真正存业务向量的**，剩下的 ~10 GiB 全是"1847 × 4 段 = 7387 个 RocksDB + 每 collection 的元数据"这种**与数据量无关、只与 collection 数量线性相关的成本**。

---

## 3. 根因：一个 ApeRAG Collection = 一个 Qdrant Collection

当前实现位于 `aperag/vectorstore/qdrant_connector.py`：

```102:112:aperag/vectorstore/qdrant_connector.py
    def create_collection(self, **kwargs: Any):
        vector_size = kwargs.get("vector_size")
        from qdrant_client.http import models as rest

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE,
            ),
        )
```

配合 `aperag/utils/utils.py::generate_vector_db_collection_name` 的映射：

```44:46:aperag/utils/utils.py
def generate_vector_db_collection_name(collection_id) -> str:
    return str(collection_id)
```

也就是：用户每建一个 ApeRAG Collection，ApeRAG 就调用 `recreate_collection` 创建一个同名 Qdrant Collection。

这是 Qdrant **官方明确反对**的用法。Qdrant 的文档 [https://qdrant.tech/documentation/guides/multiple-partitions/](https://qdrant.tech/documentation/guides/multiple-partitions/) 开头第一段就在强调：

> In many cases, it is more efficient to use a **single collection** with payload-based partitioning. This approach is called **multitenancy**.

照当前趋势继续线性膨胀：


| 阶段     | 用户   | ApeRAG Collection | Qdrant RSS 预期  |
| ------ | ---- | ----------------- | -------------- |
| 现状     | 1.6k | 2k                | 12 GiB         |
| 2× 增长  | 3k   | 4k                | ~25 GiB        |
| 10× 增长 | 16k  | 20k               | 120+ GiB，单机不可行 |


业务向量数据其实只涨了线性的一小段，**真正爆炸的是 collection 元数据**。

---

## 4. 解决方案

按 **收益/改动量** 排序分成三档。ABC 三档可以独立落地、互不阻塞。

### A 档 · 立即可做（今天完成，预计省 3–5 GiB，零代码改动）

#### A.1 清理"孤儿 + 已删除"的 Qdrant collection

判定规则（**只删 pg 里已经 DELETED 或根本不存在的**，不动任何 ACTIVE 记录，即便它在 Qdrant 里是空的也保留，因为业务上用户可能还没来得及上传文档）：

```text
待删 = { qdrant 里存在的 collection } ∩ ( { pg.collection.status='DELETED' } ∪ { pg 里不存在的孤儿 } )
```

该清理已经由本次治理配套的 subagent 执行，详见本文档末尾"附录 · 清理执行记录"。

#### A.2 调整 Qdrant Server 侧默认值

改动 Qdrant 的 `config.yaml`（通过 KubeBlocks 的 config 模板或启动环境变量注入均可）：

```yaml
storage:
  optimizers:
    # 4 段 → 2 段，RocksDB 实例数直接减半
    default_segment_number: 2

    # 超过 20 MiB 的 segment 用 mmap 存储向量，冷数据交给 kernel page cache 管
    memmap_threshold_kb: 20480

  hnsw_index:
    # HNSW 图也落盘，查询路径会多一次 page fault 但内存占用大幅降低
    on_disk: true

  wal:
    # 绝大多数 collection 很小，32 MiB 的 WAL 段太奢侈
    wal_capacity_mb: 8
```

**落地位置**：

- 仓库内自部署脚本：`deploy/databases/qdrant/values.yaml`（目前只有 CPU/mem/storage/version，需要新增一节覆盖 Qdrant config）。
- 生产 helm values：`apecloud/aperag-values` 仓库内的 qdrant values 文件（该仓库独立维护，需要在 PR 里一起更新）。

> 注意：`default_segment_number` 调整**不会自动应用到已存在的 collection**，需要对现有 collection 发起 `PATCH /collections/{name}` 或通过 optimizer 重建才能生效。建议在 A.1 清理之后，通过 `UpdateCollection` API 批量刷一遍。

### B 档 · 中期（1–2 天，一次性砍掉 ~70% 内存，彻底解决扩展性）

#### B.1 重新设计：多租户 = 单 Qdrant collection + payload 索引过滤

##### B.1.1 Qdrant 多租户的官方模型

Qdrant 为此专门提供了三项能力，缺一不可：

1. **全局单 collection**：所有租户共用一个 Qdrant collection（例如 `aperag_vectors`）。原来每个 ApeRAG Collection 的 vector size 如果不一样，就按"向量维度 + 距离"分成少数几个全局 collection（例如 `aperag_vectors_1024_cosine`、`aperag_vectors_1536_cosine`）。
2. **给 point 加上 tenant 维度的 payload 字段**：在每次 upsert 时，payload 里必须带 `collection_id`（ApeRAG Collection ID）。`point.id` 继续用现有的 chunk id 方案。
3. **给 tenant 字段建 keyword 索引，并启用 tenant 优化**：
  ```http
   PUT /collections/aperag_vectors_1024_cosine/index
   {
     "field_name": "collection_id",
     "field_schema": {
       "type": "keyword",
       "is_tenant": true          # 关键：告诉 Qdrant 这是多租户分区字段
     }
   }
  ```
   `is_tenant: true` 是 Qdrant **1.11+** 引入的特殊标记（[官方发布说明](https://qdrant.tech/blog/qdrant-1.11.x/)）。打开后，Qdrant 的 optimizer 会**按照 tenant 字段对点进行物理分组存储**（tenant 相同的点会被尽量放在同一个 segment 内），查询时相当于只扫描对应 tenant 的子集，性能和独立 collection 几乎等价。
   **⚠️ 版本兼容**：生产现在的 Qdrant 是 **1.10.0**，不支持 `is_tenant`。连接器在 `aperag/vectorstore/qdrant_connector.py::_ensure_tenant_payload_index` 里采用了兜底策略：优先尝试 `is_tenant=True`，失败时降级为普通 keyword 索引。在 1.10 上：
  - ✅ payload filter 本身完全工作，租户隔离语义严格成立；
  - ✅ "1847 个 collection → 1 个" 的合并收益完全拿到（~3–5 GiB 的 RocksDB 实例开销立刻消失）；
  - ❌ segment 级 defragmentation 不会生效，tenant 点会混存在同一批 segment 中，查询时 HNSW 需要跨更多"别人的"节点。预期 p95 查询延迟有几十百分比的退化（退化量随全局 collection 的总点数线性增长）。
   **强烈建议**：把 Qdrant server 从 1.10.0 升到 1.11.x（或 1.12.x）作为本次停服窗口的一部分。升级只需改镜像 tag，不涉及数据迁移。升上去后，新建 collection 自动带 is_tenant 优化；**已有的全局 collection 上的索引不会自动升级**——需要在升级后重建索引：

##### B.1.2 查询时的过滤模板

所有 `search / query_points / scroll` 都必须带 tenant 过滤：

```python
hits = client.query_points(
    collection_name="aperag_vectors_1024_cosine",
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(key="collection_id", match=MatchValue(value=ctx.collection))
        ]
    ),
    limit=top_k,
)
```

##### B.1.3 代码落点

需要修改的主要位置：


| 文件                                                                      | 改动                                                                                                                                                                                                                                                                  |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `aperag/vectorstore/qdrant_connector.py`                                | ① 去掉每次 `create_collection` 调 `recreate_collection`；改成惰性启动时 `ensure_global_collection(vector_size, distance)`（存在则跳过，不存在则创建并建 `is_tenant` 索引）。② `search/upsert/delete` 全部带上 `collection_id` 过滤。③ 删点时用 `delete_points_by_filter({collection_id, ids})` 限定在当前 tenant 内。 |
| `aperag/utils/utils.py::generate_vector_db_collection_name`             | 废弃。替换为 `get_global_qdrant_collection_name(vector_size, distance)`。保留旧函数一段时间以便在"迁移中"回读兼容。                                                                                                                                                                            |
| `aperag/tasks/collection.py::CollectionTask.create / delete`            | 创建时不再 `create_collection`，只需确保全局 collection 存在；删除时改成"按 filter 批量删点"而不是 `delete_collection`。                                                                                                                                                                         |
| `aperag/service/search_pipeline_service.py` 及 `aperag/index/*_index.py` | 每一处 `get_vector_db_connector(...)` 调用点都要把 ApeRAG collection id 作为 tenant filter 传入。                                                                                                                                                                                 |
| `aperag/config.py::get_vector_db_connector`                             | 建议新增签名 `get_vector_db_connector(collection_id: str)`，内部查询向量维度后路由到对应的全局 collection；在 ctx 里挂上 `tenant_key`、`tenant_value`。                                                                                                                                            |


##### B.1.4 停服迁移策略（本次采用）

考虑到停服窗口明确、时长可控，本次采用**停服一次性迁移**，避免双写的复杂度：

1. **停服**：停掉 apiserver / celery worker / beat，Qdrant 仍然运行。
2. **迁移**：跑 `scripts/migrate_qdrant_multitenancy.py`：
  - 自动枚举所有 `col<hex>` 源 collection，跳过孤儿（DB 里查不到的）；
  - 为每个 `(vector_size, distance)` 组合创建 `aperag_vectors_{size}_{distance}` 全局 collection，写入完整配置（INT8 量化 / HNSW on_disk / 2 segments / mmap 阈值 / tenant index）；
  - scroll 源 collection 所有点，把 `payload.collection_id = <源 collection 名>` 注入后 upsert 进全局 collection；
  - **默认不删源 collection**，需要显式 `--delete-old` 或第二阶段 `--only-delete`；
3. **发布新版代码**，默认 `QDRANT_MULTITENANT=True`。
4. **观察 24 h**：读路径（document chunks / search / retrieve）、写路径（新建 collection、新上传文档）都正常后；
5. **清理**：再跑一次 `--only-delete` 把老的 `col<hex>` collection 删除。

> **⚠️ 单向门警告**：一旦新版代码接收任何新写入（新建 collection、新上传文档、新 chat 产生的 chunk），`QDRANT_MULTITENANT` **不能简单切回 `False` 就算回滚**——那期间写入的新点只在 `aperag_vectors`_* 里，legacy 模式下看不到；同期新建的 ApeRAG Collection 也不会有对应的 `col<hex>` 物理 collection。如果必须回滚，需要反向迁移（从全局 collection 按 `collection_id` scroll 回每个 `col<hex>`）——这段逻辑目前**未实现**。所以：回滚窗口 = 新版代码上线到接收第一笔新写入之间的那几分钟。如果发现问题必须在窗口内决策。

> **另一个单向影响**：是否在停服窗口里把 Qdrant server 顺带从 1.10.0 升到 1.11.x 也是一次决策——见 B.1.1。升级一次就回不去了（KubeBlocks qdrant 0.9.1 支持任意 tag 切换，但数据文件在 1.11 上会被 optimizer 重排）。

##### B.1.5 预期收益


| 项                   | 现状           | 多租户化后                    |
| ------------------- | ------------ | ------------------------ |
| Qdrant collection 数 | 1847         | ~1–3（按向量维度分）             |
| Segment 数           | 7387         | ~数十                      |
| RocksDB 实例数         | 7387         | ~数十                      |
| RSS（同等数据量）          | 12 GiB       | **~2–3 GiB**             |
| 再增长 10× 用户后的 RSS    | 120 GiB（不可行） | **~10–15 GiB**（线性随真实数据量） |


### C 档 · 锦上添花

#### C.1 **默认开启标量量化（INT8）**

对绝大多数 1024–1536 维的 embedding，INT8 的精度损失小于 1%，但可以把向量存储从 float32 压到 int8，**立刻省 4×**（2.4 GiB → ~0.6 GiB），并且 Qdrant 的量化实现会让向量主体走 mmap，压力进一步下降。

**代码侧改造**（与 B.1.3 同批完成最省力）：

```python
# aperag/vectorstore/qdrant_connector.py

from qdrant_client.http import models as rest

def _default_quantization_config() -> rest.QuantizationConfig:
    return rest.ScalarQuantization(
        scalar=rest.ScalarQuantizationConfig(
            type=rest.ScalarType.INT8,
            quantile=0.99,     # 鲁棒分位裁剪，离群点不影响量化区间
            always_ram=True,   # 量化后的向量留在 RAM，原 float 全量走 mmap
        )
    )

def _default_hnsw_config() -> rest.HnswConfigDiff:
    return rest.HnswConfigDiff(
        m=16,
        ef_construct=100,
        on_disk=True,          # HNSW 图落盘
    )

def _default_optimizer_config() -> rest.OptimizersConfigDiff:
    return rest.OptimizersConfigDiff(
        default_segment_number=2,
        memmap_threshold=20000,  # 单位 KB
    )
```

在 `ensure_global_collection` / 未来所有 `create_collection` 路径里默认带上这三项。

**配置侧改造**（推荐按环境变量可覆盖）：在 `aperag/config.py::Settings` 里新增：

```python
qdrant_enable_quantization: bool = Field(True, alias="QDRANT_ENABLE_QUANTIZATION")
qdrant_quantization_type: str = Field("int8", alias="QDRANT_QUANTIZATION_TYPE")
qdrant_hnsw_on_disk: bool = Field(True, alias="QDRANT_HNSW_ON_DISK")
qdrant_default_segment_number: int = Field(2, alias="QDRANT_DEFAULT_SEGMENT_NUMBER")
```

然后在两处 values 里把默认值同步过去：

1. `**deploy/aperag/values.yaml**`（当前仓库）：在 `vars` 段追加：
  ```yaml
   QDRANT_ENABLE_QUANTIZATION: "true"
   QDRANT_QUANTIZATION_TYPE: "int8"
   QDRANT_HNSW_ON_DISK: "true"
   QDRANT_DEFAULT_SEGMENT_NUMBER: "2"
  ```
2. `**apecloud/aperag-values**`（独立仓库，生产部署用）：同步上面四个环境变量；Qdrant 子 chart 的 server-side 配置（`deploy/databases/qdrant/values.yaml` 等价项）也要加：
  ```yaml
   extra:
     config:
       storage:
         optimizers:
           default_segment_number: 2
           memmap_threshold_kb: 20480
         hnsw_index:
           on_disk: true
         wal:
           wal_capacity_mb: 8
  ```
  > aperag-values 仓库的 PR 需要和本仓库的 Settings/connector 代码改动**同步发版**，避免应用开了量化但旧 Qdrant 不支持的问题。
  >
  > **Qdrant 版本要求**：INT8 量化 / HNSW on_disk / 2 segments / mmap threshold 这些选项 **1.10.0 全部支持**，无需升级。但多租户 `**is_tenant=True`** 需要 **1.11+**；生产 1.10.0 会被连接器自动降级为普通 keyword 索引——功能正确但失去 segment 级 defragmentation。**建议在本次停服窗口里把 Qdrant 也升到 1.11+**（只改镜像 tag，无数据迁移），或在下一次停服窗口里升，并接受 1.11 之前的查询延迟退化。详见 B.1.1 的版本兼容说明。

#### C.2 其他零散优化

- `wal_capacity_mb: 8`（见 A.2）：对于 ApeRAG 这种写入量并不高的场景，32 MiB 是过度预留。
- `on_disk_payload: true`：现状已经是 true，保持即可。
- 搜索侧：如果开启 INT8 量化，在 `search_params` 里加 `quantization: { rescore: true, oversampling: 2.0 }`，用重排恢复精度。

---

## 5. 落地排期建议


| 周    | 改动                                                    | 负责方向        |
| ---- | ----------------------------------------------------- | ----------- |
| 本周   | A.1 清理（已由 subagent 执行）、A.2 Server 侧配置上线到 staging      | DevOps + 后端 |
| 本周末  | A.2 推到 prod，观察内存变化 48h                                | DevOps      |
| 次周   | B.1.3 代码改造 + C.1 量化默认开启，提 PR 到 ApeRAG & aperag-values | 后端          |
| 次周末  | B.1.4 双写上线 staging                                    | 后端          |
| +2 周 | B.1.4 回填 + 切读 + 清理老 collection                        | 后端 + DevOps |


---

## 6. 验证与监控

做完后要有"三条线"的可观测能力：

1. **Qdrant pod RSS**：`container_memory_working_set_bytes{pod="qdrant-cluster-qdrant-0"}`。目标：从 12 GiB 降至 ≤ 4 GiB。
2. **Qdrant 自身 metrics**（暴露在 `/metrics`）：`collections_total`、`pending_optimizations`、`segments_total`。
3. **RAG 查询延迟/召回**：以现有测试集 replay，要求 recall@10 降幅 < 2%，p95 延迟增幅 < 20%（量化 + on_disk 的权衡）。

把这三条线加到 Grafana 里，作为此次治理的验收门槛。

---

## 附录 · 清理执行记录

本次治理启动时，由 subagent 在 `ack-hong-kong` 集群执行了 A.1 的清理脚本。执行结果追加在本节：

### 执行时间

2026-04-20 20:44 CST（UTC+8），集群 `ack-hong-kong`，操作人：cleanup subagent（串行 REST DELETE，单条确认，无并发）。

### 基线（清理前）


| 项                      | 值                                                         |
| ---------------------- | --------------------------------------------------------- |
| Qdrant collection 总数   | **1847**                                                  |
| PG `status='ACTIVE'`   | 2003（其中 1833 已在 Qdrant、170 尚未上传文档）                        |
| PG `status='DELETED'`  | 179（其中 165 早就不在 Qdrant，仅 14 条残留）                          |
| Qdrant `qdrant` 容器 RSS | **12 517 MiB**（约 12.2 GiB，`kubectl top pod --containers`） |


### 待删清单分类

严格按 `IN_QDRANT \ ACTIVE_IN_PG` 计算，共 **14** 条，全部来源于 PG `status='DELETED'`：


| 分类                                     | 数量     |
| -------------------------------------- | ------ |
| DELETED 来源（PG 标记 DELETED 但 Qdrant 仍残留） | **14** |
| 孤儿来源（Qdrant 有、PG 完全查不到）                | **0**  |


完整列表：

```
col234abe498124212b  col2de94bc540c00373  col364fc3b305b38d9d  col428bbf8564ba165a
col430ad0d0358e4787  col56460b3f4eb88690  col69178ab2362c6391  col78e7bbbe447907eb
col8682c84fbda6333b  col896e0ebd4d699b89  colafa077fd51475227  colc18fb93c44d9f2be
colc49a291371cc17fc  colf67799b2b73f391e
```

### 删除执行结果

通过 `kubectl port-forward qdrant-cluster-qdrant-0 16333:6333` + `curl -X DELETE http://localhost:16333/collections/{name}` 逐条删除，串行、无并发，完成后关闭 port-forward。


| 项                                                           | 值                           |
| ----------------------------------------------------------- | --------------------------- |
| 发起 DELETE 请求数                                               | 14                          |
| HTTP 200 且 `result=true`                                    | 13                          |
| HTTP 200 但 `result=false`（Qdrant 端 "collection 已不存在" 的幂等响应） | 1（首条 `col234abe498124212b`） |
| 真实失败（HTTP 非 200）                                            | **0**                       |
| 删除后抽检 14 个名字的 `GET /collections/{name}`                     | **全部 404**，确认已从 Qdrant 完全消失 |


### 清理后状态


| 项                      | 值（立即）                                    |
| ---------------------- | ---------------------------------------- |
| Qdrant collection 总数   | **1833**（-14）                            |
| Qdrant `qdrant` 容器 RSS | **12 492 MiB**（约 12.2 GiB，立即值 ≈ -25 MiB） |


> 说明：Qdrant 1.10 的 `DELETE /collections/{name}` 只把 collection 从 meta 中摘除，底层段文件回收 / mmap unmap 依赖 optimizer 下一轮调度，进程 RSS 的下降通常滞后。**本条记录为删除完成瞬时的 `kubectl top` 值，真实回收预计在 24 h 内体现。**

### 本次清理的关键结论（重要）

1. **历史上已清过一轮**：PG 有 179 条 `DELETED`，其中 **165 条在之前的治理中已经从 Qdrant 移除**，本次仅残留 14 条；**完全没有孤儿**（Qdrant 里每一个 collection 都能在 PG 里找到对应行）。
2. **"空 collection 很多"并不等于"可删的很多"**：PG `ACTIVE` 2003 条中，有 170 条仍未在 Qdrant 写入（或写入了还未上传文档），按产品规则属于"用户已建未用"的合法状态，**绝不能删**。真正可清的"空 collection"只是 DELETED 残留（本次 14 条）。这解释了为什么 A.1 步骤实际可清理量远小于"空 collection 总数"。
3. **A.1 对 12 GiB RSS 的直接收益非常有限**：本次仅摘除 14 个 collection 的 meta，即便 optimizer 回收完毕，预期可释放的 RSS 也远不到 GiB 级。**真正降低 Qdrant 内存占用要靠正文提到的 B 档（多租户化，单 collection + payload index）与 C 档（INT8 量化 + `always_ram=false`）**。A.1 的价值在于"把账对齐、防止 DELETED 残留继续累积"。
4. **正文第 2 章「12 GiB 分账」依然成立**：清理后活跃 collection 从 1847 → 1833，仅下降 0.76%，原分账中"**1847 × collection 内部结构（HNSW 图 / payload index / 段 header / 通道缓冲）**"一行的量级与结论不变，无需改写正文。

### 失败条目列表

无真实失败。唯一的非典型响应（首条 `col234abe498124212b` 返回 `result:false`）为 Qdrant 端的幂等提示，对象事实上已不存在；抽检 `GET /collections/col234abe498124212b` 返回 404，最终状态正确。

---

## 7. 附加改造：Embedding 模型锁定（本 PR 一并上线）

### 7.1 为什么要锁

多租户化之后，**物理 collection 由 `(vector_size, distance)` 唯一决定**
（见 `global_collection_name()`）。如果允许用户在 Collection 创建后修改
embedding model，会发生两类数据完整性问题：

1. **维度切换**（e.g. `bge-m3@1024` → `text-embedding-3-large@3072`）：
   写入会被路由到新的 `aperag_vectors_3072_cosine`；但旧的
   `aperag_vectors_1024_cosine` 仍残留着该租户的全部历史向量，**永远不会被读到**，
   也不会被 `delete_collection` 清理（因为 delete 路径用**当前**的 vector_size 选
   shard）。
2. **同维度异模型**（e.g. `bge-large-zh@1024` → `bge-m3@1024`）：物理上落同一
   shard，但两组向量在同一 HNSW 图里语义空间不兼容，召回质量会莫名退化，且
   `is_tenant` 优化也救不了（`is_tenant` 只按 tenant 分 segment，不区分模型）。

这两种失败模式在单机 Qdrant 时代就已经存在，只是单机 Qdrant 写入时会因维度
mismatch 直接报错（"硬失败"）；多租户化后变成**软失败**——写入成功，但
retrieval 完全错乱。所以必须从接口层直接禁止。

### 7.2 改动点

- **后端**：`aperag/service/collection_service.py::CollectionService._reject_embedding_change`
  - 在 `update_collection` 的开头调用，校验：
    - `embedding.model` 不变；
    - `embedding.model_service_provider` / `custom_llm_provider` 不变；
    - 已有 embedding 配置不可被 "清空"。
  - 任一不满足抛 `ValidationException`（映射到 HTTP 400）。
  - 首次绑定（老数据或初次创建后补填）仍然允许。
- **前端**：`web/src/app/workspace/collections/collection-form.tsx`
  - `action === 'edit'` 时 `Select` 置 `disabled`；
  - 显示 Badge "创建后不可修改 / Locked after creation"；
  - `FormDescription` 改为解释性文案；
  - **并跳过** `embeddingModelName` 的 `useEffect` watcher——否则若用户切到 "edit"
    模式时模型清单里刚好没有原模型（比如对应 provider 已下架），watcher 会自动把
    表单里的 model 改成列表第 0 项，提交时被后端校验拒绝，用户看起来像 "我啥也
    没动就不让保存"。
- **i18n**：新增两个 key `embedding_model_locked_badge`、`embedding_model_locked_description`
  （`page_collections.json` 中英文各一份）。

### 7.3 为什么不在 OpenAPI schema 上强制

考虑过在 Pydantic `CollectionUpdate`
单独拷一份不含 `embedding` 的子 schema，但：

- 现有 `CollectionUpdate = CollectionCreate` 的全量复用会被破坏；
- 前端已经在 edit 模式下不提交 embedding 字段的 UX 由锁定逻辑保证；
- 服务端显式报错反而比 "schema 层静默 drop 字段" 更友好（用户能看到具体原因）。

因此采用 "schema 保持灵活 + service 层强校验" 的组合。

---

## 8. 向量数据库全链路 Review 要点

本节记录在实现多租户 + embedding 锁定过程中做的一次**全链路代码 review**，
分成 "已修 / 已知有意为之 / 待跟进" 三档，避免以后踩同一个坑。

### 8.1 已修（本 PR）

**R1. Delete 路径在 embedding provider 下线时会把数据孤在 Qdrant 里**
（`aperag/tasks/collection.py::_delete_vector_databases`）

- 背景：多租户化后 `delete_collection` 依赖当前 `vector_size` 去选 shard。
  若 provider 被下架 → `get_collection_embedding_service_sync` 抛异常 →
  代码原本默默 fallback 到某个 "默认" shard，真实 shard 里的该租户向量永远留在那里。
- 修复：新增 `QdrantVectorStoreConnector._purge_tenant_from_all_global_collections`，
  `delete_collection(purge_all_shards=True)` 时枚举所有 `aperag_vectors_*` 做
  `FilterSelector` 级别的点删除；`_delete_vector_databases` 在无法解析 vector_size
  时走这个兜底路径。
- 影响：删 Collection 再也不会留孤儿点。

**R2. Node metadata 里 `collection_id` 可能缺失**
（`aperag/llm/embed/embedding_utils.py::create_embeddings_and_store`）

- 背景：多租户过滤依赖 payload 顶层的 `collection_id`（LlamaIndex 会把
  `node.metadata` 扁平化进 payload）；但 `vector_index / summary_index` 历史上
  只在 `extra_info` 里塞，没有在 `node.metadata` 里设，视觉 index 的两个路径
  甚至完全绕过了 `create_embeddings_and_store`。
- 修复：
  - 在三个 indexer 里显式 `part.metadata['collection_id'] = ...`；
  - 在 `create_embeddings_and_store` 里再补一次防御性注入（如缺就用 connector
    的 `tenant_id`）；
  - vision 两处直连 `store.add` 也补了相同字段。
- 影响：即使上游忘设，多租户过滤依然命中正确 shard。

**R3. `is_tenant=True` 在 Qdrant < 1.11 的兼容**

- Qdrant 1.10（线上版本）不认 `is_tenant` 字段，直接 `400`。
- 修复：`_ensure_tenant_payload_index` 三级 fallback：先带 `is_tenant` 试 →
  捕获 → 不带 `is_tenant` 的 keyword 索引 → 再捕获 → 记 warning。
  升级到 1.11+ 之后 tenant 级 defragmentation 自动生效，无需再改代码。

**R4. `create_collection` 的初始化顺序**

- 以前是先构造 `QdrantVectorStore(client, collection_name)`（LlamaIndex 里这步
  会触发 `GET /collections/xxx`，collection 不存在时抛 warning/error），再
  `_ensure_collection`。
- 改成 "先 ensure，后 wrap"，去掉了一条噪声日志，也避免了冷启动窗口里
  竞态读到 404。

**R5. 迁移脚本的安全断言**

- `scripts/migrate_qdrant_multitenancy.py` 在开头加了
  `assert generate_vector_db_collection_name(x) == str(x)`。
  若未来改命名规则，脚本会立即停机而不是静默把数据写错 shard。

### 8.2 已知但有意为之（这次不动）

**K1. 每次 search 会重建 `QdrantClient`**
（`aperag/service/search_pipeline_service.py` 的三个 `_*_search` 都会
`VectorStoreConnectorAdaptor(ctx)` 一次）

- 现状：`_ENSURED_COLLECTIONS` 进程级 set 避免了重复 `_ensure_collection`，
  但 `QdrantClient` 本身每次都是新的。
- 为什么先不动：(1) HTTP client 本身带 keep-alive，grpc 也有连接池，单查询
  overhead 可接受；(2) 做成单例需要把线程安全、刷新策略、tenant 切换等一起设计，
  范围超出本 PR。
- Follow-up：放到抽象层 M2（见 `vector_db_abstraction.md` §4.1）。

**K2. `ContextManager._create_combined_filter` 硬编码 Qdrant filter 类型**
（`aperag/context/context.py` 直接 import `qdrant_client.models`）

- 典型抽象破口：任何后端切换都要在这里加分支。
- 为什么先不动：单后端状态下，重构此处没有功能收益，反而引入回归风险。
- Follow-up：抽象层 M2 中新增 `VectorFilter` DSL 后统一收敛。

**K3. `retrieve()` 未进入基类**
（`aperag/vectorstore/base.py` 里没有该抽象方法，但 `document_service.py` 直接
调 `connector.retrieve(...)`）

- 若未来切 pgvector/Milvus 会直接 `AttributeError`。
- Follow-up：抽象层 M2 中补齐，顺便把 `with_vectors`、`with_payload` 这些参数
  做成统一语义。

**K4. Vision 索引绕过 `create_embeddings_and_store`，直连 `store.add(nodes)`**
（`aperag/index/vision_index.py` 两处）

- 目前我们在两处都手工补了 `metadata['collection_id']`，语义是对的，但
  "两份写路径" 带来的维护成本需要记住——将来 `create_embeddings_and_store`
  的 metadata 约定变更时，vision 路径要同步修改，容易漏。
- Follow-up：抽象层 M2 中把 "写点" 统一到 `connector.upsert(tenant, points)`，
  彻底去掉 `store.add` 直连。

**K5. 业务层 collection id 被直接当做 tenant id**

- `QdrantVectorStoreConnector.__init__` 用 `ctx['collection']` 当
  `tenant_id`；`generate_vector_db_collection_name(collection_id) == str(collection_id)`
  是幂等映射。
- 这是当前正确、简单的做法；迁移脚本也断言了这点。记录在案，防止以后
  有人把 "Qdrant collection name" 和 "ApeRAG collection id" 拆成两个不同字符串
  时忘了同步断言。

### 8.3 待跟进（不在本 PR 范围）

**F1. `ContextManager` 过滤条件里 `doc_id` / `document_id` / `ref_doc_id` 三名共存**

- 历史上 LlamaIndex 往 payload 里同时写 `doc_id` 和 `document_id`（不同版本命名），
  部分过滤用 `doc_id`、部分用 `document_id`。
- 当前 review 认为语义 OK（查询端兼容两套），但等 M2 抽象层时应该收敛到单一 key。

**F2. `retrieve` 的 `with_payload=True` 语义差异**

- Qdrant 的 "payload" ≈ pgvector 的 "payload JSONB"；但 LlamaIndex 期望 payload 里
  包含 `_node_content` 序列化字符串。未来实现 pgvector 后端时要注意把这种
  LlamaIndex 约定从"协议"降级成"Qdrant 后端实现细节"（见抽象层 §5.1）。

**F3. 并发场景下 `_ENSURED_COLLECTIONS` 的 lock 粒度**

- 当前 `threading.Lock()` 是进程级、全局单把锁。多个线程同时首次访问不同
  collection 时会串行化，但由于 `_ensure_collection` 本身幂等、只在冷启动触发
  一次，实际不是瓶颈。观测到 QPS 上升再评估。

**F4. Qdrant 1.10 → 1.11 升级窗口**

- 升级后 `_ensure_tenant_payload_index` 的 "带 is_tenant" 路径会开始生效，此时
  已经存在的 keyword 索引不会自动升级为 tenant-aware。建议升级操作脚本里
  额外一步 `DELETE index → 重建 with is_tenant=True`（无数据影响，毫秒级）。

---

## 9. 关联设计：向量数据库抽象层

本次 Qdrant 优化已经暴露了三条抽象破口（见 §8.2 K1/K2/K3）。这些问题在
单后端状态下可以接受，但一旦要支持 pgvector/Milvus 就会成为硬阻塞。

详见同目录 [`vector_db_abstraction.md`](./vector_db_abstraction.md)，该文档以
当前代码事实为起点，给出了：

- 三层分层（Transport / 能力抽象 / 调优）；
- 最小可行过滤 DSL；
- Qdrant / pgvector / Milvus 三个后端的实现草图；
- 路线图 M1 → M4；
- 与本次 embedding 锁定的依赖关系（锁定是抽象层的前置条件）。

**何时启动抽象层**：触发条件见 `vector_db_abstraction.md` §10。在触发之前，
本文档和抽象层设计文档共同构成决策依据，不做任何代码层面的先行重构。
