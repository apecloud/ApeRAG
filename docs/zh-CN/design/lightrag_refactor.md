# LightRAG 模块重构方案

> Status: **设计与计划文档（仅文档，无代码改动）**。本文与同目录
> [`graph_db_abstraction.md`](./graph_db_abstraction.md) 是姊妹文档，
> 两份一起读才能判断先做哪个、还是一起做。

---

## 0. 本文回答三个问题

1. **LightRAG 模块现在"跟 ApeRAG 架构不搭"具体是什么？**（§3 事实清单）
2. **应该怎么重构它？**（§5 目标形态 + §6 分阶段方案）
3. **LightRAG 重构 vs 图数据库抽象层，先做哪个？还是一起做？**（§7）

---

## 1. 背景

用户原话：

> 我的这个 ApeRAG 项目一开始的 GraphRAG 模块是使用的是 LightRAG 的代码，
> 当时为了快速启动，直接把他们的代码库拿了进来并进行了深度修改。
>
> 但这带来了一个历史负担：
>   1. 他们的代码写得不是特别好
>   2. 作为一个相对独立的模块存在于我的系统中，跟我的架构并不是特别搭
>
> 理论上来说，LightRAG 应该成为一个类似于 Web 架构中的 service 模块，
> 这样其他模块调用和使用它都会变得更顺畅一些。

解构一下，其实是三个独立的问题叠在一起：

| 抱怨 | 实际症结 | 解决手段 |
|---|---|---|
| 他们代码写得不好 | 内部实现质量（方法过多、默认实现坑、命名混乱） | 内部重构（**可延后**） |
| 跟我的架构不搭 | 没有清晰的对外接口 / 类型跨模块泄漏 | 定义 facade（**应先做**） |
| 应该是 service 模块 | 生命周期笨重 / 调用方直接 new 实例 | facade + 生命周期接管（**应先做**） |

**关键判断**：三件事里，"定义对外接口"是**根因**；把它做掉，另外两个
问题一半以上自动消解；而即便只做这一件也能立刻看到架构清晰度的提升。
这是本文建议先做的部分。

---

## 2. 术语澄清："service 模块"是什么

> "类似于 Web 架构中的 service 模块"

在 Web 后端架构里，"service 模块"通常指**进程内的服务层**：夹在
controller/view 与 repository 之间，负责组织业务流程、对外暴露一组
清晰方法、对内协调多个数据源。**不**等于独立部署的微服务（microservice）。

**本文按这个口径定义 LightRAG 重构目标**：

- 目标形态 = ApeRAG 进程内的一个 **service 模块**（in-process）；
- 对外暴露的接口 = 一组方法 + 一组 DTO；
- 对内管理 LightRAG 的全部实现细节，业务代码不再直接 import 它的内部
  类型；
- **物理拆成独立 service** 是**后续可选**步骤（§6.3），不是本次重构的
  硬目标。

这个区分很重要，因为——

- 如果目标是 microservice，现在就要设计 OpenAPI、HTTP 客户端、数据序列
  化、部署流水线。工作量 ≥ 两周。
- 如果目标是 in-process service 模块，**工作量是两天**：新增一个
  facade 文件 + 搬运几处 import。

**本文推荐先做 in-process 形态**，并在 §6.3 说明什么时候再升级成
microservice。

---

## 3. 现状事实清单：跨模块耦合

这一节基于对代码库的实测扫描。LightRAG 的内部类型泄漏到 ApeRAG
其他模块的位置（截至 2026-04-22）：

### 3.1 业务层直接 import LightRAG 内部类型

```text
aperag/tasks/collection.py:27           from aperag.graph import lightrag_manager
aperag/tasks/document.py:112,214,302    from aperag.graph.lightrag_manager import process_document_for_celery, delete_document_for_celery
aperag/service/graph_service.py:22      from aperag.graph import lightrag_manager
aperag/service/graph_service.py:23      from aperag.graph.lightrag.types import KnowledgeGraph
aperag/service/search_pipeline_service.py:265   from aperag.graph import lightrag_manager
aperag/service/search_pipeline_service.py:266   from aperag.graph.lightrag import QueryParam
aperag/service/prompt_template_service.py:156   from aperag.graph.lightrag.prompt import PROMPTS
aperag/db/repositories/graph.py:23      from aperag.graph.lightrag.prompt import GRAPH_FIELD_SEP
```

8 处泄漏，三种不同形态：

1. **manager / 工厂** 被直接 import（4 处）—— 业务代码自己 `create_lightrag_instance`
   并管理 `try/finally`；这就是 §3.3 bug 的温床。
2. **内部 DTO** 被直接 import：`KnowledgeGraph`、`QueryParam`（2 处）——
   换实现就得同步改 DTO 定义。
3. **内部常量** 被直接 import：`PROMPTS`、`GRAPH_FIELD_SEP`（2 处）——
   `aperag/db/repositories/graph.py` 作为最底层都要知道 LightRAG 的分
   隔符，这是**明显的层次反转**。

### 3.2 生命周期管理散落

LightRAG 实例的 `try/finally finalize_storages()` 模式在 6 个业务 handler
里重复：

```text
aperag/service/graph_service.py           5 处 (get_graph_labels / get_knowledge_graph / generate_merge_suggestions / _execute_merge_operation / export_for_kg_eval)
aperag/tasks/collection.py                1 处 (_delete_lightrag)
aperag/graph/lightrag_manager.py          2 处 (_process_document_async / _delete_document_async)
```

重复 = 容易漏。实测里就有 1 处漏写：

### 3.3 已被坐实的 bug：`_graph_search` 没有 finalize

`aperag/service/search_pipeline_service.py:265-273`：

```python
rag = await lightrag_manager.create_lightrag_instance(collection)
param = QueryParam(mode="hybrid", only_need_context=True, top_k=top_k)
context = await rag.aquery_context(query=query, param=param)
if not context:
    return []
return [DocumentWithScore(text=context, metadata={"recall_type": "graph_search"})]
```

**没有 `try/finally`，没有 `finalize_storages()`**。这条代码路径每次
图检索都泄漏一批 storage 对象的引用，等 GC。

**这就是"没有 service 模块接管生命周期"的直接代价**——生命周期责任散在
8 个业务 handler 里，漏 1 个就是 bug，没办法靠 code review 长期盯住。

### 3.4 配置通过环境变量传递

`lightrag_manager.create_lightrag_instance` 每次调用都：

```python
kv_storage = os.environ.get("GRAPH_INDEX_KV_STORAGE")
vector_storage = os.environ.get("GRAPH_INDEX_VECTOR_STORAGE")
graph_storage = os.environ.get("GRAPH_INDEX_GRAPH_STORAGE")
```

—— 函数内部读全局环境变量。后果：

- 单元测试想换后端就要改环境变量，不能用 DI；
- 同一进程不能用不同后端（测试想同时跑 pg 和 neo4j 场景时难办）；
- 配置来源隐式，看函数签名完全不知道它会读 env。

这是 12-factor 的典型 anti-pattern：**配置应该在构造时注入，不是在每次
调用时读**。

### 3.5 `lightrag_manager` 与 ApeRAG globals 的耦合

`aperag/graph/lightrag_manager.py` 导入：

```python
from aperag.db.models import Collection                                 # ApeRAG schema
from aperag.db.ops import db_ops                                        # ApeRAG repository
from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync  # ApeRAG LLM
from aperag.schema.utils import parseCollectionConfig                   # ApeRAG schema
```

换句话说："LightRAG 模块"知道 ApeRAG 的 collection schema、db_ops、embedding
service 是什么。这就是"**跟我的架构不是特别搭**"的具体症状：**不是
ApeRAG 依赖 LightRAG，而是 LightRAG 依赖 ApeRAG**——模块边界被反向
穿透了。

一个清晰的 service 模块应该倒过来：ApeRAG 业务代码 → 调 service 模块 →
service 模块内部用独立的小函数去做事，这些小函数接受注入的
embedding/LLM/storage 工厂，而不是 `from aperag.db.ops import db_ops`。

### 3.6 内部代码体量

```text
aperag/graph/lightrag/            ~8700 行
  ├─ lightrag.py                  1800  (LightRAG 主类)
  ├─ operate.py                   2368  (抽取 / 合并 / 查询 operations)
  ├─ utils.py                     731
  ├─ utils_graph.py               680
  ├─ base.py                      659   (BaseGraphStorage / BaseKVStorage / BaseVectorStorage)
  ├─ prompt.py                    493
  ├─ kg/ (3 后端 + kv + vector)   ~2900
  └─ ...
aperag/graph/lightrag_manager.py   347
```

**~10k 行代码**。这是"作为一个相对独立的模块存在于我的系统中"的规模。
任何深度重构都要考虑这个量级——不是一个下午能搞定的。

---

## 4. 诊断：三类问题，三类解决手段

把 §3 的事实归类：

| 类别 | 症状 | 根因 | 建议 |
|---|---|---|---|
| **接口泄漏**（§3.1、§3.2、§3.3） | 8 处跨模块导入 + 生命周期散落 + 因此产生的 bug | 没有对外 facade | **Phase 1，必做** |
| **配置劫持**（§3.4） | 函数内读 env | 没有依赖注入 | **Phase 1 附带** |
| **反向依赖**（§3.5） | LightRAG 知道 ApeRAG schema | 历史为了快速搬代码的捷径 | **Phase 2，可延后** |
| **内部代码质量**（§3.6） | 巨文件、命名、默认实现坑 | 上游代码被深度修改后没整理 | **Phase 2/3，可延后** |

**Phase 1 必做** 是因为不做就持续踩 §3.3 那类 bug；**Phase 2 可延后**
是因为"他们代码不好"不是紧迫问题——**在一个对外接口干净的模块内部**，
代码再糟都可以慢慢修，**不影响 ApeRAG 其他模块**。

---

## 5. 重构后的目标形态

### 5.1 对外（ApeRAG 业务代码看到的）

**唯一的 public API 位于一个文件**：

```python
# aperag/graph/service.py (NEW — or reuse existing aperag/graph/__init__.py)

from aperag.graph.dto import (
    KnowledgeGraph, GraphLabels, GraphContext,
    MergeSuggestion, MergedNode, KGEvalExport,
    IndexDocumentResult, DeleteDocumentResult,
)


class GraphIndexService:
    """Business-facing service module for knowledge-graph operations.

    Every ApeRAG module that needs to read / write / query the knowledge
    graph imports ONLY from this class and the DTOs next to it. Anything
    else in ``aperag/graph/`` is implementation detail.
    """

    def __init__(self, *, config: GraphIndexConfig) -> None: ...

    async def index_document(self, collection, doc_id, content, file_path) -> IndexDocumentResult: ...
    async def delete_document(self, collection, doc_id) -> DeleteDocumentResult: ...
    async def query_context(self, collection, query, top_k) -> GraphContext: ...
    async def get_labels(self, collection) -> GraphLabels: ...
    async def get_knowledge_graph(self, collection, label, max_depth, max_nodes) -> KnowledgeGraph: ...
    async def generate_merge_suggestions(self, collection, top_k) -> list[MergeSuggestion]: ...
    async def merge_nodes(self, collection, source_ids, target_id) -> MergedNode: ...
    async def export_for_kg_eval(self, collection) -> KGEvalExport: ...
```

**对外就是这 9 个方法 + 约 9 个 DTO**。`PROMPTS` / `GRAPH_FIELD_SEP` 如果
业务层真的需要（§3.1 的 2 处），re-export 到 `aperag/graph/dto.py`；更
好的选择是把它们用到的地方提取成 service 方法（不让常量跨模块）。

### 5.2 对内（不变的 + 微调的）

```
aperag/graph/
├── service.py              <- NEW: GraphIndexService 入口
├── dto.py                  <- NEW: 9 个 DTO 集中定义
├── config.py               <- NEW: GraphIndexConfig（注入式，不再读 env）
├── lifecycle.py            <- NEW: request-scoped rag 缓存 / pool
├── lightrag_manager.py     <- 保留：包成 service 内部的工厂，不再被业务 import
└── lightrag/               <- 保留：全部 ~8700 行代码原样不动
    ├── lightrag.py
    ├── operate.py
    ├── base.py
    ├── kg/
    └── ...
```

**Phase 1 不动 `aperag/graph/lightrag/` 内部的任何一行代码**。这是本方案
的核心克制：重构是外部接口，不是内部实现。

### 5.3 整体图

```text
┌──────────────────────────────────────────────────────┐
│                   ApeRAG 业务代码                     │
│  (graph_service / search_pipeline / tasks / ...)     │
└──────────────────┬───────────────────────────────────┘
                   │ 依赖 import
                   ▼
┌──────────────────────────────────────────────────────┐
│  aperag/graph/service.py  (GraphIndexService)        │  <- facade
│  aperag/graph/dto.py      (DTOs)                     │  <- types
│  aperag/graph/config.py   (GraphIndexConfig)         │  <- config
└──────────────────┬───────────────────────────────────┘
                   │ 内部实现
                   ▼
┌──────────────────────────────────────────────────────┐
│  aperag/graph/lightrag_manager.py  (工厂)            │
│  aperag/graph/lifecycle.py         (生命周期)        │
└──────────────────┬───────────────────────────────────┘
                   │ 包装
                   ▼
┌──────────────────────────────────────────────────────┐
│  aperag/graph/lightrag/  (~8700 lines, 原样不动)     │
│  ├─ LightRAG + operate + base + prompt               │
│  └─ kg/ (PG / Neo4j / Nebula)                        │
└──────────────────────────────────────────────────────┘
```

**两条边界线**：

1. **facade 边界**（`service.py`）：业务代码不能越过这条线往下看。
2. **engine 边界**（`lightrag/` 目录）：service 模块内部不需要关心这个
   目录里怎么实现，只通过 `lightrag_manager` + `lifecycle` 使用它。

---

## 6. 分阶段执行方案

### Phase 1：建立 facade（推荐**先做**，~3 天）

落地清单：

- [ ] 新增 `aperag/graph/service.py`：`GraphIndexService` 类，9 个方法
  的身体是"调 `create_lightrag_instance`  + `try/finally
  finalize_storages()`"的直接搬运。
- [ ] 新增 `aperag/graph/dto.py`：9 个 DTO。
  - `GraphContext`、`KnowledgeGraph`、`GraphLabels`、`MergeSuggestion`、
    `MergedNode`、`KGEvalExport`、`IndexDocumentResult`、
    `DeleteDocumentResult`；
  - `aperag/graph/lightrag/types.py::KnowledgeGraph` 不删，在 DTO 层
    薄包装或直接 re-export。
- [ ] 新增 `aperag/graph/config.py`：`GraphIndexConfig` dataclass。接受
  `kv_storage / vector_storage / graph_storage` 等字段；**从 env 读一次**
  存到单例里，整个 service 生命周期只读这一次。
- [ ] 新增 `aperag/graph/lifecycle.py`：两件事
  - request-scoped `rag` cache（FastAPI dependency）——同一请求内多次
    调 `GraphIndexService` 复用同一个 LightRAG 实例；
  - 进程级 `GraphIndexService` 单例入口。
- [ ] 迁移 8 处业务层 import（§3.1）：
  - `graph_service.py` / `search_pipeline_service.py` / `tasks/collection.py`
    / `tasks/document.py` 改为 `from aperag.graph.service import
    graph_index_service`；
  - `prompt_template_service.py` / `db/repositories/graph.py` 的常量依赖
    单独处理（re-export 或消除）。
- [ ] `search_pipeline_service._graph_search` 的 finalize 漏洞（§3.3）
  **自然消失**——生命周期不再是业务代码的责任。
- [ ] 单元测试：`GraphIndexService` 的契约测试，mock LightRAG 内部。

**工作量估计**：
- 纯添加 facade：半天；
- 迁移 8 处导入 + 回归测试：1 天；
- 生命周期 + 配置注入：1 天；
- 代码 review + 小修小补：0.5 天。

**Phase 1 产出**：业务代码 `grep -r "lightrag" aperag/service aperag/tasks
aperag/db` 为 0 命中。这就是"跟我的架构搭了"的客观指标。

### Phase 2：内部整理（按需，~1 周）

在 Phase 1 完成之后，LightRAG 内部代码怎么烂都**不影响 ApeRAG 其他模块**。
所以 Phase 2 的东西**完全可以不做**，做也只在有明确收益时做：

- [ ] 改名：`aperag/graph/lightrag/` → `aperag/graph/engine/`（或别的
  不带 "LightRAG" 字样的名字）。一次性改所有 import，~30 个文件的字符
  串替换，纯机械工作。
- [ ] 消除 §3.5 的反向依赖：
  - `lightrag_manager.py` 不再 import `aperag.db.models.Collection`；
    改成接受 "workspace_id + collection_config + embedding_func +
    llm_func" 四个原始参数；
  - `kg/pg_ops_sync_*.py` 内部不再 `from aperag.db.ops import db_ops`；
    通过构造注入一个 `DbOps` 接口；
  - 这一步让"engine"模块**理论上**可以独立发布成 package；**实际上**
    本次不做这一步，标在 §6.3 未来拆服务时做。
- [ ] 拆分 `lightrag.py`（1800 行）和 `operate.py`（2368 行）：两个文件
  各自拆成 3~5 个小文件。**看需要做**。
- [ ] 清理 R1/R2/R3 / R4/R5/R6 的内部问题（详见
  [`graph_db_abstraction.md`](./graph_db_abstraction.md) §3）。

**Phase 2 最大的收益**不在代码质量本身，而在于**让"这块代码将来真的能
搬出去"变得可能**。Phase 1 只是画了门，Phase 2 是确保门后的房子可以被
整体搬走。

### Phase 3：物理拆成独立 service（远期，独立项目）

当且仅当以下**至少一个**信号出现再启动：

- 图索引的 LLM 调用集中消耗了应用进程的 GIL / 异步调度预算；
- 不同租户的图索引需要**独立的资源配额**（CPU / 内存隔离）；
- 要让图索引集群独立水平扩展（比如接海量文档摄入）；
- 有独立的 lightrag team 要专门维护这个 service。

如果以上都没有，**拆成独立 service 是负收益**：新增运维组件、RPC 延迟、
版本兼容面。

Phase 3 本身的设计见 [`graph_db_abstraction.md`](./graph_db_abstraction.md) §4.1
的拓扑图；本文档不再展开。

---

## 7. 先做 LightRAG 重构还是先做图 DB 抽象？

**推荐：先做 LightRAG 重构的 Phase 1**。理由如下。

### 7.1 两份方案的关系重读

| | 图 DB 抽象方案（姊妹文档） | LightRAG 重构方案（本文） |
|---|---|---|
| **Layer A**（`BaseGraphStorage` + 3 后端） | M3（按需清理） | Phase 2 内部整理（按需） |
| **Layer B**（`GraphIndexService`） | M2 | **Phase 1 核心交付物** |
| **小清洁**（R1/R2/R3） | M1 | Phase 1 自然消化 |
| **物理拆 service** | M4 | Phase 3 |

**Layer B 就是 LightRAG 重构的 Phase 1 对外面**——不是两件事，是一件事
的两个视角。

### 7.2 先做 LightRAG 重构的理由

1. **"存储抽象"这件事的价值**在 LightRAG 没有 facade 的时候**是负的**。
   你辛苦把存储层抽象干净了，业务代码依然直接调 `rag.get_knowledge_graph`
   / `rag.aquery_context`，表面上看不出抽象的好处。
2. **facade 一旦建立，storage 抽象的问题大部分消失**：Layer A 是 LightRAG
   内部的事，对 ApeRAG 业务层不可见，怎么设计都无所谓。
3. **R1/R2/R3 的 bug 会被 Phase 1 自然修掉**。分开做反而增加冲突。
4. **Phase 1 的 diff 很小、风险很低**：约 3 天，纯添加 + 改 import；
   不改内部实现。不会破现有行为。
5. **Phase 2 以后的任何"内部清理"**都有一个干净的外部边界做保护——可以
   放心大胆修内部。没有 Phase 1 这个边界，任何内部清理都有外泄风险。

### 7.3 先做图 DB 抽象的"反论"和回应

**反论 A**："图 DB 抽象的图后端场景已经实际存在（PG/Neo4j/Nebula 都在
用），更紧迫。"

回应：三个后端**已经在 work**（通过 `GRAPH_INDEX_GRAPH_STORAGE` env
切换）。紧迫的不是抽象层，是**业务代码不受内部重构影响**——这恰恰是
facade 解决的问题。

**反论 B**："Layer A 的设计已经基本稳定（`BaseGraphStorage` + 25 测试），
立刻落地没风险。"

回应：**现状已经稳定，没有必要现在动它**。M3 的接口分层/清理是"锦上添花"，
不做不会挂。

**反论 C**："先做小的（storage）练手，再做大的（facade）？"

回应：facade 的 diff **不比** storage 清理大。而且 facade 做完之后，
storage 清理的收益变成"纯内部的代码整洁度"，不是架构级收益。

### 7.4 顺序建议（明确）

**1.** Phase 1（LightRAG 重构 facade）——~3 天，必做
**2.** 暂停，观察 1~2 个月
**3.** 按需做 Phase 2（内部整理）或图 DB 抽象 M3（存储层清理）——两者
   都成为"内部装修"，随意排序，取决于哪块先出现痛点
**4.** 远期如有触发信号，启动 Phase 3（物理拆 service）

**不推荐同时做 Phase 1 + 图 DB 抽象 M2/M3**。原因：
- Phase 1 本身的 diff 已经能波及 8 个文件；
- 同时改内部会让 review 难度和 rollback 风险陡增；
- 本来就是同一个方向，没必要并发。

---

## 8. 反过度设计：什么时候**不**做

一如既往地列一下不做的条件：

- **ApeRAG 只有 1 人维护，且图功能不是卖点** → Phase 1 的 3 天也省了，
  现状能 work 就行。
- **短期内（半年）没有新增图后端 / 新增调用方 / 大规模扩容的计划** →
  Phase 2 永远不做都行。
- **确信 LightRAG 生命周期内都是内嵌的 ApeRAG 一部分，没打算拆出去** →
  Phase 3 不存在。

**只有当至少一个下列信号出现时，启动 Phase 1**（实际上几乎一定会有）：

- 又出现一次类似 §3.3 的生命周期漏写 bug；
- 新增一个业务模块要调图能力，又得重新 import `lightrag_manager`；
- LightRAG 内部代码要做任何超过 100 行的改动，并且担心外溢。

---

## 9. Open questions

不影响本次判断、但落地 Phase 1 时要答：

### Q1. `GraphIndexService` 是**单例**还是**per-collection 实例**？

两种做法的取舍：

- **单例**：进程级一个 service 对象；`create_lightrag_instance(collection)`
  被封在方法内，每次调用按 collection 构造 rag 实例（+ request-scoped
  cache）。优点：调用简单 `graph_index_service.query_context(collection, ...)`。
- **Per-collection 实例**：service 工厂 `make_graph_index_service(collection)`，
  返回的 service 对象绑定 collection。优点：方法签名更简洁 `svc.query_context(...)`，
  不用每次传 collection。

倾向 **单例** + 方法接受 `collection` 参数——配合 FastAPI 的 DI，单例注入
很自然；per-collection 实例在 Celery 任务里生命周期管理复杂。

### Q2. Phase 1 的 DTO 层和 LightRAG 的 types 如何协调？

`aperag/graph/lightrag/types.py::KnowledgeGraph` 已经存在并被业务层
import。Phase 1 的选项：

1. 直接 re-export：`from .lightrag.types import KnowledgeGraph` in `dto.py`。
   零成本、零破坏。
2. 复制定义：`dto.py` 定义自己的 `KnowledgeGraph`，内部做 adapter。
   解耦彻底但前期一次性成本。

倾向 **方案 1**：Phase 1 追求"零 diff 引入"；真要解耦等 Phase 2 做内部
重命名时顺手做。

### Q3. Celery 路径怎么融入？

`aperag/graph/lightrag_manager.py::process_document_for_celery` /
`delete_document_for_celery` 是同步入口（为 Celery task 设计）。这两个
函数**就是 Celery 任务入口**，迁移时不能简单换成 `graph_index_service.index_document`
的 async 版本——Celery worker 不跑 asyncio 主循环。

方案：`GraphIndexService` 提供**同步包装方法** `index_document_sync()` /
`delete_document_sync()`，内部用 `_run_in_new_loop`（现在
`lightrag_manager` 里已经有了）。Celery 任务调同步方法，FastAPI 调异
步方法，同一个 service 对象。

### Q4. Phase 1 的回归测试怎么做？

- 单元层：mock 掉 LightRAG，只测 service 的 9 个方法签名 + 委托行为。
- 集成层：复用 `tests/integration/graphstorage/` 已有的 3 个后端测试集，
  再加一层：通过 `GraphIndexService` 跑一遍端到端（create collection →
  index doc → query context → delete doc）。

---

## 10. 与向量抽象层重构的方法论对比

| | 向量抽象（已落地，PR #1556） | LightRAG 重构 Phase 1（本文） |
|---|---|---|
| 起点 | 只有 Qdrant 一个后端 + 即将加 pgvector | 三个后端已在用 + 不会新增后端 |
| 重构驱动力 | "要加 pgvector 就不得不抽象" | "不加 facade 就没法修内部 bug" |
| 工作量 | ~1 周（Qdrant + pgvector 一起做完） | ~3 天（仅 facade） |
| 对外 API 稳定性 | 历史较乱，本次一次性重设 | 仅 re-export 已有类型 |
| 内部改造范围 | 深度（引入 DTO + 抽象 upsert/search 全套） | **刻意最小化**（不动 ~8700 行 lightrag/ 内部） |

方法论上的共同原则：**一次做完，不留尾巴，干净的外部边界保护内部折腾
自由**。差别在于向量层可以一次干掉、LightRAG 这块体量太大只能分阶段。

---

## 11. 结论

### 先做什么

**Phase 1：LightRAG facade**。~3 天工作量，定义 `GraphIndexService` +
9 个 DTO + 生命周期封装，迁移 8 处跨模块 import。

### 再做什么

观察 1~2 个月，看 Phase 1 之后真实使用中出现的痛点，决定：

- Phase 2 内部整理 vs 图 DB 抽象 M3 清理 —— 两者都是 "内部装修"，按痛点
  优先级排。
- 如果出现 §6.3 / §8 末尾列出的信号 —— 启动 Phase 3（独立 service）。

### 不要做什么

- **不要** 现在就全面重写 LightRAG 内部；
- **不要** 在做 Phase 1 之前做图 DB 抽象层的 M2/M3；
- **不要** 在没有触发信号时启动 Phase 3。

---

## 12. 落地清单（可直接转为 issue）

Phase 1 的任务拆解，按依赖顺序：

1. `aperag/graph/dto.py` —— 9 个 DTO，re-export 现有类型为主。
2. `aperag/graph/config.py` —— `GraphIndexConfig`，从 env 读一次。
3. `aperag/graph/lifecycle.py` —— request-scoped rag 缓存、单例入口。
4. `aperag/graph/service.py` —— `GraphIndexService` 类，9 个方法。
5. 迁移 `aperag/service/graph_service.py`（5 处调用）。
6. 迁移 `aperag/service/search_pipeline_service.py`（1 处，修掉 §3.3 bug）。
7. 迁移 `aperag/tasks/collection.py`、`aperag/tasks/document.py`。
8. 处理 `aperag/service/prompt_template_service.py`、
   `aperag/db/repositories/graph.py` 的常量依赖。
9. 回归测试：现有 `tests/integration/graphstorage/` + 新增
   `tests/unit_test/graph/test_graph_index_service.py`。
10. 更新文档：`graph_db_abstraction.md` 标注 Layer B = `GraphIndexService`
    已落地；本文档标注 Phase 1 已完成。

**预估总工作量：3 天（1 个 PR，diff 规模 ~500 行净增 + ~200 行迁移改动）**。
