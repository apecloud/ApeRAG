---
title: 开发指南
description: ApeRAG 开发环境设置、工作流程、模块化后代码落位 + 边界门禁
---

# 🛠️ 开发指南

本指南面向想从源代码跑 ApeRAG 或贡献代码的开发者，讲三件事：

1. **开发环境怎么搭**（§1）
2. **模块化之后新代码写在哪、边界怎么守**（§2、§3 —— Phase 0→6 重构的核心结果）
3. **常见开发任务怎么做**（§4）

如果你只是想快速跑起来，直接看 §1。如果你要给某个 domain 加功能，**先读 §2 再动手**，否则很容易撞 boundary gate 失败。

---

## 1. 🚀 开发环境设置

按顺序执行，每一步都会验证上一步。

### 1.1 📂 克隆仓库并配置环境变量

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
```

`.env` 默认值和下一步 `make infra-up` 起的本地数据库匹配，一般无需改。接入真实 LLM provider 时再改 `OPENAI_API_KEY` 之类的。

### 1.2 📋 系统前提

- **Node.js** ≥ 20（前端开发要；仅后端开发可以不装）— [下载](https://nodejs.org/)
- **Docker + Docker Compose**（本地数据库服务要）— [下载](https://docs.docker.com/get-docker/)
- **Python 3.11** —— 不用手动装，下一步的 `uv` 会自动拉起来

### 1.3 🗄️ 启动数据库服务

```bash
# PostgreSQL + Redis + Qdrant + Elasticsearch
make infra-up
```

后台启动所有必须的数据库。`.env` 默认连接字符串已经指向它们。

<details>
<summary><strong>高级数据库选项</strong></summary>

```bash
# 用 Neo4j 代替 PostgreSQL 作图存储
make infra-up WITH_NEO4J=1
```

</details>

### 1.4 ⚙️ 设置 Python 开发环境

```bash
make env-dev
```

这一步会：

- 如果还没装，下载 `uv`
- 创建 Python 3.11 虚拟环境（位置 `.venv/`）
- 安装后端依赖

**激活虚拟环境**：

```bash
source .venv/bin/activate
```

终端提示符里出现 `(.venv)` 代表激活成功。

### 1.5 📦 安装依赖

```bash
make env-install
```

一次性装齐：`pyproject.toml` 里所有 Python 依赖 + `web/` 前端的 `yarn install`。

### 1.6 🔄 应用数据库迁移

```bash
make db-migrate
```

Alembic 会按 `aperag/migration/versions/` 的顺序把 schema 推到最新。

### 1.7 ▶️ 启动开发服务

在不同终端窗口跑：

| 终端 | 命令 | 做什么 |
| --- | --- | --- |
| 1 | `make serve-api` | FastAPI 后端，`http://localhost:8000`，代码改动自动 reload |
| 2 | `make serve-worker` | Celery worker，处理异步后台任务 |
| 3（可选） | `make serve-web` | 前端 dev server，`http://localhost:3000`，hot reload |

### 1.8 🌐 访问入口

- 前端 UI：<http://localhost:3000>（启动了 web 才有）
- 后端 API：<http://localhost:8000>
- API 文档：<http://localhost:8000/docs>

### 1.9 ⏹️ 停止服务

```bash
# 停数据库服务，保留数据
make stack-down

# 停数据库服务并删除所有 data volumes（⚠️ 永久删数据）
make stack-down REMOVE_VOLUMES=1
```

开发服务（`make serve-api` / `serve-worker` / `serve-web`）在各自终端 `Ctrl+C`。

**验证数据是否被清掉**：

```bash
docker volume ls | grep aperag
# REMOVE_VOLUMES=1 后这里应该什么都不返回
```

现在本地环境跑起来了 🎉，继续往下读之前，**请先读 §2**。

---

## 2. 📦 模块化后的代码布局

Phase 0→6 重构把后端从「所有业务堆在 `aperag/service/*` + `aperag/db/models.py` + `aperag/schema/view_models.py` 三个大聚合层」拆成了 **12 个业务 domain**，新代码几乎都该落在 `aperag/domains/<domain>/` 下。

### 2.1 12 Domain 清单

| Domain | 做什么 |
| --- | --- |
| `identity` | 用户账号 / Role / OAuth / fastapi-users |
| `governance` | API Key 管理 + 审计日志 |
| `model_platform` | LLM provider / 模型配置 / 默认模型（v1 + v2 两个 router） |
| `marketplace` | 公开 collection 发布 / 订阅 |
| `knowledge_base` | Collection / Document / CollectionSummary（主领域） |
| `indexing` | 索引 reconciler + 各类 worker（vector / fulltext / graph / summary / vision） |
| `retrieval` | 检索 pipeline 编排 + chunk 聚合 + reranking |
| `knowledge_graph` | 实体 / 关系 ORM + Nebula + `graphindex` reconciler |
| `conversation` | Bot / Chat / TurnFeedback + 聊天发起编排 |
| `agent_runtime` | Agent turn / SSE / artifact 存储 |
| `evaluation` | 数据集 / 评估 run / judge |
| `web_access` | 爬虫 / URL 阅读 / 网络搜索 |

一段话定位 + cross-ref 到各 domain 细节见 [`docs/zh-CN/architecture/domains.md`](../architecture/domains.md)，完整英文 canonical source-of-truth 在 [`docs/modularization/architecture.md`](../../modularization/architecture.md)。

### 2.2 Per-domain 目录契约

```
aperag/domains/<domain>/
├── db/
│   └── models.py        # SQLAlchemy ORM + 本 domain 拥有的 Enum
├── schemas.py           # 本 domain 的 Pydantic schema（走 dual-hook 绑回 view_models）
├── ports.py             # consumer-owned Protocol（对别人的依赖声明）
├── service/             # 业务逻辑；或直接 service.py
├── api/
│   └── routes.py        # FastAPI 路由；或按前缀切成 <feature>_routes.py
└── __init__.py
```

不是每个 domain 都有全套（比如 `web_access` 没有 DB，`indexing` 没有 API 路由）。

### 2.3 新代码落位决策树

```
要加的功能归属哪个业务 domain？
├─ 归属明确 → aperag/domains/<domain>/ 对应槽位
│                 ├─ 要加 ORM → db/models.py
│                 ├─ 要加 Pydantic → schemas.py（注意 dual-hook）
│                 ├─ 要加业务逻辑 → service/
│                 └─ 要加 HTTP endpoint → api/routes.py（或 <feature>_routes.py）
├─ 归属模糊 → 读 docs/zh-CN/architecture/domains.md §3，对照 12 domain 一段话定位
└─ 不属于任何业务 domain（跨切面基础设施） → 保留在 top-level:
     ├─ LLM 封装 → aperag/llm/
     ├─ DB session helper → aperag/db/base.py
     ├─ 工具 / 常量 → aperag/utils/
     └─ （legacy）aperag/service/*.py 下的 shim + 3 条 legacy-only 服务（quota_service / prompt_template_service / search_pipeline_service 等），不要新增
```

**不要**在 `aperag/domains/` 下造新的 "shared" 子目录（例如 `aperag/domains/common/`）。跨 domain 共享的 Pydantic 原语走 `aperag/schema/common.py`（准入标准严格：≥2 domain 都用 + 纯值对象，见 canonical SSoT Section 2.3）。

### 2.4 Pydantic schema 放哪

- 只有本 domain 用 → `aperag/domains/<d>/schemas.py`，通过 **dual-hook Scenario A**（canonical SSoT Section 3.3）绑回 `aperag.schema.view_models`，老调用方仍能 `from aperag.schema.view_models import X` 导入。
- ≥2 domain 都用 + 纯值对象（无 ORM 依赖 / 无 domain 特定语义） → `aperag/schema/common.py`，准入由 CR 人审（没有 AST 自动检查 —— 故意的，避免规则退化成 catch-all）。

dual-hook 在域文件尾部已有 `_bind_view_models_reexports()` 模板，在新 schema 加完后记得把类名加进 `__all__`。

### 2.5 跨 domain 调用的两种形态

1. **Provider 已搬进 `aperag/domains/`** → 直接 import：

   ```python
   from aperag.domains.knowledge_base.service.document_service import document_service
   await document_service.get_by_id(session, doc_id)
   ```

2. **Provider 还在 `aperag/service/*.py`**（或永久 standalone-infra） → **consumer-owned Protocol + DI 槽**：

   - consumer 在自己的 `ports.py` 写 `Protocol`
   - consumer 在 service 模块级留 `_ops: Optional[XOps]` + `set_x_ops()` + `_get_x_ops()`
   - `aperag/app.py` 启动时把 provider（或 adapter）注入槽
   - **provider 不能 import consumer 的 `ports.py`** —— Protocol 由 consumer 拥有，lesson 9a-quad

   今天 legacy 未搬 provider 还剩一些（如 `search_pipeline_service` 待分类），两条永久 standalone-infra seam：`QuotaOps`（provider = `aperag.service.quota_service`）、`PromptTemplateOps`（provider = `aperag.service.prompt_template_service`）。

更细节的 Protocol + DI pattern、两个子类（A 过渡 / B 永久）、`sys.modules.get(...)` 绕 G1 AST 扫描的技巧，都写在 canonical SSoT Section 3。

---

## 3. 🛡️ Domain 边界门禁 G1–G19 速查

所有边界规则都写成了 `tests/unit_test/test_modularization_boundaries.py` 里的 pytest 测试，跑 `make test-unit` 会全部检查。下面是速查表，详细原因和 AST 扫描实现细节见 canonical SSoT Section 4。

| Gate | 禁 / 要求 | 关键触发点 |
| --- | --- | --- |
| **G1** | `aperag/domains/**` 禁 import `aperag.service.*` / `aperag.schema.view_models` / `aperag.db.models` | 任何你 "偷懒" 从旧聚合路径拿东西 |
| **G4** | domain API handler 的 `required_user` / `current_user` 参数必须有 `Protocol` 类型注解（不允许 `Any`） | 写 route 时类型漏了 |
| **G10 / G3** | `retrieval ↔ knowledge_graph` 单向 —— retrieval 走 `graphindex.integration` 窄口；knowledge_graph 禁反向 import retrieval | KG 代码里不小心 import 了 retrieval |
| **G14** | Phase 2 之后 `aperag/views/collections.py` / `aperag/views/graph.py` 里不能有 retrieval / KG 的 route 装饰器（除一条 410-Gone tombstone） | 老 route 漏删 |
| **KB consumer-owned** | `marketplace_service` / `marketplace_collection_service` / `search_pipeline_service` / `quota_service` 等 legacy provider 禁 import `aperag.domains.knowledge_base.ports` | provider 反向 import consumer 的 Protocol |
| **KB DI smoke** | `import aperag.app` 后 KB 4 个 `_*_ops` 槽必须 non-`None` | 启动 wire-up 漏了一条 |
| **G15** | 非 identity domain 禁 import `Role` | 别处用 `Role.ADMIN`；改成字符串比较 `user.role == "admin"` |
| **G16** | 非 identity domain 禁 import `User` ORM | 读走 `AuthenticatedUser(Protocol)` / `UserView`；写走 `identity_user_ops.*` facade（lesson 9a-sexdec 三层优先级） |
| **G17** | `import aperag.app` 后 7 个 Phase 3+4 槽必须 non-`None`（4 KB + 3 identity `*InitOps`） | startup wire-up 顺序或缺项 |
| **G18 alt** | `import aperag.app` 后 2 个 Phase 5/6 permanent 槽必须 non-`None`（`bot_service._quota_ops` / `runtime._prompt_template_ops`） | 上面两条永久 seam 漏了 |
| **G19** | `aperag/domains/**/api/routes.py` 禁 `from __future__ import annotations` | PEP 563 + FastAPI 204 组合失败，lesson 9a-quatuordec |

实战经验：

- **G1 最容易撞**。写新 domain 时顺手写 `from aperag.db.models import ...` 就红。改成 `from aperag.domains.<d>.db.models import ...`。
- **G15 / G16 撞上来就是改成 Protocol**。别 import `User` / `Role` 本身，读字段走 `Protocol` 声明的窄口。
- **G17 / G18 alt 红意味着启动 wire-up 漏了一条** —— `aperag/app.py` 里对应 `set_x_ops(...)` 那行没跑到，通常是 `import` 顺序问题。
- **G19 撞上来改成普通注解**（去掉文件头 `from __future__ import annotations`）。

---

## 4. ❓ 常见开发任务

### Q: 🔧 如何添加或修改 REST API 端点？

**完整流程**：

1. 定位 domain —— 按 §2.1 / §2.3 决定这个 endpoint 归哪个 domain。
2. 请求 / 响应 schema 在 `aperag/domains/<d>/schemas.py` 加（或如果是共享原语，`aperag/schema/common.py`）。
3. 业务逻辑在 `aperag/domains/<d>/service/` 里加方法。
4. 在 `aperag/domains/<d>/api/routes.py` 加 `@router.post(...)` / `@router.get(...)`。注意：
   - 路由参数用 `Annotated[AuthenticatedUser, Depends(required_user)]`，**不要** 用 `User`（会撞 G16）。
   - **不要** 在 route module 顶部加 `from __future__ import annotations`（G19）。
5. 导出 OpenAPI + 前端 typed client 更新：

   ```bash
   make openapi-generate   # 写 openapi.full.json + openapi.public.json
   make openapi-check      # 验证导出链路干净
   ```

6. 跑测试：

   ```bash
   make test-unit       # 包含 boundary tests（G1-G19）
   make test-http-smoke # HTTP 黑盒 smoke，验证 endpoint 回正常
   ```

### Q: 🗃️ 如何修改数据库模型 / schema？

**迁移工作流程**：

1. 定位 domain —— 实体属于哪个 domain。
2. 改 `aperag/domains/<d>/db/models.py` 里的 SQLAlchemy 类。
3. 生成 migration（Alembic autogenerate 能跨 domain 识别，是因为所有 `db/models.py` 共用 `aperag/db/base.py::Base`）：

   ```bash
   make db-revision  # 在 migration/versions/ 创建新 migration 文件
   ```

4. 人工检查生成的 migration；autogenerate 对 index/constraint rename 和某些复合变更可能出错，必要时手动修正。
5. 应用：

   ```bash
   make db-migrate
   ```

6. 更新 `aperag/domains/<d>/service/` 里相关服务（**不要** 走 `aperag/service/*` shim 加业务逻辑）。
7. 验证：

   ```bash
   make test-all
   ```

> 老 `aperag/db/models.py` 只是 re-export shim（把各 domain 的 ORM 重新导出，让旧 import 路径仍可用）。**不要**在 `aperag/db/models.py` 里加新 class —— 会绕开 domain ownership。

### Q: ⚡ 如何添加带后台处理的新功能？

**流程**：

1. 后端逻辑 → `aperag/domains/<d>/service/`
2. Celery 任务 → `aperag/tasks/`（tasks 文件顺序按旧习惯保留在 top-level，不是 domain-owned；它们 import domain service 执行）
3. 数据库模型 → `aperag/domains/<d>/db/models.py`
4. 生成迁移 + 验证 API 契约：

   ```bash
   make db-revision
   make db-migrate
   make openapi-check
   ```

5. 质量：

   ```bash
   make format && make lint && make test-all
   ```

如果你发现 `aperag/tasks/<x>.py` 里要调 `evaluation` 或 `agent_runtime` 而又想避免 import 循环，参考 `aperag/domains/evaluation/worker.py::dispatch_fn` 的 late-import 模式（canonical SSoT Section 2.5）。

### Q: 🧪 如何运行单元测试和 e2e 测试？

**单元测试**（快，无外部依赖；包含 G1-G19 边界测试）：

```bash
# 跑全部
make test-unit

# 跑单个文件
uv run pytest tests/unit_test/test_model_service.py -v

# 跑单个测试函数
uv run pytest tests/unit_test/test_model_service.py::TestModelService::test_get_models -v

# 只跑边界测试（G1-G19）
uv run pytest tests/unit_test/test_modularization_boundaries.py -v
```

**E2E 测试**（需要服务运行）：

```bash
# 准备：先起数据库 + API
make infra-up
make serve-api                 # 单独终端

# pytest e2e
make test-e2e

# HTTP 黑盒 smoke
make test-http-smoke

# 集成测试
make test-integration

# 跑单个 e2e 文件
uv run pytest tests/e2e_pytest/test_chat.py -v

# 带 -s 看实时 print
uv run pytest tests/e2e_pytest/test_chat.py -v -s

# 性能基准
make test-e2e-perf
```

**全跑**：

```bash
# 单元 + e2e
make test-all

# 用不同后端配置跑
make infra-up WITH_NEO4J=1
make test-all
```

### Q: 🐛 如何调试失败的测试？

1. 单独重跑并打开详细输出：

   ```bash
   uv run pytest tests/unit_test/test_failing.py::test_specific_function -v -s

   # 第一次失败就停
   uv run pytest tests/unit_test/ -x --tb=short
   ```

2. e2e 测试失败通常是服务没起全：

   ```bash
   make infra-up
   make serve-api
   make serve-worker   # 如果测异步任务
   ```

3. 调试工具：

   ```bash
   # pdb 断点
   uv run pytest tests/unit_test/test_failing.py --pdb

   # 捕获日志
   uv run pytest tests/e2e_pytest/test_chat.py --log-cli-level=DEBUG
   ```

4. 修完再验证：

   ```bash
   make format   # 自动修样式
   make lint
   uv run pytest tests/path/to/fixed_test.py -v
   ```

### Q: 📦 如何安全地更新依赖？

**Python**：

1. 改 `pyproject.toml`
2. `make env-install` 同步所有 group + extras
3. `make test-all` 验证兼容

**前端**：

1. 改 `web/package.json`
2. `cd web && yarn install`
3. `make serve-web` 或 `yarn build` 检查编译

### Q: 🚀 如何准备代码上生产？

发布前清单：

1. 代码质量：

   ```bash
   make format        # 自动修
   make lint          # 无 style 违规
   make static-check  # mypy 类型检查
   ```

2. 全面测试：

   ```bash
   make test-all
   make test-e2e-perf
   ```

3. API 一致性：

   ```bash
   make openapi-check
   ```

4. 数据库迁移：

   ```bash
   make db-revision
   ```

5. 类生产栈集成验证：

   ```bash
   make stack-up WITH_NEO4J=1 WITH_DOCRAY=1
   # 手动在 http://localhost:3000/web/ 测
   make stack-down
   ```

### Q: 🔄 如何完全重置开发环境？

**核选项**（会清数据）：

```bash
make stack-down REMOVE_VOLUMES=1
make env-clean

# 从头
make infra-up
make db-migrate
make serve-api
make serve-worker
```

**软重置**（保留数据）：

```bash
make stack-down
make infra-up
make db-migrate
```

**只重置 Python 环境**：

```bash
rm -rf .venv/
make env-dev
source .venv/bin/activate
```

---

## 5. 📚 拓展阅读

- [`docs/modularization/architecture.md`](../../modularization/architecture.md) — canonical 英文 SSoT，所有边界 / permanent seam / shim 清单的权威版本
- [`docs/zh-CN/architecture/overview.md`](../architecture/overview.md) — 架构入口（中文）
- [`docs/zh-CN/architecture/domains.md`](../architecture/domains.md) — 12 domain 通览（中文）
- [`docs/modularization/breaking-changes/`](../../modularization/breaking-changes/) — 各 phase 的变更说明，改老代码前可读
- [`tests/unit_test/test_modularization_boundaries.py`](../../../tests/unit_test/test_modularization_boundaries.py) — 20 条边界测试实现，是理解 G1–G19 最硬的权威

---

*文档基线：`origin/main @ 10cabcf`（PR #1637 merged）。Module-owner / G-gate 定义若发生变化，先改 canonical SSoT，再回头更新本文。*
