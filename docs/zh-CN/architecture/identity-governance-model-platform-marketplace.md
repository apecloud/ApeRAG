# Architecture: `identity` / `governance` / `model_platform` / `marketplace` Domains

> **读者定位**：后端架构师、Phase 4 四域代码 owner、需要理解"用户身份 / 访问凭证 / 模型配置 / 市场订阅"之间 dependency 边界的开发者。
>
> **范围**：ApeRAG 12 canonical domain 中的 4 个 Phase 4 域：`identity` / `governance` / `model_platform` / `marketplace`。本文只写 **current state**。Phase 0→6 演进史见 `docs/modularization/architecture.md` 和 `docs/modularization/breaking-changes/phase4.md`。
>
> **Baseline**：`origin/main @ 28a9f531` 代码面 + `docs/modularization/architecture.md` Section 2.1–2.4 / 3 / 4（canonical SSoT）。

## 目录

1. [Why 4 个域放一起讲](#why-4-个域放一起讲)
2. [`identity` 域](#identity-域)
3. [`governance` 域](#governance-域)
4. [`model_platform` 域](#model_platform-域)
5. [`marketplace` 域](#marketplace-域)
6. [跨域 dependency graph](#跨域-dependency-graph)
7. [Canonical 规则（G15 / G16 / G17）](#canonical-规则g15--g16--g17)
8. [Boundary gates](#boundary-gates)
9. [Legacy shims 与 Phase 7+ 去向](#legacy-shims-与-phase-7-去向)
10. [相关文档](#相关文档)

---

## Why 4 个域放一起讲

这 4 个域在 Phase 4 被同时从 `aperag/db/models.py` 和 `aperag/views/*.py` 聚合模块里 **hard-cut** 出来（Step 4-S2a/b/c/d + 4-S5a/b/c/d）。它们共享若干 canonical 约定：

- **都声明自己的 `AuthenticatedUser(Protocol)`** — 14 份（其他域也有），刻意不合并（见 Section "Canonical 规则"）。
- **governance / model_platform / marketplace 都禁止 import `identity.Role` 枚举**（G15 边界规则），admin 判断走 `user.role == "admin"` 字面量。
- **没有任何一个非 identity 域能持有 `identity.User` ORM 实例**（G16），要读 User 字段就走 `UserView` Protocol 或者 `identity_user_ops` 写 facade。
- **3 条 identity-side DI adapter** 把 `UserManager.on_after_register` 的三个副作用（默认 bot / 默认 chat / quota 初始化）串到其他域，是 Phase 4 为什么单独引入 G17 的原因。

把这 4 个域放在一篇说，能把这些横向约定讲清楚、不重复。

---

## `identity` 域

### 职责

"谁是谁" — 认证 + 授权主体 + OAuth 账户绑定。

### 目录结构

```
aperag/domains/identity/
├── __init__.py
├── db/
│   └── models.py              # User / OAuthAccount / Role
├── schemas.py
├── ports.py                   # AuthenticatedUser + BotInitOps + ChatInitOps + QuotaInitOps
└── service/
    ├── user_manager.py        # fastapi-users UserManager + on_after_register
    └── identity_user_ops.py   # User 写 facade（G16-legal 入口）
```

注意 **identity 没有自己的 `api/routes.py`**：fastapi-users 自己提供 `/auth/*`、`/users/*` 路由；domain 只提供 ORM + service + port 层。

### ORM 实体

| 实体 | 说明 |
| --- | --- |
| `User` | 主认证实体，fastapi-users-backed；字段含 `id` / `username` / `email` / `role` / `is_active` / `is_verified` / `is_superuser` / `hashed_password` / `chat_collection_id` / `gmt_created` / `gmt_deleted` 等 |
| `OAuthAccount` | 第三方 OAuth 绑定，一个 User 可以绑定多个 provider（GitHub / Google / Microsoft / ...）；外键回 `User.id` |
| `Role` | 字符串枚举 `admin` / `rw` / `ro`，保存在 `User.role` |

### `Role` 枚举的特殊位置

`Role` 定义在 `aperag/domains/identity/db/models.py`，但它**同时**被 `aperag/db/models.py`（legacy 聚合）的 `Invitation` 类 class-body 引用：

```python
# aperag/db/models.py
from aperag.domains.identity.db.models import Role  # ← G15 允许的唯一一次 top-level import
...
class Invitation(Base):
    ...
    role = Column(EnumColumn(Role), nullable=False)  # class-body 加载期就需要 Role
```

**原因**：SQLAlchemy `Column(EnumColumn(Role), ...)` 在类定义时就要求 `Role` 已解析成具体 class，不能延迟到方法内部。`Invitation` 是 Phase 4 结束时还没迁出 legacy 聚合的 transitional 表（Phase 7+ 候选），因此要容忍这一 special case。G15 gate 里对 `aperag/db/models.py` 这个 top-level import 做了白名单豁免。

其他任何非 identity 文件若 import `Role` 都会被 G15 判为违规。详见 Section "Boundary gates"。

### `UserManager.on_after_register` 副作用

fastapi-users 注册用户后调用 `on_after_register`。`aperag/domains/identity/service/user_manager.py` 在这里触发 **4 个初始化副作用**：

1. **第一个注册用户自动提权为 admin**（`Role.ADMIN`）— identity 内部，`user_db.session` 直接写
2. **创建两条 API Key**：一个 `is_system=true`（系统内部），一个 `is_system=false`（默认用户 key）— 直接 `async_db_ops.create_api_key`，governance 域 ORM 的写入走 `db_ops`
3. **初始化用户 quota**：`_get_quota_init_ops().initialize_user_quota(user_id)`
4. **创建默认资源**：
   - `_get_bot_init_ops().create_default_bot_for_user(user_id)` → 默认 Agent Bot（`title="Default Agent Bot"`）
   - `_get_chat_init_ops().create_default_chat_for_user(user_id)` → 默认 chat collection

步骤 3 + 4 里的 `*InitOps` 是 **consumer-owned Protocol**：identity 域声明接口，具体 adapter 在 `aperag/app.py` 启动时注入。这是 Phase 4 引入的"identity consumer / legacy provider"跨域形态。

### 三条 identity DI slot（G17 CRITICAL_WIRINGS）

`aperag/domains/identity/ports.py` 定义：

```python
@runtime_checkable
class BotInitOps(Protocol):
    async def create_default_bot_for_user(self, user_id: str) -> None: ...

@runtime_checkable
class ChatInitOps(Protocol):
    async def create_default_chat_for_user(self, user_id: str) -> None: ...

@runtime_checkable
class QuotaInitOps(Protocol):
    async def initialize_user_quota(self, user_id: str) -> None: ...
```

`user_manager.py` 提供 setter：

```python
def set_bot_init_ops(ops: BotInitOps) -> None: ...
def set_chat_init_ops(ops: ChatInitOps) -> None: ...
def set_quota_init_ops(ops: QuotaInitOps) -> None: ...
```

wire-up 发生在 `aperag/app.py`（import-time 执行，FastAPI 启动前）：

```python
class _BotInitOpsAdapter:
    async def create_default_bot_for_user(self, user_id: str) -> None:
        from aperag.db.models import BotType
        from aperag.schema.view_models import BotCreate
        from aperag.service.bot_service import bot_service
        await bot_service.create_bot(user_id, BotCreate(..., bot_type=BotType.AGENT, title="Default Agent Bot"), skip_quota_check=True)

class _ChatInitOpsAdapter:
    async def create_default_chat_for_user(self, user_id: str) -> None:
        from aperag.service.chat_collection_service import chat_collection_service
        ...

class _QuotaInitOpsAdapter:
    async def initialize_user_quota(self, user_id: str) -> None:
        from aperag.service.quota_service import quota_service
        await quota_service.initialize_user_quotas(user_id)

_id_set_bot_init_ops(_BotInitOpsAdapter())
_id_set_chat_init_ops(_ChatInitOpsAdapter())
_id_set_quota_init_ops(_QuotaInitOpsAdapter())
```

关键点：

- **3 个 adapter 都 lazy-import** provider 模块（`bot_service` / `chat_collection_service` / `quota_service`），避免 `identity` 域 startup 时强依赖未移入 domain 的 legacy 模块。
- 3 个 provider 目前都 **不在 identity 域**：`bot_service` 实际被 Phase 5 移到 conversation 域（通过 `A is B is C` 三元 shim 让 `aperag.service.bot_service` == `aperag.domains.conversation.service.bot_service`）；`quota_service` / `chat_collection_service` 仍在 `aperag/service/` 作为 standalone-infra。
- 即便 provider 是 domain-moved，identity 这边**仍通过 adapter 走 Protocol+DI**，而不是直接 import — 因为 identity 域保守处理："consumer-owned Protocol" 是 identity 首选契约，遵循 lesson 9a-quad。

这 3 条 slot 合称 **G17 的 3 条 identity adapter**，加上 Phase 3 knowledge_base 的 4 条 slot，共 **7 条**构成 G17 registry。详见 Section "Boundary gates" 的 G17。

### `identity_user_ops` — User 字段的写 facade

G16 禁止非 identity 域 import `User` ORM 做写入。但业务总会有跨域场景需要改 User 字段（例如 knowledge_base 在用户创建默认 chat collection 时要 set `User.chat_collection_id`）。

解决方案：`aperag/domains/identity/service/identity_user_ops.py`：

```python
async def set_chat_collection(session: AsyncSession, user_id: str, collection_id: str) -> None:
    user = await session.get(User, user_id)
    if user is None:
        return
    user.chat_collection_id = collection_id
    session.add(user)
```

消费方（如 `chat_collection_service`）这样用：

```python
from aperag.domains.identity.service.identity_user_ops import set_chat_collection
await set_chat_collection(session, user_id, collection_id)
```

这是 **lesson 9a-sexdec（User write hierarchy）的终态 level-1**：

1. **首选**：走 `identity_user_ops.<method>` facade（identity-owned 写入）
2. **次选**（单调用点 + PM 确认）：inline text SQL
3. **禁止**（G16）：非 identity 域 `from aperag.db.models import User` 做 ORM 写

Phase 4 结束时 hierarchy level-1 已经覆盖所有真实写入路径；剩余只是 Phase 6 cleanup 继续收缩 inline SQL。

### `AuthenticatedUser` Protocol（identity 域本地声明）

即使在 identity 自己的 routes（fastapi-users 提供）里需要类型收窄，identity 也**不**把 `AuthenticatedUser` 推广为跨域共享合约：每个域声明自己的 `AuthenticatedUser(Protocol)`，只 pin 自己实际读取的属性。14 份刻意重复（不是忘了合并，是 Phase 4 design-lock 的 canonical choice）。

见 SSoT Section 4 和本文 Section "Canonical 规则"。

---

## `governance` 域

### 职责

"谁在什么时间做了什么" — API 访问凭证 + 审计日志。

### 目录结构

```
aperag/domains/governance/
├── __init__.py
├── db/
│   └── models.py              # ApiKey / AuditLog / ApiKeyStatus / AuditResource
├── schemas.py
├── ports.py                   # AuthenticatedUser + UserView
├── service/
│   ├── api_key_service.py
│   └── audit_service.py
└── api/
    └── routes.py              # /apikeys/* + /audit-logs/*
```

### ORM 实体

| 实体 | 说明 |
| --- | --- |
| `ApiKey` | Bearer Token，形如 `sk-<hex>`；字段含 `user` / `description` / `status` / `is_system` / `last_used_at` / `gmt_*` |
| `AuditLog` | 变更操作审计，字段含 `user_id` / `username` / `resource_type` / `resource_id` / `api_name` / `http_method` / `path` / `status_code` / `request_data` / `response_data` / `start_time` / `end_time` 等 |
| `ApiKeyStatus` | `ACTIVE` / `DELETED` |
| `AuditResource` | 资源类型枚举（collection / document / bot / chat / ... 19 种） |

用户面接口见 [`admin-guide/api-keys.md`](../admin-guide/api-keys.md) + [`admin-guide/audit-log.md`](../admin-guide/audit-log.md)。

### `UserView` Protocol — governance 读 User 的唯一路径

governance 的审计 / admin 检查偶尔需要展示 `User.username` 等识别字段。按 G16，governance 不能 import `User` ORM。所以 `ports.py` 声明：

```python
@runtime_checkable
class UserView(Protocol):
    """Read-only user view consumed by governance services."""
    id: Any
    role: str
    # 其他 pin 的属性按实际需求补
```

identity 域的 `User` ORM 结构上满足这个 Protocol（`User.id` + `User.role` 存在），governance 不 import identity，identity 也不 import governance — consumer 侧声明 Protocol + provider 侧 duck-typed 满足，典型 lesson 9a-quad 模式。

### G15 literal compare（governance 的 admin 判定）

governance handler 做 admin 检查时**不** import `Role.ADMIN`：

```python
# aperag/domains/governance/api/routes.py
if user.role == "admin":
    stmt = select(AuditLog).where(AuditLog.id == audit_id)
else:
    stmt = select(AuditLog).where(AuditLog.id == audit_id, AuditLog.user_id == user.id)
```

`user.role` 的类型就是 Protocol 里声明的 `str`。identity 内部 `Role` 枚举的成员字符串值（`"admin"` / `"rw"` / `"ro"`）作为 canonical 约定散在各域的 literal compare 里。要加新 role 就同时改 enum + 各 domain 的 literal compare — 这点 Phase 4 被 PM 确认为**可接受的代价**，换来的好处是 enum 类型不需要跨域传播。

### 跨域 dependency

governance 对其他域的 import：

- ✅ 无 inbound ORM / service import 到 identity / model_platform / marketplace / KB
- ✅ 使用 `aperag.utils.audit_decorator.@audit`（基础设施，跨域使用）

governance 被谁消费：

- 业务 handler 通过 `@audit(resource_type=..., api_name=...)` 装饰器写 AuditLog — 内部走 `aperag.utils.audit_decorator` → `audit_service`；调用方不需要直接 import governance
- identity `UserManager.on_after_register` 直接 `async_db_ops.create_api_key(...)` 写 ApiKey（不走 governance service，是 db_ops 层直接写。**这条是 identity → governance ORM 的直接读写，符合 G16 的"identity 对本域外 ORM 只读写一次"特权例外**）

---

## `model_platform` 域

### 职责

"模型平台配置" — LLM provider 信息 + per-provider 模型清单。

**不是**"runtime LLM 调用"：`aperag/llm/*`（embedding / rerank / completion 的 HTTP 封装）是跨域共享基础设施，**不属于** model_platform 域。见 SSoT Section 2.3 注释。

### 目录结构

```
aperag/domains/model_platform/
├── __init__.py
├── db/
│   └── models.py              # LLMProvider / LLMProviderModel / APIType
├── schemas.py
├── ports.py                   # AuthenticatedUser
├── service/
│   ├── llm_provider_service.py       # provider CRUD
│   ├── llm_available_model_service.py # per-provider models 查询
│   └── default_model_service.py       # system default model config
└── api/
    ├── llm_routes.py                  # v1 API: /llm/embeddings, /llm/rerank
    └── providers_v2_routes.py         # v2 API: /llm/providers, /llm/providers/{id}/models, /default-models
```

### ORM 实体

| 实体 | 说明 |
| --- | --- |
| `LLMProvider` | 某个 provider（OpenAI / Anthropic / Alibaba Bailian / ...）的配置：dialect / base_url / 默认 API key / per-user override 规则 / public vs private |
| `LLMProviderModel` | provider 下某个具体 model 的元数据：`name` / `api`（completion / embedding / rerank）/ context window / tags / 能力标记 |
| `APIType` | `completion` / `embedding` / `rerank` |

`ModelServiceProvider` 是一张历史表（用户 / admin 自定义的模型服务 endpoint），仍留在 `aperag/db/models.py`，Phase 7+ 决定去向。

### 2-router 拆分

```python
# aperag/app.py
from aperag.domains.model_platform.api.llm_routes import router as llm_routes_router
from aperag.domains.model_platform.api.providers_v2_routes import router as providers_v2_router
...
app.include_router(llm_routes_router, prefix="/api/v1")
app.include_router(providers_v2_router, prefix="/api/v1")
```

两份 router 都在同一个 `model_platform.api` 子包下，区别：

- `llm_routes.py` — 运行时推理类接口（`POST /llm/embeddings` / `POST /llm/rerank`），消费方是业务代码（不是前端 UI）
- `providers_v2_routes.py` — 配置管理类接口（`GET|POST|PUT|DELETE /llm/providers`、`/llm/providers/{id}/models`、`/default-models`），消费方是前端 admin / 用户 UI

**为什么 2 router 不合并**：inference vs config 两条路径的生命周期、安全策略、OpenAPI tag 都不一样。合并会引入一份难以维护的超大 router；Phase 5 conversation 域的 `chat_router` + `bots_router` 复用了同样的 2-router pattern。

### Service 层

三个 service 各司其职：

- `llm_provider_service`：provider CRUD + API key 管理（含 per-user override 逻辑）
- `llm_available_model_service`：列出 provider 下的可用 model，支持 tag 过滤
- `default_model_service`：system default model 配置（admin 设置全局默认 embedding / completion / rerank model）

`llm_provider_service.py` 里的函数并非类方法，而是**模块级函数**（`create_llm_provider` / `get_llm_provider` / ...），为了兼容 Phase 4 前的 legacy 调用形态；Phase 6+ 可能统一收进类。

---

## `marketplace` 域

### 职责

"知识库的公开分享与订阅" — publish / unpublish / subscribe / access check。

### 目录结构

```
aperag/domains/marketplace/
├── __init__.py
├── db/
│   └── models.py              # CollectionMarketplace / UserCollectionSubscription
├── schemas.py
├── ports.py                   # AuthenticatedUser 只
├── service/
│   ├── marketplace_service.py                 # publish / unpublish / list
│   └── marketplace_collection_service.py      # subscriber-access 读路径
└── api/
    └── routes.py              # /marketplace/*
```

### ORM 实体

| 实体 | 说明 |
| --- | --- |
| `CollectionMarketplace` | 某个 Collection 的 publish 状态（`DRAFT` / `PUBLISHED`），unique on `collection_id` |
| `UserCollectionSubscription` | 某用户对某已发布 collection 的订阅记录 |
| `CollectionMarketplaceStatusEnum` | `DRAFT`（只 owner 可见） / `PUBLISHED`（公开可见） |

### ports.py 极简

`marketplace/ports.py` 只声明 `AuthenticatedUser(Protocol)`，且只 pin `id` 一个属性 — marketplace handler 只读 owner id，不做 admin-only 检查，不读其他 User 字段。也没有 `UserView` 这种跨域读 User 的 Protocol（因为 marketplace 根本不展示 user info）。

### Q2 rename — `check_marketplace_access` 的公开化

Phase 4 msg=6ab7d211 Q2 的一次 canonical 澄清：

- 原名 `_check_marketplace_access`（前导下划线，semi-private）
- 新名 `check_marketplace_access`（公开方法名）

改动原因：KB 的 consumer Protocol `knowledge_base.ports.MarketplaceCollectionOps` 声明这个方法为公开合约的一部分，service 侧的 `_` 前缀变成矛盾；改名后 service 的类 structurally 满足 Protocol，**不再需要** `aperag/app.py` 的 `_MarketplaceCollectionOpsAdapter`。

这是 Phase 4 canonical 落实的典型例子："方法命名 / 可见性是 consumer Protocol 合约的一部分，不只是 provider 的私事"。

### KB consumer 消费 marketplace 的方式

knowledge_base 域有自己的 `ports.py` 声明 5 条 consumer Protocol，其中一条是：

```python
# aperag/domains/knowledge_base/ports.py
@runtime_checkable
class MarketplaceCollectionOps(Protocol):
    async def check_marketplace_access(
        self, user_id: str, collection_id: str, ...
    ) -> bool: ...
    ...
```

`aperag/app.py`：

```python
from aperag.domains.knowledge_base.service.collection_service import (
    set_marketplace_collection_ops as _kb_set_marketplace_collection_ops,
)
_kb_set_marketplace_collection_ops(_legacy_marketplace_collection_service)
```

这里 `_legacy_marketplace_collection_service` 是 `aperag/service/marketplace_collection_service.py` 的 shim 实例；shim 内部 delegate 到已经 domain-moved 的 `aperag/domains/marketplace/service/marketplace_collection_service.py`。通过 `A is B is C` 三元 shim，wire 的实际对象就是 domain service。shim 保留是 Phase 7+ 去除候选（见 SSoT Section 6）。

### 跨域 dependency

marketplace service 的 inbound：

- ✅ 只 import `knowledge_base.schemas` 的 view models（`Document` / `DocumentList` / `DocumentPreview`）做订阅方文档 preview — 属于 domain-moved 直接 import
- ❌ 不 import identity / governance / model_platform

marketplace 被消费：

- knowledge_base 通过 `MarketplaceCollectionOps` Protocol + DI 消费（见 KB 章节架构）

---

## 跨域 dependency graph

聚焦 4 个 Phase 4 域 + 它们的非本域消费方（KB / conversation），忽略 infra / shared:

```
       ┌──────────────────────────────────────────────────┐
       │       aperag/app.py（adapter + DI wire-up）       │
       └──┬───────────────────┬──────────────────┬────────┘
          │ _*InitOpsAdapter  │ MktCollAdapter   │ KB quota/etc
          ▼                   ▼                  ▼
     ┌──────────┐        ┌─────────────┐    ┌──────────────┐
     │ identity │◀──Protocol─────────────────│ identity     │
     │  (User / │        │             │    │  ports       │
     │   Role)  │───identity_user_ops─▶│    │ (BotInitOps  │
     └────┬─────┘        │             │    │  ChatInitOps │
          │              │             │    │  QuotaInit)  │
          │              │             │    └──────────────┘
          │              │             │
          │    ┌─────────┴──┐    ┌─────┴────────┐    ┌─────────────┐
          │    │ governance │    │ model_plat.. │    │ marketplace │
          │    │ (ApiKey /  │    │  (LLMProv..) │    │ (CollMkpl.) │
          │    │ AuditLog)  │    │              │    │             │
          │    └──────┬─────┘    └──────────────┘    └──────┬──────┘
          │           │                                      │
          │           │ UserView                             │ check_marketplace_access
          │           │ Protocol                             │ (Q2 public rename)
          │           ▼                                      ▼
          │      (structural)                     ┌────────────────┐
          │                                       │ knowledge_base │
          │                                       │  (consumer)    │
          └──────── User ORM read/write ──────────┤ ports          │
                                                  └────────────────┘
```

说明：

- **实线箭头**：直接代码依赖（import / 消费）
- **虚线 / "Protocol"**：consumer-owned Protocol，provider 结构满足，不互相 import
- **`aperag/app.py`**：adapter + DI wire 的唯一物理位置，import-time 执行

---

## Canonical 规则（G15 / G16 / G17）

### 14 份 `AuthenticatedUser(Protocol)` 刻意不合并

每个 domain（`identity` / `governance` / `model_platform` / `marketplace` / `knowledge_base` / `conversation` / `agent_runtime` / `evaluation` / `indexing` / `retrieval` / `knowledge_graph` / `web_access` + `chat_collection_service` 局部 + `auth.py` 本地）各自声明自己的 `AuthenticatedUser(Protocol)`，pin 各自 handler 读的属性。

理由：

1. **最窄契约原则**：每个 handler 真正需要的属性差别大（marketplace 只 pin `id`；governance pin `id` + `role`；KB 的有些 handler pin `id` + `role` + `is_superuser`）。合并到一个全字段 Protocol 会让所有 handler 名义上依赖所有字段，违反最窄契约。
2. **跨域依赖方向控制**：共享 Protocol 需要放在某个域，那个域就变成所有域的上游依赖源；目前每个域 own 自己的 Protocol，避免形成 dependency graph 中的特权节点。
3. **重复成本可接受**：14 份 3-5 行代码 ≤ 70 行，维护成本显著低于引入共享抽象的复杂度代价。

未来若统一契约，会作为 Phase 7+ F5 candidate 处理，不在 Phase 4 scope。详见 SSoT Section 4 和 `docs/modularization/breaking-changes/phase4.md`。

### G15 — `Role` enum 仅限 identity 域持有

`aperag/domains/identity/db/models.py` 的 `Role` 枚举**不允许**被 identity 外的任何文件 import（包括 governance / model_platform / marketplace / KB / conversation / agent_runtime / evaluation / indexing / retrieval / KG / web_access）。

唯一例外：`aperag/db/models.py` 的 `Invitation` 类 class-body 引用，通过 G15 白名单豁免。

实施方式：`tests/unit_test/test_modularization_boundaries.py` 的 G15 gate AST-walk 扫 `ImportFrom(module="aperag.domains.identity.db.models", names=[..., "Role", ...])`，把 identity 和白名单文件外的 import 都判为违规。

admin 判断的 canonical 写法：

```python
# ✅ Right（G15 compliant）
if user.role == "admin":
    ...

# ❌ Wrong（G15 violation）
from aperag.domains.identity.db.models import Role
if user.role == Role.ADMIN:
    ...
```

### G16 — `User` ORM 仅限 identity 域持有

同样方式：禁止非 identity 文件 `from aperag.db.models import User` 或 `from aperag.domains.identity.db.models import User`。

读 User 字段 → 走 consumer-owned `UserView(Protocol)`（governance 有这一 Protocol 就是典型）。
写 User 字段 → 走 `identity_user_ops.<method>` facade。
特殊写 → inline text SQL（单调用点 + PM ack）。

### G17 — Phase 4 DI smoke（3 identity + 4 KB = 7 条）

`tests/unit_test/test_modularization_boundaries.py::test_phase4_di_critical_wirings_at_app_startup`：import `aperag.app` 后断言 7 个 CRITICAL_WIRINGS 的 DI slot 都已经 set 到非 None：

1. `aperag.domains.identity.service.user_manager._bot_init_ops` ≠ None
2. `aperag.domains.identity.service.user_manager._chat_init_ops` ≠ None
3. `aperag.domains.identity.service.user_manager._quota_init_ops` ≠ None
4. `aperag.domains.knowledge_base.service.collection_service._marketplace_collection_ops` ≠ None
5. `aperag.domains.knowledge_base.service.collection_service._search_pipeline_ops` ≠ None
6. `aperag.domains.knowledge_base.service.collection_service._quota_ops` ≠ None
7. `aperag.domains.knowledge_base.service.collection_service._embedding_ops`（或类似名）≠ None

G17 **不包含** Phase 5 的 2 条 permanent standalone-infra seam — 那是 G18 alt 的职责。G17 / G18 alt 分离的理由见 SSoT Section 4。

---

## Boundary gates

### G1 — legacy aggregate import ban

- 禁止 `from aperag.service.<...> import ...`（除 shim 白名单 3 条：quota / prompt_template / search_pipeline）
- 禁止 `from aperag.schema.view_models import ...` 在 domain 代码里
- 禁止 `from aperag.db.models import ...` 在 domain 代码里（`Invitation` 例外）

### G14 — 禁止 legacy route decorator

domain 的 `api/routes.py` 里不允许用 `@aperag.views.<old_api_router>.get(...)` 这种 legacy 路由装饰器 — 必须用 domain 本地的 `APIRouter()`。

### G19 — 禁止 `from __future__ import annotations` 在 FastAPI 路由文件里

lesson 9a-quatuordec：PEP 563 延迟 annotation 求值会让 FastAPI `response_class` + `status_code=204` 的组合返回 500（annotation 里 `None` 被字符串化后 FastAPI 无法识别）。AST gate 扫 routes.py 有没有这条 import。

这条 gate 是 Phase 4 给 identity / governance / model_platform / marketplace 四域的 routes.py **特别校验**的 — 之前 `aperag/views/` 里有一些文件带着这条 import，Phase 4 hard-cut 时顺带扫清。

### Gate 索引

完整 gate 清单（G1 / G4 / G10 / G14 / KB consumer-owned / KB DI smoke / G15 / G16 / G17 / G18 alt / G19）见 `docs/modularization/architecture.md` Section 4 + `tests/unit_test/test_modularization_boundaries.py`。

---

## Legacy shims 与 Phase 7+ 去向

### `aperag/db/models.py`

保留 53 个 re-export symbol + `Role` top-level import（给 `Invitation` class-body 用）+ `Invitation` class 本身。Phase 7+ 看是否把 `Invitation` 迁到某个域（identity 或者独立）一起清。

### `aperag/schema/view_models.py`

6 个 dual-hook try-block 用于 Scenario A identity preservation —— 保证 domain `schemas.X is view_models.X`。运行时通过 `sys.modules.get("aperag.schema.view_models")` string lookup 避免 G1 AST 扫描判违规。是 Phase 4 + Phase 5 的 canonical choice，Phase 7+ 删除候选。

### `aperag/service/*_service.py`

- **19 条普通 shim**（给 Phase 4 前的 `aperag.service.<domain>_service` import 保留入口）：Phase 7+ 删除候选，需要先把所有 import 改成 domain 路径。
- **3 条 standalone-infra shim**（quota / prompt_template / search_pipeline）：**不是** Phase 7+ 候选 — 这 3 条 service 的 canonical home 就在 `aperag/service/`，走 Protocol+DI seam。对应 G18 alt 的 2 条 permanent CRITICAL_WIRINGS。

### `aperag/views/*.py`

- **13 条普通 shim**：Phase 7+ 候选
- 其余 non-domain legacy（settings / prompts / auth）：长期保留

### `ModelServiceProvider` 表

仍在 `aperag/db/models.py`。Phase 7+ 再看是归入 model_platform 还是独立域。

---

## 相关文档

- **canonical SSoT**：`docs/modularization/architecture.md`
  - Section 2.1–2.4 — identity / governance / model_platform / marketplace domain 定义
  - Section 3 — canonical 规则（direct import / Protocol+DI / User write hierarchy）
  - Section 4 — boundary gates
  - Section 5 — permanent DI seam（G18 alt，2 条，**不在** 本文 4 域内）
  - Section 6 — legacy shim lifecycle
  - Section 8 — Phase 7+ F1-F15 候选

- **admin guide（本 4 域的用户面操作）**：
  - [`admin-guide/api-keys.md`](../admin-guide/api-keys.md) — API Key 管理
  - [`admin-guide/audit-log.md`](../admin-guide/audit-log.md) — 审计日志
  - [`admin-guide/quota-system.md`](../admin-guide/quota-system.md) — 配额管理（跨 G18 alt seam）

- **user guide（marketplace 用户面）**：
  - [`user-guide/collection-marketplace.md`](../user-guide/collection-marketplace.md) — 发布 / 订阅流程（起稿中）

- **其他架构**：
  - [`architecture/overview.md`](./overview.md) — 入口
  - [`architecture/domains.md`](./domains.md) — 12 域通览
  - [`architecture/conversation-agent-evaluation.md`](./conversation-agent-evaluation.md) — Phase 5/6 三域架构
  - [`architecture/indexing-retrieval-kg.md`](./indexing-retrieval-kg.md) — Phase 3 四域架构
  - [`architecture/web-access.md`](./web-access.md) — web_access 域

- **历史 / 决策**：
  - `docs/modularization/breaking-changes/phase4.md` — Phase 4 hard-cut breaking changes
  - `docs/modularization/` — Phase 0→6 完整演进记录
