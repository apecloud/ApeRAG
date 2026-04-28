# 配额管理（Quota System）

> **读者定位**：ApeRAG 系统管理员（`role == "admin"`）、SRE、运维。
>
> **范围**：用户配额的查询、调整、重算、系统默认值。架构层 Protocol+DI（`quota_service` 为 2 条 permanent CRITICAL_WIRINGS 之一）见 [`architecture/conversation-agent-evaluation.md`](../architecture/conversation-agent-evaluation.md) 或 canonical SSoT `docs/modularization/architecture.md`。

## 概述

ApeRAG 使用**每用户配额**来限制可创建的核心资源数量。配额分两部分：

- **limit**（上限）：由系统默认值初始化，admin 可按用户单独调整。
- **usage**（当前用量）：由业务层在创建 / 删除资源时原子地 `check_and_consume_quota` / `release_quota` 更新。

配额检查走事务内 `SELECT ... FOR UPDATE` 行锁，避免并发创建导致的超额；超限时抛 `QuotaExceededException`，接口返回 403。

## 当前支持的配额类型

| Key | 含义 | 默认值 |
| --- | --- | --- |
| `max_collection_count` | 每用户知识库数量上限 | 20 |
| `max_document_count` | 每用户跨所有知识库的文档总数上限 | 4000 |
| `max_document_count_per_collection` | 每个知识库内的文档数量上限 | 200 |
| `max_bot_count` | 每用户智能体（Bot）数量上限，不含系统默认 bot | 10 |

> 默认值存在 `ConfigModel` 表中 key 为 `system_default_quotas` 的一行 JSON；若该行不存在则回退到代码里的硬编码默认（上表）。

### 不在配额内的资源

- Chat、Message、AgentTurn：不计入配额，数量由聊天使用自然产生。
- ApiKey：不计入配额（管理侧另有 `is_system` 标记区分系统生成 key）。
- Subscription（marketplace 订阅）：不计入配额。
- Model Service Provider：不计入配额，由 admin 集中管理。

## 数据模型

`UserQuota` 实体（`aperag/db/models.py`，复合主键 `(user, key)`）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `user` | 字符串 | 用户 ID，主键之一 |
| `key` | 字符串 | 配额类型（上表 Key 列），主键之一 |
| `quota_limit` | 整数 | 上限 |
| `current_usage` | 整数 | 当前用量 |
| `gmt_created` / `gmt_updated` / `gmt_deleted` | 时间戳 | 审计字段 |

`ConfigModel`（同文件）里 key=`system_default_quotas` 的行保存 JSON：

```json
{
  "max_collection_count": 20,
  "max_document_count": 4000,
  "max_document_count_per_collection": 200,
  "max_bot_count": 10
}
```

## 查询接口

### 当前用户的配额

```http
GET /api/v2/quotas
Authorization: Bearer sk-<user-key>
```

返回 `UserQuotaInfo`：`user_id` / `username` / `email` / `role` / `quotas[]`，其中 `quotas[i]` 形如：

```json
{
  "quota_type": "max_collection_count",
  "quota_limit": 10,
  "current_usage": 3,
  "remaining": 7
}
```

普通用户只能查自己；如果 `user_id` 或 `search` 参数传进来但 caller 不是 admin，会返回 403。

### admin 查某个用户

```http
GET /api/v2/quotas?user_id=<uid>
Authorization: Bearer sk-<admin-key>
```

### admin 按用户名 / 邮箱 / ID 搜索

```http
GET /api/v2/quotas?search=<term>
```

`search` 是精确匹配 username / email / user.id 之一（注意不是模糊搜索），返回 `UserQuotaList`。当查询结果为 0 时返回 404；其他情况返回 list（即便只有一条结果）。

## 管理操作

### 调整单个用户的配额

```http
PUT /api/v2/quotas/{user_id}
Content-Type: application/json
Authorization: Bearer sk-<admin-key>

{
  "max_collection_count": 20,
  "max_document_count": 2000
}
```

- 支持单个和批量更新（请求体里只传要改的 key）。
- 如果某个配额 key 之前不存在，会创建新行。
- `null` 值字段会被忽略（不要想用 null 来清空某个配额；当前没有清空的语义）。
- 接口只改 `quota_limit`，**不会重置 `current_usage`**。

响应返回每项的 `old_limit` → `new_limit`：

```json
{
  "success": true,
  "message": "Quotas updated successfully",
  "user_id": "u-xxx",
  "updated_quotas": [
    {"quota_type": "max_collection_count", "old_limit": 10, "new_limit": 20}
  ]
}
```

### 重算当前用量

```http
POST /api/v2/quotas/{user_id}/recalculate
Authorization: Bearer sk-<admin-key>
```

用于**校正漂移**。接口会实际扫描数据库计算真实用量：

- `max_collection_count`：`SELECT COUNT(*) FROM collection WHERE user=? AND status != 'DELETED'`
- `max_document_count`：跨所有非删除 collection 的非删除 document 总数
- `max_bot_count`：该用户的非删除 bot，且排除系统默认 bot（`title != "Default Agent Bot"`）

然后把 `current_usage` 写回 UserQuota 行。**注意**：重算不包括 `max_document_count_per_collection`（那是 per-collection 限制，没有单值可以汇总）。

> 什么时候需要重算：业务异常崩溃后（资源已创建但 quota 没 consume），或者从外部脚本批量插入了数据之后。正常路径下不需要手动触发。

### 调整系统默认配额

```http
GET /api/v2/system/default-quotas
PUT /api/v2/system/default-quotas
Content-Type: application/json

{
  "quotas": {
    "max_collection_count": 15,
    "max_document_count": 1500,
    "max_document_count_per_collection": 150,
    "max_bot_count": 8
  }
}
```

- 新注册用户会按当前 system default 初始化自己的 UserQuota 行（由 identity 域的 `on_after_register` 钩子调用 `quota_service.initialize_user_quotas`）。
- **改系统默认值不会影响已存在用户**。已存在用户若要跟上新默认，走 `PUT /api/v2/quotas/{user_id}` 单个改。

## 运行时行为

### 消耗配额

业务创建资源时调用 `quota_service.check_and_consume_quota(user_id, quota_type, amount, session)`：

1. `SELECT ... FOR UPDATE` 锁定该 user+key 行。
2. 若 `current_usage + amount > quota_limit`，抛 `QuotaExceededException`（接口层转成 HTTP 403）。
3. 否则 `current_usage += amount`，flush。

调用方**必须**把自己的 session 传进去，确保配额更新和资源创建在同一事务里；否则事务回滚时 quota 不会自动回退。

### 释放配额

软删除 / 硬删除资源时调用 `quota_service.release_quota(user_id, quota_type, amount, session)`：

- `current_usage = max(0, current_usage - amount)`（不会变负）。
- 同样建议在资源删除事务里调用，保证一致性。

### 新用户初始化

`on_after_register` 钩子调用 `initialize_user_quotas`：对每个系统默认 key，若 UserQuota 行不存在则按系统默认 limit 创建，`current_usage=0`。重复调用安全（已存在的行不会被覆盖）。

## 权限模型

- **普通用户**：只能 `GET /api/v2/quotas`（本人）。所有其他 quota 接口返回 403。
- **admin（`role == "admin"`）**：
  - 查任意用户 / 搜索
  - 改任意用户 limit
  - 触发任意用户 recalculate
  - 读 / 写 system default

quota 接口层现在与 governance 其他接口对齐，使用 `AuthenticatedUser` port 读取当前用户，并以 `current_user.role != "admin"` 做 admin 判定。`quota_service` 的 canonical implementation 已收编到 `aperag/domains/governance/service/quota_service.py`；`aperag/service/quota_service.py` 仅保留兼容 shim，供现有 Protocol/DI consumer 继续导入。

## 常见运维场景

### 给某用户临时放宽

```http
PUT /api/v2/quotas/<uid>
{"max_document_count": 5000}
```

### 排查"为什么创建 collection 失败 403"

1. `GET /api/v2/quotas?user_id=<uid>` 看 `max_collection_count` 的 `current_usage` vs `quota_limit`。
2. 若 `current_usage` 大于实际 collection 数（漂移），`POST /api/v2/quotas/<uid>/recalculate` 修正。
3. 若 `quota_limit` 太小，`PUT /api/v2/quotas/<uid>` 提高上限。

### 批量提高所有新用户默认

`PUT /api/v2/system/default-quotas` — 改完只影响此后注册的用户，老用户不变。

### 统计各配额的饱和度

目前没有直接接口，需要从 DB 跑 SQL：

```sql
SELECT key,
       COUNT(*) AS users,
       SUM(current_usage) AS total_used,
       SUM(quota_limit) AS total_limit,
       AVG(current_usage::float / NULLIF(quota_limit, 0)) AS avg_saturation
FROM user_quota
GROUP BY key;
```

## 跨 domain 边界

`quota_service` 是 **2 条 permanent Protocol+DI seam** 之一（另一条是 `prompt_template_service`），在 `aperag/app.py` 启动时注入到 `conversation.bot_service._quota_ops`（G18 alt CRITICAL_WIRINGS registry）。

设计原因：quota 能力由 governance 域拥有，但被 knowledge_base / conversation / agent_runtime 多个域交叉消费；跨域调用仍通过消费方的 `ports.py` 显式声明为 Protocol，并由启动 wiring 注入。旧的 `aperag/service/quota_service.py` 只作为 shim 指向 governance canonical implementation，避免跨域代码直接依赖治理域内部实现。

详见 `docs/modularization/architecture.md` Section 4（canonical rules：direct import / Protocol+DI transitional / standalone-infra permanent）和 Section 5（Runtime seams）。

## 相关文档

- [`admin-guide/api-keys.md`](./api-keys.md) — API Key 管理
- [`admin-guide/audit-log.md`](./audit-log.md) — 审计日志（配额变更会被 `@audit` 装饰器记录）
- [`architecture/conversation-agent-evaluation.md`](../architecture/conversation-agent-evaluation.md) — `quota_service` 在 conversation 域的 DI wire-up
- `docs/modularization/architecture.md` — canonical SSoT
