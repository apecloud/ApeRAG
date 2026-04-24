# 审计日志（Audit Log）

> **读者定位**：ApeRAG 系统管理员（`role == "admin"`）、SRE、合规审计人员。
>
> **范围**：审计日志的数据模型、查询接口、权限边界、敏感字段过滤。架构层实现细节见 [`architecture/identity-governance-model-platform-marketplace.md`](../architecture/identity-governance-model-platform-marketplace.md) 的 governance 章节。

## 概述

`AuditLog` 记录系统里所有**变更类 HTTP 操作**（POST / PUT / DELETE）的执行痕迹，用于：

- 事后追溯（"这条数据是谁、什么时候、通过什么请求改掉的"）
- 异常排查（HTTP 4xx / 5xx 的请求体、响应体、错误信息）
- 安全合规（接入点、IP、User-Agent、Request ID 关联）

审计日志由 `@audit(resource_type=..., api_name=...)` 装饰器自动写入，业务 handler 无需手动调用。装饰器实现位于 `aperag/utils/audit_decorator.py`，服务层位于 `aperag/domains/governance/service/audit_service.py`。

## 数据模型

`AuditLog` 实体（`aperag/domains/governance/db/models.py`）字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `id` | UUID 字符串 | 主键 |
| `user_id` | 字符串，可空 | 操作者用户 ID（匿名接口可能为空） |
| `username` | 字符串，可空 | 操作者用户名（冗余存储，避免 join） |
| `resource_type` | `AuditResource` 枚举 | 见下节枚举列表 |
| `resource_id` | 字符串 | 从 `path` 中解析出来的资源 ID（查询时解析，不在写入时落库） |
| `api_name` | 字符串 | 由 `@audit` 装饰器传入，形如 `CreateCollection` |
| `http_method` | 字符串 | `POST` / `PUT` / `DELETE` 等 |
| `path` | 字符串 | 请求路径（如 `/api/v1/collections/col-abc123`） |
| `status_code` | 整数 | HTTP 响应码 |
| `request_data` | JSON 字符串 | 请求体，敏感字段已过滤 |
| `response_data` | JSON 字符串 | 响应体，敏感字段已过滤 |
| `error_message` | 字符串 | 失败时保留异常信息 |
| `ip_address` | 字符串 | 客户端 IP |
| `user_agent` | 字符串 | 浏览器 / CLI 标识 |
| `request_id` | 字符串 | 用于跨服务 trace 关联 |
| `start_time` / `end_time` | 毫秒时间戳 | `end_time - start_time` 即耗时 |
| `gmt_created` | 时间戳 | 落库时间 |

### AuditResource 枚举

写入的资源类型由 `@audit(resource_type="...")` 传入，取值必须在 `AuditResource` 枚举内：

`collection` / `document` / `bot` / `chat` / `message` / `api_key` / `llm_provider` / `llm_provider_model` / `model_service_provider` / `user` / `config` / `invitation` / `auth` / `chat_completion` / `search` / `llm` / `flow` / `system` / `index`

## 敏感字段过滤

写入前，`audit_service` 会递归扫描 `request_data` / `response_data`，将以下字段名（忽略大小写）的值替换为 `***FILTERED***`：

`password` / `token` / `api_key` / `secret` / `authorization` / `access_token` / `refresh_token` / `private_key` / `credential`

> 注意：过滤基于**字段名子串匹配**，不会扫描字段值本身。若自定义字段需要脱敏，用上面任一关键词命名即可（例如 `oauth_token`）。

## 查询接口

### 列表查询

```http
GET /api/v1/audit-logs
Authorization: Bearer sk-<admin-key>
```

支持的 query 参数：

| 参数 | 说明 |
| --- | --- |
| `user_id` | 按用户 ID 过滤。**非 admin 时该参数被忽略，强制回填为当前用户** |
| `username` | 按用户名过滤 |
| `resource_type` | 任一 `AuditResource` 枚举值 |
| `resource_id` | 资源 ID 精确匹配 |
| `api_name` | API 操作名精确匹配 |
| `http_method` | `POST` / `PUT` / `DELETE` |
| `status_code` | HTTP 响应码精确匹配 |
| `start_date` / `end_date` | ISO-8601 时间区间 |
| `page` / `page_size` | 分页，默认 `page=1` / `page_size=20`，`page_size` 上限 100 |
| `sort_by` / `sort_order` | 排序字段 + `asc` / `desc`，默认按时间倒序 |
| `search` | 模糊搜索（在 api_name / path 上做 LIKE） |

### 单条查询

```http
GET /api/v1/audit-logs/{audit_id}
```

返回完整的 `AuditLog` + 运行时计算的 `resource_id`（从 `path` 正则提取）和 `duration_ms`。

### 权限规则

```
if user.role == "admin":
    # 可见所有人的日志
    pass
else:
    # 强制只看自己的日志，忽略 user_id / username 参数
    filter_user_id = user.id
```

该 literal 比较**不依赖** identity 域的 `Role` 枚举（G15 边界规则 — governance 域不能 import `Role`）。admin 角色名以字符串 `"admin"` 写死在 governance handler 里。

## 常见管理场景

### 追溯某个知识库被谁删除

```http
GET /api/v1/audit-logs?resource_type=collection&api_name=DeleteCollection&page_size=100
```

按时间倒序扫，结合 `resource_id`（会从 `/collections/{id}` 中解析出来）定位目标记录，记录中的 `user_id` + `username` + `ip_address` 即操作者身份。

### 排查 5xx 错误

```http
GET /api/v1/audit-logs?status_code=500&start_date=2026-04-20T00:00:00Z
```

返回结果中的 `error_message` 字段会保留异常 repr；`request_id` 可以在日志系统中 grep 到完整调用栈。

### 统计 API 调用量

```http
GET /api/v1/audit-logs?api_name=CreateChatCompletion&start_date=2026-04-01T00:00:00Z
```

后端目前只提供 listing，不直接提供 aggregation 接口。如果要做 per-day / per-user 的统计报表，需要从 AuditLog 表直接跑 SQL（或者把表同步到 OLAP）。

### 查看某个用户的近期活动

```http
GET /api/v1/audit-logs?user_id=<uid>&page_size=50
```

admin 视角可以看任何用户；普通用户只能看自己（后端强制覆盖）。

## 保留与清理策略

当前版本没有内置的保留期清理任务。所有 `AuditLog` 记录永久保存在业务数据库中。如需控制表大小：

- **推荐**：把 AuditLog 定期（月 / 季度）导出到冷存储（S3 / 对象存储），然后按保留策略删除。
- 不推荐直接 `DELETE FROM audit_log WHERE gmt_created < ?`，因为会导致合规窗口内的证据缺失；必须在导出归档之后再清。

> 未来可能会提供自动归档 Job，目前先靠运维侧脚本实现。

## 跨 domain 边界（为什么 AuditLog 属于 governance 不属于 identity）

AuditLog 记录"谁在什么时间做了什么"，语义上归 **governance** 域。`user_id` 和 `username` 字段是冗余存储（不是 FK 到 identity.User），这个选择有两个原因：

1. **审计日志必须在用户被删除后仍然保留**。如果用 FK 关联，用户删除会触发级联或者 SET NULL，污染审计证据。
2. **governance 域不持有 identity.User ORM 实例**（G16 边界规则）。governance 读侧若需要展示 `username` 等识别信息，走 `governance.ports.UserView` Protocol，由 identity 域在启动时注入。

详见 [`architecture/identity-governance-model-platform-marketplace.md`](../architecture/identity-governance-model-platform-marketplace.md) governance 章节。

## 相关文档

- [`admin-guide/api-keys.md`](./api-keys.md) — API Key 管理（同属 governance 域）
- [`architecture/identity-governance-model-platform-marketplace.md`](../architecture/identity-governance-model-platform-marketplace.md) — governance 域架构
- `docs/modularization/architecture.md` — 12 域 canonical SSoT（boundary gates / UserView Protocol / G15 literal compare 规则）
