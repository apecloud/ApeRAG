# API Key 管理

> **读者定位**：ApeRAG 系统管理员 / 高级用户。
>
> **范围**：API Key 的创建、轮换、查询、吊销 admin 操作。技术面（ORM / 服务层 / DI）见 [`architecture/identity-governance-model-platform-marketplace.md`](../architecture/identity-governance-model-platform-marketplace.md) 的 governance 章节。

## 概述

API Key 是 ApeRAG 提供的程序化访问凭证，与 fastapi-users 的 Cookie 会话认证并存：

- **人工访问**：Web UI / Cookie 认证（登录态）
- **程序化访问**：API Key（`Authorization: Bearer sk-...`）

所有通过 API Key 发起的 HTTP 请求会被映射到 key 所属用户身份，权限与该用户一致。API Key 不可跨用户共享。

## 实体模型（管理员视角）

API Key 由 governance 域拥有，字段要点：

| 字段 | 含义 | 说明 |
| --- | --- | --- |
| `id` | Key 记录主键 | 形如 `keyXXXX...`，由系统生成 |
| `key` | Bearer Token 明文 | 形如 `sk-<32 位十六进制>`，仅创建时返回给用户，后续仅保存在数据库 |
| `user` | 所属用户 ID | 对应 identity 域的 `User.id` |
| `description` | 备注 | 可选 |
| `status` | 状态 | `ACTIVE` / `DELETED`（软删除） |
| `is_system` | 是否系统生成 | 区分用户自建与系统内部使用的 key |
| `last_used_at` | 最近一次使用时间 | 便于发现闲置 key |
| `gmt_created` / `gmt_updated` / `gmt_deleted` | 审计时间戳 | 软删除不物理移除记录 |

## 常见管理操作

### 查询某个用户的所有 API Key

```http
GET /api/v1/apikeys
Authorization: Bearer sk-<admin-key>
```

返回的列表中，管理员默认只能看到自己的 key。若需跨用户查询，走审计日志（见 [audit-log.md](./audit-log.md)）。

### 创建 API Key

```http
POST /api/v1/apikeys
Content-Type: application/json
Authorization: Bearer sk-<owner-key>

{
  "description": "CI runner for apecloud/aperag"
}
```

响应会包含一次性的 `key` 明文（形如 `sk-xxxxxxxxxxxx...`）。

> ⚠️ **只会返回这一次**。保存到 CI / 密钥管理器后再也取不回。如果丢失，必须走「吊销 + 重建」流程，不要考虑明文恢复。

### 更新备注

```http
PUT /api/v1/apikeys/{apikey_id}
Content-Type: application/json

{
  "description": "rotated 2026-04 per security policy"
}
```

只能改 `description`。如果要换 token，请参考下面的「轮换（rotate）」节。

### 吊销（软删除）

```http
DELETE /api/v1/apikeys/{apikey_id}
```

此操作会：

1. 把 `status` 置为 `DELETED`，填写 `gmt_deleted`
2. 之后使用该 token 的所有请求都会被拒绝（401）
3. **不会**清除 `key` 字段，便于事后审计追溯哪个 token 曾发起过什么操作
4. 审计日志中的历史记录保留

## 轮换（rotate）

ApeRAG 目前没有原子化的「保留 id、换 token」接口。推荐的轮换步骤：

1. `POST /apikeys`：创建新 key（得到新 `sk-...`）
2. 将新 key 分发到调用方（CI、外部系统）
3. 观察几天，确认旧 key 不再产生流量（通过 `last_used_at` 或审计日志）
4. `DELETE /apikeys/{old_id}`：吊销旧 key

建议将轮换写进组织的安全 playbook，不依赖内存。

## 系统生成的 API Key

`is_system = true` 的 API Key 是 ApeRAG 内部为其他能力自动签发的凭证（例如某些一次性工具流）。这些 key：

- **不应被人工吊销**，否则会影响系统功能
- 通常没有 `description`，但会绑定明确的 `user`
- 在管理员查询接口中会显式标出 `is_system`

## 跨 domain 边界（为什么 API Key 属于 governance 不属于 identity）

从概念上，API Key 是「某个用户的访问凭证」；但在 ApeRAG 12 域模型里，它归 **governance** 域而非 **identity** 域，原因是：

- `identity` 域管理「谁是谁」（User / Role / OAuthAccount）
- `governance` 域管理「谁在什么时间做了什么」（API Key / Audit Log / Audit Resource）

API Key 更接近 **访问凭证 + 操作归属**，语义上和审计日志同源，因此在 Phase 4 模块化重构中归 governance 域（`aperag/domains/governance/db/models.py`）。

跨域读侧通过 `governance.ports.UserView` 读取 `User.username` 等展示字段，不持有 `User` ORM 实例（遵守 G16 边界规则，详见 architecture 章节）。

## 相关文档

- [`architecture/identity-governance-model-platform-marketplace.md`](../architecture/identity-governance-model-platform-marketplace.md) — governance 域架构实现 + `UserView` Protocol
- [`admin-guide/audit-log.md`](./audit-log.md) — 审计日志查询
- [`reference/prompt-api.md`](../reference/prompt-api.md) — 示例 API 调用
