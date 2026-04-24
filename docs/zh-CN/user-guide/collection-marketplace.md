# 知识库市场（Collection Marketplace）

> **读者定位**：两类最终用户 — (1) 希望把自己的知识库公开分享给他人的 **所有者（owner）**；(2) 希望发现 / 订阅他人已发布知识库的 **订阅者（subscriber）**。
>
> **范围**：发布 / 取消发布、浏览、订阅 / 取消订阅、订阅后的只读访问。架构侧（CollectionMarketplace / UserCollectionSubscription ORM + G16 边界）见 [`architecture/identity-governance-model-platform-marketplace.md`](../architecture/identity-governance-model-platform-marketplace.md) 的 marketplace 章节。

## 概念

ApeRAG 的 **Collection Marketplace**（知识库市场）让你把自己的知识库公开给其他用户浏览和订阅：

- **所有者**：可以把自己的 collection `publish` 到市场，公开可见；也可以随时 `unpublish` 下架。
- **订阅者**：可以浏览市场里所有已发布的 collection，`subscribe` 感兴趣的 collection 后以**只读**方式访问里面的文档、知识图谱等。

市场行为与 RBAC 权限体系是正交的：订阅不会给你该 collection 的写权限；它只是把原 owner 的 collection 内容以只读视图开放给你。原 owner 对数据的任何变更（添加文档、重建索引、解除分享）都会即时反映到订阅者那边。

## 状态模型

一个 collection 在市场里只有两种状态：

- **`DRAFT`（草稿 / 未发布）**：只 owner 可见。这是所有新创建 collection 的默认状态。
- **`PUBLISHED`（已发布）**：所有登录用户都可以浏览；任何其他用户都可以订阅。

`unpublish` 把状态从 `PUBLISHED` 转回 `DRAFT`（技术上是软删除对应的 marketplace 记录）。一旦取消发布，已订阅的用户会在下次访问时被拒绝。

## 所有者视角

### 发布知识库到市场

在 ApeRAG Web UI 的 collection 详情页点击"分享到市场"按钮；或走 HTTP：

```http
POST /api/v1/collections/{collection_id}/sharing
Authorization: Bearer sk-<your-key>
```

返回 204 No Content。发布后：

- 该 collection 出现在 `GET /api/v1/marketplace/collections` 列表
- 其他登录用户可以订阅
- 未登录 / 匿名用户也能看到它出现在公开列表（当前设计允许匿名浏览市场目录）

### 查看发布状态

```http
GET /api/v1/collections/{collection_id}/sharing
```

返回：

```json
{
  "is_published": true,
  "published_at": "2026-04-15T08:23:00Z"
}
```

### 取消发布

```http
DELETE /api/v1/collections/{collection_id}/sharing
```

返回 204。取消后：

- 该 collection 从市场列表里消失
- **现有订阅者不会被强制退订**，但他们访问该 collection 时会收到 403 / 404
- 原 owner 的使用不受影响

### 发布前的准备

- **检查内容**：所有文档 / 链接 / 知识图谱都会被订阅者看到，下架前先确认没有敏感内容。
- **加 collection summary**：在详情页点"生成摘要"或走 `POST /api/v1/collections/{id}/summary`；订阅者能在市场列表里看到摘要，有助于发现。
- **调整标题和描述**：这些字段是订阅者看到的第一眼信息。
- **确认索引完整**：发布时不会自动触发索引重建；若某些文档还在 `PENDING` / `FAILED` 状态，订阅者访问时会看到空白或错误。

## 订阅者视角

### 浏览市场目录

```http
GET /api/v1/marketplace/collections?page=1&page_size=30
```

不需要认证也能调用（匿名浏览）；若带 token，响应里的 subscription status 会标记你是否已订阅。

返回 `SharedCollectionList`：分页列表，每项包含 collection 标题、摘要、owner username、发布时间、订阅人数等。

### 订阅一个 collection

```http
POST /api/v1/marketplace/collections/{collection_id}/subscribe
Authorization: Bearer sk-<your-key>
```

成功返回 `SharedCollection`（订阅后的只读视图）。

失败情况：

| HTTP | 说明 |
| --- | --- |
| 400 | `Collection is not published to marketplace` — 该 collection 已取消发布 |
| 400 | `Cannot subscribe to your own collection` — 不能订阅自己的 collection |
| 409 | `Already subscribed to this collection` — 已订阅过，不需要重复 |

### 查看自己订阅的列表

```http
GET /api/v1/marketplace/collections/subscriptions?page=1&page_size=30
Authorization: Bearer sk-<your-key>
```

返回你当前**未取消**的所有订阅。

### 取消订阅

```http
DELETE /api/v1/marketplace/collections/{collection_id}/subscribe
Authorization: Bearer sk-<your-key>
```

返回 `{"message": "Successfully unsubscribed"}`。再次订阅同一 collection 是允许的（如果它还在 `PUBLISHED` 状态）。

### 访问订阅后的 collection 内容

订阅后，你可以通过 marketplace 专用 endpoints 只读访问：

| Endpoint | 返回 |
| --- | --- |
| `GET /api/v1/marketplace/collections/{id}` | 元数据（title / description / summary / owner） |
| `GET /api/v1/marketplace/collections/{id}/documents` | 文档列表（分页 / 排序 / 搜索） |
| `GET /api/v1/marketplace/collections/{id}/documents/{doc_id}/preview` | 文档 preview |
| `GET /api/v1/marketplace/collections/{id}/documents/{doc_id}/object` | 原始文档字节（支持 Range header） |
| `GET /api/v1/marketplace/collections/{id}/graph` | 知识图谱（节点 / 边结构）|

所有这些接口都走相同的 access 检查：
1. 若 collection 未发布 → 404
2. 若 collection 已发布但订阅方式不符（目前所有登录用户 = 允许读）→ 403
3. 否则返回数据

**重要**：所有 marketplace 文档 / 图谱查询在后端都用 **owner 的 user_id** 去读 KB 内容，而不是订阅者自己的 user_id。这意味着你看到的是 owner 最新状态，包括 owner 刚添加的文档、刚重建的索引。

### marketplace 内容不能被订阅者修改

下面的操作在订阅的 marketplace collection 上都会返回 403 / 404：

- 添加文档 / 删除文档
- 触发重建索引
- 编辑 collection 描述
- 生成 / 编辑 collection summary
- 修改知识图谱

唯一的写操作是订阅 / 取消订阅本身。

## 在 Chat / Bot / Agent 里使用订阅的 collection

订阅的 collection 可以直接在聊天 / bot 配置里**作为数据源使用**（前提是你已订阅）：

- 在 Bot 配置的"知识库"选择框里，订阅的 collection 会出现在"已订阅"分组
- 在 Chat 侧边栏直接发起基于订阅 collection 的对话
- Agent Runtime 在检索工具调用时会根据 Bot 配置自动走订阅权限

如果 owner 取消发布 / 下架，Bot 下次调用时会拿到空结果（不会静默失败，UI 会显示"该知识库不再可用"）。

## 知识图谱（KG）视角

如果 owner 在自己的 collection 里启用了 Knowledge Graph（详见 [`architecture/indexing-retrieval-kg.md`](../architecture/indexing-retrieval-kg.md)），订阅者可以通过 `GET /api/v1/marketplace/collections/{id}/graph?label=*&max_nodes=1000&max_depth=3` 只读查询：

- `label`：过滤节点 label（`*` 表示全部）
- `max_nodes`：返回节点数上限（默认 1000，上限 10000）
- `max_depth`：BFS 深度（默认 3，上限 10）

与非 marketplace collection 的 graph 查询参数保持一致，只是底层访问权限走 `marketplace_collection_service.check_marketplace_access` 验证。

## 实体与数据模型（简版）

| 表 | 字段要点 | 说明 |
| --- | --- | --- |
| `CollectionMarketplace` | `collection_id` (unique) / `status` (`DRAFT` \| `PUBLISHED`) / `gmt_created` / `gmt_deleted` | 每个 collection 最多 1 条活跃 marketplace 记录 |
| `UserCollectionSubscription` | `user_id` / `collection_id` / `gmt_created` / `gmt_deleted` | 每 (user, collection) 在同一时刻最多 1 条活跃订阅 |

这两张表在 `aperag/domains/marketplace/db/models.py`。

**注意**：`UserCollectionSubscription` 只记录"订阅了什么"，**不记录**"阅读历史"或"对订阅内容的个人标注"— marketplace 目前是纯目录 + 订阅关系，不持有订阅者侧的用户数据。

## 发布 / 订阅 / 匿名访问的权限矩阵

| 操作 | 匿名 | 登录非 owner | 登录 owner | 已订阅非 owner | admin |
| --- | --- | --- | --- | --- | --- |
| 列出市场 collections | ✅ | ✅ | ✅ | ✅ | ✅ |
| 查看 collection metadata（marketplace endpoint） | ❌ 403 | ✅ | ✅ | ✅ | ✅ |
| 订阅 | — | ✅ | ❌ 400（self） | ❌ 409（dup） | ✅ |
| 取消订阅 | — | ❌（没订阅） | — | ✅ | ✅ |
| 读文档 / 图谱 | ❌ 403 | ✅ | ✅ | ✅ | ✅ |
| 发布 / 取消发布 | ❌ | ❌ 403 | ✅ | — | ✅ |
| 写 / 修改数据 | ❌ | ❌ 403 | ✅（走普通 KB 接口） | ❌ 403 | ✅ |

> ⚠️ 注意"列出市场 collections"允许匿名；"查看 metadata"和"读文档 / 图谱"**需要登录**但不一定需要订阅。订阅目前主要用途是"把这个 collection 放进我的订阅列表"便于快速访问，不是 hard access gate。可能随安全策略调整。

## 常见问题

### 发布后别人看得到我的原始文件吗？

能。`GET /marketplace/collections/{id}/documents/{doc_id}/object` 允许登录用户按 owner 权限读原始文件字节。发布前请确认没有敏感文件（合同 / PII / 密钥）。

### 取消发布后，已订阅用户的 bot 会怎样？

下次访问时拿到 404 / 403；Bot UI 会标记该知识库"不再可用"。建议发布前和订阅者沟通，避免突发下架。

### owner 删除了文档，订阅者能看到吗？

不能。订阅者看到的是 owner 侧的当前状态：owner 删了就没了。软删除的文档（`DocumentStatus=DELETED`）也被 filter 掉。

### 一个用户可以订阅多少个 collection？

当前没有硬上限；每个订阅一条 `UserCollectionSubscription` 行。如果需要限制，可以走 quota system 扩展（目前 quota 不含订阅计数）。

### collection 的 summary 谁看得到？

所有能看到 collection metadata 的用户都能看到 summary。summary 是发布面向订阅者的主要发现路径。

### 可以只对特定用户分享吗？

**目前不支持**。marketplace 只有"全公开 / 不公开"两档。如果需要细粒度分享（特定用户 / 组），可以走自定义应用层集成（例如用你自己的 API key 模式代理访问）。未来版本可能加 unlisted share link 选项。

### 分享到市场会影响 quota 吗？

不会。Marketplace 订阅不计入订阅者的 `max_collection_count` quota — 那个 quota 只数自己 owned 的 collection。

## 相关文档

- [`user-guide/document-upload.md`](./document-upload.md) — 发布前准备的文档上传流程
- [`user-guide/content-import.md`](./content-import.md) — URL / 文本导入
- [`user-guide/knowledge-export.md`](./knowledge-export.md) — 打包导出（订阅者不能导出，只 owner 可）
- [`architecture/identity-governance-model-platform-marketplace.md`](../architecture/identity-governance-model-platform-marketplace.md) — marketplace 架构
- `docs/modularization/architecture.md` — 12 域 canonical SSoT
