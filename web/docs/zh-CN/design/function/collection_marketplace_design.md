---
title: 分享与市场
description:
keywords:
---

## 设计文档：Collection 分享与市场 (MVP)

**版本:** 1.6 **关联 Issue:** [#1127](https://github.com/apecloud/ApeRAG/issues/1127)

### 1. 概述

本文档旨在为 ApeRAG 设计并实现一个 Collection（知识库）分享与市场的最小可行产品 (MVP)。核心目标是允许用户将自己的 Collection 发布到一个公共市场，其他用户可以发现并以**严格只读**的模式访问这些共享的 Collection。

MVP 阶段将专注于实现最核心的发布、浏览和只读访问流程，省略复杂的审核、分类、评级、统计分析、用户评价、热度排序、专门的订阅管理页面等功能，以便快速验证核心价值。

**核心功能范围:**

- Collection 所有者可以发布和取消发布自己的 Collection
- 已发布的 Collection 出现在公共市场页面供所有用户浏览
- 非所有者用户可以以严格只读模式访问已发布的 Collection
- 只读模式包括：查看文档列表、阅读文档内容、浏览知识图谱、使用聊天机器人搜索
- 只读模式禁止：添加/删除/修改文档、修改 Collection 设置、任何写操作

### 2. 数据库 Schema 设计

基于 Subscribe 模式的需求，我们需要新增两个表来支持 Collection 分享和用户订阅功能。

#### 2.1. 新增表设计

**表1: `collection_marketplace` - Collection 分享状态表**

用于记录 Collection 的分享状态和发布信息。

```sql
CREATE TABLE collection_marketplace (
    id VARCHAR(24) PRIMARY KEY DEFAULT ('market_' || substr(md5(random()::text), 1, 16)),
    collection_id VARCHAR(24) NOT NULL,  -- 关联collections表，应用层维护关联关系

    -- 分享状态：使用VARCHAR存储，不使用数据库enum类型，应用层校验
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT',

    -- 时间戳字段
    gmt_created TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    gmt_updated TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),  -- 代码层更新
    gmt_deleted TIMESTAMP WITH TIME ZONE NULL,

    -- 约束
    CONSTRAINT uq_collection_marketplace_collection UNIQUE (collection_id)
);

-- 注意：gmt_updated字段需要在应用代码中手动更新
-- 在SQLModel中更新记录时，手动设置: gmt_updated = datetime.utcnow()

-- 索引优化
CREATE INDEX idx_collection_marketplace_status ON collection_marketplace(status);
CREATE INDEX idx_collection_marketplace_gmt_deleted ON collection_marketplace(gmt_deleted);
CREATE INDEX idx_collection_marketplace_collection_id ON collection_marketplace(collection_id);
-- 查询市场列表时的复合索引
CREATE INDEX idx_collection_marketplace_list ON collection_marketplace(status, gmt_created DESC);
```

**表2: `user_collection_subscription` - 用户订阅表**

用于记录用户对已发布 Collection 的订阅关系，采用 Subscribe 模式。

```sql
CREATE TABLE user_collection_subscription (
    id VARCHAR(24) PRIMARY KEY DEFAULT ('sub_' || substr(md5(random()::text), 1, 16)),
    user_id VARCHAR(24) NOT NULL,                      -- 关联users表，应用层维护关联关系
    collection_marketplace_id VARCHAR(24) NOT NULL,    -- 关联collection_marketplace表，应用层维护关联关系

    -- 时间戳字段
    gmt_subscribed TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    gmt_deleted TIMESTAMP WITH TIME ZONE NULL,  -- 软删除：NULL表示活跃订阅

    -- 注意：活跃订阅的唯一性通过复合唯一索引实现，包含gmt_deleted字段
    -- 级联删除逻辑需要在应用代码中处理，删除Collection时同时删除相关订阅记录
);

-- 索引优化
CREATE UNIQUE INDEX idx_user_marketplace_history_unique ON user_collection_subscription(user_id, collection_marketplace_id, gmt_deleted);  -- 允许多条历史记录，但活跃订阅(gmt_deleted=NULL)保持唯一
CREATE INDEX idx_user_subscription_marketplace ON user_collection_subscription(collection_marketplace_id);
CREATE INDEX idx_user_subscription_user ON user_collection_subscription(user_id);
CREATE INDEX idx_user_subscription_gmt_deleted ON user_collection_subscription(gmt_deleted);
```

#### 2.2. 数据库约束说明

**业务约束:**

1. **唯一性约束**: 每个 Collection 只能有一条分享记录
2. **订阅约束**: 一个用户对同一发布实例（collection_marketplace）只能有一个活跃订阅，但可以保留多条历史记录
3. **所有权约束**: 用户无法订阅自己是所有者的 Collection（业务逻辑禁止）
4. **应用层级联**: Collection 删除时，需要在代码中同时软删除相关的分享和订阅记录
5. **状态检查**: 分享状态只能是 'DRAFT' 或 'PUBLISHED'（应用层校验）
6. **发布状态绑定**: 订阅关系与Collection的发布状态绑定，Collection取消发布时订阅失效，重新发布时需要重新订阅

**性能优化:**

1. **简单索引策略**: 使用常规索引，与项目现有表保持一致，降低维护复杂度
2. **复合索引**:
   - 用户+Collection+删除状态组合查询优化（订阅检查场景）
   - 状态+时间复合索引（市场列表查询场景）
   - Collection关联查询索引（按collection_id查询优化）
3. **应用层更新**: `gmt_updated` 字段在代码中手动更新，保持项目一致性
4. **数据规范化**: 遵循项目惯例，不使用外键约束，通过应用层维护数据一致性

#### 2.3. 数据生命周期

**分享生命周期:**

- **创建**: 用户首次发布 Collection 时创建记录，状态为 'PUBLISHED'
- **取消发布**: 将 collection_marketplace 记录状态改为 'DRAFT'（保留记录），应用层处理相关订阅失效
- **重新发布**: 将现有 collection_marketplace 记录状态改回 'PUBLISHED'，之前的订阅关系不会自动恢复
- **删除处理**: Collection 删除时，需要在代码中同时软删除 collection_marketplace 记录

**订阅生命周期:**

- **订阅**: 用户订阅已发布的 Collection，创建订阅记录（关联到具体的发布实例）
- **取消订阅**: 设置 `gmt_deleted = NOW()`，保留历史记录
- **自动失效**: Collection 取消发布时，collection_marketplace 记录状态改为 'DRAFT'，应用层查询并软删除相关订阅记录
- **级联删除**: Collection 删除时，在代码中批量软删除相关的 marketplace 和订阅记录

### 3. 系统架构与业务流程

#### 3.1. 技术架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端 (UmiJS + React)                      │
├─────────────────────────────────────────────────────────────────┤
│  /marketplace     │ /collections      │ /collections/{collection_id}            │
│  (市场浏览页面)     │ (统一工作台)       │ (Collection详情, 区分owner/订阅者)         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ HTTP/HTTPS
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    后端 API (FastAPI)                            │
├─────────────────────────────────────────────────────────────────┤
│  MarketplaceView              │ CollectionView                  │
│  - 市场Collection列表          │ - Collection CRUD API          │
│  - 订阅/取消订阅API           │ - 发布/取消发布API               │
│  - 用户订阅列表API            │ - 分享状态查询API                │
│                              │ - 权限控制集成                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Service Layer
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        服务层 (Business Logic)                   │
├─────────────────────────────────────────────────────────────────┤
│  MarketplaceService           │ CollectionService               │
│  - 发布/取消发布              │ - 现有CRUD操作 (保持不变)        │
│  - 订阅/取消订阅              │ - 添加分享状态字段               │
│  - 用户订阅列表               │                                 │
│  - 市场Collection列表         │ MarketplaceCollectionService    │
│  - 分享状态查询               │ - 订阅权限检查                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                │ Database Layer
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        数据库层 (PostgreSQL)                     │
├─────────────────────────────────────────────────────────────────┤
│  collections              │ collection_marketplace              │
│  - 原有Collection数据      │ - 分享状态表                        │
│                           │                                     │
│  user_collection_subscription                                   │
│  - 用户订阅关系表                                                │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2. 核心业务流程

**流程1: Collection 发布流程**

```
用户A (Collection所有者)
    │
    ├─ 1. POST /api/v1/collections/{collection_id}/sharing
    │     │
    │     ├─ 验证用户身份和所有权
    │     ├─ 创建/更新 collection_marketplace 记录
    │     └─ 状态设置为 'PUBLISHED'
    │
    └─ 2. Collection 出现在市场列表
           │
           └─ 其他用户可以在 /marketplace 看到
```

**流程2: 用户订阅流程**

```
用户B (非所有者)
    │
    ├─ 1. 浏览市场 (GET /api/v1/marketplace/collections)
    │     │
    │     └─ 看到用户A发布的Collection
    │
    ├─ 2. 点击订阅 (POST /api/v1/marketplace/collections/{collection_id}/subscribe)
    │     │
    │     ├─ 验证Collection已发布
    │     ├─ 验证用户不是Collection所有者（防止订阅自己的Collection）
    │     ├─ 检查是否已订阅
    │     ├─ 创建 user_collection_subscription 记录
    │     └─ 返回订阅成功，自动跳转到Collection详情页
    │
    └─ 3. 访问订阅的Collection内容
           │
           ├─ 3a. 在Collection列表页面查看订阅的Collection
           │     │
           │     ├─ 页面: /collections (主Collection列表页面)
           │     ├─ API调用:
           │     │   ├─ GET /api/v1/collections (获取自有Collection)
           │     │   └─ GET /api/v1/marketplace/collections/subscriptions (获取订阅Collection)
           │     ├─ 前端合并: 两个接口响应合并显示在同一页面
           │     ├─ 区分显示: 订阅Collection显示"已订阅"标签，自有Collection显示"我的"标签
           │     └─ 点击进入: 智能路由（根据用户关系跳转到不同页面）
           │

           └─ 3b. MarketplaceCollection详情页只读访问
                 │
                 ├─ 页面: /marketplace/collections/{collection_id}
                 ├─ API: GET /api/v1/marketplace/collections/{collection_id}
                 ├─ 权限检查: _check_subscription_access() 验证订阅状态
                 ├─ 响应类型: SharedCollection（专用只读接口）
                 ├─ UI显示: 顶部显示只读Banner
                 ├─ 功能权限: 可查看文档、图谱，可使用聊天Bot
                 └─ 操作限制: 隐藏所有编辑、删除、上传按钮
```

**流程3: MarketplaceCollection接口权限检查流程（新增）**

```
用户请求访问 /api/v1/marketplace/collections/{id}
    │
    └─ _check_subscription_access()
           │
           ├─ 检查Collection是否存在且已发布
           │   └─ 否 → 403 "Collection no longer available" ❌
           │
           ├─ 检查用户是否已订阅 (gmt_deleted IS NULL)
           │   └─ 否 → 403 "Need to subscribe first" ❌
           │
           └─ 是 → 只读访问权限 ✅
```

**流程4: 用户取消订阅流程**

```
用户B (已订阅用户)
    │
    ├─ 1. 在MarketplaceCollection详情页点击"取消订阅"
    │     │
    │     ├─ 页面: /marketplace/collections/{collection_id}
    │     ├─ UI元素: 详情页面显示"取消订阅"按钮
    │     └─ 确认对话框: "确定要取消订阅此知识库吗？"
    │
    ├─ 2. 执行取消订阅 (DELETE /api/v1/marketplace/collections/{collection_id}/subscribe)
    │     │
    │     ├─ 验证用户身份和订阅状态
    │     ├─ 验证用户确实已订阅该Collection (gmt_deleted IS NULL)
    │     ├─ 软删除订阅记录 (设置 gmt_deleted = current_timestamp)
    │     └─ 返回取消成功响应
    │
    ├─ 3. 立即失去访问权限
    │     │
    │     ├─ 权限检查: marketplace/collections接口的 _check_subscription_access() 立即返回403
    │     ├─ 前端处理: 自动跳转到市场页面或首页
    │     └─ 提示消息: "已成功取消订阅"
    │
    └─ 4. Collection从用户工作区移除
           │
           ├─ API影响: GET /api/v1/marketplace/collections/subscriptions 不再返回该Collection
           ├─ 前端更新: Collection列表页面不再显示该Collection
           ├─ 重新订阅: 用户可以在市场页面重新订阅
           └─ 历史保留: 数据库保留订阅历史记录（便于审计）
```

**流程5: Collection取消发布流程**

```
用户A (Collection所有者)
    │
    ├─ 1. 在Collection详情页点击"取消发布"
    │     │
    │     ├─ 页面: /collections/{collection_id}
    │     ├─ UI元素: 分享控制组件显示"取消发布"按钮
    │     ├─ 确认对话框: "取消发布后，所有订阅用户将失去访问权限，确定继续吗？"
    │     └─ 风险提示: 显示当前订阅用户数量
    │
    ├─ 2. 执行取消发布 (DELETE /api/v1/collections/{collection_id}/sharing)
    │     │
    │     ├─ 验证用户身份和所有权
    │     ├─ 将 collection_marketplace 记录状态改为 'DRAFT'（设置 status='DRAFT', gmt_updated=current_timestamp）
    │     ├─ 应用层查询并软删除相关订阅记录（设置 gmt_deleted）
    │     └─ 返回取消发布成功响应
    │
    ├─ 3. 立即从市场移除
    │     │
    │     ├─ 市场API: GET /api/v1/marketplace/collections 不再返回该Collection
    │     ├─ 搜索结果: 市场搜索无法找到该Collection
    │     └─ 直接访问: 非所有者访问将返回403 "Collection not published"
    │
    ├─ 4. 所有订阅用户失去访问权限
    │     │
    │     ├─ 权限检查: marketplace/collections接口的 _check_subscription_access() 对所有订阅用户返回403
    │     ├─ 活跃连接: 正在使用的用户会在下次请求时收到403错误
    │     ├─ 前端处理: 订阅用户的Collection列表自动移除该项
    │     └─ 通知机制: (可选) 向订阅用户发送取消发布通知
    │
    └─ 5. 重新发布支持
           │
           ├─ 状态恢复: 所有者可以重新发布 (POST /api/v1/collections/{collection_id}/sharing)
           ├─ 订阅恢复: 重新发布后不会自动恢复之前的订阅关系
           ├─ 用户重新订阅: 之前的订阅用户需要重新手动订阅
           └─ 历史记录: 保留所有发布/取消发布的历史记录
```

#### 3.3. 安全设计

**权限控制策略:**

1. **严格的所有权验证**: 只有Collection所有者可以发布/取消发布
2. **订阅前置检查**: 非所有者必须订阅才能访问内容
3. **只读强制执行**: 订阅用户无法进行任何写操作
4. **自动权限回收**: 取消发布时自动失效所有订阅

**数据安全:**

1. **级联删除**: Collection删除时自动清理相关记录
2. **软删除审计**: 保留订阅历史记录便于审计
3. **状态一致性**: 通过事务确保分享状态和订阅状态一致

#### 3.4. 性能考虑

**数据库优化:**

1. **索引策略**: 为高频查询场景创建专门索引，使用简单一致的索引设计
2. **分页查询**: 所有列表接口支持分页，避免大数据量查询
3. **复合索引**: 针对多字段查询场景创建复合索引，提升查询效率

**查询优化:**

```sql
-- 高效的市场列表查询（利用复合索引 idx_collection_marketplace_list）
SELECT cm.id, c.title, c.description, u.username, cm.gmt_created
FROM collection_marketplace cm
JOIN collections c ON cm.collection_id = c.id
JOIN users u ON c.user_id = u.id
WHERE cm.status = 'PUBLISHED' AND cm.gmt_deleted IS NULL
ORDER BY cm.gmt_created DESC
LIMIT 12 OFFSET ?;

-- 高效的订阅检查查询（利用唯一索引 idx_user_marketplace_active_unique）
SELECT ucs.id FROM user_collection_subscription ucs
JOIN collection_marketplace cm ON ucs.collection_marketplace_id = cm.id
WHERE ucs.user_id = ? AND cm.collection_id = ? AND ucs.gmt_deleted IS NULL AND cm.gmt_deleted IS NULL
LIMIT 1;

-- 获取用户订阅的Collection详情（通过marketplace表关联）
SELECT c.id, c.title, c.description, u.username, ucs.id as subscription_id, ucs.gmt_subscribed
FROM user_collection_subscription ucs
JOIN collection_marketplace cm ON ucs.collection_marketplace_id = cm.id
JOIN collections c ON cm.collection_id = c.id
JOIN users u ON c.user_id = u.id
WHERE ucs.user_id = ? AND ucs.gmt_deleted IS NULL AND cm.gmt_deleted IS NULL
ORDER BY ucs.gmt_subscribed DESC;
```

### 4. 后端设计

遵循**软件架构分层原则**，按照从底层到高层的顺序进行设计：数据模型 → 服务层 → API层。

#### 4.1. 数据模型设计 (OpenAPI / `view_models.py`)

**4.1.1 新增数据库模型 (aperag/db/models.py):**

- **`CollectionMarketplaceStatusEnum`**: 分享状态枚举 (Python enum，用于代码逻辑)
  - `DRAFT = "DRAFT"`: 未发布状态，仅所有者可见
  - `PUBLISHED = "PUBLISHED"`: 已发布状态，公开可见

- **`CollectionMarketplace`**: Collection 分享状态表 (SQLAlchemy 模型)
  - `id: str`: 分享记录的唯一标识符
  - `collection_id: str`: 关联的 Collection ID
  - `status: str`: 当前分享状态 (VARCHAR存储，值为'DRAFT'或'PUBLISHED')
  - `gmt_created: datetime`: 分享记录创建时间
  - `gmt_updated: datetime`: 分享记录最后更新时间
  - `gmt_deleted: Optional[datetime]`: 软删除时间（NULL表示活跃记录）

- **`UserCollectionSubscription`**: 用户订阅 Collection 表 (SQLAlchemy 模型)
  - `id: str`: 订阅记录的唯一标识符
  - `user_id: str`: 订阅用户 ID
  - `collection_marketplace_id: str`: 关联的发布实例 ID（collection_marketplace.id）
  - `gmt_subscribed: datetime`: 订阅时间
  - `gmt_deleted: Optional[datetime]`: 取消订阅时间（NULL表示活跃订阅）

**4.1.2 新增视图模型 (aperag/schema/view_models.py):**

> 在aperag/api/components/schemas/marketplace.yaml中定义注意需要用make generate-models和make generate-frontend-sdk生成前后端代码

- **`SharedCollection`**: 共享的 Collection 信息（视图模型）
  - `id: str`: Collection ID
  - `title: str`: Collection 标题
  - `description: str`: Collection 描述
  - `owner_user_id: str`: 原所有者用户ID
  - `owner_username: str`: 原所有者用户名
  - `subscription_id: Optional[str]`: 订阅记录ID（有值表示已订阅，None表示未订阅）
  - `gmt_subscribed: Optional[datetime]`: 订阅时间（仅在已订阅时有值）

- **`SharedCollectionList`**: 共享 Collection 列表响应
  - `items: List[SharedCollection]`: 共享的 Collection 列表
  - `total: int`: 总数量（用于分页）
  - `page: int`: 当前页码
  - `page_size: int`: 每页大小

**4.1.3 修改现有模型:**

- **`Collection`**: 扩展现有 Collection model（采用语义化设计）
  - `is_published: bool`: 是否已发布到市场
  - `published_at: Optional[datetime]`: 发布时间，未发布时为null

**4.1.4 OpenAPI Schema 组织:**

所有新增的 model 定义将放置在 `aperag/api/components/schemas/marketplace.yaml` 文件中，现有 Collection model 的扩展将在 `aperag/api/components/schemas/collection.yaml` 中添加新字段。

#### 4.2. 服务层设计 (Business Logic)

**4.2.1 新增服务模块: `aperag/service/marketplace_service.py`**

```python
class MarketplaceService:
    """
    Marketplace业务逻辑服务
    职责: 处理所有与市场和分享相关的业务逻辑
    """

    async def publish_collection(self, user_id: str, collection_id: str) -> None:
        """发布Collection到市场"""
        # 验证用户所有权
        # 创建或更新collection_marketplace记录
        # 状态设置为PUBLISHED

    async def unpublish_collection(self, user_id: str, collection_id: str) -> None:
        """从市场下架Collection"""
        # 验证用户所有权
        # 将collection_marketplace记录状态改为'DRAFT'（设置status='DRAFT', gmt_updated=datetime.utcnow()）
        # 应用层查询并软删除相关订阅记录（设置gmt_deleted）
        # 注意：需要使用事务确保数据一致性

    async def get_sharing_status(self, collection_id: str) -> tuple[bool, Optional[datetime]]:
        """获取Collection的分享状态"""
        # 返回 (is_published, published_at) 元组

    async def get_raw_sharing_status(self, collection_id: str) -> Optional[CollectionMarketplace]:
        """获取原始分享状态（供权限检查使用）"""

    async def list_published_collections(self, user_id: str, page: int, page_size: int) -> SharedCollectionList:
        """列出市场中所有已发布的Collection"""
        # 查询PUBLISHED状态的Collection
        # 计算当前用户的订阅状态
        # 支持分页

    async def subscribe_collection(self, user_id: str, collection_id: str) -> SharedCollection:
        """订阅Collection"""
        # 1. 查找Collection对应的已发布marketplace记录 (status = 'PUBLISHED', gmt_deleted IS NULL)
        # 2. 验证用户不是Collection所有者 (user_id != collection.user)
        # 3. 检查是否已订阅该marketplace实例，防止重复订阅
        # 4. 创建user_collection_subscription记录（关联collection_marketplace_id）
        # 异常: 如果用户是所有者，抛出 SelfSubscriptionError("Cannot subscribe to your own collection")

    async def unsubscribe_collection(self, user_id: str, collection_id: str) -> None:
        """取消订阅Collection"""
        # 验证用户已订阅该Collection
        # 软删除订阅记录(设置gmt_deleted)

    async def get_user_subscription(self, user_id: str, collection_id: str) -> Optional[UserCollectionSubscription]:
        """获取用户对指定Collection的活跃订阅状态"""
        # 通过collection_id查找已发布的marketplace记录，再查找对应的订阅记录
        # 供权限检查函数调用
        # 返回None表示未订阅或已取消订阅

    async def list_user_subscribed_collections(self, user_id: str, page: int, page_size: int) -> SharedCollectionList:
        """获取用户所有活跃订阅的Collection"""
        # 查询WHERE gmt_deleted IS NULL
        # 关联查询获取Collection详细信息和原所有者信息
        # 支持分页
```

**4.2.2 级联删除处理**

由于新增了marketplace相关数据，需要在Collection删除时进行级联处理：

```python
# 在现有的collection删除逻辑中添加
async def delete_collection(self, user_id: str, collection_id: str):
    # ... 现有的删除逻辑

    # 新增：级联软删除marketplace相关记录
    await marketplace_service.cleanup_collection_marketplace_data(collection_id)
    # 该方法会：
    # 1. 软删除collection_marketplace记录 (设置gmt_deleted)
    # 2. 批量软删除user_collection_subscription记录 (设置gmt_deleted)
    # 3. 使用事务确保数据一致性
```

**4.2.3 接口权限策略:**

采用"专门接口分离"的设计，权限职责明确：

**现有Collection接口族** (`/api/v1/collections/*`)：

- 保持现有权限逻辑不变
- 只需在Collection model响应中添加 `is_published` 和 `published_at` 字段

**新增MarketplaceCollection接口族** (`/api/v1/marketplace/collections/*`)：

- **权限策略**: 仅允许有效订阅用户只读访问
- **权限检查**: 在 `marketplace_collection_service._check_subscription_access` 中统一处理
- **适用功能**:
  - 文档列表查看和预览
  - 知识图谱只读浏览
  - 聊天Bot查询（如果支持）

**核心优势**：

- ✅ **不影响现有逻辑**: 现有Collection接口保持不变
- ✅ **职责清晰**: 新增接口专门处理订阅访问
- ✅ **安全可靠**: 权限检查在接口层面就被分离
- ✅ **易于维护**: marketplace功能独立，便于后续扩展

#### 4.3. API 端点设计 (View 层)

基于服务层的业务逻辑，设计RESTful API端点，遵循统一的URL命名规范和错误处理模式。

**4.3.1 新增 API 端点**

设计采用混合 URL 模式：marketplace 相关的浏览功能使用 `/marketplace` 路径，而具体 Collection 的分享操作作为 Collection 的子资源管理。

- **`GET /api/v1/marketplace/collections`**: 列出市场中所有公开的 Collection
  - **功能**: 返回所有状态为 `PUBLISHED` 的 Collection 列表（包括当前用户自己发布的Collection）
  - **权限**: 任何已登录用户都可以访问
  - **响应**: `SharedCollectionList` 类型，包含每个 Collection 的基本信息、所有者用户名；`subscription_id` 字段表示当前用户是否已订阅
  - **分页**: 支持 `page` 和 `page_size` 参数

- **`POST /api/v1/collections/{collection_id}/sharing`**: 发布一个 Collection 到市场
  - **功能**: 将指定 Collection 的状态设置为 `PUBLISHED`
  - **权限**: 仅限 Collection 所有者
  - **行为**: 在 `collection_marketplace` 表中创建记录或更新状态
  - **响应**: 返回 204 No Content

- **`DELETE /api/v1/collections/{collection_id}/sharing`**: 从市场下架一个 Collection
  - **功能**: 将指定 Collection 的状态设置为 `DRAFT`（不删除记录，仅改变状态）
  - **权限**: 仅限 Collection 所有者
  - **行为**: 立即停止其他用户对该 Collection 的访问，应用层处理相关订阅失效
  - **响应**: 返回 204 No Content

- **`GET /api/v1/collections/{collection_id}/sharing`**: 获取指定 Collection 的分享状态
  - **功能**: 返回 Collection 的当前分享状态和相关信息
  - **权限**: 仅限 Collection 所有者
  - **响应**: 简洁的分享状态对象

  ```json
  {
    "is_published": true,
    "published_at": "2024-01-15T10:30:00Z"
  }
  ```

- **`POST /api/v1/marketplace/collections/{collection_id}/subscribe`**: 订阅一个已发布的 Collection
  - **功能**: 将指定的已发布 Collection 添加到用户的订阅列表
  - **权限**: 任何已登录用户（除 Collection 所有者外）
  - **业务限制**: 用户无法订阅自己是所有者的 Collection
  - **行为**: 在 `user_collection_subscription` 表中创建订阅记录
  - **响应**: 返回 `SharedCollection` 信息
  - **错误处理**:
    - 如果已订阅则返回 409 Conflict
    - 如果尝试订阅自己的 Collection 则返回 400 Bad Request "Cannot subscribe to your own collection"

- **`DELETE /api/v1/marketplace/collections/{collection_id}/subscribe`**: 取消订阅 Collection
  - **功能**: 从用户的订阅列表中移除指定 Collection
  - **权限**: 仅限已订阅该 Collection 的用户
  - **行为**: 软删除订阅记录（设置 `gmt_deleted = current_timestamp`）
  - **响应**: 返回 204 No Content

- **`GET /api/v1/marketplace/collections/subscriptions`**: 获取用户订阅的 Collection 列表 (MVP核心API)
  - **功能**: 返回当前用户所有活跃订阅的 Collection（`gmt_deleted IS NULL`）
  - **权限**: 仅限当前用户（通过认证确定）
  - **响应**: Collection 列表，每个item包含订阅信息（订阅时间、原所有者等）
  - **分页**: 支持 `page` 和 `page_size` 参数
  - **设计理念**: 资源层级更清晰，subscriptions作为collections的子资源

**4.3.2 完整API结构概览**

采用方案B的RESTful API设计，marketplace作为资源命名空间：

```
# Marketplace 相关API（新增）
GET    /api/v1/marketplace/collections                    # 市场列表
GET    /api/v1/marketplace/collections/subscriptions     # 用户订阅列表
POST   /api/v1/marketplace/collections/{id}/subscribe    # 订阅Collection
DELETE /api/v1/marketplace/collections/{id}/subscribe    # 取消订阅
GET    /api/v1/marketplace/collections/{id}              # 订阅Collection详情（只读）
GET    /api/v1/marketplace/collections/{id}/documents    # 订阅Collection的文档列表（只读）
GET    /api/v1/marketplace/collections/{id}/documents/{doc_id}/preview  # 文档预览（只读）
GET    /api/v1/marketplace/collections/{id}/graph        # 知识图谱（只读）

# Collection 管理API（现有，增强）
GET    /api/v1/collections                               # 用户自有Collection列表
GET    /api/v1/collections/{id}                          # Collection详情（含分享状态）
POST   /api/v1/collections/{id}/sharing                  # 发布到市场
DELETE /api/v1/collections/{id}/sharing                  # 从市场下架
GET    /api/v1/collections/{id}/sharing                  # 查询分享状态
# ... 其他现有Collection管理接口保持不变
```

**API设计优势**：

- ✅ **命名空间清晰**: `/marketplace/` 明确表示市场功能
- ✅ **RESTful规范**: 资源层级结构合理
- ✅ **职责分离**: 管理和浏览功能完全分开
- ✅ **扩展性好**: 便于后续添加其他marketplace功能

**4.3.3 MarketplaceCollection 专用 API 接口**

为了确保数据隔离和API语义清晰，我们为SharedCollection（订阅的Collection）设计专门的只读API接口。这些接口：

- 只返回订阅者需要的字段，避免敏感信息泄露
- 提供清晰的只读语义，用户不会意外调用编辑接口
- 简化权限逻辑，只需验证订阅关系

**核心设计原则**：

- **数据隔离**: SharedCollection接口返回的字段与Collection接口不同，隐藏配置、统计等敏感信息
- **权限简单**: 只需验证用户是否有有效订阅，无需复杂的所有者判断
- **功能专注**: 只提供内容浏览必需的4个核心接口

**MarketplaceCollection 专用接口列表**：

- **`GET /api/v1/marketplace/collections/{collection_id}`**: 获取MarketplaceCollection详情
  - **功能**: 返回订阅者视角的Collection信息
  - **权限检查**: 验证当前用户是否已订阅该Collection（`user_collection_subscription.gmt_deleted IS NULL`）
  - **响应类型**: `SharedCollection`
  - **数据隔离**:
    - ✅ 包含: `id`, `title`, `description`, `owner_username`, `subscription_id`, `gmt_subscribed`
    - ❌ 隐藏: Collection的详细配置、内部统计、所有者ID等敏感信息
  - **错误处理**:
    - 未订阅: `403 Forbidden "You need to subscribe to this collection first"`
    - Collection不存在: `404 Not Found`
    - Collection未发布: `403 Forbidden "This collection is no longer available"`

- **`GET /api/v1/marketplace/collections/{collection_id}/documents`**: 获取MarketplaceCollection的文档列表
  - **功能**: 返回文档基本信息列表（只读模式）
  - **权限检查**: 验证订阅关系
  - **响应格式**: 标准文档列表，但隐藏创建者、编辑历史等内部信息
  - **分页支持**: `page`, `page_size` 参数
  - **过滤支持**: `search`, `file_type` 等基础过滤

- **`GET /api/v1/marketplace/collections/{collection_id}/documents/{document_id}/preview`**: 预览MarketplaceCollection中的文档
  - **功能**: 获取文档内容预览（与原接口相同的预览功能）
  - **权限检查**: 验证订阅关系
  - **响应格式**: 文档预览数据，格式与原接口相同
  - **支持格式**: 所有原有支持的文档格式（PDF、Word、图片等）

- **`GET /api/v1/marketplace/collections/{collection_id}/graph`**: 获取MarketplaceCollection的知识图谱
  - **功能**: 返回知识图谱数据（只读模式）
  - **权限检查**: 验证订阅关系
  - **响应格式**: 图谱节点和边的数据，与原接口格式相同
  - **查询参数**: 支持 `node_limit`, `depth` 等图谱查询参数
  - **注意**: 不提供图谱编辑相关的接口（如合并建议、节点编辑等）

**权限验证逻辑**（所有MarketplaceCollection接口通用）：

```python
async def _check_subscription_access(user_id: str, collection_id: str) -> bool:
    """检查用户是否有权访问SharedCollection"""

    # 1. 检查Collection是否存在且已发布
    collection_marketplace = await db.get_collection_marketplace_by_collection_id(collection_id)
    if not collection_marketplace or collection_marketplace.status != "PUBLISHED":
        raise HTTPException(status_code=403, detail="This collection is no longer available")

    # 2. 检查用户订阅状态
    subscription = await db.get_user_subscription(user_id, collection_id)
    if not subscription or subscription.gmt_deleted:
        raise HTTPException(status_code=403, detail="You need to subscribe to this collection first")

    return True
```

**与原有Collection接口的区别**：

| 方面 | Collection接口 | SharedCollection接口 |
| --- | --- | --- |
| **用途** | 用户自有Collection的完整管理 | 订阅Collection的只读访问 |
| **权限** | 验证所有权 | 验证订阅关系 |
| **功能** | 完整CRUD + 高级功能 | 仅内容浏览（4个接口） |
| **数据** | 完整字段，包含配置和统计 | 精简字段，隐藏敏感信息 |
| **路径** | `/api/v1/collections/{id}` | `/api/v1/marketplace/collections/{id}` |

### 5. 错误处理策略

**API 错误处理分层:**

```python
# 1. 业务逻辑层错误 (Service Layer)
class MarketplaceError(Exception):
    """市场相关业务错误基类"""
    pass

class CollectionNotPublishedError(MarketplaceError):
    """Collection未发布错误"""
    pass

class AlreadySubscribedError(MarketplaceError):
    """重复订阅错误"""
    pass

class SubscriptionNotFoundError(MarketplaceError):
    """订阅不存在错误"""
    pass

class SelfSubscriptionError(MarketplaceError):
    """尝试订阅自己Collection错误"""
    pass

# 2. API层错误转换 (View Layer)
@router.post("/marketplace/collections/{collection_id}/subscribe")
async def subscribe_collection(collection_id: str, user: User = Depends(current_user)):
    try:
        result = await marketplace_service.subscribe_collection(user.id, collection_id)
        return result
    except CollectionNotPublishedError:
        raise HTTPException(status_code=400, detail="Collection is not published")
    except SelfSubscriptionError:
        raise HTTPException(status_code=400, detail="Cannot subscribe to your own collection")
    except AlreadySubscribedError:
        raise HTTPException(status_code=409, detail="Already subscribed to this collection")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in subscribe_collection: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

**数据库事务一致性:**

```python
# marketplace_service.py 中的事务处理
async def unpublish_collection(self, user_id: str, collection_id: str):
    async with self.db_session.begin():  # 事务开始
        try:
            # 1. 验证所有权
            collection = await self._verify_ownership(user_id, collection_id)

            # 2. 更新分享状态
            await self._update_sharing_status(collection_id, 'DRAFT')

            # 3. 批量失效订阅
            await self._invalidate_subscriptions(collection_id)

            # 事务自动提交
            return {"message": "Collection unpublished successfully"}

        except Exception as e:
            # 事务自动回滚
            logger.error(f"Failed to unpublish collection {collection_id}: {e}")
            raise
```

**前端错误处理:**

```typescript
// 前端错误处理中间件
const handleApiError = (error: any) => {
  if (error.response?.status === 403) {
    if (error.response.data.detail?.includes('subscribe')) {
      return { type: 'SUBSCRIPTION_REQUIRED', message: '请先订阅此知识库' };
    }
    return { type: 'PERMISSION_DENIED', message: '权限不足' };
  }

  if (error.response?.status === 409) {
    return { type: 'CONFLICT', message: '您已订阅此知识库' };
  }

  return { type: 'UNKNOWN', message: '操作失败，请重试' };
};

// 组件中的错误处理
const subscribeCollection = async (collectionId: string) => {
  try {
    await api.subscribeCollection(collectionId);
    message.success('订阅成功');
    refresh();
  } catch (error) {
    const errorInfo = handleApiError(error);
    message.error(errorInfo.message);
  }
};
```

### 6. 前端设计

#### 6.1. 页面与路由设计

**A. 新增市场页面**

- **路由**: `/marketplace`
- **文件位置**: `frontend/src/pages/marketplace/index.tsx`
- **页面功能**:
  - 展示所有已发布的 Collection 卡片列表
  - 支持分页浏览，默认每页显示 12 个卡片
  - 每个卡片包含：Collection 标题、描述、所有者用户名、发布时间
  - 点击卡片跳转到对应的 Collection 详情页（只读模式）
- **UI 设计**:

**B. Collection列表页面增强 (MVP核心功能)**

- **路由**: `/collections` (现有页面增强)
- **文件位置**: `frontend/src/pages/collections/index.tsx`
- **API 调用策略**: 同时调用两个专门的API接口
  ```typescript
  // 并行调用两个接口
  const [ownedCollections, sharedCollections] = await Promise.all([
    api.getCollections(pagination), // 获取自有Collection，返回Collection[]
    api.getMarketplaceCollectionsSubscriptions(pagination), // 获取订阅Collection，返回SharedCollection[]
  ]);
  ```
- **设计理念**: 聚焦marketplace核心概念，避免workspace抽象，双接口专职专责
- **页面功能增强**:
  - 前端合并显示用户自有Collection + 订阅的Collection
  - 新增Collection类型标签：`我的` / `已订阅`
  - 新增筛选器：`全部` / `我的知识库` / `已订阅` (前端筛选实现)
  - 订阅Collection显示特殊图标和样式区分
  - 在订阅Collection上提供取消订阅操作
  - **UI 设计增强**:
  ```typescript
  // Collection 卡片 Props - 支持两种类型
  interface OwnedCollectionCardProps {
    type: 'owned';
    collection: Collection;
  }

  interface SharedCollectionCardProps {
    type: 'shared';
    collection: SharedCollection;
  }

  type CollectionCardProps =
    | OwnedCollectionCardProps
    | SharedCollectionCardProps;
  ```

  - **Collection卡片左上角显示类型标签**:
    - 自有Collection: 绿色标签 "我的"
    - 订阅Collection: 蓝色标签 "已订阅"
  - **订阅Collection卡片样式区分**:
    - 边框颜色: `#1890ff` (蓝色)
    - 卡片背景: `#f6f9ff` (浅蓝色背景)
    - 标题前添加订阅图标 `<ShareAltOutlined />`
  - **悬浮信息显示**:
    - 自有Collection: 显示创建时间
    - 订阅Collection: 显示 "来自 @{owner_username} • 订阅于 {gmt_subscribed相对时间}"
  - **操作菜单差异化**:
    - 自有Collection: 编辑、删除、分享设置、查看详情
    - 订阅Collection: 查看详情、取消订阅
  - **筛选器实现**:
    ```typescript
    const [filter, setFilter] = useState<'all' | 'owned' | 'shared'>('all');
    // 前端合并两种类型的Collection进行筛选
    const allCollections = [
      ...ownedCollections.map((col) => ({
        type: 'owned' as const,
        collection: col,
      })),
      ...sharedCollections.map((col) => ({
        type: 'shared' as const,
        collection: col,
      })),
    ];
    const filteredCollections = allCollections.filter((item) => {
      if (filter === 'owned') return item.type === 'owned';
      if (filter === 'shared') return item.type === 'shared';
      return true; // 'all'
    });
    ```

**C. Collection详情页面（所有者专用）**

- **路由**: `/collections/{collection_id}` (保持现有路由)
- **文件位置**: `frontend/src/pages/collections/$collectionId/index.tsx`
- **API调用**: `GET /api/v1/collections/{collection_id}` (仅限所有者)
- **权限检查**: 使用现有的所有者权限验证
- **功能特性**:
  - 显示SharingControl组件（发布/取消发布开关）
  - 显示完整的编辑功能（文档管理、设置、删除等）
  - 可查看分享状态信息（通过 `is_published` 和 `published_at` 字段）
  - 如果Collection已发布，显示订阅用户数量（可选功能）

**D. MarketplaceCollection详情页面（订阅用户专用）**

- **路由**: `/marketplace/collections/{collection_id}` (新增专门路由)
- **文件位置**: `frontend/src/pages/marketplace/collections/$collectionId/index.tsx`
- **API调用**: `GET /api/v1/marketplace/collections/{collection_id}` (仅限订阅用户)
- **权限检查**: 后端 `_check_subscription_access()` 验证用户是否已订阅
- **功能特性**:
  - 页面顶部显示ReadOnlyBanner组件
  - 显示订阅信息："来自 @{owner_username} 的共享知识库"
  - 提供取消订阅按钮
  - 隐藏所有编辑按钮：编辑Collection、上传文档、删除文档、重建索引
  - 隐藏设置页面入口
  - 文档列表只显示查看、预览按钮
  - 图谱页面隐藏合并节点等编辑功能
  - 聊天Bot可正常使用（只读查询）

**路由跳转逻辑**：

```typescript
// 智能路由跳转逻辑
const handleCollectionClick = (
  collection: SharedCollection,
  currentUser: User,
) => {
  if (collection.owner_user_id === currentUser.id) {
    // 自己的Collection → 所有者管理页面
    navigate(`/collections/${collection.id}`);
  } else if (collection.subscription_id) {
    // 已订阅的Collection → Marketplace详情页面
    navigate(`/marketplace/collections/${collection.id}`);
  } else {
    // 未订阅的Collection → 显示订阅对话框
    showSubscribeModal(collection.id);
  }
};

// 从Collection列表点击（工作台页面）
const handleCollectionClick = (item: CollectionListItem) => {
  if (item.type === 'owned') {
    // 自有Collection → 所有者管理页面
    navigate(`/collections/${item.collection.id}`);
  } else {
    // 订阅Collection → Marketplace详情页面
    navigate(`/marketplace/collections/${item.collection.id}`);
  }
};
```

#### 6.2. 组件设计

**A. CollectionMarketplaceCard 组件**

- **文件位置**: `frontend/src/components/CollectionMarketplaceCard.tsx`
- **Props 接口**:
  ```typescript
  interface CollectionMarketplaceCardProps {
    collection: SharedCollection;
    onClick: (collectionId: string) => void;
  }
  ```
- **UI 元素**:
  - Collection 标题（加粗显示）
  - Collection 描述（最多显示 150 字符，超出显示省略号）
  - 所有者用户名（小号字体，灰色显示）
  - 订阅状态显示（根据 `subscription_id` 是否有值判断已订阅/未订阅）
  - 悬浮效果和点击交互

**B. 只读模式提示 Banner**

- **文件位置**: `frontend/src/components/ReadOnlyBanner.tsx`
- **显示条件**: 当响应类型为 `SharedCollection` 时显示
- **Props 接口**:
  ```typescript
  interface ReadOnlyBannerProps {
    ownerUsername: string;
  }
  ```
- **UI 设计**:
  - 位置：页面顶部，在页面标题下方
  - 样式：使用 Ant Design Alert 组件，type="info"
  - 文案："您正在以只读模式浏览来自 @{ownerUsername} 的共享知识库，无法进行修改操作"
  - 图标：信息图标
  - 可关闭：否

**C. 分享控制组件**

- **文件位置**: `frontend/src/components/SharingControl.tsx`
- **显示条件**: 仅当用户是 Collection 所有者时显示
- **UI 元素**:
  - 分享状态开关（Switch 组件）
  - 状态标签："已发布到市场" / "未发布"
  - 确认对话框：发布和取消发布操作都需要用户确认

#### 6.3. 状态管理

**A. Collection Model 扩展**

- **文件位置**: `frontend/src/models/collection.ts`
- **新增状态字段**:
  ```typescript
  interface CollectionState {
    // 现有字段...

    // 新增字段
    marketplaceCollections: SharedCollection[];
    subscribedCollections: SharedCollection[];
    marketplaceLoading: boolean;
    subscribedLoading: boolean;
    marketplacePagination: {
      current: number;
      pageSize: number;
      total: number;
    };
    subscribedPagination: {
      current: number;
      pageSize: number;
      total: number;
    };
  }
  ```
- **新增 Effects**:
  - `fetchMarketplaceCollections`: 获取市场 Collection 列表
  - `fetchSubscribedCollections`: 获取用户订阅的 Collection 列表
  - `subscribeCollection`: 订阅 Collection
  - `unsubscribeCollection`: 取消订阅 Collection
  - `publishCollection`: 发布 Collection 到市场
  - `unpublishCollection`: 从市场下架 Collection
  - `fetchSharingStatus`: 获取 Collection 分享状态

**B. 全局状态更新**

- **文件位置**: `frontend/src/models/global.ts`
- **导航菜单**: 新增 "知识库市场" 菜单项，链接到 `/marketplace`

#### 6.4. UI 交互逻辑

**A. 只读模式下的 UI 限制**

需要在以下组件中根据响应类型（`SharedCollection`）禁用或隐藏相关功能：

```typescript
// 类型判断逻辑
const isReadOnly = isSharedCollection(collectionData);

// 在组件中使用
if (isReadOnly) {
  // 隐藏编辑功能
} else {
  // 显示完整功能
}
```

- **文档管理页面**:
  - 隐藏 "上传文档" 按钮
  - 隐藏文档操作菜单（编辑、删除）
  - 禁用批量操作功能
- **Collection 设置页面**:
  - 完全隐藏设置页面入口
  - 或显示设置但所有表单字段设为只读
- **知识图谱页面**:
  - 保持正常显示，图谱本身就是只读的
- **聊天页面**:
  - 保持正常功能，允许查询和对话

**B. 分享操作的用户体验**

- **发布确认**:
  - 弹出确认对话框
  - 说明发布后其他用户可以访问这个知识库
  - 提供 "发布" 和 "取消" 按钮
- **取消发布确认**:
  - 弹出确认对话框
  - 说明取消发布后其他用户将无法访问
  - 提供 "确认下架" 和 "取消" 按钮
- **操作反馈**:
  - 操作成功后显示成功提示
  - 操作失败后显示错误信息
  - 操作进行中显示加载状态

**C. 前端代码示例（简化设计的使用）**

```typescript
// SharingControl 组件使用示例
interface SharingControlProps {
  collection: Collection;
  onToggle: (published: boolean) => Promise<void>;
}

const SharingControl: React.FC<SharingControlProps> = ({ collection, onToggle }) => {
  return (
    <div className="sharing-control">
      <Switch
        checked={collection.is_published}
        onChange={onToggle}
        loading={loading}
      />
      <span className="sharing-status">
        {collection.is_published ? '已发布到市场' : '未发布'}
      </span>
      {collection.is_published && collection.published_at && (
        <Text type="secondary" className="publish-time">
          发布于 {formatRelativeTime(collection.published_at)}
        </Text>
      )}
    </div>
  );
};

// 在Collection详情页面中的使用
const CollectionDetail: React.FC = () => {
  const handleTogglePublish = async (shouldPublish: boolean) => {
    if (shouldPublish) {
      await api.publishCollection(collection.id);
    } else {
      await api.unpublishCollection(collection.id);
    }
    // 刷新Collection数据
    refreshCollection();
  };

  return (
    <div>
      {/* 只有所有者才显示分享控制 */}
      {isOwner && (
        <SharingControl
          collection={collection}
          onToggle={handleTogglePublish}
        />
      )}
    </div>
  );
};
```

### 7. 详细实施计划 (TODO List)

#### **Phase 1: 后端 - 数据库与核心服务**

- [x] **1.1. 数据库模型与迁移**
  - [x] 在 `aperag/db/models.py` 中定义数据库模型：
    - `CollectionMarketplace` (SQLAlchemy模型)：分享状态记录，包含状态和时间字段
    - `UserCollectionSubscription` (SQLAlchemy模型)：用户订阅记录，关联collection_marketplace_id，使用 `gmt_deleted` 字段实现软删除
    - `CollectionMarketplaceStatusEnum` (Python enum)：分享状态枚举，用于代码逻辑，数据库使用VARCHAR存储
    - 包含所有必要字段、约束和索引（特别注意 `gmt_deleted` 的索引优化）
    - 注意：`status` 字段使用 `Column(String(20))` 而非 `EnumColumn`，确保数据库层使用VARCHAR
    - 重点：UserCollectionSubscription表关联collection_marketplace_id而不是collection_id
  - [x] 运行 `make makemigration` 生成新的数据库迁移脚本
  - [x] 检查生成的迁移脚本（位于 `aperag/migration/versions/`）确保 SQL 语法正确性和索引创建
  - [x] 运行 `make migrate` 将数据库 schema 变更应用到开发环境
  - [x] 验证新表创建成功，检查约束和索引是否正确建立

- [x] **1.2. OpenAPI Schema 定义**
  - [x] 创建 `aperag/api/components/schemas/marketplace.yaml`，定义以下视图模型：
    - `CollectionMarketplaceStatusEnum`
    - `SharedCollection` (共享Collection模型，用于市场浏览和订阅访问)
    - `SharedCollectionList` (共享Collection列表响应模型)
    - `SharingStatusResponse` (简洁的分享状态响应模型，包含is_published和published_at字段)
  - [x] 创建 `aperag/api/paths/marketplace.yaml`，定义以下端点的完整规范：
    - `GET /api/v1/marketplace/collections`：获取市场Collection列表
    - `GET /api/v1/marketplace/collections/subscriptions`：获取当前用户订阅的Collection列表
    - `POST /api/v1/marketplace/collections/{collection_id}/subscribe`：订阅Collection
    - `DELETE /api/v1/marketplace/collections/{collection_id}/subscribe`：取消订阅Collection
  - [x] 修改 `aperag/api/paths/collections.yaml`，添加 sharing 相关端点：
    - `GET /api/v1/collections/{collection_id}/sharing`
    - `POST /api/v1/collections/{collection_id}/sharing`
    - `DELETE /api/v1/collections/{collection_id}/sharing`

  - [x] 修改 `aperag/api/components/schemas/collection.yaml`，在 Collection schema 中添加 `is_published` 和 `published_at` 字段
  - [x] 运行 `make generate-models` 生成更新后的 `aperag/schema/view_models.py`
  - [x] 验证生成的 Pydantic 模型类型注解正确

- [x] **1.3. 服务层 - Marketplace Service**
  - [x] 创建 `aperag/service/marketplace_service.py` 文件和 MarketplaceService 类
  - [x] 实现 `publish_collection(user_id: str, collection_id: str)` 方法：
    - 验证用户是 Collection 所有者
    - 创建或更新 collection_marketplace 记录为 PUBLISHED 状态，手动设置 `gmt_updated = datetime.utcnow()`
    - 处理重复发布的情况（如果已经是 PUBLISHED 状态，应返回成功但不执行任何操作）
  - [x] 实现 `unpublish_collection(user_id: str, collection_id: str)` 方法：
    - 验证用户是 Collection 所有者
    - 将 collection_marketplace 记录状态改为 'DRAFT'（设置 `status='DRAFT', gmt_updated=datetime.utcnow()`）
    - 应用层查询并软删除相关订阅记录（设置 `gmt_deleted`）
    - 使用数据库事务确保数据一致性
  - [x] 实现 `get_sharing_status(collection_id: str)` 方法：
    - 返回指定 Collection 的分享状态信息
  - [x] 实现 `get_raw_sharing_status(collection_id: str)` 内部方法：
    - 供权限检查函数调用，不进行额外的权限验证
  - [x] 实现 `list_published_collections(user_id: str, page: int, page_size: int)` 方法：
    - 查询所有 PUBLISHED 状态的 Collection
    - 支持分页功能
    - 关联查询获取 Collection 基本信息和所有者用户名
    - 计算当前用户的订阅状态（通过subscription_id字段）
  - [x] 实现订阅相关方法：
    - `subscribe_collection(user_id: str, collection_id: str)` 方法：
      - 查找Collection对应的已发布marketplace记录 (status = 'PUBLISHED', gmt_deleted IS NULL)
      - 验证用户不是 Collection 所有者，如果是则抛出 SelfSubscriptionError
      - 检查是否已订阅该marketplace实例，防止重复订阅
      - 创建用户订阅记录（关联collection_marketplace_id）
    - `unsubscribe_collection(user_id: str, collection_id: str)` 方法：
      - 验证用户已订阅该 Collection
      - 软删除订阅记录（设置 gmt_deleted = current_timestamp）
    - `get_user_subscription(user_id: str, collection_id: str)` 方法：
      - 通过collection_id查找已发布的marketplace记录，再查找对应的订阅记录
      - 获取用户对指定 Collection 的活跃订阅状态（`WHERE gmt_deleted IS NULL`）
      - 供权限检查函数调用，返回 None 表示未订阅或已取消订阅
    - `list_user_subscribed_collections(user_id: str, page: int, page_size: int)` 方法：
      - 查询用户所有活跃订阅的 Collection（`WHERE gmt_deleted IS NULL`）
      - 关联查询获取 Collection 详细信息和原所有者信息
      - 返回包含订阅信息的 Collection 列表
      - 支持分页功能

- [x] **1.4. 服务层 - MarketplaceCollection Service**
  - [x] 创建 `aperag/service/marketplace_collection_service.py` 文件和 MarketplaceCollectionService 类
  - [x] 实现 `_check_subscription_access(user_id: str, collection_id: str)` 方法：
    - 验证 Collection 是否存在且已发布（status = 'PUBLISHED'）
    - 验证用户是否已订阅且订阅有效（gmt_deleted IS NULL）
    - 返回有效的订阅记录或抛出相应的 HTTPException
  - [x] 实现 MarketplaceCollection 专用业务方法：
    - `get_marketplace_collection(user_id: str, collection_id: str)` 方法：
      - 调用 `_check_subscription_access` 验证权限
      - 返回 SharedCollection 数据（只包含订阅者需要的字段）
    - `list_marketplace_collection_documents(user_id: str, collection_id: str, page: int, page_size: int)` 方法：
      - 调用 `_check_subscription_access` 验证权限
      - 返回文档列表（隐藏内部信息如创建者、编辑历史等）
    - `get_marketplace_collection_document_preview(user_id: str, collection_id: str, document_id: str)` 方法：
      - 调用 `_check_subscription_access` 验证权限
      - 返回文档预览数据
    - `get_marketplace_collection_graph(user_id: str, collection_id: str, **params)` 方法：
      - 调用 `_check_subscription_access` 验证权限
      - 返回知识图谱数据（只读模式）

#### **Phase 2: 后端 - API 视图与前端集成**

- [x] **2.1. API 视图层实现**
  - [x] 创建 `aperag/views/marketplace.py` 文件：
    - 实现 `list_marketplace_collections_view` 函数
    - 处理分页参数验证和默认值设置
    - 调用 `marketplace_service.list_published_collections`
    - 返回标准化的分页响应格式
  - [x] 修改 `aperag/views/collections.py`（或相关视图文件）实现 sharing 相关端点：
    - `get_collection_sharing_status_view`: 获取分享状态（仅所有者）
    - `publish_collection_view`: 发布 Collection 到市场
    - `unpublish_collection_view`: 从市场下架 Collection
    - 为每个端点添加用户身份验证、所有权验证和异常错误处理
  - [x] 在 `aperag/app.py` 中注册新的路由：
    - 添加 `marketplace` 路由组，tag 设为 "marketplace"
    - 添加 `marketplace-collections` 路由组，tag 设为 "marketplace-collections"
    - 集成到主应用的路由配置中
  - [x] 创建 `aperag/views/marketplace_collections.py` 文件：
    - 实现 `get_marketplace_collection_view`: 获取MarketplaceCollection详情
    - 实现 `list_marketplace_collection_documents_view`: 获取文档列表
    - 实现 `get_marketplace_collection_document_preview_view`: 文档预览
    - 实现 `get_marketplace_collection_graph_view`: 知识图谱
    - 为每个端点添加订阅权限验证和异常错误处理

- [x] **2.2. 前端 - 生成 SDK 与状态管理**
  - [x] 运行 `make generate-frontend-sdk` 更新前端 API client
  - [x] 验证 `frontend/src/api/` 目录中的新增内容：
    - 检查 `apis/` 目录下是否生成了 marketplace 相关的 API 函数
    - 检查 `models/` 目录下是否生成了新的 TypeScript 接口
    - 验证现有 Collection 接口是否正确更新
  - [ ] 更新前端类型定义：
    - 修改 `frontend/src/models/collection.ts` 中的 Collection 接口
    - 在 `frontend/src/types/` 中添加或更新相关类型定义
    - 确保 `SharedCollection` 类型正确
    - 实现类型守卫函数 `isSharedCollection`

#### **Phase 3: 前端 - UI 实现**

- [x] **3.1. Marketplace 页面开发**
  - [x] 创建页面文件 `frontend/src/pages/marketplace/index.tsx`：
    - 实现基础页面结构和布局
    - 添加页面标题 "知识库市场" 和功能说明
    - 集成分页组件和加载状态管理
  - [x] 实现 API 数据获取逻辑：
    - 在页面加载时调用 marketplace API
    - 处理分页参数和状态更新
    - 实现错误状态处理和重试机制
  - [x] 创建 `CollectionMarketplaceCard` 组件（`frontend/src/components/CollectionMarketplaceCard.tsx`）：
    - 设计卡片布局（标题、描述、所有者、发布时间）
    - 实现悬浮效果和点击交互
    - 处理描述文本截断（最多 150 字符）
    - 添加相对时间格式化功能
    - 使用 `SharedCollection` 类型作为 Props
    - 实现订阅状态显示逻辑：
      - 根据 `subscription_id` 字段是否有值判断订阅状态
      - 如果当前用户是Collection所有者，显示 "我的" 标签
      - 如果当前用户非所有者且未订阅（`subscription_id` 为空），显示 "订阅" 按钮
      - 如果当前用户已订阅（`subscription_id` 有值），显示 "已订阅" 状态
  - [x] 实现网格布局和响应式设计：
    - 桌面端：4 列网格布局
    - 平板端：2-3 列网格布局
    - 手机端：1 列布局
  - [x] 添加到导航菜单：
    - 在 `frontend/src/layouts/sidebar.tsx` 中添加 "知识库市场" 菜单项
    - 设置市场图标（如ShopOutlined）和路由链接

- [ ] **3.2. MarketplaceCollection 专用页面开发**
  - [ ] 创建 `frontend/src/pages/marketplace/collections/$collectionId/index.tsx`：
    - 实现MarketplaceCollection详情页面基础结构
    - 使用专用的marketplace/collections API接口
    - 显示只读模式Banner和取消订阅功能
  - [ ] 创建 `ReadOnlyBanner` 组件（`frontend/src/components/ReadOnlyBanner.tsx`）：
    - 使用 Ant Design Alert 组件
    - 设计醒目的提示样式（蓝色信息提示）
    - 添加信息图标（InfoCircleOutlined）和提示文案
    - 接收 `ownerUsername` 作为 Props
    - 集成"取消订阅"按钮功能
  - [ ] 实现MarketplaceCollection页面功能：
    - 文档列表展示（只读模式）
    - 文档预览功能
    - 知识图谱查看（只读模式）
    - 聊天Bot功能（如果支持）
  - [ ] 创建共享组件用于复用：
    - `DocumentListReadOnly`: 只读文档列表组件
    - `GraphViewReadOnly`: 只读图谱查看组件
    - `CollectionInfoReadOnly`: 只读基本信息展示

- [x] **3.3. Collection 详情页 - 分享功能实现（所有者专用）**
  - [x] 创建 `SharingControl` 组件（`frontend/src/components/SharingControl.tsx`）：
    - 使用 Switch 组件控制发布状态（基于 `collection.is_published`）
    - 显示当前分享状态标签和发布时间（使用 `collection.published_at`）
    - 仅在Collection详情页面（`/collections/{id}`）中显示
  - [x] 实现分享操作确认对话框：
    - 发布确认：说明发布后其他用户可以访问
    - 下架确认：说明下架后其他用户将无法访问
    - 使用 Ant Design Modal 组件
  - [x] 集成分享状态管理：
    - 在 Collection model 中添加相关 Effects
    - 实现发布/取消发布的 API 调用（调用 `/api/v1/collections/{id}/sharing`）
    - 处理操作成功/失败的反馈提示
  - [x] 在 Collection 详情页面中集成 SharingControl：
    - 在Collection标题右侧区域展示分享控制组件
    - 确保只有在Collection接口（所有者模式）下才显示
    - 实现状态变更后的页面刷新

**注意**：MarketplaceCollection详情页面（`/marketplace/collections/{id}`）不需要SharingControl组件，它们有专门的ReadOnlyBanner和取消订阅功能。
