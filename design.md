## 设计文档：Collection 分享与市场 (MVP)

**版本:** 1.6
**关联 Issue:** [#1127](https://github.com/apecloud/ApeRAG/issues/1127)

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
    
    -- 分享状态枚举: DRAFT, PUBLISHED
    status VARCHAR(20) NOT NULL DEFAULT 'DRAFT' CHECK (status IN ('DRAFT', 'PUBLISHED')),
    
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
CREATE INDEX idx_collection_marketplace_status ON collection_marketplace(status) 
    WHERE gmt_deleted IS NULL;
CREATE INDEX idx_collection_marketplace_published ON collection_marketplace(gmt_created) 
    WHERE status = 'PUBLISHED' AND gmt_deleted IS NULL;
-- 查询市场列表时的复合索引
CREATE INDEX idx_collection_marketplace_list ON collection_marketplace(status, gmt_created DESC) 
    WHERE gmt_deleted IS NULL;
-- Collection关联查询索引
CREATE INDEX idx_collection_marketplace_collection_id ON collection_marketplace(collection_id) 
    WHERE gmt_deleted IS NULL;
```

**表2: `user_collection_subscription` - 用户订阅表**

用于记录用户对已发布 Collection 的订阅关系，采用 Subscribe 模式。

```sql
CREATE TABLE user_collection_subscription (
    id VARCHAR(24) PRIMARY KEY DEFAULT ('sub_' || substr(md5(random()::text), 1, 16)),
    user_id VARCHAR(24) NOT NULL,        -- 关联users表，应用层维护关联关系
    collection_id VARCHAR(24) NOT NULL,  -- 关联collections表，应用层维护关联关系
    
    -- 时间戳字段
    gmt_subscribed TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    gmt_deleted TIMESTAMP WITH TIME ZONE NULL,  -- 软删除：NULL表示活跃订阅
    
    -- 注意：活跃订阅的唯一性通过部分唯一索引实现，而非表级约束
    -- 级联删除逻辑需要在应用代码中处理，删除Collection时同时删除相关订阅记录
);

-- 索引优化
CREATE UNIQUE INDEX idx_user_collection_active_unique ON user_collection_subscription(user_id, collection_id) 
    WHERE gmt_deleted IS NULL;  -- 部分唯一索引：确保活跃订阅唯一性
CREATE INDEX idx_user_subscription_collection ON user_collection_subscription(collection_id) 
    WHERE gmt_deleted IS NULL;
CREATE INDEX idx_user_subscription_user ON user_collection_subscription(user_id) 
    WHERE gmt_deleted IS NULL;
CREATE INDEX idx_user_subscription_deleted ON user_collection_subscription(gmt_deleted) 
    WHERE gmt_deleted IS NOT NULL;
```

#### 2.2. 数据库约束说明

**业务约束:**
1. **唯一性约束**: 每个 Collection 只能有一条分享记录
2. **订阅约束**: 一个用户对同一 Collection 只能有一个活跃订阅
3. **所有权约束**: 用户无法订阅自己是所有者的 Collection（业务逻辑禁止）
4. **应用层级联**: Collection 删除时，需要在代码中同时软删除相关的分享和订阅记录
5. **状态检查**: 分享状态只能是 'DRAFT' 或 'PUBLISHED'

**性能优化:**
1. **部分索引**: 只为活跃记录（`gmt_deleted IS NULL`）创建索引，大幅减少索引空间
2. **复合索引**: 
   - 用户+Collection 组合查询优化（订阅检查场景）
   - 状态+时间复合索引（市场列表查询场景）
   - Collection关联查询索引（按collection_id查询优化）
3. **应用层更新**: `gmt_updated` 字段在代码中手动更新，保持项目一致性
4. **数据规范化**: 遵循项目惯例，不使用外键约束，通过应用层维护数据一致性

#### 2.3. 数据生命周期

**分享生命周期:**
- **创建**: 用户首次发布 Collection 时创建记录，状态为 'PUBLISHED'
- **取消发布**: 状态改为 'DRAFT'，需要在代码中批量失效所有相关订阅（设置 `gmt_deleted`）
- **重新发布**: 状态改回 'PUBLISHED'，用户需要重新订阅
- **删除处理**: Collection 删除时，需要在代码中同时软删除 collection_marketplace 记录

**订阅生命周期:**
- **订阅**: 用户订阅已发布的 Collection，创建订阅记录
- **取消订阅**: 设置 `gmt_deleted = NOW()`，保留历史记录
- **自动失效**: Collection 取消发布时，在代码中批量设置相关订阅的 `gmt_deleted`
- **级联删除**: Collection 删除时，在代码中批量软删除相关订阅记录

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
│  - 发布/取消发布              │ - 权限检查 (_check_read/write)   │
│  - 订阅/取消订阅              │ - Collection CRUD操作           │
│  - 用户订阅列表               │                                 │
│  - 市场Collection列表         │                                 │
│  - 分享状态查询               │                                 │
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
           │     └─ 点击进入: 路由到 /collections/{collection_id}
           │

           └─ 3b. Collection详情页只读访问
                 │
                 ├─ 页面: /collections/{collection_id} (同自有Collection)
                 ├─ API: GET /api/v1/collections/{collection_id}
                 ├─ 权限检查: _check_read_access() 验证订阅状态
                 ├─ 响应字段: is_readonly_view=true, access_type="subscribed"
                 ├─ UI显示: 顶部显示只读Banner
                 ├─ 功能权限: 可查看文档、图谱、搜索，可使用聊天Bot
                 └─ 操作限制: 隐藏所有编辑、删除、上传按钮
```

**流程3: 权限检查流程**
```
用户请求访问Collection
    │
    ├─ _check_read_access()
    │     │
    │     ├─ 检查用户是否为Collection所有者
    │     │   └─ 是 → 完全访问权限 ✅
    │     │
    │     ├─ 检查Collection是否已发布
    │     │   └─ 否 → 403 Forbidden ❌
    │     │
    │     ├─ 检查用户是否已订阅 (gmt_deleted IS NULL)
    │     │   └─ 否 → 403 "请先订阅" ❌
    │     │
    │     └─ 是 → 只读访问权限 ✅
    │
    └─ _check_write_access()
           │
           ├─ 检查用户是否为Collection所有者
           │   └─ 是 → 写权限 ✅
           │
           └─ 否 → 403 "只读共享Collection" ❌
```

**流程4: 用户取消订阅流程**
```
用户B (已订阅用户)
    │
    ├─ 1. 在Collection详情页点击"取消订阅"
    │     │
    │     ├─ 页面: /collections/{collection_id}
    │     ├─ UI元素: 详情页面显示"取消订阅"按钮（因为 is_readonly_view=true）
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
    │     ├─ 权限检查: _check_read_access() 立即返回403
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
    │     ├─ 更新 collection_marketplace 状态为 'DRAFT'
    │     ├─ 批量失效所有相关订阅
    │     │   └─ UPDATE user_collection_subscription 
    │     │       SET gmt_deleted = current_timestamp 
    │     │       WHERE collection_id = ? AND gmt_deleted IS NULL
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
    │     ├─ 权限检查: _check_read_access() 对所有非所有者返回403
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
1. **索引策略**: 为高频查询场景创建专门索引
2. **分页查询**: 所有列表接口支持分页，避免大数据量查询
3. **部分索引**: 只为活跃记录创建索引，节省存储空间


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

-- 高效的订阅检查查询（利用唯一索引 idx_user_collection_active_unique）
SELECT id FROM user_collection_subscription 
WHERE user_id = ? AND collection_id = ? AND gmt_deleted IS NULL
LIMIT 1;

-- 获取用户订阅的Collection详情（通过collection_id关联，无需冗余外键）
SELECT c.id, c.title, c.description, u.username, ucs.gmt_subscribed
FROM user_collection_subscription ucs
JOIN collections c ON ucs.collection_id = c.id
JOIN users u ON c.user_id = u.id
WHERE ucs.user_id = ? AND ucs.gmt_deleted IS NULL
ORDER BY ucs.gmt_subscribed DESC;
```

### 4. 后端设计

遵循**软件架构分层原则**，按照从底层到高层的顺序进行设计：数据模型 → 服务层 → API层。

#### 4.1. 数据模型设计 (OpenAPI / `view_models.py`)

**4.1.1 新增数据库模型:**

- **`CollectionMarketplaceStatusEnum`**: 分享状态枚举
    - `DRAFT`: 未发布状态，仅所有者可见
    - `PUBLISHED`: 已发布状态，公开可见

- **`CollectionMarketplaceItem`**: Collection 分享状态记录（数据库模型）
    - `id: str`: 分享记录的唯一标识符
    - `collection_id: str`: 关联的 Collection ID
    - `status: CollectionMarketplaceStatusEnum`: 当前分享状态
    - `gmt_created: datetime`: 分享记录创建时间
    - `gmt_updated: datetime`: 分享记录最后更新时间
    - `gmt_deleted: Optional[datetime]`: 软删除时间（NULL表示活跃记录）

- **`UserCollectionSubscription`**: 用户订阅 Collection 记录（数据库模型）
    - `id: str`: 订阅记录的唯一标识符
    - `user_id: str`: 订阅用户 ID
    - `collection_id: str`: 被订阅的 Collection ID
    - `gmt_subscribed: datetime`: 订阅时间
    - `gmt_deleted: Optional[datetime]`: 取消订阅时间（NULL表示活跃订阅）

**4.1.2 新增视图模型:**

- **`CollectionMarketplaceDetail`**: 市场页面展示的 Collection 信息（视图模型）
    - `collection_id: str`: Collection ID
    - `title: str`: Collection 标题
    - `description: str`: Collection 描述
    - `owner_username: str`: 所有者用户名
    - `gmt_published: datetime`: 首次发布时间（对应数据库中的 gmt_created 字段）
    - `is_subscribed: bool`: 当前用户是否已订阅（非数据库字段，在服务层计算）

- **`CollectionMarketplaceDetailList`**: 市场 Collection 列表响应
    - `items: List[CollectionMarketplaceDetail]`: Collection 列表
    - `total: int`: 总数量（用于分页）
    - `page: int`: 当前页码
    - `page_size: int`: 每页大小

- **`UserSubscription`**: 用户订阅信息（视图模型）
    - `subscription_id: str`: 订阅记录 ID
    - `collection_id: str`: Collection ID
    - `collection_title: str`: Collection 标题
    - `collection_description: str`: Collection 描述
    - `owner_username: str`: 原所有者用户名
    - `gmt_subscribed: datetime`: 订阅时间

- **`UserSubscriptionList`**: 用户订阅列表响应 (专门的订阅Collection API响应)
    - `items: List[UserSubscription]`: 订阅列表
    - `total: int`: 总数量

**4.1.3 修改现有模型:**

- **`Collection`**: 扩展现有 Collection model
    - `sharing_info: Optional[CollectionMarketplaceItem]`: 分享信息，仅在所有者查看时返回
    - `is_readonly_view: bool`: 是否为只读视图，非数据库字段，在服务层计算
        - 计算逻辑：当前用户不是所有者且通过订阅访问时为 `true`
        - 用于前端判断是否显示只读模式 UI
    - `subscription_info: Optional[UserSubscription]`: 订阅信息，仅在通过订阅访问时返回
    - `access_type: str`: 访问类型，枚举值：`owner`（所有者）、`subscribed`（订阅访问）

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
    
    async def publish_collection(self, user_id: str, collection_id: str) -> CollectionMarketplaceItem:
        """发布Collection到市场"""
        # 验证用户所有权
        # 创建或更新collection_marketplace记录
        # 状态设置为PUBLISHED
        
    async def unpublish_collection(self, user_id: str, collection_id: str) -> None:
        """从市场下架Collection"""
        # 验证用户所有权
        # 更新collection_marketplace状态为DRAFT，同时更新gmt_updated字段
        # 批量失效相关订阅(设置gmt_deleted = datetime.utcnow())
        # 注意：需要使用事务确保数据一致性
        
    async def get_sharing_status(self, collection_id: str) -> Optional[CollectionMarketplaceItem]:
        """获取Collection的分享状态"""
        
    async def get_raw_sharing_status(self, collection_id: str) -> Optional[CollectionMarketplaceItem]:
        """获取原始分享状态（供权限检查使用）"""
        
    async def list_published_collections(self, user_id: str, page: int, page_size: int) -> CollectionMarketplaceDetailList:
        """列出市场中所有已发布的Collection"""
        # 查询PUBLISHED状态的Collection
        # 计算当前用户的订阅状态
        # 支持分页
        
    async def subscribe_collection(self, user_id: str, collection_id: str) -> UserSubscription:
        """订阅Collection"""
        # 1. 验证Collection已发布 (status = 'PUBLISHED')
        # 2. 验证用户不是Collection所有者 (user_id != collection.user)
        # 3. 检查是否已订阅，防止重复订阅
        # 4. 创建user_collection_subscription记录
        # 异常: 如果用户是所有者，抛出 SelfSubscriptionError("Cannot subscribe to your own collection")
        
    async def unsubscribe_collection(self, user_id: str, collection_id: str) -> None:
        """取消订阅Collection"""
        # 验证用户已订阅该Collection
        # 软删除订阅记录(设置gmt_deleted)
        
    async def get_user_subscription(self, user_id: str, collection_id: str) -> Optional[UserCollectionSubscription]:
        """获取用户对指定Collection的活跃订阅状态"""
        # 供权限检查函数调用
        # 返回None表示未订阅或已取消订阅
        
    async def list_user_subscribed_collections(self, user_id: str, page: int, page_size: int) -> UserSubscriptionList:
        """获取用户所有活跃订阅的Collection"""
        # 查询WHERE gmt_deleted IS NULL
        # 关联查询获取Collection详细信息和原所有者信息
        # 支持分页
```

**4.2.2 修改现有服务: `aperag/service/collection_service.py`**

核心变更是在所有Collection相关操作的入口处增加**权限检查**：

        ```python
class CollectionService:
    
    async def _check_read_access(self, user_id: str, collection_id: str) -> db_models.Collection:
        """
        检查用户是否有权限读取指定的Collection
        
        权限规则（Subscribe模式）：
        1. Collection所有者有完全读权限
        2. 非所有者必须订阅已发布的Collection才能读取
        3. 未订阅的用户无法访问任何非自有的Collection
        """
        collection = await self.db_ops.query_collection_by_id(collection_id)
            if not collection:
                raise HTTPException(status_code=404, detail="Collection not found")

        # 1. 所有者有完全访问权限
            if collection.user == user_id:
                return collection

        # 2. 非所有者需要检查订阅状态
        from aperag.service.marketplace_service import marketplace_service
        
        # 首先检查Collection是否已发布
            sharing_info = await marketplace_service.get_raw_sharing_status(collection_id)
        is_published = sharing_info and sharing_info.status == CollectionMarketplaceStatusEnum.PUBLISHED

        if not is_published:
            raise HTTPException(status_code=403, detail="Collection not published")
        
        # 检查用户是否已订阅该Collection
        subscription = await marketplace_service.get_user_subscription(user_id, collection_id)
        if not subscription or subscription.gmt_deleted is not None:
            # 区分未订阅和订阅已失效的情况
            if not subscription:
                raise HTTPException(
                    status_code=403, 
                    detail="Access denied. Please subscribe to this collection first."
                )
            else:
                raise HTTPException(
                    status_code=403, 
                    detail="Access denied. Your subscription to this collection has expired."
                )
        
        return collection

    async def _check_write_access(self, user_id: str, collection_id: str) -> db_models.Collection:
        """
        检查用户是否有权限修改指定的Collection
        
        权限规则（Subscribe模式）：
        1. 只有Collection所有者有写权限
        2. 订阅的Collection对订阅者严格只读
        3. 非所有者和非订阅者无任何访问权限
        """
        collection = await self.db_ops.query_collection_by_id(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")

        # 只有所有者才有写权限
        if collection.user == user_id:
                return collection
            
        # 检查是否为共享Collection，提供更具体的错误信息
        from aperag.service.marketplace_service import marketplace_service
        sharing_info = await marketplace_service.get_raw_sharing_status(collection_id)
        is_published = sharing_info and sharing_info.status == CollectionMarketplaceStatusEnum.PUBLISHED

        if is_published:
            raise HTTPException(
                status_code=403, 
                detail="Permission denied. This is a read-only shared collection."
            )
        else:
            raise HTTPException(status_code=403, detail="Permission denied")

    # 使用示例：
    async def get_collection(self, user_id: str, collection_id: str):
        collection = await self._check_read_access(user_id, collection_id)
        # ... 构建响应数据并计算 is_readonly_view 字段
        
    async def update_collection(self, user_id: str, collection_id: str, updates: dict):
        collection = await self._check_write_access(user_id, collection_id)
        # ... 执行更新逻辑
        
        async def delete_collection(self, user_id: str, collection_id: str):
        collection = await self._check_write_access(user_id, collection_id)
        # ... 执行删除逻辑
        # 注意：删除Collection时需要级联软删除相关记录：
        # 1. 软删除collection_marketplace记录 (设置gmt_deleted)
        # 2. 批量软删除user_collection_subscription记录 (设置gmt_deleted)
        # 3. 使用事务确保数据一致性
```

**4.2.3 其他相关服务的权限集成:**

- **`aperag/service/document_service.py`**: 所有涉及Collection内文档的读取操作（如`list_documents`, `get_document`）必须调用`collection_service._check_read_access`
- **`aperag/service/document_service.py`**: 所有涉及Collection内文档的写操作（如`create_document`, `update_document`, `delete_document`）必须调用`collection_service._check_write_access`
- **`aperag/service/graph_service.py`**: 图相关的读取操作（如`get_graph`）必须调用`collection_service._check_read_access`
- **`aperag/service/chat_service.py`**: 聊天查询操作必须调用`collection_service._check_read_access`
- **`aperag/service/bot_service.py`**: Bot相关操作必须检查关联Collection的权限
- **`aperag/service/search_service.py`**: 搜索相关操作必须调用权限检查函数

#### 4.3. API 端点设计 (View 层)

基于服务层的业务逻辑，设计RESTful API端点，遵循统一的URL命名规范和错误处理模式。

**4.3.1 新增 API 端点**

设计采用混合 URL 模式：marketplace 相关的浏览功能使用 `/marketplace` 路径，而具体 Collection 的分享操作作为 Collection 的子资源管理。

- **`GET /api/v1/marketplace/collections`**: 列出市场中所有公开的 Collection
    - **功能**: 返回所有状态为 `PUBLISHED` 的 Collection 列表（包括当前用户自己发布的Collection）
    - **权限**: 任何已登录用户都可以访问
    - **响应**: `CollectionMarketplaceDetailList` 类型，包含每个 Collection 的基本信息、所有者用户名、发布时间
    - **分页**: 支持 `page` 和 `page_size` 参数

- **`POST /api/v1/collections/{collection_id}/sharing`**: 发布一个 Collection 到市场
    - **功能**: 将指定 Collection 的状态设置为 `PUBLISHED`
    - **权限**: 仅限 Collection 所有者
    - **行为**: 在 `collection_marketplace` 表中创建记录或更新状态
    - **响应**: 返回更新后的 `CollectionMarketplaceItem` 信息

- **`DELETE /api/v1/collections/{collection_id}/sharing`**: 从市场下架一个 Collection
    - **功能**: 将指定 Collection 的状态设置为 `DRAFT`（不删除记录，仅改变状态）
    - **权限**: 仅限 Collection 所有者
    - **行为**: 立即停止其他用户对该 Collection 的访问，批量失效所有相关订阅
    - **响应**: 返回 204 No Content

- **`GET /api/v1/collections/{collection_id}/sharing`**: 获取指定 Collection 的分享状态
    - **功能**: 返回 Collection 的当前分享状态和相关信息
    - **权限**: 仅限 Collection 所有者
    - **响应**: `CollectionMarketplaceItem` 类型，包含状态、发布时间

- **`POST /api/v1/marketplace/collections/{collection_id}/subscribe`**: 订阅一个已发布的 Collection
    - **功能**: 将指定的已发布 Collection 添加到用户的订阅列表
    - **权限**: 任何已登录用户（除 Collection 所有者外）
    - **业务限制**: 用户无法订阅自己是所有者的 Collection
    - **行为**: 在 `user_collection_subscription` 表中创建订阅记录
    - **响应**: 返回 `UserSubscription` 信息
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

**4.3.2 修改现有 API 行为**

现有的 Collection 相关端点需要集成新的权限控制逻辑，支持 Subscribe 模式的访问控制。

**核心变更**：
- 用户必须先订阅 Collection 才能访问其内容（除了所有者）
- 订阅关系为用户提供只读访问权限
- 当 Collection 被取消发布时，相关订阅会自动失效（设置 `gmt_deleted`），但保留历史记录

- **只读端点（Read-Only Endpoints）**:
    - **`GET /api/v1/collections/{collection_id}`**: 获取 Collection 详情
        - **权限检查**: 调用 `_check_read_access`
        - **响应变更**: 新增 `is_readonly_view: bool` 字段和 `sharing_info` 字段
        - **`is_readonly_view` 计算逻辑**: 当访问用户不是所有者且通过订阅访问时为 `true`
        - **`access_type` 计算逻辑**: `owner`（所有者）或 `subscribed`（订阅访问）
    - **`GET /api/v1/collections/{collection_id}/documents`**: 获取文档列表
        - **权限检查**: 调用 `_check_read_access`
        - **行为**: 权限通过后正常返回文档列表
    - **`GET /api/v1/collections/{collection_id}/documents/{document_id}`**: 获取文档内容
        - **权限检查**: 调用 `_check_read_access`
        - **行为**: 权限通过后正常返回文档内容
    - **`GET /api/v1/collections/{collection_id}/graph`**: 获取知识图谱
        - **权限检查**: 调用 `_check_read_access`
        - **行为**: 权限通过后正常返回图谱数据
    - **`GET /api/v1/collections/{collection_id}/searches`**: 获取搜索历史
        - **权限检查**: 调用 `_check_read_access`
        - **行为**: 权限通过后正常返回搜索历史
    - **`POST /api/v1/collections/{collection_id}/searches`**: 执行搜索查询
        - **权限检查**: 调用 `_check_read_access`
        - **行为**: 权限通过后正常执行搜索并返回结果
    - **`GET /api/v1/collections/{collection_id}/documents/{document_id}/preview`**: 预览文档
        - **权限检查**: 调用 `_check_read_access`
        - **行为**: 权限通过后正常返回文档预览
    - **`GET /api/v1/collections/{collection_id}/documents/{document_id}/object`**: 获取文档对象
        - **权限检查**: 调用 `_check_read_access`
        - **行为**: 权限通过后正常返回文档对象数据
    - **`GET /api/v1/collections/{collection_id}/graphs/labels`**: 获取知识图谱标签
        - **权限检查**: 调用 `_check_read_access`
        - **行为**: 权限通过后正常返回图谱标签列表
    - **`GET /api/v1/collections/{collection_id}/graphs/merge-suggestions`**: 获取图谱合并建议
        - **权限检查**: 调用 `_check_read_access`
        - **行为**: 权限通过后正常返回合并建议

- **写操作端点（Write Endpoints）**:
    - **`POST /api/v1/collections`**: 创建 Collection
        - **权限**: 仅对已登录用户开放，不受分享机制影响
    - **`PUT /api/v1/collections/{collection_id}`**: 更新 Collection
        - **权限检查**: 调用 `_check_write_access`
        - **行为**: 非所有者访问将返回 `403 Forbidden`
    - **`DELETE /api/v1/collections/{collection_id}`**: 删除 Collection
        - **权限检查**: 调用 `_check_write_access`
        - **行为**: 非所有者访问将返回 `403 Forbidden`
        - **应用层级联删除**: 
            - 软删除 `collection_marketplace` 记录（设置 `gmt_deleted`）
            - 批量软删除所有相关的 `user_collection_subscription` 记录
            - 使用数据库事务确保操作原子性
        - **注意**: 订阅用户将立即失去对该Collection的访问权限
    - **`POST /api/v1/collections/{collection_id}/documents`**: 创建文档
        - **权限检查**: 调用 `_check_write_access`
        - **行为**: 非所有者访问将返回 `403 Forbidden`，错误信息明确说明这是只读共享 Collection
    - **`PUT /api/v1/collections/{collection_id}/documents/{document_id}`**: 更新文档
        - **权限检查**: 调用 `_check_write_access`
        - **行为**: 非所有者访问将返回 `403 Forbidden`
    - **`DELETE /api/v1/collections/{collection_id}/documents/{document_id}`**: 删除文档
        - **权限检查**: 调用 `_check_write_access`
        - **行为**: 非所有者访问将返回 `403 Forbidden`
    - **`POST /api/v1/collections/{collection_id}/summary/generate`**: 生成 Collection 摘要
        - **权限检查**: 调用 `_check_write_access`
        - **行为**: 非所有者访问将返回 `403 Forbidden`
    - **`POST /api/v1/collections/{collection_id}/documents/{document_id}/rebuild_indexes`**: 重建文档索引
        - **权限检查**: 调用 `_check_write_access`
        - **行为**: 非所有者访问将返回 `403 Forbidden`
    - **`DELETE /api/v1/collections/{collection_id}/searches/{search_id}`**: 删除搜索记录
        - **权限检查**: 调用 `_check_write_access`
        - **行为**: 非所有者访问将返回 `403 Forbidden`
    - **`POST /api/v1/collections/{collection_id}/graphs/nodes/merge`**: 合并知识图谱节点
        - **权限检查**: 调用 `_check_write_access`
        - **行为**: 非所有者访问将返回 `403 Forbidden`
    - **`POST /api/v1/collections/{collection_id}/graphs/merge-suggestions/{suggestion_id}/action`**: 执行图谱合并操作
        - **权限检查**: 调用 `_check_write_access`
        - **行为**: 非所有者访问将返回 `403 Forbidden`

**4.3.3 Bot 和 Chat 相关端点权限控制**

由于 Bot 通常与特定的 Collection 关联，需要在 Bot 相关操作中检查关联 Collection 的权限：

**⚠️ 重要边界情况**: 如果 Bot 关联的 Collection 被删除（`gmt_deleted` 不为 NULL），所有 Bot 相关操作应返回 `404 Not Found` 或 `403 Forbidden`，并提供明确的错误信息。

- **Bot 管理端点**:
    - **`GET /api/v1/bots/{bot_id}`**: 获取 Bot 详情
        - **权限检查**: 检查 Bot 关联的 Collection 读权限
        - **行为**: 如果 Bot 关联到共享 Collection，非所有者可以查看（用于聊天）
    - **`PUT /api/v1/bots/{bot_id}`**: 更新 Bot
        - **权限检查**: 检查 Bot 关联的 Collection 写权限
        - **行为**: 非所有者无法修改关联到共享 Collection 的 Bot
    - **`DELETE /api/v1/bots/{bot_id}`**: 删除 Bot
        - **权限检查**: 检查 Bot 关联的 Collection 写权限
        - **行为**: 非所有者无法删除关联到共享 Collection 的 Bot

- **Chat 相关端点**:
    - **`GET /api/v1/bots/{bot_id}/chats`**: 获取聊天列表
        - **权限检查**: 检查 Bot 关联的 Collection 读权限
        - **行为**: 非所有者可以查看自己与共享 Collection Bot 的聊天记录
    - **`POST /api/v1/bots/{bot_id}/chats`**: 创建新聊天
        - **权限检查**: 检查 Bot 关联的 Collection 读权限
        - **行为**: 非所有者可以与共享 Collection 的 Bot 创建聊天
    - **`GET /api/v1/bots/{bot_id}/chats/{chat_id}`**: 获取聊天详情
        - **权限检查**: 检查 Bot 关联的 Collection 读权限 + 聊天所有权
        - **行为**: 用户只能查看自己的聊天记录
    - **`PUT /api/v1/bots/{bot_id}/chats/{chat_id}`**: 更新聊天
        - **权限检查**: 检查聊天所有权（不需要 Collection 写权限）
        - **行为**: 用户只能修改自己的聊天记录
    - **`DELETE /api/v1/bots/{bot_id}/chats/{chat_id}`**: 删除聊天
        - **权限检查**: 检查聊天所有权（不需要 Collection 写权限）
        - **行为**: 用户只能删除自己的聊天记录


### 5. 错误处理与测试策略

#### 5.1. 错误处理策略

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

#### 5.2. 测试策略

**单元测试覆盖 (使用 pytest):**

```python
# tests/unit_test/test_marketplace_service.py
import pytest
from aperag.service.marketplace_service import MarketplaceService
from aperag.service.marketplace_service import AlreadySubscribedError

class TestMarketplaceService:
    
    @pytest.fixture
    def service(self, mock_db_session):
        return MarketplaceService(mock_db_session)
    
    async def test_subscribe_collection_success(self, service, mock_user, mock_collection):
        # 测试正常订阅流程
        result = await service.subscribe_collection(mock_user.id, mock_collection.id)
        assert result.user_id == mock_user.id
        assert result.collection_id == mock_collection.id
    
    async def test_subscribe_already_subscribed(self, service, mock_user, mock_collection):
        # 测试重复订阅错误
        await service.subscribe_collection(mock_user.id, mock_collection.id)
        
        with pytest.raises(AlreadySubscribedError):
            await service.subscribe_collection(mock_user.id, mock_collection.id)
    
    async def test_subscribe_own_collection_error(self, service, mock_user):
        # 测试用户无法订阅自己的Collection
        own_collection = create_mock_collection(owner_id=mock_user.id)
        
        with pytest.raises(SelfSubscriptionError):
            await service.subscribe_collection(mock_user.id, own_collection.id)
    
    async def test_permission_check_owner_vs_subscriber(self, service):
        # 测试权限检查逻辑
        # ... 详细的权限测试用例
```

**集成测试 (E2E 测试):**

```python
# tests/e2e_test/test_marketplace_integration.py
import pytest
from httpx import AsyncClient

class TestMarketplaceIntegration:
    
    async def test_complete_subscription_workflow(self, async_client: AsyncClient):
        """完整的订阅工作流程测试"""
        
        # 1. 用户A创建Collection
        collection_data = {"title": "Test Collection", "description": "Test"}
        response = await async_client.post("/collections", json=collection_data)
        collection = response.json()
        
        # 2. 用户A发布Collection
        response = await async_client.post(f"/collections/{collection['id']}/sharing")
        assert response.status_code == 200
        
        # 3. 用户B订阅Collection
        response = await async_client.post(f"/marketplace/collections/{collection['id']}/subscribe")
        assert response.status_code == 200
        
        # 4. 用户B访问Collection内容 (只读)
        response = await async_client.get(f"/collections/{collection['id']}")
        assert response.status_code == 200
        data = response.json()
        assert data['is_readonly_view'] == True
        assert data['access_type'] == 'subscribed'
        
        # 5. 用户B尝试写操作 (应该失败)
        response = await async_client.put(f"/collections/{collection['id']}", 
                                         json={"title": "Modified"})
        assert response.status_code == 403
```

**性能测试:**

```python
# tests/performance/test_marketplace_load.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def test_concurrent_subscriptions():
    """测试并发订阅场景"""
    
    # 模拟100个用户同时订阅同一个Collection
    async def subscribe_user(user_id: int, collection_id: str):
        async with AsyncClient() as client:
            return await client.post(f"/marketplace/collections/{collection_id}/subscribe")
    
    tasks = [subscribe_user(i, "test_collection_id") for i in range(100)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 验证数据库一致性
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    assert success_count <= 100  # 不应该有重复订阅
```

**前端测试策略:**

```typescript
// frontend/src/pages/marketplace/__tests__/index.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import MarketplacePage from '../index';

describe('MarketplacePage', () => {
  test('displays published collections', async () => {
    render(<MarketplacePage />);
    
    await waitFor(() => {
      expect(screen.getByText('知识库市场')).toBeInTheDocument();
    });
    
    // 验证Collection卡片显示
    const collectionCards = screen.getAllByTestId('collection-card');
    expect(collectionCards.length).toBeGreaterThan(0);
  });
  
  test('handles subscription action', async () => {
    const mockSubscribe = jest.fn().mockResolvedValue({});
    render(<MarketplacePage />);
    
    const subscribeButton = screen.getByText('订阅');
    fireEvent.click(subscribeButton);
    
    await waitFor(() => {
      expect(mockSubscribe).toHaveBeenCalled();
    });
  });
});
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
    const [ownedCollections, subscribedCollections] = await Promise.all([
      api.getCollections(pagination),                            // 获取自有Collection
      api.getMarketplaceCollectionsSubscriptions(pagination)    // 获取订阅Collection
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
    // Collection 卡片增强 Props
    interface EnhancedCollectionCardProps {
      collection: Collection;
      access_type: 'owner' | 'subscribed';
      subscription_info?: {
        owner_username: string;
        gmt_subscribed: string;
      };
    }
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
        - 订阅Collection: 显示 "来自 @{owner_username} • 订阅于 {相对时间}"
    - **操作菜单差异化**:
        - 自有Collection: 编辑、删除、分享设置、查看详情
        - 订阅Collection: 查看详情、取消订阅
    - **筛选器实现**:
        ```typescript
        const [filter, setFilter] = useState<'all' | 'owned' | 'subscribed'>('all');
        const filteredCollections = collections.filter(col => {
          if (filter === 'owned') return col.access_type === 'owner';
          if (filter === 'subscribed') return col.access_type === 'subscribed';
          return true; // 'all'
        });
        ```

**C. 修改 Collection 详情页**

- **路由**: `/collections/{collection_id}` (复用现有路由)
- **文件位置**: `frontend/src/pages/collections/$collectionId/index.tsx`
- **API调用**: `GET /api/v1/collections/{collection_id}` (现有接口)
- **权限检查**: 后端 `_check_read_access()` 验证用户是否有权限访问
- **响应字段扩展**:
    ```typescript
    interface CollectionDetail {
      // ... 现有字段
      is_readonly_view: boolean;           // 是否只读模式
      access_type: 'owner' | 'subscribed' | 'public'; // 访问类型
      sharing_info?: {                     // 分享信息 (仅所有者可见)
        status: 'DRAFT' | 'PUBLISHED';
        gmt_created: string;
      };
      subscription_info?: {                // 订阅信息 (仅订阅者可见)
        gmt_subscribed: string;
        owner_username: string;
      };
    }
    ```
- **功能增强**:
    - **只读模式** (`is_readonly_view: true`):
        - 页面顶部显示ReadOnlyBanner组件
        - 隐藏所有编辑按钮：编辑Collection、上传文档、删除文档、重建索引
        - 隐藏设置页面入口
        - 文档列表只显示查看、预览按钮
        - 图谱页面隐藏合并节点等编辑功能
        - 聊天Bot可正常使用（只读查询）
    - **所有者模式** (`access_type: 'owner'`):
        - 显示SharingControl组件（发布/取消发布开关）
        - 显示完整的编辑功能
        - 可查看分享统计信息
    - **订阅模式** (`access_type: 'subscribed'`):
        - 显示订阅信息："订阅自 @{owner_username}，{订阅时间}"
        - 提供取消订阅按钮
        - 所有内容只读访问

#### 6.2. 组件设计

**A. CollectionMarketplaceCard 组件**

- **文件位置**: `frontend/src/components/CollectionMarketplaceCard.tsx`
- **Props 接口**:
    ```typescript
    interface CollectionMarketplaceCardProps {
      collection: CollectionMarketplaceDetail;
      onClick: (collectionId: string) => void;
    }
    ```
- **UI 元素**:
    - Collection 标题（加粗显示）
    - Collection 描述（最多显示 150 字符，超出显示省略号）
    - 所有者用户名（小号字体，灰色显示）
    - 发布时间（相对时间格式，如 "3 天前"）
    - 悬浮效果和点击交互

**B. 只读模式提示 Banner**

- **文件位置**: `frontend/src/components/ReadOnlyBanner.tsx`
- **显示条件**: 当 `is_readonly_view` 为 `true` 时显示
- **UI 设计**:
    - 位置：页面顶部，在页面标题下方
    - 样式：使用 Ant Design Alert 组件，type="info"
    - 文案："您正在以只读模式浏览一个共享知识库，无法进行修改操作"
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
      marketplaceCollections: CollectionMarketplaceDetail[];
      marketplaceLoading: boolean;
      marketplacePagination: {
        current: number;
        pageSize: number;
        total: number;
      };
    }
    ```
- **新增 Effects**:
    - `fetchMarketplaceCollections`: 获取市场 Collection 列表
    - `publishCollection`: 发布 Collection 到市场
    - `unpublishCollection`: 从市场下架 Collection
    - `fetchSharingStatus`: 获取 Collection 分享状态

**B. 全局状态更新**

- **文件位置**: `frontend/src/models/global.ts`
- **导航菜单**: 新增 "知识库市场" 菜单项，链接到 `/marketplace`

#### 4.4. UI 交互逻辑

**A. 只读模式下的 UI 限制**

需要在以下组件中根据 `is_readonly_view` 字段禁用或隐藏相关功能：

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

### 7. 详细实施计划 (TODO List)

#### **Phase 1: 后端 - 数据库与核心服务**

- [ ] **1.1. 数据库模型与迁移**
    - [ ] 在 `aperag/db/models.py` 中定义数据库模型：
        - `CollectionMarketplaceItem` SQLModel：分享状态记录，包含状态和时间字段
        - `UserCollectionSubscription` SQLModel：用户订阅记录，使用 `gmt_deleted` 字段实现软删除
        - 包含所有必要字段、约束和索引（特别注意 `gmt_deleted` 的索引优化）
    - [ ] 运行 `make makemigration` 生成新的数据库迁移脚本
    - [ ] 检查生成的迁移脚本（位于 `aperag/migration/versions/`）确保 SQL 语法正确性和索引创建
    - [ ] 运行 `make migrate` 将数据库 schema 变更应用到开发环境
    - [ ] 验证新表创建成功，检查约束和索引是否正确建立

- [ ] **1.2. OpenAPI Schema 定义**
    - [ ] 创建 `aperag/api/components/schemas/marketplace.yaml`，定义以下模型：
        - `CollectionMarketplaceStatusEnum`
        - `CollectionMarketplaceItem`
        - `CollectionMarketplaceDetail`
        - `CollectionMarketplaceDetailList`
        - `UserCollectionSubscription`
        - `UserSubscription` (用于订阅Collection API)
        - `UserSubscriptionList` (用于订阅Collection API)
    - [ ] 创建 `aperag/api/paths/marketplace.yaml`，定义以下端点的完整规范：
        - `GET /api/v1/marketplace/collections`：获取市场Collection列表
        - `GET /api/v1/marketplace/collections/subscriptions`：获取当前用户订阅的Collection列表
        - `POST /api/v1/marketplace/collections/{collection_id}/subscribe`：订阅Collection
        - `DELETE /api/v1/marketplace/collections/{collection_id}/subscribe`：取消订阅Collection
    - [ ] 修改 `aperag/api/paths/collections.yaml`，添加 sharing 相关端点：
        - `GET /api/v1/collections/{collection_id}/sharing`
        - `POST /api/v1/collections/{collection_id}/sharing`
        - `DELETE /api/v1/collections/{collection_id}/sharing`

    - [ ] 修改 `aperag/api/components/schemas/collection.yaml`，在 Collection schema 中添加 `sharing_info` 和 `is_readonly_view` 字段
    - [ ] 运行 `make generate-models` 生成更新后的 `aperag/schema/view_models.py`
    - [ ] 验证生成的 Pydantic 模型类型注解正确

- [ ] **1.3. 服务层 - Marketplace Service**
    - [ ] 创建 `aperag/service/marketplace_service.py` 文件和 MarketplaceService 类
    - [ ] 实现 `publish_collection(user_id: str, collection_id: str)` 方法：
        - 验证用户是 Collection 所有者
        - 创建或更新 collection_marketplace 记录为 PUBLISHED 状态，手动设置 `gmt_updated = datetime.utcnow()`
        - 处理重复发布的情况（如果已经是 PUBLISHED 状态，应返回成功但不执行任何操作）
    - [ ] 实现 `unpublish_collection(user_id: str, collection_id: str)` 方法：
        - 验证用户是 Collection 所有者
        - 将 collection_marketplace 记录状态更新为 DRAFT，同时手动设置 `gmt_updated = datetime.utcnow()`
        - 批量失效所有相关订阅（批量设置 `gmt_deleted = datetime.utcnow()`）
        - 使用数据库事务确保数据一致性
    - [ ] 实现 `get_sharing_status(collection_id: str)` 方法：
        - 返回指定 Collection 的分享状态信息
    - [ ] 实现 `get_raw_sharing_status(collection_id: str)` 内部方法：
        - 供权限检查函数调用，不进行额外的权限验证
    - [ ] 实现 `list_published_collections(user_id: str, page: int, page_size: int)` 方法：
        - 查询所有 PUBLISHED 状态的 Collection
        - 支持分页功能
        - 关联查询获取 Collection 基本信息和所有者用户名
        - 计算当前用户的订阅状态（is_subscribed 字段）
    - [ ] 实现订阅相关方法：
        - `subscribe_collection(user_id: str, collection_id: str)` 方法：
            - 验证 Collection 已发布 (status = 'PUBLISHED')
            - 验证用户不是 Collection 所有者，如果是则抛出 SelfSubscriptionError
            - 检查是否已订阅，防止重复订阅
            - 创建用户订阅记录
        - `unsubscribe_collection(user_id: str, collection_id: str)` 方法：
            - 验证用户已订阅该 Collection
            - 软删除订阅记录（设置 gmt_deleted = current_timestamp）
        - `get_user_subscription(user_id: str, collection_id: str)` 方法：
            - 获取用户对指定 Collection 的活跃订阅状态（`WHERE gmt_deleted IS NULL`）
            - 供权限检查函数调用，返回 None 表示未订阅或已取消订阅
        - `list_user_subscribed_collections(user_id: str, page: int, page_size: int)` 方法：
            - 查询用户所有活跃订阅的 Collection（`WHERE gmt_deleted IS NULL`）
            - 关联查询获取 Collection 详细信息和原所有者信息
            - 返回包含订阅信息的 Collection 列表
            - 支持分页功能

- [ ] **1.4. 服务层 - 权限控制**
    - [ ] 在 `aperag/service/collection_service.py` 中实现 `_check_read_access` 方法（Subscribe 模式）：
        - 检查 Collection 是否存在
        - 判断用户是否为所有者，所有者具有完全访问权限
        - 非所有者时检查 Collection 是否已发布
        - 非所有者时检查用户是否已订阅该 Collection
        - 只有已订阅的用户才能访问已发布的 Collection
        - 返回 Collection 实例或抛出相应的 HTTPException
    - [ ] 在 `aperag/service/collection_service.py` 中实现 `_check_write_access` 方法：
        - 检查 Collection 是否存在
        - 验证只有所有者具有写权限
        - 订阅用户对 Collection 严格只读
        - 为共享 Collection 提供更具体的错误信息
    - [ ] 修改 `collection_service.py` 中的现有方法集成权限检查：
        - `get_collection`: 调用 `_check_read_access`
        - `update_collection`: 调用 `_check_write_access`
        - `delete_collection`: 调用 `_check_write_access`，并实现级联软删除相关的marketplace和订阅记录
        - `list_collections`: 保持现有逻辑（仅返回所有者的 Collection）
    - [ ] 修改 `aperag/service/document_service.py` 集成权限检查：
        - `list_documents`: 调用 `collection_service._check_read_access`
        - `get_document`: 调用 `collection_service._check_read_access`
        - `create_document`: 调用 `collection_service._check_write_access`
        - `update_document`: 调用 `collection_service._check_write_access`
        - `delete_document`: 调用 `collection_service._check_write_access`
    - [ ] 修改 `aperag/service/graph_service.py` 集成权限检查：
        - `get_graph`: 调用 `collection_service._check_read_access`
    - [ ] 修改相关聊天服务集成权限检查：
        - 在访问 Collection 相关信息时调用 `collection_service._check_read_access`
    - [ ] 修改 `aperag/service/search_service.py` 集成权限检查：
        - `get_search_history`: 调用 `collection_service._check_read_access`
        - `execute_search`: 调用 `collection_service._check_read_access`
        - `delete_search_record`: 调用 `collection_service._check_write_access`
    - [ ] 修改 `aperag/service/bot_service.py` 集成权限检查：
        - `get_bot`: 检查 Bot 关联的 Collection 读权限
        - `update_bot`: 检查 Bot 关联的 Collection 写权限
        - `delete_bot`: 检查 Bot 关联的 Collection 写权限
        - 实现 `get_bot_collections()` 方法获取 Bot 关联的 Collection 列表
    - [ ] 修改相关的其他写操作端点：
        - `collection_summary_generate`: 调用 `collection_service._check_write_access`
        - `document_rebuild_indexes`: 调用 `collection_service._check_write_access`
        - `graph_nodes_merge`: 调用 `collection_service._check_write_access`
        - `graph_suggestion_action`: 调用 `collection_service._check_write_access`

#### **Phase 2: 后端 - API 视图与前端集成**

- [ ] **2.1. API 视图层实现**
    - [ ] 创建 `aperag/views/marketplace.py` 文件：
        - 实现 `list_marketplace_collections_view` 函数
        - 处理分页参数验证和默认值设置
        - 调用 `marketplace_service.list_published_collections`
        - 返回标准化的分页响应格式
    - [ ] 修改 `aperag/views/collections.py`（或相关视图文件）实现 sharing 相关端点：
        - `get_collection_sharing_status_view`: 获取分享状态（仅所有者）
        - `publish_collection_view`: 发布 Collection 到市场
        - `unpublish_collection_view`: 从市场下架 Collection
        - 为每个端点添加用户身份验证、所有权验证和异常错误处理
    - [ ] 在 `aperag/app.py` 中注册新的路由：
        - 添加 `marketplace` 路由组，tag 设为 "marketplace"
        - 集成到主应用的路由配置中
    - [ ] 修改现有的 `get_collection_view` 视图逻辑：
        - 使用新的 `_check_read_access` 权限检查
        - 计算并填充 `is_readonly_view` 字段
        - 为所有者返回 `sharing_info` 信息
        - 处理非所有者访问共享 Collection 的情况

- [ ] **2.2. 前端 - 生成 SDK 与状态管理**
    - [ ] 运行 `make generate-frontend-sdk` 更新前端 API client
    - [ ] 验证 `frontend/src/api/` 目录中的新增内容：
        - 检查 `apis/` 目录下是否生成了 marketplace 相关的 API 函数
        - 检查 `models/` 目录下是否生成了新的 TypeScript 接口
        - 验证现有 Collection 接口是否正确更新
    - [ ] 更新前端类型定义：
        - 修改 `frontend/src/models/collection.ts` 中的 Collection 接口
        - 在 `frontend/src/types/` 中添加或更新相关类型定义
        - 确保 `sharing_info` 和 `is_readonly_view` 字段类型正确

#### **Phase 3: 前端 - UI 实现**

- [ ] **3.1. Marketplace 页面开发**
    - [ ] 创建页面文件 `frontend/src/pages/marketplace/index.tsx`：
        - 实现基础页面结构和布局
        - 添加页面标题 "知识库市场" 和功能说明
        - 集成分页组件和加载状态管理
    - [ ] 实现 API 数据获取逻辑：
        - 在页面加载时调用 marketplace API
        - 处理分页参数和状态更新
        - 实现错误状态处理和重试机制
    - [ ] 创建 `CollectionMarketplaceCard` 组件（`frontend/src/components/CollectionMarketplaceCard.tsx`）：
        - 设计卡片布局（标题、描述、所有者、发布时间）
        - 实现悬浮效果和点击交互
        - 处理描述文本截断（最多 150 字符）
        - 添加相对时间格式化功能
        - 实现订阅按钮逻辑：
            - 如果当前用户是Collection所有者，显示 "我的" 标签，不显示订阅按钮
            - 如果当前用户非所有者且未订阅，显示 "订阅" 按钮
            - 如果当前用户已订阅，显示 "已订阅" 状态
    - [ ] 实现网格布局和响应式设计：
        - 桌面端：4 列网格布局
        - 平板端：2-3 列网格布局
        - 手机端：1 列布局
    - [ ] 添加到导航菜单：
        - 在 `frontend/src/layouts/sidebar.tsx` 中添加 "知识库市场" 菜单项
        - 设置市场图标（如ShopOutlined）和路由链接

- [ ] **3.2. Collection 详情页 - 只读模式实现**
    - [ ] 创建 `ReadOnlyBanner` 组件（`frontend/src/components/ReadOnlyBanner.tsx`）：
        - 使用 Ant Design Alert 组件
        - 设计醒目的提示样式（蓝色信息提示）
        - 添加信息图标（InfoCircleOutlined）和提示文案
    - [ ] 修改 Collection 详情页面：
        - 在页面顶部集成 ReadOnlyBanner 组件
        - 根据 `is_readonly_view` 字段控制组件显示
    - [ ] 实现写操作 UI 的禁用逻辑：
        - 文档管理页面：隐藏 "上传文档"、"批量操作"、文档编辑/删除按钮
        - Collection 设置页面：完全隐藏设置页面入口或设置表单为只读
        - 其他页面的编辑、删除、添加功能按钮全部禁用
    - [ ] 保持只读功能正常：
        - 文档列表查看功能正常
        - 文档内容阅读功能正常
        - 知识图谱浏览功能正常
        - 聊天查询功能正常

- [ ] **3.3. Collection 详情页 - 分享功能实现**
    - [ ] 创建 `SharingControl` 组件（`frontend/src/components/SharingControl.tsx`）：
        - 使用 Switch 组件控制发布状态
        - 显示当前分享状态标签
        - 仅在用户是所有者时显示该组件
    - [ ] 实现分享操作确认对话框：
        - 发布确认：说明发布后其他用户可以访问
        - 下架确认：说明下架后其他用户将无法访问
        - 使用 Ant Design Modal 组件
    - [ ] 集成分享状态管理：
        - 在 Collection model 中添加相关 Effects
        - 实现发布/取消发布的 API 调用
        - 处理操作成功/失败的反馈提示
    - [ ] 在 Collection 详情页面中集成 SharingControl：
        - 在Collection标题右侧区域展示分享控制组件
        - 根据用户权限控制组件可见性
        - 实现状态变更后的页面刷新
