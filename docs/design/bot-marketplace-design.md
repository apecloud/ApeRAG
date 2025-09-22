# Bot Marketplace 功能设计文档

## 1. 概述

### 1.1 背景
基于现有的Collection Marketplace功能，为ApeRAG系统增加Bot Marketplace功能，允许用户发布自己的Bot到市场，其他用户可以订阅并使用这些Bot进行对话，但不能编辑Bot配置。

### 1.2 目标
- 允许Bot创建者将Bot发布到公共市场
- 允许其他用户浏览、订阅和使用发布的Bot
- 保持与Collection Marketplace一致的用户体验
- 确保权限控制：订阅者只能使用Bot对话，不能编辑

### 1.3 核心功能对比

| 功能 | Collection Marketplace | Bot Marketplace |
|-----|---------------------|-----------------|
| 发布到marketplace | ✅ | ✅ |
| 从marketplace取消发布 | ✅ | ✅ |
| 查看发布状态 | ✅ | ✅ |
| 浏览marketplace | ✅ | ✅ |
| 订阅/取消订阅 | ✅ | ✅ |
| 查看已订阅列表 | ✅ | ✅ |
| 只读访问 | ✅ (文档浏览) | ✅ (对话功能) |
| 编辑权限控制 | ✅ (订阅者无编辑权) | ✅ (订阅者无编辑权) |

## 2. 数据库设计

### 2.1 新增枚举类型

```python
class BotMarketplaceStatusEnum(str, Enum):
    """Bot marketplace sharing status enumeration"""
    DRAFT = "DRAFT"  # Not published, only owner can see
    PUBLISHED = "PUBLISHED"  # Published to marketplace, publicly visible
```

### 2.2 新增数据表

#### 2.2.1 Bot Marketplace 表

```python
class BotMarketplace(Base):
    """Bot sharing status table"""
    
    __tablename__ = "bot_marketplace"
    __table_args__ = (
        UniqueConstraint("bot_id", name="uq_bot_marketplace_bot"),
        Index("idx_bot_marketplace_status", "status"),
        Index("idx_bot_marketplace_gmt_deleted", "gmt_deleted"),
        Index("idx_bot_marketplace_bot_id", "bot_id"),
        Index("idx_bot_marketplace_list", "status", "gmt_created"),
    )
    
    id = Column(String(24), primary_key=True, default=lambda: "bot_market_" + random_id()[:14])
    bot_id = Column(String(24), nullable=False)
    
    # Sharing status: use VARCHAR storage, not database enum type
    status = Column(String(20), nullable=False, default=BotMarketplaceStatusEnum.DRAFT.value)
    
    # Timestamp fields
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)
```

#### 2.2.2 User Bot Subscription 表

```python
class UserBotSubscription(Base):
    """User subscription to published bots table"""
    
    __tablename__ = "user_bot_subscription"
    __table_args__ = (
        # Allow multiple history records, but active subscription (gmt_deleted=NULL) must be unique
        UniqueConstraint(
            "user_id", "bot_marketplace_id", "gmt_deleted", 
            name="idx_user_bot_marketplace_history_unique"
        ),
        Index("idx_user_bot_subscription_marketplace", "bot_marketplace_id"),
        Index("idx_user_bot_subscription_user", "user_id"),
        Index("idx_user_bot_subscription_gmt_deleted", "gmt_deleted"),
    )
    
    id = Column(String(24), primary_key=True, default=lambda: "bot_sub_" + random_id()[:14])
    user_id = Column(String(24), nullable=False)
    bot_marketplace_id = Column(String(24), nullable=False)
    
    # Timestamp fields
    gmt_subscribed = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True)
```

## 3. API设计

### 3.1 Bot管理API扩展

在现有Bot API基础上，新增sharing相关endpoints：

#### 3.1.1 获取Bot发布状态

```yaml
GET /bots/{bot_id}/sharing
```

**参数：**
- `bot_id`: Bot ID (path parameter)

**响应：**
```json
{
  "is_published": true,
  "published_at": "2025-01-01T00:00:00Z"
}
```

#### 3.1.2 发布Bot到Marketplace

```yaml
POST /bots/{bot_id}/sharing
```

**权限：** 仅Bot所有者

**响应：** 204 No Content

#### 3.1.3 从Marketplace取消发布Bot

```yaml
DELETE /bots/{bot_id}/sharing
```

**权限：** 仅Bot所有者

**响应：** 204 No Content

### 3.2 Bot Marketplace API

#### 3.2.1 浏览Marketplace中的Bot

```yaml
GET /marketplace/bots
```

**参数：**
- `page`: 页码 (query, default: 1)
- `page_size`: 页大小 (query, default: 12, max: 100)
- `type`: Bot类型筛选 (query, optional: knowledge/common/agent)

**响应：**
```json
{
  "items": [
    {
      "id": "bot123",
      "title": "AI Assistant Bot",
      "description": "A helpful AI assistant",
      "type": "knowledge",
      "owner_user_id": "user123",
      "owner_username": "john_doe",
      "subscription_id": null,
      "gmt_subscribed": null,
      "config": {
        "agent": {
          "completion": {...},
          "system_prompt_template": "...",
          "collections": [...]
        }
      },
      "is_subscribed": false,
      "is_owner": false
    }
  ],
  "total": 50,
  "page": 1,
  "page_size": 12
}
```

#### 3.2.2 获取用户订阅的Bot列表

```yaml
GET /marketplace/bots/subscriptions
```

**权限：** 需要登录

**参数：**
- `page`: 页码 (query, default: 1)
- `page_size`: 页大小 (query, default: 12, max: 100)

**响应：** 同上，但`is_subscribed`均为true

#### 3.2.3 订阅Bot

```yaml
POST /marketplace/bots/{bot_id}/subscribe
```

**权限：** 需要登录，不能订阅自己的Bot

**响应：**
```json
{
  "id": "bot123",
  "title": "AI Assistant Bot",
  "subscription_id": "sub456",
  "gmt_subscribed": "2025-01-01T00:00:00Z",
  ...
}
```

#### 3.2.4 取消订阅Bot

```yaml
DELETE /marketplace/bots/{bot_id}/subscribe
```

**权限：** 需要登录

**响应：**
```json
{
  "message": "Successfully unsubscribed"
}
```

#### 3.2.5 获取Marketplace Bot详情

```yaml
GET /marketplace/bots/{bot_id}
```

**权限：** 需要订阅该Bot或者是Bot所有者

**响应：** 返回Bot详情（只读）

### 3.3 用户Bot列表API扩展

扩展现有的`GET /bots`API，支持返回用户自有Bot和订阅Bot的混合列表：

```yaml
GET /bots?include_subscribed=true
```

**新增参数：**
- `include_subscribed`: 是否包含订阅的Bot (query, default: true)

**响应扩展：**
```json
{
  "items": [
    {
      "id": "bot123",
      "title": "My Bot",
      "is_owner": true,
      "is_subscribed": false,
      "owner_user_id": null,
      "subscription_id": null,
      ...
    },
    {
      "id": "bot456", 
      "title": "Subscribed Bot",
      "is_owner": false,
      "is_subscribed": true,
      "owner_user_id": "user789",
      "owner_username": "alice",
      "subscription_id": "sub123",
      ...
    }
  ],
  ...
}
```

## 4. Schema设计

### 4.1 新增Marketplace Schema

```yaml
# marketplace.yaml 扩展

BotMarketplaceStatusEnum:
  type: string
  enum:
    - DRAFT
    - PUBLISHED
  description: Bot marketplace sharing status

SharedBotConfig:
  type: object
  description: Configuration settings for shared bot (read-only version)
  properties:
    agent:
      type: object
      properties:
        completion:
          $ref: './model.yaml#/modelSpec'
        system_prompt_template:
          type: string
        query_prompt_template:
          type: string
        collections:
          type: array
          items:
            $ref: './marketplace.yaml#/SharedCollection'
    flow:
      $ref: './flow.yaml#/components/schemas/WorkflowDefinition'

SharedBot:
  type: object
  description: Shared bot information for marketplace users
  properties:
    id:
      type: string
      description: Bot ID
    title:
      type: string
      description: Bot title
    description:
      type: string
      nullable: true
      description: Bot description
    type:
      type: string
      enum: ["knowledge", "common", "agent"]
      description: Bot type
    owner_user_id:
      type: string
      description: Original owner user ID
    owner_username:
      type: string
      nullable: true
      description: Original owner username
    subscription_id:
      type: string
      nullable: true
      description: Subscription record ID
    gmt_subscribed:
      type: string
      format: date-time
      nullable: true
      description: Subscription time
    config:
      $ref: '#/SharedBotConfig'
      description: Bot configuration (read-only)
    is_subscribed:
      type: boolean
      description: Whether current user has subscribed
    is_owner:
      type: boolean
      description: Whether current user is the owner
  required:
    - id
    - title
    - type
    - owner_user_id
    - config
    - is_subscribed
    - is_owner

SharedBotList:
  type: object
  properties:
    items:
      type: array
      items:
        $ref: '#/SharedBot'
    total:
      type: integer
    page:
      type: integer
    page_size:
      type: integer
  required:
    - items
    - total
    - page
    - page_size
```

### 4.2 扩展Bot Schema

```yaml
# bot.yaml 扩展

bot:
  type: object
  properties:
    id:
      type: string
    title:
      type: string
    description:
      type: string
    type:
      type: string
      enum: ["knowledge", "common", "agent"]
    config:
      $ref: '#/botConfig'
    created:
      type: string
      format: date-time
    updated:
      type: string
      format: date-time
    # 新增字段
    is_subscribed:
      type: boolean
      description: Whether this is a subscribed bot
      default: false
    is_owner:
      type: boolean
      description: Whether current user owns this bot
      default: true
    owner_user_id:
      type: string
      nullable: true
      description: Owner user ID (for subscribed bots)
    owner_username:
      type: string
      nullable: true
      description: Owner username (for subscribed bots)
    subscription_id:
      type: string
      nullable: true
      description: Subscription ID (for subscribed bots)
```

## 5. 服务层设计

### 5.1 BotMarketplaceService

参考`MarketplaceService`的设计，创建`BotMarketplaceService`：

```python
class BotMarketplaceService:
    """Bot marketplace business logic service"""
    
    async def publish_bot(self, user_id: str, bot_id: str) -> None:
        """Publish bot to marketplace"""
        
    async def unpublish_bot(self, user_id: str, bot_id: str) -> None:
        """Remove bot from marketplace"""
        
    async def get_sharing_status(self, user_id: str, bot_id: str) -> Tuple[bool, Optional[datetime]]:
        """Get bot sharing status"""
        
    async def list_published_bots(self, user_id: str, page: int = 1, page_size: int = 12, bot_type: str = None) -> view_models.SharedBotList:
        """List all published bots in marketplace"""
        
    async def subscribe_bot(self, user_id: str, bot_id: str) -> view_models.SharedBot:
        """Subscribe to bot"""
        
    async def unsubscribe_bot(self, user_id: str, bot_id: str) -> None:
        """Unsubscribe from bot"""
        
    async def list_user_subscribed_bots(self, user_id: str, page: int = 1, page_size: int = 12) -> view_models.SharedBotList:
        """Get user's subscribed bots"""
        
    async def get_user_subscription(self, user_id: str, bot_id: str) -> Optional[db_models.UserBotSubscription]:
        """Get user's subscription status for bot"""
        
    async def cleanup_bot_marketplace_data(self, bot_id: str) -> None:
        """Cleanup marketplace data when bot is deleted"""
```

### 5.2 BotService扩展

扩展现有的`BotService`以支持marketplace功能：

```python
class BotService:
    # 现有方法...
    
    async def list_user_bots_with_subscriptions(self, user_id: str, page: int = 1, page_size: int = 10, include_subscribed: bool = True) -> view_models.BotList:
        """List user's own bots and subscribed bots"""
        
    async def get_bot_with_permission_check(self, user_id: str, bot_id: str) -> Tuple[view_models.Bot, bool]:
        """Get bot details with permission info (is_owner, is_subscribed)"""
```

## 6. 数据访问层设计

### 6.1 新增Repository方法

在`aperag/db/repositories/`下新增或扩展相关Repository：

#### 6.1.1 BotMarketplaceRepository

```python
class BotMarketplaceRepository:
    """Bot marketplace data access layer"""
    
    async def create_or_update_bot_marketplace(self, bot_id: str, status: str) -> db_models.BotMarketplace:
        """Create or update bot marketplace record"""
        
    async def get_bot_marketplace_by_bot_id(self, bot_id: str) -> Optional[db_models.BotMarketplace]:
        """Get bot marketplace record by bot_id"""
        
    async def unpublish_bot(self, bot_id: str) -> Optional[db_models.BotMarketplace]:
        """Unpublish bot and invalidate subscriptions"""
        
    async def list_published_bots_with_subscription_status(self, user_id: str, page: int, page_size: int, bot_type: str = None) -> Tuple[List[Dict], int]:
        """List published bots with user's subscription status"""
        
    async def soft_delete_bot_marketplace(self, bot_id: str) -> None:
        """Soft delete bot marketplace data"""
```

#### 6.1.2 UserBotSubscriptionRepository

```python
class UserBotSubscriptionRepository:
    """User bot subscription data access layer"""
    
    async def create_subscription(self, user_id: str, bot_marketplace_id: str) -> db_models.UserBotSubscription:
        """Create bot subscription"""
        
    async def get_user_subscription_by_bot_id(self, user_id: str, bot_id: str) -> Optional[db_models.UserBotSubscription]:
        """Get user's subscription by bot_id"""
        
    async def get_user_subscription_by_marketplace_id(self, user_id: str, bot_marketplace_id: str) -> Optional[db_models.UserBotSubscription]:
        """Get user's subscription by marketplace_id"""
        
    async def unsubscribe_bot(self, user_id: str, bot_id: str) -> None:
        """Unsubscribe from bot (soft delete)"""
        
    async def list_user_subscribed_bots(self, user_id: str, page: int, page_size: int) -> Tuple[List[Dict], int]:
        """List user's subscribed bots"""
```

## 7. 权限控制

### 7.1 权限矩阵

| 操作 | Bot Owner | Subscribed User | Anonymous User |
|-----|-----------|----------------|----------------|
| 查看Bot详情 | ✅ | ✅ | ❌ |
| 编辑Bot配置 | ✅ | ❌ | ❌ |
| 删除Bot | ✅ | ❌ | ❌ |
| 发布到Marketplace | ✅ | ❌ | ❌ |
| 取消发布 | ✅ | ❌ | ❌ |
| 与Bot对话 | ✅ | ✅ | ❌ |
| 订阅Bot | ❌ (不能订阅自己的) | ✅ | ❌ |
| 取消订阅 | ❌ | ✅ | ❌ |
| 浏览Marketplace | ✅ | ✅ | ✅ (可选) |

### 7.2 权限检查实现

```python
async def check_bot_access_permission(user_id: str, bot_id: str) -> Tuple[bool, bool]:
    """
    Check user's access permission to bot
    
    Returns:
        Tuple[is_owner, can_access]: (是否是所有者, 是否可以访问)
    """
    bot = await bot_service.get_bot(bot_id)
    if not bot:
        return False, False
        
    is_owner = bot.user == user_id
    if is_owner:
        return True, True
        
    # Check subscription
    subscription = await bot_marketplace_service.get_user_subscription(user_id, bot_id)
    can_access = subscription is not None
    
    return False, can_access
```

## 8. 异常处理

### 8.1 新增异常类型

```python
class BotNotPublishedError(Exception):
    """Bot is not published to marketplace"""
    pass

class SelfBotSubscriptionError(Exception):
    """Cannot subscribe to own bot"""
    pass

class AlreadySubscribedToBotError(Exception):
    """Already subscribed to this bot"""
    pass

class BotNotFoundException(Exception):
    """Bot not found"""
    pass
```

### 8.2 异常处理映射

| 异常类型 | HTTP状态码 | 错误消息 |
|---------|-----------|---------|
| BotNotPublishedError | 400 | "Bot is not published to marketplace" |
| SelfBotSubscriptionError | 400 | "Cannot subscribe to your own bot" |
| AlreadySubscribedToBotError | 409 | "Already subscribed to this bot" |
| BotNotFoundException | 404 | "Bot not found" |
| PermissionDeniedError | 403 | "Permission denied" |

## 9. 实现步骤

### 9.1 Phase 1: 数据库和模型层
1. 添加新的枚举类型到`models.py`
2. 添加`BotMarketplace`和`UserBotSubscription`模型
3. 生成并应用数据库迁移

### 9.2 Phase 2: 数据访问层
1. 扩展`aperag/db/ops.py`添加bot marketplace相关操作
2. 或创建新的repository文件

### 9.3 Phase 3: 服务层
1. 创建`BotMarketplaceService`
2. 扩展现有的`BotService`
3. 添加异常类型

### 9.4 Phase 4: API层
1. 扩展`views/bots.py`添加sharing endpoints
2. 扩展`views/marketplace.py`添加bot相关endpoints
3. 更新OpenAPI schema定义

### 9.5 Phase 5: 前端集成
1. 更新Bot列表页面支持显示订阅的Bot
2. 添加Bot发布/取消发布功能
3. 创建Bot Marketplace页面
4. 添加订阅/取消订阅功能

### 9.6 Phase 6: 测试和优化
1. 单元测试
2. 集成测试
3. 性能优化
4. 文档更新

## 10. 注意事项

### 10.1 数据一致性
- 当Bot被删除时，需要清理相关的marketplace和subscription数据
- 当Bot从marketplace取消发布时，需要处理现有订阅

### 10.2 性能考虑
- 对常用查询字段添加数据库索引
- 考虑对热门Bot进行缓存
- 分页查询避免性能问题

### 10.3 安全性
- 确保用户只能访问有权限的Bot
- 防止恶意用户通过API绕过权限检查
- 对敏感操作进行审计日志记录

### 10.4 用户体验
- 清晰区分用户自有Bot和订阅Bot
- 提供友好的错误提示
- 支持Bot搜索和筛选功能

## 11. 未来扩展

### 11.1 可能的增强功能
- Bot评分和评论系统
- Bot使用统计和分析
- Bot分类和标签系统
- Bot推荐算法
- 付费Bot支持

### 11.2 技术债务
- 考虑将marketplace相关代码抽象为通用组件
- 统一Collection和Bot的marketplace实现
- 优化数据库查询性能

---

**文档版本:** 1.0  
**创建日期:** 2025-01-20  
**最后更新:** 2025-01-20  
**作者:** ApeRAG开发团队
