# ApeRAG 标签系统与批量授权设计

## 概述

标签系统是一个轻量级的**分组工具**，用于批量管理用户和知识库的访问关系。

**核心理念**：
- 标签只用于分组，不参与权限判断
- 批量授权 = 批量创建订阅记录
- 完全复用现有的订阅机制
- 只新增 2 张表，不修改现有逻辑
- 通用的标签关系表，易于扩展

## 业务场景

### 场景 1：批量授权
```
需求：让研发团队（20人）访问技术文档库
操作：
  1. 给 20 人打上"研发团队"标签
  2. 点击"授权给研发团队标签"
结果：
  系统批量创建 20 条订阅记录
```

### 场景 2：批量订阅
```
需求：新员工需要访问 10 个入门知识库
操作：
  1. 给新员工打上"新员工"标签
  2. 给 10 个知识库打上"新员工必读"标签
  3. 点击"批量订阅"
结果：
  创建订阅记录，新员工可访问这些知识库
```

### 场景 3：临时协作
```
需求：双十一项目组需要共享资料
操作：
  1. 创建"双十一项目"标签
  2. 给项目成员打标签
  3. 给项目知识库打标签
  4. 批量授权
结果：
  项目组成员可访问项目知识库
```

## 系统架构

### 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                             │
│  用户管理：给用户打标签                                        │
│  知识库设置：给知识库打标签、批量授权                           │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  tag_service.py              - 标签 CRUD                     │
│  tag_relation_service.py     - 打标签（通用）                │
│  batch_permission_service.py - 批量授权（新增）              │
│      ↓ 调用                                                  │
│  marketplace_service.py      - 创建订阅记录（复用）           │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    PostgreSQL                               │
│  tag (新增)                   - 标签表                       │
│  tag_relation (新增)          - 标签关系表（通用）            │
│  user_collection_subscription - 订阅表（复用，权限的实际存储） │
└─────────────────────────────────────────────────────────────┘
```

### 批量授权流程

```
用户点击"授权给研发团队标签"
    ↓
查询标签下的所有用户
SELECT target_id FROM tag_relation 
WHERE tag_id='研发团队' AND target_type='user'
    ↓
过滤已有订阅的用户
    ↓
批量创建订阅记录
INSERT INTO user_collection_subscription ...
    ↓
清除权限缓存
    ↓
返回结果：成功 15人，已有权限 3人
```

## 数据模型

### 新增表（2张）

#### 1. tag - 标签表
```sql
CREATE TABLE tag (
    id               VARCHAR(24) PRIMARY KEY,
    name             VARCHAR(50) NOT NULL,
    description      VARCHAR(200),
    user             VARCHAR(256) NOT NULL,  -- 创建者
    gmt_created      TIMESTAMP NOT NULL,
    gmt_updated      TIMESTAMP NOT NULL,
    gmt_deleted      TIMESTAMP
);

-- 同一创建者下标签名称唯一
CREATE UNIQUE INDEX uq_tag_name_user ON tag(name, user, gmt_deleted);
CREATE INDEX idx_tag_user ON tag(user);
```

**字段说明**：
- `name`: 标签名称，如"研发团队"、"新员工必读"
- `description`: 标签描述，可选
- `user`: 创建者，系统管理员创建的标签

#### 2. tag_relation - 标签关系表（通用）
```sql
CREATE TABLE tag_relation (
    id           VARCHAR(24) PRIMARY KEY,
    tag_id       VARCHAR(24) NOT NULL,
    target_type  VARCHAR(20) NOT NULL,  -- 'user', 'collection', 'document', 'bot' ...
    target_id    VARCHAR(24) NOT NULL,
    gmt_created  TIMESTAMP NOT NULL,
    FOREIGN KEY (tag_id) REFERENCES tag(id) ON DELETE CASCADE
);

-- 防止重复打标签
CREATE UNIQUE INDEX uq_tag_relation ON tag_relation(tag_id, target_type, target_id);

-- 查询某个资源的所有标签
CREATE INDEX idx_tag_relation_target ON tag_relation(target_type, target_id);

-- 查询某个标签关联的所有资源（批量操作关键索引）
CREATE INDEX idx_tag_relation_tag_type ON tag_relation(tag_id, target_type);
```

**字段说明**：
- `tag_id`: 标签ID
- `target_type`: 目标资源类型，当前支持：
  - `user`: 用户
  - `collection`: 知识库
  - 未来可扩展：`document`, `bot`, `chat` 等
- `target_id`: 目标资源ID

**扩展性**：
```sql
-- 给用户打标签
INSERT INTO tag_relation (tag_id, target_type, target_id) 
VALUES ('tag_123', 'user', 'user_456');

-- 给知识库打标签
INSERT INTO tag_relation (tag_id, target_type, target_id) 
VALUES ('tag_123', 'collection', 'col_789');

-- 未来给文档打标签
INSERT INTO tag_relation (tag_id, target_type, target_id) 
VALUES ('tag_123', 'document', 'doc_abc');
```

### 复用表（不修改）

- `user_collection_subscription` - 订阅表（权限的实际存储）
- `collection` - 知识库表
- `user` - 用户表

### 表关系

```
tag (1)  ───────< (N) tag_relation
              │
              ├─── target_type='user' ──> user (N)
              ├─── target_type='collection' ──> collection (N)
              └─── target_type='document' ──> document (N)  // 未来
```

## API 设计

### 1. 标签管理

```http
# 创建标签
POST /api/v1/tags
{
  "name": "研发团队",
  "description": "研发相关人员"
}

# 获取标签列表
GET /api/v1/tags?search=研发&page=1&page_size=20

# 更新标签
PUT /api/v1/tags/{tag_id}
{
  "name": "研发团队（新）",
  "description": "更新后的描述"
}

# 删除标签（会级联删除所有 tag_relation 记录）
DELETE /api/v1/tags/{tag_id}
```

### 2. 用户标签

```http
# 给用户添加标签
POST /api/v1/users/{user_id}/tags
{
  "tag_ids": ["tag_abc123", "tag_def456"]
}

# 移除用户标签
DELETE /api/v1/users/{user_id}/tags/{tag_id}

# 获取用户的所有标签
GET /api/v1/users/{user_id}/tags

# 获取标签下的所有用户
GET /api/v1/tags/{tag_id}/users
```

### 3. 知识库标签

```http
# 给知识库添加标签
POST /api/v1/collections/{collection_id}/tags
{
  "tag_ids": ["tag_abc123"]
}

# 移除知识库标签
DELETE /api/v1/collections/{collection_id}/tags/{tag_id}

# 获取知识库的所有标签
GET /api/v1/collections/{collection_id}/tags
```

### 4. 批量授权（核心）

```http
# 批量授权知识库给用户标签
POST /api/v1/collections/{collection_id}/grant-to-tag
{
  "tag_id": "tag_abc123"
}

Response:
{
  "collection_id": "col_123",
  "tag_id": "tag_abc123",
  "tag_name": "研发团队",
  "total_users": 18,
  "new_granted": 15,
  "already_granted": 3,
  "failed": 0
}

# 批量订阅知识库标签
POST /api/v1/users/me/subscribe-tag
{
  "tag_id": "tag_def456"  // 知识库标签
}

Response:
{
  "tag_id": "tag_def456",
  "tag_name": "新员工必读",
  "total_collections": 10,
  "new_subscribed": 8,
  "already_subscribed": 2
}

# 批量撤销授权
POST /api/v1/collections/{collection_id}/revoke-from-tag
{
  "tag_id": "tag_abc123"
}
```

## 实现要点

### 1. 批量授权实现

```python
def grant_collection_to_user_tag(collection_id: str, tag_id: str, operator_id: str):
    # 1. 权限检查：只有 owner 可以授权
    collection = get_collection(collection_id)
    if collection.user != operator_id:
        raise PermissionDenied()
    
    # 2. 查询标签下的所有用户
    user_ids = db.query(TagRelation.target_id).filter(
        TagRelation.tag_id == tag_id,
        TagRelation.target_type == 'user'
    ).all()
    
    # 3. 过滤已有订阅的用户
    existing = db.query(UserCollectionSubscription.user_id).filter(
        UserCollectionSubscription.user_id.in_(user_ids),
        # ... 其他条件
    ).all()
    
    # 4. 批量创建订阅记录
    new_user_ids = [u for u in user_ids if u not in existing]
    for user_id in new_user_ids:
        create_subscription(user_id, collection_id)
    
    # 5. 清除权限缓存
    clear_permission_cache(new_user_ids, collection_id)
    
    return {
        'total': len(user_ids),
        'new_granted': len(new_user_ids),
        'already_granted': len(existing)
    }
```

### 2. 权限检查（不变）

```python
def check_permission(user_id: str, collection_id: str) -> bool:
    """
    权限检查逻辑保持不变，不查询标签表
    """
    # 1. 检查 owner
    if collection.user == user_id:
        return True
    
    # 2. 检查订阅（包括通过标签批量授权创建的订阅）
    subscription = db.query(UserCollectionSubscription).filter(
        UserCollectionSubscription.user_id == user_id,
        # ... 其他条件
    ).first()
    
    if subscription:
        return True
    
    # 3. 检查公开知识库
    if collection.is_published:
        return True
    
    return False
```

### 3. 性能优化

**批量插入优化**：
```python
# 使用 bulk_insert_mappings 批量插入
subscriptions = [
    {
        'id': f'sub_{random_id()}',
        'user_id': user_id,
        'collection_marketplace_id': marketplace_id,
        'gmt_subscribed': utc_now()
    }
    for user_id in new_user_ids
]
db.bulk_insert_mappings(UserCollectionSubscription, subscriptions)
db.commit()
```

**缓存策略**：
```python
# 缓存标签成员列表
redis.setex(f'tag_users:{tag_id}', 1800, json.dumps(user_ids))

# 缓存权限检查结果
redis.setex(f'permission:{user_id}:{collection_id}', 300, 'true')
```

## 权限控制

### 操作权限

| 操作 | 权限要求 |
|------|---------|
| 创建标签 | is_superuser=true |
| 给用户打标签 | is_superuser=true |
| 给知识库打标签 | 知识库 owner |
| 批量授权 | 知识库 owner |
| 批量订阅 | 任何用户（订阅自己） |

### 安全限制

```python
# 批量操作数量限制
MAX_BATCH_SIZE = 1000

if user_count > MAX_BATCH_SIZE:
    raise BadRequest("Too many users, max 1000")

# 审计日志
audit_log.record(
    user_id=operator_id,
    action='batch_grant_to_tag',
    details={
        'tag_id': tag_id,
        'affected_users': 15
    }
)
```

## 迁移方案

### 数据库迁移

```sql
-- 1. 创建标签表
CREATE TABLE tag (
    id VARCHAR(24) PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    description VARCHAR(200),
    user VARCHAR(256) NOT NULL,
    gmt_created TIMESTAMP NOT NULL,
    gmt_updated TIMESTAMP NOT NULL,
    gmt_deleted TIMESTAMP
);

-- 2. 创建标签关系表（通用）
CREATE TABLE tag_relation (
    id VARCHAR(24) PRIMARY KEY,
    tag_id VARCHAR(24) NOT NULL,
    target_type VARCHAR(20) NOT NULL,
    target_id VARCHAR(24) NOT NULL,
    gmt_created TIMESTAMP NOT NULL,
    FOREIGN KEY (tag_id) REFERENCES tag(id) ON DELETE CASCADE
);

-- 3. 创建索引
CREATE UNIQUE INDEX uq_tag_name_user ON tag(name, user, gmt_deleted);
CREATE INDEX idx_tag_user ON tag(user);
CREATE UNIQUE INDEX uq_tag_relation ON tag_relation(tag_id, target_type, target_id);
CREATE INDEX idx_tag_relation_target ON tag_relation(target_type, target_id);
CREATE INDEX idx_tag_relation_tag_type ON tag_relation(tag_id, target_type);

-- 完成，无需数据迁移
```

### 部署步骤

1. **Phase 1**：部署后端 API
2. **Phase 2**：前端添加标签管理 UI
3. **Phase 3**：灰度测试
4. **Phase 4**：全量上线

完全兼容现有系统，无需修改现有代码。

## 未来扩展（可选）

1. **智能推荐**：根据用户行为推荐标签
2. **标签模板**：预定义常用标签组合，一键应用
3. **统计分析**：标签使用情况和趋势分析
4. **外部同步**：与钉钉/LDAP 同步组织架构
5. **审批流程**：批量授权触发审批流程

## 总结

**核心特点**：
- 标签只是分组工具，不参与权限判断
- 批量授权 = 批量创建订阅记录
- 只新增 2 张表，完全兼容现有系统
- 通用的 tag_relation 表，易于扩展到其他资源
- 实现简单，易于维护

**实施路径**：
1. 实现标签 CRUD
2. 实现批量授权核心功能
3. 优化性能和体验
4. 根据反馈迭代
