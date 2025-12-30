# ApeRAG 文档导出架构设计

## 概述

本文档详细说明 ApeRAG 项目中文档导出模块的架构设计，包括单个文档下载和知识库级别导出的功能设计。

**核心设计理念**：
- **后端代理模式**：所有文件下载都通过后端代理，不直接暴露对象存储链接
- **两种导出模式**：
  - 单个文档：同步流式下载
  - 知识库导出：异步打包，生成后端下载 URL
- **简单可靠**：采用成熟的异步任务方案，避免复杂的流式打包

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                             │
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  文档列表页面     │         │  集合设置页面     │        │
│  │  - 单文档下载     │         │  - 知识库导出     │        │
│  └──────────────────┘         └──────────────────┘        │
└─────────┬──────────────────────────┬────────────────────────┘
          │                          │
          │ GET /documents/{id}/download        (同步，流式返回)
          │ POST /collections/{id}/export       (异步，生成下载链接)
          ▼                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      View Layer                             │
│  - download_document_view()          (同步流式)              │
│  - export_collection_view()          (创建导出任务)          │
│  - get_export_task_view()            (查询任务状态)          │
│  - download_export_result_view()     (下载 ZIP)             │
└─────────┬────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  - export_service.py                                        │
│    ├─ download_document()           (流式代理单个文档)       │
│    ├─ export_collection()           (创建导出任务)          │
│    └─ download_export_result()      (流式下载 ZIP)          │
└─────────┬────────────────────────────────────────────────────┘
          │
          ├─ 单个文档下载（同步流式）
          │  └─► 从对象存储读取 → 流式返回客户端
          │      └─► StreamingResponse (chunked transfer)
          │
          └─ 知识库导出（异步）
             └─► 创建 ExportTask 记录
                 └─► 触发 Celery 任务
                     │
                     ▼
          ┌─────────────────────────┐
          │   Celery Worker         │
          │  1. 下载文档到临时目录   │
          │  2. 打包压缩 ZIP         │
          │  3. 上传到对象存储       │
          │  4. 清理临时文件         │
          │  5. 更新任务状态         │
          └────────┬────────────────┘
                   │
                   ▼
          ┌─────────────────────────┐
          │   Object Store          │
          │  - exports/             │
          │    └─ user-{id}/        │
          │       └─ export-{id}.zip│
          │  (7天后自动清理)         │
          └─────────────────────────┘
                   │
                   ▼
          任务完成，返回下载 URL：
          /api/v1/export-tasks/{id}/download
                   │
                   ▼
          用户点击链接下载
                   │
                   ▼
          后端从对象存储读取 ZIP
                   │
                   ▼
          后端流式返回给客户端
```

### 为什么采用后端代理模式？

**场景分析**：

1. **内网部署场景**
   - 对象存储（MinIO/S3）只在内网可访问
   - 用户浏览器无法直接访问对象存储 URL
   - 必须通过后端代理

2. **安全合规要求**
   - 所有文件访问需要经过审计
   - 需要在后端层面进行权限校验
   - 防止通过签名 URL 绕过权限控制

3. **统一访问控制**
   - 后端可以记录下载日志
   - 可以实施下载速率限制
   - 可以统计下载次数

## 导出场景

系统提供两种导出模式：

| 场景 | API | 模式 | 说明 |
|------|-----|------|------|
| **单个文档下载** | `GET /documents/{id}/download` | 同步流式 | 直接返回文件流 |
| **知识库导出** | `POST /collections/{id}/export` | 异步 | 生成后端下载 URL |

## 核心流程详解

### 场景 1: 单个文档下载（同步流式）

#### 1.1 流程图

```
用户点击"下载"按钮
    │
    ▼
GET /api/v1/documents/{document_id}/download
    │
    ▼
后端处理：
    │
    ├─► 验证用户身份（JWT）
    │
    ├─► 验证文档访问权限
    │
    ├─► 查询 Document 记录
    │
    ├─► 从 doc_metadata 获取 object_path
    │
    ├─► 从对象存储读取文件（流式）
    │   └─ 路径：user-{user_id}/{collection_id}/{doc_id}/original.pdf
    │
    └─► 返回 StreamingResponse
        ├─ Content-Type: application/octet-stream
        ├─ Content-Disposition: attachment; filename="xxx.pdf"
        └─ Transfer-Encoding: chunked (流式传输)
    │
    ▼
文件通过后端流式传输给客户端
    │
    ▼
浏览器触发下载
```

#### 1.2 API 定义

**请求**：
```http
GET /api/v1/documents/{document_id}/download
Authorization: Bearer {token}
```

**响应**：
```http
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Content-Disposition: attachment; filename="user_manual.pdf"
Content-Length: 5242880
Transfer-Encoding: chunked

[文件二进制流]
```

#### 1.3 关键特性

- **流式读取**：从对象存储按块读取（chunk size = 64KB）
- **流式写入**：边读边写到 HTTP 响应流
- **内存优化**：最大内存占用 = chunk size × 并发下载数
- **超时控制**：设置合理的读取超时（如 30 分钟）
- **权限控制**：每次下载都验证用户权限
- **审计日志**：记录下载操作（用户、时间、文档）

### 场景 2: 知识库导出（异步打包）

#### 2.1 流程图

```
用户在集合设置页点击"导出知识库"
    │
    ▼
POST /api/v1/collections/{collection_id}/export
    │
    ▼
后端处理：
    │
    ├─► 验证权限（必须是集合所有者）
    │
    ├─► 查询集合下所有文档
    │   └─ WHERE status NOT IN (DELETED, UPLOADED)
    │
    ├─► 检查并发限制（最多 3 个并发任务）
    │
    ├─► 创建 ExportTask 记录
    │   ├─ id = export{random_id}
    │   ├─ status = PENDING
    │   ├─ collection_id = xxx
    │   └─ document_count = N
    │
    └─► 触发 Celery 任务
    │
    ▼
返回给客户端：
{
  "export_task_id": "export_abc123",
  "status": "PENDING",
  "estimated_time_seconds": 120,
  "document_count": 45
}
    │
    ▼
前端开始轮询任务状态（每 2 秒）
    │
    ▼
═══════════════════════════════════════════════════════════
后台 Celery Worker 异步处理：
═══════════════════════════════════════════════════════════
    │
    ▼
1. 更新任务状态：PROCESSING
    │
    ▼
2. 创建临时目录：/tmp/export_temp_{task_id}/
    │
    ▼
3. 遍历文档列表（45 个）：
    │
    ├─► 文档 1：
    │   ├─ 从对象存储下载 → /tmp/export_temp_.../doc1.pdf
    │   └─ 更新进度：2%
    │
    ├─► 文档 2：
    │   ├─ 从对象存储下载 → /tmp/export_temp_.../doc2.docx
    │   └─ 更新进度：4%
    │
    └─► ... (并发下载，最多 5 个)
    │
    ▼
4. 打包 ZIP 文件：
    │
    ├─ 使用 zipfile 库压缩
    ├─ 输出：/tmp/export_{task_id}.zip
    └─ 更新进度：90%
    │
    ▼
5. 上传到对象存储：
    │
    ├─ 路径：exports/user-{user_id}/export_{task_id}.zip
    └─ 更新进度：95%
    │
    ▼
6. 清理本地临时文件
    │
    ├─ 删除临时目录：/tmp/export_temp_{task_id}/
    ├─ 删除本地 ZIP：/tmp/export_{task_id}.zip
    └─ 更新进度：98%
    │
    ▼
7. 更新任务状态：COMPLETED
    │
    ├─ status = COMPLETED
    ├─ object_store_path = exports/user-{user_id}/export_{task_id}.zip
    ├─ file_size = 52428800
    └─ gmt_expires = now + 7 days
═══════════════════════════════════════════════════════════
    │
    ▼
前端轮询到 status = COMPLETED
    │
    ▼
显示下载 URL：/api/v1/export-tasks/{task_id}/download
    │
    ▼
用户点击下载
    │
    ▼
GET /api/v1/export-tasks/{export_task_id}/download
    │
    ▼
后端处理：
    │
    ├─► 验证权限和任务状态
    │
    ├─► 从对象存储读取 ZIP（流式）
    │   └─ 路径：exports/user-{user_id}/export_{task_id}.zip
    │
    └─► 流式返回给客户端（StreamingResponse）
    │
    ▼
浏览器下载完成
```

#### 2.2 API 定义

**创建导出任务**：
```http
POST /api/v1/collections/{collection_id}/export
Authorization: Bearer {token}
```

**响应**：
```json
{
  "export_task_id": "export_abc123",
  "status": "PENDING",
  "estimated_time_seconds": 120,
  "document_count": 45,
  "collection_name": "MyCollection"
}
```

**查询任务状态**：
```http
GET /api/v1/export-tasks/{export_task_id}
Authorization: Bearer {token}
```

**响应（进行中）**：
```json
{
  "export_task_id": "export_abc123",
  "status": "PROCESSING",
  "progress": 65,
  "message": "Processing 30 of 45 documents",
  "gmt_created": "2025-12-30T10:00:00Z"
}
```

**响应（完成）**：
```json
{
  "export_task_id": "export_abc123",
  "status": "COMPLETED",
  "progress": 100,
  "download_url": "/api/v1/export-tasks/export_abc123/download",
  "file_size": 52428800,
  "gmt_created": "2025-12-30T10:00:00Z",
  "gmt_completed": "2025-12-30T10:02:15Z",
  "gmt_expires": "2025-01-06T10:02:15Z"
}
```

**下载导出结果**：
```http
GET /api/v1/export-tasks/{export_task_id}/download
Authorization: Bearer {token}
```

**响应**：
```http
HTTP/1.1 200 OK
Content-Type: application/zip
Content-Disposition: attachment; filename="MyCollection_2025-12-30.zip"
Content-Length: 52428800
Transfer-Encoding: chunked

[ZIP 文件流]
```

#### 2.3 导出文件结构

```
MyCollection_2025-12-30.zip
├── document_1.pdf
├── document_2.docx
├── presentation.pptx
├── spreadsheet.xlsx
└── ...
```

**文件名冲突处理**：
- 如果有重名文件，自动添加序号
- 示例：`document.pdf`, `document_1.pdf`, `document_2.pdf`

#### 2.4 关键特性

- **异步处理**：不阻塞用户操作，后台打包
- **进度追踪**：实时更新任务进度（0-100%）
- **并发下载**：从对象存储并发下载多个文件（最多 5 个）
- **下载 URL**：生成后端下载链接，可多次下载
- **自动清理**：ZIP 文件 7 天后自动删除
- **错误处理**：任务失败时记录错误信息
- **权限控制**：只有集合所有者可以导出

## 数据库设计

### 表：export_task

```sql
CREATE TABLE export_task (
    -- 主键
    id VARCHAR(24) PRIMARY KEY,              -- export{random_id}
    
    -- 用户和范围
    user VARCHAR(256) NOT NULL,              -- 用户 ID
    collection_id VARCHAR(24) NOT NULL,      -- 集合 ID
    
    -- 任务状态
    status VARCHAR(32) NOT NULL,             -- 任务状态
    progress INTEGER DEFAULT 0,              -- 进度 0-100
    message TEXT,                            -- 当前处理信息
    error_message TEXT,                      -- 错误信息
    
    -- 导出结果
    object_store_path TEXT,                  -- ZIP 文件在对象存储的路径
    file_size BIGINT,                        -- 文件大小（字节）
    document_count INTEGER,                  -- 文档数量
    
    -- 时间管理
    gmt_created TIMESTAMP NOT NULL,          -- 创建时间
    gmt_updated TIMESTAMP NOT NULL,          -- 更新时间
    gmt_completed TIMESTAMP,                 -- 完成时间
    gmt_expires TIMESTAMP,                   -- 过期时间（7天后）
    
    -- 索引
    INDEX idx_user_status (user, status),
    INDEX idx_status_expires (status, gmt_expires),
    INDEX idx_created (gmt_created),
    INDEX idx_collection (collection_id)
);
```

**字段说明**：

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `id` | VARCHAR(24) | 任务 ID | `export_abc123` |
| `collection_id` | VARCHAR(24) | 集合 ID | `col_abc123` |
| `status` | VARCHAR(32) | 任务状态 | `PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`, `EXPIRED` |
| `object_store_path` | TEXT | ZIP 在对象存储的路径 | `exports/user-xxx/export_abc123.zip` |
| `document_count` | INTEGER | 文档数量 | 45 |
| `progress` | INTEGER | 进度百分比 | 0-100 |
| `message` | TEXT | 当前处理信息 | `Processing 30 of 45 documents` |
| `error_message` | TEXT | 错误信息 | `Failed to download document: xxx` |
| `gmt_expires` | TIMESTAMP | 过期时间 | `2025-01-06T10:02:15Z` |

**状态枚举**：

| 状态 | 说明 | 何时设置 | 可下载 |
|------|------|----------|--------|
| `PENDING` | 等待处理 | 创建任务时 | ❌ |
| `PROCESSING` | 处理中 | Celery Worker 开始处理 | ❌ |
| `COMPLETED` | 已完成 | ZIP 文件生成完成 | ✅ |
| `FAILED` | 失败 | 处理过程中出错 | ❌ |
| `EXPIRED` | 已过期 | 7 天后定时任务清理 | ❌ |

**状态转换图**：

```
[创建任务] → PENDING → PROCESSING → COMPLETED
                           ↓
                        FAILED
                           ↓
            (7天后) → EXPIRED (文件被删除)
```

## 性能优化和限制

### 1. 并发控制

**每用户并发限制**：
- 最多 3 个并发导出任务
- 超出限制返回 HTTP 429 Too Many Requests
- 防止单个用户占用过多资源

**Worker 并发下载**：
- 每个导出任务内部并发下载文件
- 最多 5 个文件并发下载
- 避免对象存储压力过大

### 2. 文件大小限制

```
知识库导出的限制：
- 最多文档数：10,000 个
- 最大总大小：50 GB
- 单个文档大小：100 MB（继承上传限制）
```

### 3. 超时控制

```
Celery 任务超时：
- 软超时：55 分钟（提前警告）
- 硬超时：60 分钟（强制终止）
- 单个文档下载超时：5 分钟
```

### 4. 存储管理

**临时文件清理策略**：
- 导出文件保留期：7 天
- 清理时机：每天凌晨 3 点执行定时任务
- 清理内容：
  - 从对象存储删除过期的 ZIP 文件
  - 更新数据库任务状态为 `EXPIRED`

**对象存储空间管理**：
- 预估导出文件大小
- 检查对象存储剩余空间（如果支持）
- 空间不足时返回错误提示

## 监控和审计

### 1. 审计日志

**记录内容**：
- 操作类型：`DownloadDocument`, `ExportCollection`
- 用户信息：user_id, IP 地址
- 资源信息：document_id, collection_id, export_task_id
- 时间戳：操作时间
- 文件信息：文件大小、文件名

**日志用途**：
- 安全审计
- 用户行为分析
- 问题排查

### 2. 监控指标

**关键指标**：
- 导出任务成功率
- 平均导出时间
- 导出文件大小分布
- 存储空间使用量
- 下载速率

**告警规则**：
- 导出任务失败率 > 10%
- 磁盘使用率 > 80%
- 平均导出时间 > 5 分钟（小于 100 个文档）

### 3. 错误处理

**常见错误**：

| 错误类型 | HTTP 状态码 | 触发场景 | 处理建议 |
|---------|------------|----------|----------|
| `ResourceNotFoundException` | 404 | 集合/文档不存在 | 检查 ID 是否正确 |
| `PermissionDeniedException` | 403 | 非集合所有者 | 检查用户权限 |
| `QuotaExceededException` | 429 | 并发任务超限 | 等待现有任务完成 |
| `InsufficientStorageException` | 507 | 对象存储空间不足 | 联系管理员清理空间 |
| `ResourceExpiredException` | 410 | 导出文件已过期 | 重新创建导出任务 |

## 为什么上传到对象存储？

### 架构优势

**1. 多实例部署友好**

场景：后端有 3 个实例（Pod/容器）
- Worker A 打包了 ZIP
- 用户下载请求路由到 Worker B
- Worker B 从对象存储读取 ZIP → 成功

如果 ZIP 只在本地：
- Worker A 有文件
- Worker B 找不到文件 → 失败 ❌

**2. 服务器重启不丢失**

- 服务器重启、容器重启、扩缩容
- ZIP 文件在对象存储中，不受影响
- 用户可以继续下载

**3. 统一存储管理**

- 所有文件都在对象存储中
- 统一的备份和容灾策略
- 统一的访问控制和监控

### 完整的对象存储路径设计

```
对象存储根目录/
├── user-{user_A}/
│   ├── {collection_1}/
│   │   └── {document_1}/
│   │       └── original.pdf        (原始文档)
│   └── {collection_2}/
│       └── {document_2}/
│           └── original.docx       (原始文档)
│
└── exports/                         (导出文件专用目录)
    └── user-{user_A}/
        ├── export_{task_1}.zip     (知识库导出 1)
        ├── export_{task_2}.zip     (知识库导出 2)
        └── ...
```

**路径规则**：
- 原始文档：`user-{user_id}/{collection_id}/{document_id}/original{suffix}`
- 导出文件：`exports/user-{user_id}/export_{task_id}.zip`

### 仍需后端代理下载的原因

虽然 ZIP 在对象存储中，但**不直接生成对象存储的 presigned URL**，而是通过后端代理：

**原因**：
1. **权限控制**：下载时验证用户是否仍有权限
2. **审计日志**：记录每次下载操作
3. **内网部署**：对象存储可能只在内网，浏览器无法直接访问
4. **统一管理**：所有 API 调用都经过后端，便于监控和限流

## API 接口汇总

### 完整 API 列表

| 方法 | 路径 | 说明 | 模式 |
|------|------|------|------|
| GET | `/documents/{id}/download` | 下载单个文档 | 同步流式 |
| POST | `/collections/{id}/export` | 知识库导出 | 异步 |
| GET | `/export-tasks/{id}` | 查询导出任务状态 | - |
| GET | `/export-tasks/{id}/download` | 下载导出结果 | 同步流式 |
| GET | `/export-tasks` | 获取用户导出历史 | - |
| DELETE | `/export-tasks/{id}` | 取消/删除导出任务 | - |

## 技术选型

### 1. 异步任务框架

**选择**：Celery + Redis
- **Celery**：成熟的 Python 异步任务框架
- **Redis**：作为消息队列和结果存储

**优势**：
- 成熟稳定，生产环境验证
- 支持任务重试、超时控制
- 易于监控和调试

### 2. 文件压缩

**选择**：Python zipfile 库
- 标准库，无需额外依赖
- 支持流式压缩
- 压缩率适中，速度快

**压缩级别**：
- 使用 `ZIP_DEFLATED` 压缩算法
- 压缩级别：6（默认，平衡速度和压缩率）

### 3. 流式传输

**选择**：FastAPI StreamingResponse
- 支持 chunked transfer encoding
- 内存占用低
- 支持大文件传输

**参数配置**：
- Chunk size：64 KB
- 超时：30 分钟
- 最大并发下载：根据服务器配置（建议 50-100）

## 安全考虑

### 1. 权限控制

**单个文档下载**：
- 验证用户是否有访问文档所在集合的权限
- 验证文档是否处于可访问状态（非 DELETED）

**知识库导出**：
- 只有集合所有者（creator）可以导出
- 验证集合是否处于 ACTIVE 状态

### 2. 防止滥用

**限流措施**：
- 每用户最多 3 个并发导出任务
- 下载速率限制（可选）：10 MB/s per user
- 同一集合导出间隔限制（可选）：5 分钟

**Quota 管理**：
- 记录用户导出次数
- 可设置每月导出次数限制
- 超出限制后需要升级套餐

### 3. 数据安全

**传输安全**：
- 强制使用 HTTPS
- JWT Token 验证
- 下载 URL 带有签名（可选）

**存储安全**：
- 临时文件存储在受保护的目录
- 文件权限：只有应用进程可访问
- 自动清理过期文件

## 相关文件索引

### 核心实现

- **View 层**：`aperag/views/export.py`
- **Service 层**：`aperag/service/export_service.py`
- **数据库模型**：`aperag/db/models.py`
- **Celery 任务**：`aperag/tasks/export.py`

### 对象存储

- **接口定义**：`aperag/objectstore/base.py`
- **Local 实现**：`aperag/objectstore/local.py`
- **S3 实现**：`aperag/objectstore/s3.py`

### 前端实现

- **文档列表**：`web/src/app/workspace/collections/[collectionId]/documents/page.tsx`
- **导出组件**：`web/src/components/documents/export-dialog.tsx`

## 总结

ApeRAG 的文档导出模块采用**后端代理 + 流式传输 + 异步打包**的架构设计：

**核心特性**：
1. ✅ **后端代理模式**：所有下载通过后端，不暴露对象存储链接，支持内网部署
2. ✅ **两种导出方式**：单个文档同步下载，知识库异步打包
3. ✅ **流式传输**：文件边读边写，内存占用低
4. ✅ **下载 URL**：异步任务完成后生成后端下载链接，可多次下载
5. ✅ **自动清理**：导出文件 7 天后自动删除，节省存储空间
6. ✅ **并发控制**：限制每用户最多 3 个并发导出任务
7. ✅ **进度追踪**：实时更新任务进度，提升用户体验
8. ✅ **审计日志**：记录所有下载操作，满足合规要求
9. ✅ **权限控制**：严格的访问权限验证，确保数据安全
10. ✅ **简单可靠**：采用成熟的异步任务方案，生产环境验证

**适用场景**：
- ✅ 内网部署环境
- ✅ 私有云场景
- ✅ 需要严格权限控制的企业环境
- ✅ 需要审计日志的合规环境

这种设计既保证了安全性和可控性，又提供了良好的性能和用户体验，特别适合企业级部署场景。
