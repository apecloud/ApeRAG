---
title: 文档上传架构设计
description: 详细说明ApeRAG文档上传模块的完整架构设计，包括上传流程、临时存储配置、文档解析、格式转换、数据库设计等
keywords: [document upload, architecture, object store, parser, index building, two-phase commit]
---

# ApeRAG 文档上传架构设计

## 概述

本文档详细说明 ApeRAG 项目中文档上传模块的完整架构设计，涵盖从文件上传、临时存储、文档解析、格式转换到最终索引构建的全链路流程。

**核心设计理念**：采用**两阶段提交**模式，将文件上传（临时存储）和文档确认（正式添加）分离，提供更好的用户体验和资源管理能力。

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                             │
│                       (Next.js)                             │
└────────┬───────────────────────────────────┬────────────────┘
         │                                   │
         │ Step 1: Upload                    │ Step 2: Confirm
         │ POST /documents/upload            │ POST /documents/confirm
         ▼                                   ▼
┌─────────────────────────────────────────────────────────────┐
│  View Layer: aperag/views/collections.py                    │
│  - HTTP请求处理                                              │
│  - JWT身份验证                                               │
│  - 参数验证                                                  │
└────────┬───────────────────────────────────┬────────────────┘
         │                                   │
         │ document_service.upload_document() │ document_service.confirm_documents()
         ▼                                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Service Layer: aperag/service/document_service.py          │
│  - 业务逻辑编排                                              │
│  - 文件验证（类型、大小）                                     │
│  - SHA-256 哈希去重                                          │
│  - Quota 检查                                               │
│  - 事务管理                                                  │
└────────┬───────────────────────────────────┬────────────────┘
         │                                   │
         │ Step 1                            │ Step 2
         ▼                                   ▼
┌────────────────────────┐     ┌────────────────────────────┐
│  1. 创建 Document 记录  │     │  1. 更新 Document 状态     │
│     status=UPLOADED    │     │     UPLOADED → PENDING     │
│  2. 保存到 ObjectStore │     │  2. 创建 DocumentIndex 记录│
│  3. 计算 content_hash  │     │  3. 触发索引构建任务        │
└────────┬───────────────┘     └────────┬───────────────────┘
         │                              │
         ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                            │
│                                                             │
│  ┌───────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  PostgreSQL   │  │  Object Store    │  │  Vector DB  │ │
│  │               │  │                  │  │             │ │
│  │ - document    │  │ - Local/S3       │  │ - Qdrant    │ │
│  │ - document_   │  │ - 原始文件        │  │ - 向量索引  │ │
│  │   index       │  │ - 转换后的文件    │  │             │ │
│  └───────────────┘  └──────────────────┘  └─────────────┘ │
│                                                             │
│  ┌───────────────┐  ┌──────────────────┐                  │
│  │ Elasticsearch │  │   Neo4j/PG       │                  │
│  │               │  │                  │                  │
│  │ - 全文索引     │  │ - 知识图谱       │                  │
│  └───────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
               ┌───────────────────┐
               │  Celery Workers   │
               │                   │
               │  - 文档解析        │
               │  - 格式转换        │
               │  - 内容提取        │
               │  - 文档分块        │
               │  - 索引构建        │
               └───────────────────┘
```

### 分层架构

```
┌─────────────────────────────────────────────┐
│  View Layer (views/collections.py)         │  HTTP 处理、认证、参数验证
└─────────────────┬───────────────────────────┘
                  │ 调用
┌─────────────────▼───────────────────────────┐
│  Service Layer (service/document_service.py)│  业务逻辑、事务编排、权限控制
└─────────────────┬───────────────────────────┘
                  │ 调用
┌─────────────────▼───────────────────────────┐
│  Repository Layer (db/ops.py, objectstore/) │  数据访问抽象、对象存储接口
└─────────────────┬───────────────────────────┘
                  │ 访问
┌─────────────────▼───────────────────────────┐
│  Storage Layer (PG, S3, Qdrant, ES, Neo4j) │  数据持久化
└─────────────────────────────────────────────┘
```

## 核心流程详解

### 阶段 0: API 接口定义

系统提供三个主要接口：

1. **上传文件**（两阶段模式 - 第一步）
   - 接口：`POST /api/v1/collections/{collection_id}/documents/upload`
   - 功能：上传文件到临时存储，状态为 `UPLOADED`
   - 返回：`document_id`、`filename`、`size`、`status`

2. **确认文档**（两阶段模式 - 第二步）
   - 接口：`POST /api/v1/collections/{collection_id}/documents/confirm`
   - 功能：确认已上传的文档，触发索引构建
   - 参数：`document_ids` 数组
   - 返回：`confirmed_count`、`failed_count`、`failed_documents`

3. **一步上传**（传统模式，兼容旧版）
   - 接口：`POST /api/v1/collections/{collection_id}/documents`
   - 功能：上传并直接添加到知识库，状态直接为 `PENDING`
   - 支持批量上传

### 阶段 1: 文件上传与临时存储

#### 1.1 上传流程

```
用户选择文件
    │
    ▼
前端调用 upload API
    │
    ▼
View 层验证身份和参数
    │
    ▼
Service 层处理业务逻辑：
    │
    ├─► 验证集合存在且激活
    │
    ├─► 验证文件类型和大小
    │
    ├─► 读取文件内容
    │
    ├─► 计算 SHA-256 哈希
    │
    └─► 事务处理：
        │
        ├─► 重复检测（按文件名+哈希）
        │   ├─ 完全相同：返回已存在文档（幂等）
        │   ├─ 同名不同内容：抛出冲突异常
        │   └─ 新文档：继续创建
        │
        ├─► 创建 Document 记录（status=UPLOADED）
        │
        ├─► 上传到对象存储
        │   └─ 路径：user-{user_id}/{collection_id}/{document_id}/original{suffix}
        │
        └─► 更新文档元数据（object_path）
```

#### 1.2 文件验证

**支持的文件类型**：
- 文档：`.pdf`, `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx`
- 文本：`.txt`, `.md`, `.html`, `.json`, `.xml`, `.yaml`, `.yml`, `.csv`
- 图片：`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.tif`
- 音频：`.mp3`, `.wav`, `.m4a`
- 压缩包：`.zip`, `.tar`, `.gz`, `.tgz`

**大小限制**：
- 默认：100 MB（可通过 `MAX_DOCUMENT_SIZE` 环境变量配置）
- 解压后总大小：5 GB（`MAX_EXTRACTED_SIZE`）

#### 1.3 重复检测机制

采用**文件名 + SHA-256 哈希**双重检测：

| 场景 | 文件名 | 哈希值 | 系统行为 |
|------|--------|--------|----------|
| 完全相同 | 相同 | 相同 | 返回已存在文档（幂等操作） |
| 文件名冲突 | 相同 | 不同 | 抛出 `DocumentNameConflictException` |
| 新文档 | 不同 | - | 创建新文档记录 |

**优势**：
- ✅ 支持幂等上传：网络重传不会创建重复文档
- ✅ 避免内容冲突：同名不同内容会提示用户
- ✅ 节省存储空间：相同内容只存储一次

### 阶段 2: 临时存储配置

#### 2.1 对象存储类型

系统支持两种对象存储后端，可通过环境变量切换：

**1. Local 存储（本地文件系统）**

适用场景：
- 开发测试环境
- 小规模部署
- 单机部署

配置方式：
```bash
# 开发环境
OBJECT_STORE_TYPE=local
OBJECT_STORE_LOCAL_ROOT_DIR=.objects

# Docker 环境
OBJECT_STORE_TYPE=local
OBJECT_STORE_LOCAL_ROOT_DIR=/shared/objects
```

存储路径示例：
```
.objects/
└── user-google-oauth2-123456/
    └── col_abc123/
        └── doc_xyz789/
            ├── original.pdf              # 原始文件
            ├── converted.pdf             # 转换后的 PDF
            ├── processed_content.md      # 解析后的 Markdown
            ├── chunks/                   # 分块数据
            │   ├── chunk_0.json
            │   └── chunk_1.json
            └── images/                   # 提取的图片
                ├── page_0.png
                └── page_1.png
```

**2. S3 存储（兼容 AWS S3/MinIO/OSS 等）**

适用场景：
- 生产环境
- 大规模部署
- 分布式部署
- 需要高可用和容灾

配置方式：
```bash
OBJECT_STORE_TYPE=s3
OBJECT_STORE_S3_ENDPOINT=http://127.0.0.1:9000  # MinIO/S3 地址
OBJECT_STORE_S3_REGION=us-east-1                # AWS Region
OBJECT_STORE_S3_ACCESS_KEY=minioadmin           # Access Key
OBJECT_STORE_S3_SECRET_KEY=minioadmin           # Secret Key
OBJECT_STORE_S3_BUCKET=aperag                   # Bucket 名称
OBJECT_STORE_S3_PREFIX_PATH=dev/                # 可选的路径前缀
OBJECT_STORE_S3_USE_PATH_STYLE=true             # MinIO 需要设置为 true
```

#### 2.2 对象存储路径规则

**路径格式**：
```
{prefix}/user-{user_id}/{collection_id}/{document_id}/{filename}
```

**组成部分**：
- `prefix`：可选的全局前缀（仅 S3）
- `user_id`：用户 ID（`|` 替换为 `-`）
- `collection_id`：集合 ID
- `document_id`：文档 ID
- `filename`：文件名（如 `original.pdf`、`page_0.png`）

**多租户隔离**：
- 每个用户有独立的命名空间
- 每个集合有独立的存储目录
- 每个文档有独立的文件夹

### 阶段 3: 文档确认与索引构建

#### 3.1 确认流程

```
用户点击"保存到集合"
    │
    ▼
前端调用 confirm API
    │
    ▼
Service 层处理：
    │
    ├─► 验证集合配置
    │
    ├─► 检查 Quota（确认阶段才扣除配额）
    │
    └─► 对每个 document_id：
        │
        ├─► 验证文档状态为 UPLOADED
        │
        ├─► 更新文档状态：UPLOADED → PENDING
        │
        ├─► 根据集合配置创建索引记录：
        │   ├─ VECTOR（向量索引，必选）
        │   ├─ FULLTEXT（全文索引，必选）
        │   ├─ GRAPH（知识图谱，可选）
        │   ├─ SUMMARY（文档摘要，可选）
        │   └─ VISION（视觉索引，可选）
        │
        └─► 返回确认结果
    │
    ▼
触发 Celery 任务：reconcile_document_indexes
    │
    ▼
后台异步处理索引构建
```

#### 3.2 Quota（配额）管理

**检查时机**：
- ❌ 不在上传阶段检查（临时存储不占用配额）
- ✅ 在确认阶段检查（正式添加才消耗配额）

**配额类型**：

1. **用户全局配额**
   - `max_document_count`：用户总文档数量限制
   - 默认：1000（可通过 `MAX_DOCUMENT_COUNT` 配置）

2. **单集合配额**
   - `max_document_count_per_collection`：单个集合文档数量限制
   - 不计入 `UPLOADED` 和 `DELETED` 状态的文档

**配额超限处理**：
- 抛出 `QuotaExceededException`
- 返回 HTTP 400 错误
- 包含当前用量和配额上限信息

### 阶段 4: 文档解析与格式转换

#### 4.1 Parser 架构

系统采用**多 Parser 链式调用**架构，每个 Parser 负责特定类型的文件解析：

```
DocParser（主控制器）
    │
    ├─► MinerUParser
    │   └─ 功能：高精度 PDF 解析（商业 API）
    │   └─ 支持：.pdf
    │
    ├─► DocRayParser
    │   └─ 功能：文档布局分析和内容提取
    │   └─ 支持：.pdf, .docx, .pptx, .xlsx
    │
    ├─► ImageParser
    │   └─ 功能：图片内容识别（OCR + 视觉理解）
    │   └─ 支持：.jpg, .png, .gif, .bmp, .tiff
    │
    ├─► AudioParser
    │   └─ 功能：音频转录（Speech-to-Text）
    │   └─ 支持：.mp3, .wav, .m4a
    │
    └─► MarkItDownParser（兜底）
        └─ 功能：通用文档转 Markdown
        └─ 支持：几乎所有常见格式
```

#### 4.2 Parser 配置

**配置方式**：通过集合配置（Collection Config）动态控制

```json
{
  "parser_config": {
    "use_mineru": false,           // 是否启用 MinerU（需要 API Token）
    "use_doc_ray": false,          // 是否启用 DocRay
    "use_markitdown": true,        // 是否启用 MarkItDown（默认）
    "mineru_api_token": "xxx"      // MinerU API Token（可选）
  }
}
```

**环境变量配置**：
```bash
USE_MINERU_API=false              # 全局启用 MinerU
MINERU_API_TOKEN=your_token       # MinerU API Token
```

#### 4.3 解析流程

```
Celery Worker 收到索引任务
    │
    ▼
1. 从对象存储下载原始文件
    │
    ▼
2. 根据文件扩展名选择 Parser
    │
    ├─► 尝试第一个匹配的 Parser
    │   ├─ 成功：返回解析结果
    │   └─ 失败：FallbackError → 尝试下一个 Parser
    │
    └─► 最终兜底：MarkItDownParser
    │
    ▼
3. 解析结果（Parts）：
    │
    ├─► MarkdownPart：文本内容
    │   └─ 包含：标题、段落、列表、表格等
    │
    ├─► PdfPart：PDF 文件
    │   └─ 用于：线性化、页面渲染
    │
    └─► AssetBinPart：二进制资源
        └─ 包含：图片、嵌入的文件等
    │
    ▼
4. 后处理（Post-processing）：
    │
    ├─► PDF 页面转图片（Vision 索引需要）
    │   └─ 每页渲染为 PNG 图片
    │   └─ 保存到 {document_path}/images/page_N.png
    │
    ├─► PDF 线性化（加速浏览器加载）
    │   └─ 使用 pikepdf 优化 PDF 结构
    │   └─ 保存到 {document_path}/converted.pdf
    │
    └─► 提取文本内容（纯文本）
        └─ 合并所有 MarkdownPart 内容
        └─ 保存到 {document_path}/processed_content.md
    │
    ▼
5. 保存到对象存储
```

#### 4.4 格式转换示例

**示例 1：PDF 文档**
```
输入：user_manual.pdf (5 MB)
    │
    ▼
解析器选择：DocRayParser / MarkItDownParser
    │
    ▼
输出 Parts：
    ├─ MarkdownPart: "# User Manual\n\n## Chapter 1\n..."
    └─ PdfPart: <原始 PDF 数据>
    │
    ▼
后处理：
    ├─ 渲染 50 页为图片 → images/page_0.png ~ page_49.png
    ├─ 线性化 PDF → converted.pdf
    └─ 提取文本 → processed_content.md
```

**示例 2：图片文件**
```
输入：screenshot.png (2 MB)
    │
    ▼
解析器选择：ImageParser
    │
    ▼
输出 Parts：
    ├─ MarkdownPart: "[OCR 提取的文字内容]"
    └─ AssetBinPart: <原始图片数据> (vision_index=true)
    │
    ▼
后处理：
    └─ 保存原图副本 → images/file.png
```

**示例 3：音频文件**
```
输入：meeting_record.mp3 (50 MB)
    │
    ▼
解析器选择：AudioParser
    │
    ▼
输出 Parts：
    └─ MarkdownPart: "[转录的会议内容文本]"
    │
    ▼
后处理：
    └─ 保存转录文本 → processed_content.md
```

### 阶段 5: 索引构建

#### 5.1 索引类型与功能

| 索引类型 | 是否必选 | 功能描述 | 存储位置 |
|---------|---------|----------|----------|
| **VECTOR** | ✅ 必选 | 向量化检索，支持语义搜索 | Qdrant / Elasticsearch |
| **FULLTEXT** | ✅ 必选 | 全文检索，支持关键词搜索 | Elasticsearch |
| **GRAPH** | ❌ 可选 | 知识图谱，提取实体和关系 | Neo4j / PostgreSQL |
| **SUMMARY** | ❌ 可选 | 文档摘要，LLM 生成 | PostgreSQL (index_data) |
| **VISION** | ❌ 可选 | 视觉理解，图片内容分析 | Qdrant (向量) + PG (metadata) |

#### 5.2 索引构建流程

```
Celery Worker: reconcile_document_indexes 任务
    │
    ▼
1. 扫描 DocumentIndex 表，找到需要处理的索引
    │
    ├─► PENDING 状态 + observed_version < version
    │   └─ 需要创建或更新索引
    │
    └─► DELETING 状态
        └─ 需要删除索引
    │
    ▼
2. 按文档分组，逐个处理
    │
    ▼
3. 对每个文档：
    │
    ├─► parse_document（解析文档）
    │   ├─ 从对象存储下载原始文件
    │   ├─ 调用 DocParser 解析
    │   └─ 返回 ParsedDocumentData
    │
    └─► 对每个索引类型：
        │
        ├─► create_index (创建/更新索引)
        │   │
        │   ├─ VECTOR 索引：
        │   │   ├─ 文档分块（Chunking）
        │   │   ├─ Embedding 模型生成向量
        │   │   └─ 写入 Qdrant
        │   │
        │   ├─ FULLTEXT 索引：
        │   │   ├─ 提取纯文本内容
        │   │   ├─ 按段落/章节分块
        │   │   └─ 写入 Elasticsearch
        │   │
        │   ├─ GRAPH 索引：
        │   │   ├─ 使用 LightRAG 提取实体
        │   │   ├─ 提取实体间关系
        │   │   └─ 写入 Neo4j/PostgreSQL
        │   │
        │   ├─ SUMMARY 索引：
        │   │   ├─ 调用 LLM 生成摘要
        │   │   └─ 保存到 DocumentIndex.index_data
        │   │
        │   └─ VISION 索引：
        │       ├─ 提取图片 Assets
        │       ├─ Vision LLM 理解图片内容
        │       ├─ 生成图片描述向量
        │       └─ 写入 Qdrant
        │
        └─► 更新索引状态
            ├─ 成功：CREATING → ACTIVE
            └─ 失败：CREATING → FAILED
    │
    ▼
4. 更新文档总体状态
    │
    ├─ 所有索引都 ACTIVE → Document.status = COMPLETE
    ├─ 任一索引 FAILED → Document.status = FAILED
    └─ 部分索引仍在处理 → Document.status = RUNNING
```

#### 5.3 文档分块（Chunking）

**分块策略**：
- 递归字符分割（RecursiveCharacterTextSplitter）
- 按自然段落、章节优先切分
- 保留上下文重叠（Overlap）

**分块参数**：
```json
{
  "chunk_size": 1000,           // 每块最大字符数
  "chunk_overlap": 200,         // 重叠字符数
  "separators": ["\n\n", "\n", " ", ""]  // 分隔符优先级
}
```

**分块结果存储**：
```
{document_path}/chunks/
    ├─ chunk_0.json: {"text": "...", "metadata": {...}}
    ├─ chunk_1.json: {"text": "...", "metadata": {...}}
    └─ ...
```

## 数据库设计

### 表 1: document（文档元数据）

**表结构**：

| 字段名 | 类型 | 说明 | 索引 |
|--------|------|------|------|
| `id` | String(24) | 文档 ID，主键，格式：`doc{random_id}` | PK |
| `name` | String(1024) | 文件名 | - |
| `user` | String(256) | 用户 ID（支持多种 IDP） | ✅ Index |
| `collection_id` | String(24) | 所属集合 ID | ✅ Index |
| `status` | Enum | 文档状态（见下表） | ✅ Index |
| `size` | BigInteger | 文件大小（字节） | - |
| `content_hash` | String(64) | SHA-256 哈希（用于去重） | ✅ Index |
| `object_path` | Text | 对象存储路径（已废弃，用 doc_metadata） | - |
| `doc_metadata` | Text | 文档元数据（JSON 字符串） | - |
| `gmt_created` | DateTime(tz) | 创建时间（UTC） | - |
| `gmt_updated` | DateTime(tz) | 更新时间（UTC） | - |
| `gmt_deleted` | DateTime(tz) | 删除时间（软删除） | ✅ Index |

**唯一约束**：
```sql
UNIQUE INDEX uq_document_collection_name_active
  ON document (collection_id, name)
  WHERE gmt_deleted IS NULL;
```
- 同一集合内，活跃文档的名称不能重复
- 已删除的文档不参与唯一性检查

**文档状态枚举**（`DocumentStatus`）：

| 状态 | 说明 | 何时设置 | 可见性 |
|------|------|----------|--------|
| `UPLOADED` | 已上传到临时存储 | `upload_document` 接口 | 前端文件选择界面 |
| `PENDING` | 等待索引构建 | `confirm_documents` 接口 | 文档列表（处理中） |
| `RUNNING` | 索引构建中 | Celery 任务开始处理 | 文档列表（处理中） |
| `COMPLETE` | 所有索引完成 | 所有索引变为 ACTIVE | 文档列表（可用） |
| `FAILED` | 索引构建失败 | 任一索引失败 | 文档列表（失败） |
| `DELETED` | 已删除 | `delete_document` 接口 | 不可见（软删除） |
| `EXPIRED` | 临时文档过期 | 定时清理任务 | 不可见 |

**文档元数据示例**（`doc_metadata` JSON 字段）：
```json
{
  "object_path": "user-xxx/col_xxx/doc_xxx/original.pdf",
  "converted_path": "user-xxx/col_xxx/doc_xxx/converted.pdf",
  "processed_content_path": "user-xxx/col_xxx/doc_xxx/processed_content.md",
  "images": [
    "user-xxx/col_xxx/doc_xxx/images/page_0.png",
    "user-xxx/col_xxx/doc_xxx/images/page_1.png"
  ],
  "parser_used": "DocRayParser",
  "parse_duration_ms": 5420,
  "page_count": 50,
  "custom_field": "value"
}
```

### 表 2: document_index（索引状态管理）

**表结构**：

| 字段名 | 类型 | 说明 | 索引 |
|--------|------|------|------|
| `id` | Integer | 自增 ID，主键 | PK |
| `document_id` | String(24) | 关联的文档 ID | ✅ Index |
| `index_type` | Enum | 索引类型（见下表） | ✅ Index |
| `status` | Enum | 索引状态（见下表） | ✅ Index |
| `version` | Integer | 索引版本号 | - |
| `observed_version` | Integer | 已处理的版本号 | - |
| `index_data` | Text | 索引数据（JSON），如摘要内容 | - |
| `error_message` | Text | 错误信息（失败时） | - |
| `gmt_created` | DateTime(tz) | 创建时间 | - |
| `gmt_updated` | DateTime(tz) | 更新时间 | - |
| `gmt_last_reconciled` | DateTime(tz) | 最后协调时间 | - |

**唯一约束**：
```sql
UNIQUE CONSTRAINT uq_document_index
  ON document_index (document_id, index_type);
```
- 每个文档的每种索引类型只有一条记录

**索引类型枚举**（`DocumentIndexType`）：

| 类型 | 值 | 说明 | 外部存储 |
|------|-----|------|----------|
| `VECTOR` | "VECTOR" | 向量索引 | Qdrant / Elasticsearch |
| `FULLTEXT` | "FULLTEXT" | 全文索引 | Elasticsearch |
| `GRAPH` | "GRAPH" | 知识图谱 | Neo4j / PostgreSQL |
| `SUMMARY` | "SUMMARY" | 文档摘要 | PostgreSQL (index_data) |
| `VISION` | "VISION" | 视觉索引 | Qdrant + PostgreSQL |

**索引状态枚举**（`DocumentIndexStatus`）：

| 状态 | 说明 | 何时设置 |
|------|------|----------|
| `PENDING` | 等待处理 | `confirm_documents` 创建索引记录 |
| `CREATING` | 创建中 | Celery Worker 开始处理 |
| `ACTIVE` | 就绪可用 | 索引构建成功 |
| `DELETING` | 标记删除 | `delete_document` 接口 |
| `DELETION_IN_PROGRESS` | 删除中 | Celery Worker 正在删除 |
| `FAILED` | 失败 | 索引构建失败 |

**版本控制机制**：
- `version`：期望的索引版本（每次文档更新时 +1）
- `observed_version`：已处理的版本号
- `version > observed_version` 时，触发索引更新

**协调器（Reconciler）**：
```python
# 查询需要处理的索引
SELECT * FROM document_index
WHERE status = 'PENDING'
  AND observed_version < version;

# 处理后更新
UPDATE document_index
SET status = 'ACTIVE',
    observed_version = version,
    gmt_last_reconciled = NOW()
WHERE id = ?;
```

### 表关系图

```
┌─────────────────────────────────┐
│         collection              │
│  ─────────────────────────────  │
│  id (PK)                        │
│  name                           │
│  config (JSON)                  │
│  status                         │
│  ...                            │
└────────────┬────────────────────┘
             │ 1:N
             ▼
┌─────────────────────────────────┐
│          document               │
│  ─────────────────────────────  │
│  id (PK)                        │
│  collection_id (FK)             │◄──── 唯一约束: (collection_id, name)
│  name                           │
│  user                           │
│  status (Enum)                  │
│  size                           │
│  content_hash (SHA-256)         │
│  doc_metadata (JSON)            │
│  gmt_created                    │
│  gmt_deleted                    │
│  ...                            │
└────────────┬────────────────────┘
             │ 1:N
             ▼
┌─────────────────────────────────┐
│       document_index            │
│  ─────────────────────────────  │
│  id (PK)                        │
│  document_id (FK)               │◄──── 唯一约束: (document_id, index_type)
│  index_type (Enum)              │
│  status (Enum)                  │
│  version                        │
│  observed_version               │
│  index_data (JSON)              │
│  error_message                  │
│  gmt_last_reconciled            │
│  ...                            │
└─────────────────────────────────┘
```

## 状态机与生命周期

### 文档状态转换

```
         ┌─────────────────────────────────────────────┐
         │                                             │
         │                                             ▼
    [上传文件] ──► UPLOADED ──► [确认] ──► PENDING ──► RUNNING ──► COMPLETE
                     │                                   │
                     │                                   ▼
                     │                                FAILED
                     │                                   │
                     │                                   ▼
                     └──────► [删除] ──────────────► DELETED
                                                         │
                     ┌───────────────────────────────────┘
                     │
                     ▼
                  EXPIRED (定时清理未确认的文档)
```

**关键转换**：
1. **UPLOADED → PENDING**：用户点击"保存到集合"
2. **PENDING → RUNNING**：Celery Worker 开始处理
3. **RUNNING → COMPLETE**：所有索引都成功
4. **RUNNING → FAILED**：任一索引失败
5. **任何状态 → DELETED**：用户删除文档

### 索引状态转换

```
  [创建索引记录] ──► PENDING ──► CREATING ──► ACTIVE
                                   │
                                   ▼
                                FAILED
                                   │
                                   ▼
                     ┌──────────► PENDING (重试)
                     │
    [删除请求] ──────┼──────────► DELETING ──► DELETION_IN_PROGRESS ──► (记录删除)
                     │
                     └──────────► (直接删除记录，如果 PENDING/FAILED)
```

## 异步任务调度（Celery）

### 任务定义

**主任务**：`reconcile_document_indexes`
- 触发时机：
  - `confirm_documents` 接口调用后
  - 定时任务（每 30 秒）
  - 手动触发（管理界面）
- 功能：扫描 `document_index` 表，处理需要协调的索引

**子任务**：
- `parse_document_task`：解析文档内容
- `create_vector_index_task`：创建向量索引
- `create_fulltext_index_task`：创建全文索引
- `create_graph_index_task`：创建知识图谱索引
- `create_summary_index_task`：创建摘要索引
- `create_vision_index_task`：创建视觉索引

### 任务调度策略

**并发控制**：
- 每个 Worker 最多同时处理 N 个文档（默认 4）
- 每个文档的多个索引可以并行构建
- 使用 Celery 的 `task_acks_late=True` 确保任务不丢失

**失败重试**：
- 最多重试 3 次
- 指数退避（1分钟 → 5分钟 → 15分钟）
- 3 次失败后标记为 `FAILED`

**幂等性**：
- 所有任务支持重复执行
- 使用 `observed_version` 机制避免重复处理
- 相同输入产生相同输出

## 设计特点与优势

### 1. 两阶段提交设计

**优势**：
- ✅ **用户体验更好**：快速上传响应，不阻塞用户操作
- ✅ **选择性添加**：批量上传后可选择性确认部分文件
- ✅ **资源控制合理**：未确认的文档不构建索引，不消耗配额
- ✅ **故障恢复友好**：临时文档可以定期清理，不影响业务

**状态隔离**：
```
临时状态（UPLOADED）：
  - 不计入配额
  - 不触发索引
  - 可以被自动清理

正式状态（PENDING/RUNNING/COMPLETE）：
  - 计入配额
  - 触发索引构建
  - 不会被自动清理
```

### 2. 幂等性设计

**文件级别幂等**：
- SHA-256 哈希去重
- 相同文件多次上传返回同一 `document_id`
- 避免存储空间浪费

**接口级别幂等**：
- `upload_document`：重复上传返回已存在文档
- `confirm_documents`：重复确认不会创建重复索引
- `delete_document`：重复删除返回成功（软删除）

### 3. 多租户隔离

**存储隔离**：
```
user-{user_A}/...  # 用户 A 的文件
user-{user_B}/...  # 用户 B 的文件
```

**数据库隔离**：
- 所有查询都带 `user` 字段过滤
- 集合级别的权限控制（`collection.user`）
- 软删除支持（`gmt_deleted`）

### 4. 灵活的存储后端

**统一接口**：
```python
AsyncObjectStore:
  - put(path, data)
  - get(path)
  - delete_objects_by_prefix(prefix)
```

**运行时切换**：
- 通过环境变量切换 Local/S3
- 无需修改业务代码
- 支持自定义存储后端（实现接口即可）

### 5. 事务一致性

**数据库 + 对象存储的两阶段提交**：
```python
async with transaction:
    # 1. 创建数据库记录
    document = create_document_record()
    
    # 2. 上传到对象存储
    await object_store.put(path, data)
    
    # 3. 更新元数据
    document.doc_metadata = json.dumps(metadata)
    
    # 所有操作成功才提交，任一失败则回滚
```

**失败处理**：
- 数据库记录创建失败：不上传文件
- 文件上传失败：回滚数据库记录
- 元数据更新失败：回滚前面的操作

### 6. 可观测性

**审计日志**：
- `@audit` 装饰器记录所有文档操作
- 包含：用户、时间、操作类型、资源 ID

**任务追踪**：
- `gmt_last_reconciled`：最后处理时间
- `error_message`：失败原因
- Celery 任务 ID：关联日志追踪

**监控指标**：
- 文档上传速率
- 索引构建耗时
- 失败率统计

## 性能优化

### 1. 异步处理

**上传不阻塞**：
- 文件上传到对象存储后立即返回
- 索引构建在 Celery 中异步执行
- 前端通过轮询或 WebSocket 获取进度

### 2. 批量操作

**批量确认**：
```python
confirm_documents(document_ids=[id1, id2, ..., idN])
```
- 一次事务处理多个文档
- 批量创建索引记录
- 减少数据库往返

### 3. 缓存策略

**解析结果缓存**：
- 解析后的内容保存到 `processed_content.md`
- 后续索引重建可直接读取，无需重新解析

**分块结果缓存**：
- 分块结果保存到 `chunks/` 目录
- 向量索引重建可复用分块结果

### 4. 并行索引构建

**多索引并行**：
```python
# VECTOR、FULLTEXT、GRAPH 可以并行构建
await asyncio.gather(
    create_vector_index(),
    create_fulltext_index(),
    create_graph_index()
)
```

## 错误处理

### 常见异常

| 异常类型 | HTTP 状态码 | 触发场景 | 处理建议 |
|---------|------------|----------|----------|
| `ResourceNotFoundException` | 404 | 集合/文档不存在 | 检查 ID 是否正确 |
| `CollectionInactiveException` | 400 | 集合未激活 | 等待集合初始化完成 |
| `DocumentNameConflictException` | 409 | 同名不同内容 | 重命名文件或删除旧文档 |
| `QuotaExceededException` | 429 | 配额超限 | 升级套餐或删除旧文档 |
| `InvalidFileTypeException` | 400 | 不支持的文件类型 | 查看支持的文件类型列表 |
| `FileSizeTooLargeException` | 413 | 文件过大 | 分割文件或压缩 |

### 异常传播

```
Service Layer 抛出异常
    │
    ▼
View Layer 捕获并转换
    │
    ▼
Exception Handler 统一处理
    │
    ▼
返回标准 JSON 响应：
{
  "error_code": "QUOTA_EXCEEDED",
  "message": "Document count limit exceeded",
  "details": {
    "limit": 1000,
    "current": 1000
  }
}
```

## 相关文件索引

### 核心实现

- **View 层**：`aperag/views/collections.py` - HTTP 接口定义
- **Service 层**：`aperag/service/document_service.py` - 业务逻辑
- **数据库模型**：`aperag/db/models.py` - Document, DocumentIndex 表定义
- **数据库操作**：`aperag/db/ops.py` - CRUD 操作封装

### 对象存储

- **接口定义**：`aperag/objectstore/base.py` - AsyncObjectStore 抽象类
- **Local 实现**：`aperag/objectstore/local.py` - 本地文件系统存储
- **S3 实现**：`aperag/objectstore/s3.py` - S3 兼容存储

### 文档解析

- **主控制器**：`aperag/docparser/doc_parser.py` - DocParser
- **Parser 实现**：
  - `aperag/docparser/mineru_parser.py` - MinerU PDF 解析
  - `aperag/docparser/docray_parser.py` - DocRay 文档解析
  - `aperag/docparser/markitdown_parser.py` - MarkItDown 通用解析
  - `aperag/docparser/image_parser.py` - 图片 OCR
  - `aperag/docparser/audio_parser.py` - 音频转录
- **文档处理**：`aperag/index/document_parser.py` - 解析流程编排

### 索引构建

- **索引管理**：`aperag/index/manager.py` - DocumentIndexManager
- **向量索引**：`aperag/index/vector_index.py` - VectorIndexer
- **全文索引**：`aperag/index/fulltext_index.py` - FulltextIndexer
- **知识图谱**：`aperag/index/graph_index.py` - GraphIndexer
- **文档摘要**：`aperag/index/summary_index.py` - SummaryIndexer
- **视觉索引**：`aperag/index/vision_index.py` - VisionIndexer

### 任务调度

- **任务定义**：`config/celery_tasks.py` - Celery 任务注册
- **协调器**：`aperag/tasks/reconciler.py` - DocumentIndexReconciler
- **文档任务**：`aperag/tasks/document.py` - DocumentIndexTask

### 前端实现

- **文档列表**：`web/src/app/workspace/collections/[collectionId]/documents/page.tsx`
- **文档上传**：`web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx`

## 总结

ApeRAG 的文档上传模块采用**两阶段提交 + 多 Parser 链式调用 + 多索引并行构建**的架构设计：

**核心特性**：
1. ✅ **两阶段提交**：上传（临时存储）→ 确认（正式添加），提供更好的用户体验
2. ✅ **SHA-256 去重**：避免重复文档，支持幂等上传
3. ✅ **灵活存储后端**：Local/S3 可配置切换，统一接口抽象
4. ✅ **多 Parser 架构**：支持 MinerU、DocRay、MarkItDown 等多种解析器
5. ✅ **格式自动转换**：PDF→图片、音频→文本、图片→OCR 文本
6. ✅ **多索引协调**：向量、全文、图谱、摘要、视觉五种索引类型
7. ✅ **配额管理**：确认阶段才扣除配额，合理控制资源
8. ✅ **异步处理**：Celery 任务队列，不阻塞用户操作
9. ✅ **事务一致性**：数据库 + 对象存储的两阶段提交
10. ✅ **可观测性**：审计日志、任务追踪、错误信息完整记录

这种设计既保证了高性能和可扩展性，又支持复杂的文档处理场景（多格式、多语言、多模态），同时具有良好的容错能力和用户体验。
