# 知识库导出功能设计

**状态**: 已实现（MVP）

---

## 1. 背景与目标

ApeRAG 文档处理管线会将用户上传的原始文件（PDF、Word 等）解析为结构化知识内容，并将所有产物存储在对象存储中。这些内容有独立的使用价值：

- 将解析结果迁移到其他 RAG 框架（如 LlamaIndex、Dify）
- 审查文档解析质量，发现截断/格式错误
- 离线分析分块策略效果
- 数据备份与合规存档

本功能在知识库操作菜单中新增「**导出知识库**」按钮，允许知识库 Owner 将对象存储中该知识库目录的全部内容打包为 ZIP 文件下载。

**MVP 范围**：
- ✅ 触发导出（异步后台打包）
- ✅ 实时进度展示（前端轮询）
- ✅ 完成后一键下载 ZIP
- ❌ 后台运行（对话框保持打开直到完成或失败）
- ❌ 导出历史页面

---

## 2. 导出内容

对象存储中每个知识库的目录结构如下：

```
.objects/
└── user-{user_id}/
    └── {collection_id}/           ← 导出此前缀下的全部内容
        ├── {document_id_1}/
        │   ├── original.pdf
        │   ├── converted.pdf
        │   ├── processed_content.md
        │   ├── chunks/
        │   │   ├── chunk_0.json
        │   │   └── chunk_1.json
        │   └── images/
        │       ├── page_0.png
        │       └── page_1.png
        └── {document_id_2}/
            └── ...
```

**导出策略**：对 `user-{user_id}/{collection_id}/` 前缀下的所有对象做全量导出，无任何过滤。

生成的 ZIP 包结构：

```
{collection_title}_export_{YYYY-MM-DD}.zip
├── manifest.json              ← 元数据（id → 标题映射）
├── {document_id_1}/
│   └── ...（与对象存储目录结构一致）
└── {document_id_2}/
    └── ...
```

`manifest.json` 格式：

```json
{
  "schema_version": "1.0",
  "collection": {
    "id": "colff4f33902752abee",
    "title": "医学文献库",
    "exported_at": "2026-03-04T10:00:00Z"
  },
  "documents": [
    { "id": "doc_xyz789", "title": "高血压诊疗指南", "status": "COMPLETE" }
  ]
}
```

> `manifest.json` 仅作信息记录，不影响哪些文件被导出。导出的 ZIP 保存在对象存储的 `exports/user-{user_id}/export_{task_id}.zip`，7 天后由定时任务清理。

---

## 3. 权限

| 角色 | 能否导出 |
|------|---------|
| Collection Owner | ✅ |
| 订阅用户（Marketplace） | ❌ |
| 未登录用户 | ❌ |

- 「导出知识库」按钮**仅对 Owner 渲染**（在 `collection-header.tsx` 中用 `collection.user === currentUser.id` 判断）
- 后端通过查询 `Collection.user == user_id` 做权限校验，非 Owner 返回 `403`

---

## 4. 系统架构

### 整体流程

```
用户点击「导出知识库」
        │
        ▼
前端弹出确认对话框
        │ 点击「开始导出」
        ▼
POST /api/v1/collections/{id}/export
        │
        ├─► 校验 Owner 权限（非 Owner → 403）
        ├─► 检查并发任务数（同一用户 > 3 → 429）
        ├─► 创建 ExportTask（status=PENDING）
        └─► 触发 Celery 任务 export_collection_task.delay(task_id)
        │
        ▼ 202 Accepted: { export_task_id, status: "PENDING" }
        │
前端切换为进度对话框，每 2 秒轮询
GET /api/v1/export-tasks/{task_id}
        │
        ▼ status=COMPLETED
前端显示「下载 ZIP」按钮
        │
        ▼
GET /api/v1/export-tasks/{task_id}/download
（后端从对象存储读取 ZIP，StreamingResponse 流式返回）
```

### Celery Worker 处理流程（`config/export_tasks.py`）

```
1. 更新 ExportTask.status → PROCESSING
2. 列举对象存储 user-{user_id}/{collection_id}/ 下所有对象（list_objects_by_prefix）
3. 创建本地临时目录 /tmp/export_{task_id}/
4. 并发下载所有文件（最多 5 个并发，ThreadPoolExecutor）
   progress = downloaded / total * 85
5. 从数据库查文档列表，生成 manifest.json → /tmp/export_{task_id}/
   progress → 90%
6. 打包 ZIP（ZIP_DEFLATED）→ /tmp/export_{task_id}.zip
   progress → 95%
7. 上传 ZIP 到对象存储 exports/user-{user_id}/export_{task_id}.zip
   progress → 98%
8. 清理本地临时文件
9. 更新 ExportTask：
   status=COMPLETED, progress=100, file_size, gmt_completed, gmt_expires=now()+7d
   （失败时：status=FAILED, error_message=<traceback>）
```

---

## 5. 数据库

### `export_task` 表

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | VARCHAR(24) | PK，格式 `export` + random_id() |
| `user` | VARCHAR(256) | 用户 ID |
| `collection_id` | VARCHAR(24) | 所属知识库 |
| `status` | VARCHAR(32) | 见下表 |
| `progress` | INTEGER | 0–100 |
| `message` | TEXT | 展示给用户的进度文字 |
| `error_message` | TEXT | 失败时的错误详情 |
| `object_store_path` | TEXT | ZIP 在对象存储的路径 |
| `file_size` | BIGINT | ZIP 大小（字节） |
| `gmt_created` | TIMESTAMP | |
| `gmt_updated` | TIMESTAMP | |
| `gmt_completed` | TIMESTAMP | |
| `gmt_expires` | TIMESTAMP | 创建后 7 天 |

### 状态枚举

| 状态 | 说明 | 可下载 |
|------|------|--------|
| `PENDING` | 等待 Worker 处理 | ❌ |
| `PROCESSING` | 正在打包中 | ❌ |
| `COMPLETED` | 打包完成 | ✅ |
| `FAILED` | 打包失败 | ❌ |
| `EXPIRED` | ZIP 已被清理 | ❌ |

```
PENDING → PROCESSING → COMPLETED
                    └→ FAILED
COMPLETED / FAILED → EXPIRED（7 天后定时任务）
```

---

## 6. API

API 定义的源文件：
- Schema：`aperag/api/components/schemas/export.yaml`
- 路径：`aperag/api/paths/export.yaml`
- 注册：`aperag/api/openapi.yaml`

修改 API 后需运行：
```bash
make generate-models          # 重新生成 aperag/schema/view_models.py
make generate-frontend-sdk    # 重新生成 web/src/api/**
```

### 接口列表

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| `POST` | `/api/v1/collections/{collection_id}/export` | 创建导出任务 | Owner Only |
| `GET` | `/api/v1/export-tasks/{task_id}` | 查询任务状态 | 任务创建者 |
| `GET` | `/api/v1/export-tasks/{task_id}/download` | 下载 ZIP | 任务创建者 |

### POST /collections/{collection_id}/export

**Response 202**
```json
{ "export_task_id": "export6f918baaa28e0180", "status": "PENDING", "progress": 0 }
```

**错误码**

| 场景 | HTTP |
|------|------|
| 非 Owner | 403 |
| 集合不存在 | 404 |
| 并发任务超限（>3） | 429 |

### GET /export-tasks/{task_id}

**处理中**
```json
{ "export_task_id": "...", "status": "PROCESSING", "progress": 58, "message": "Downloading files: 87 / 150" }
```

**完成**
```json
{
  "export_task_id": "...", "status": "COMPLETED", "progress": 100,
  "download_url": "/api/v1/export-tasks/.../download",
  "file_size": 8388608,
  "gmt_expires": "2026-03-11T10:01:30Z"
}
```

### GET /export-tasks/{task_id}/download

流式返回 ZIP 文件，`Content-Disposition: attachment; filename="{title}_export_{date}.zip"`。

---

## 7. 前端

### 入口

`web/src/app/workspace/collections/[collectionId]/collection-header.tsx`

Owner 的操作下拉菜单中：

```
┌──────────────────────┐
│  发布至市场           │
│  导出知识库  ← 新增   │
│  ─────────────────── │
│  删除知识库           │
└──────────────────────┘
```

### 组件：`CollectionExport`

路径：`web/src/components/collections/export-dialog.tsx`

组件内部实现四个状态的对话框状态机：

```
confirm → processing → completed
                   └→ failed → confirm（点击「重试」）
```

- 使用自动生成的 SDK 方法调用接口：
  - `apiClient.defaultApi.createExportTask({ collectionId })`
  - `apiClient.defaultApi.getExportTask({ taskId })`（每 2 秒轮询一次）
- 下载时直接使用 `download_url` 字段（指向 download 接口）
- 进度对话框期间禁止关闭（屏蔽 ESC 和点击外部）

### i18n

翻译键均以 `export_knowledge_base_` 为前缀，定义在：
- `web/src/i18n/en-US/page_collections.json`
- `web/src/i18n/zh-CN/page_collections.json`

---

## 8. 性能与限制

| 限制项 | 值 |
|--------|---|
| 每用户最大并发导出任务数 | 3 |
| Worker 内并发文件下载数 | 5（ThreadPoolExecutor） |
| 导出文件保留时间 | 7 天 |

---

## 9. 相关文件索引

### 新增文件

| 文件 | 说明 |
|------|------|
| `aperag/api/components/schemas/export.yaml` | API Schema 定义（源文件） |
| `aperag/api/paths/export.yaml` | API 路径定义（源文件） |
| `aperag/db/models.py` | 新增 `ExportTask` 模型和 `ExportTaskStatus` 枚举 |
| `aperag/migration/versions/20260304120000-a1b2c3d4e5f6.py` | 数据库迁移 |
| `aperag/service/export_service.py` | 导出业务逻辑 |
| `aperag/views/export.py` | FastAPI 路由（3 个接口） |
| `config/export_tasks.py` | Celery 异步打包任务 |
| `web/src/components/collections/export-dialog.tsx` | 导出对话框组件 |
| `docs/zh-CN/design/collection_knowledge_export_design.md` | 本文档 |

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `aperag/objectstore/base.py` | 新增 `list_objects_by_prefix` 抽象方法 |
| `aperag/objectstore/local.py` | 实现 `list_objects_by_prefix` |
| `aperag/objectstore/s3.py` | 实现 `list_objects_by_prefix` |
| `aperag/api/openapi.yaml` | 注册新增的 3 个路径 |
| `aperag/schema/view_models.py` | 自动生成，含 `ExportTaskResponse` |
| `aperag/app.py` | 注册 `export_router` |
| `config/celery.py` | 注册 `config.export_tasks` 模块 |
| `web/src/api/**` | 自动生成的前端 SDK |
| `web/src/app/workspace/collections/[collectionId]/collection-header.tsx` | 接入导出按钮 |
| `web/src/i18n/en-US/page_collections.json` | 新增翻译键 |
| `web/src/i18n/zh-CN/page_collections.json` | 新增翻译键 |
