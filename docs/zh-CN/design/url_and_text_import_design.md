---
title: URL 与文本导入设计
position: 4
---

# Collection 文档导入扩展：URL 抓取与文本粘贴

## 概述

本文档描述在 ApeRAG Collection 中新增两种文档来源方式的设计：

1. **URL 导入**：用户输入网址，系统自动调用 `web/read` 接口抓取页面内容，生成 Markdown 文件，走现有两阶段上传流程入库。
2. **文本导入**：用户在前端粘贴文本，前端直接将其封装为 `.txt` 文件，调用现有上传接口，**完全无需新增后端代码**。

两种方式都只是"给现有上传流程提供文件内容的方式"，confirm 及后续索引构建完全复用现有逻辑。

> **范围说明**：本期不包含"根据文字搜索网络并导入"功能，但架构设计保留此扩展空间。

---

## 设计原则

> URL 抓取和文本粘贴只是"选择文件"的替代方式。

一旦内容到手（Markdown 字符串 / 文本字符串），它就被包装成一个虚拟文件，走与普通文件上传完全相同的路径：

```
[来源]                    [统一入口]               [后续流程（不变）]
文件选择    ──────────► upload_document()  ──► UPLOADED ──► confirm ──► 索引构建
URL 抓取   ──────────►（虚拟 UploadFile）
文本粘贴   ──────────►（前端 File 对象）
```

---

## 现状与可复用组件

### 现有两阶段上传流程

```
Step 1: POST /collections/{id}/documents/upload   → status = UPLOADED（临时）
Step 2: POST /collections/{id}/documents/confirm  → status = PENDING → 触发索引构建
```

URL 导入和文本导入都将产出 `UPLOADED` 状态的文档，与文件上传无缝衔接。

### 关键可复用组件

| 组件 | 位置 | 如何复用 |
|------|------|---------|
| 文档上传服务 | `aperag/service/document_service.py` → `upload_document()` | URL 导入后端调用此方法存储抓取内容 |
| 文档确认服务 | `document_service.confirm_documents()` | 完全不变 |
| Web Read 接口 | `POST /api/v1/web/read`（`aperag/views/web.py`） | URL 导入后端通过 HTTP 调用此接口 |
| ReaderService | `aperag/websearch/reader/reader_service.py` | web/read 的底层实现（JINA + Trafilatura fallback） |
| 文档列表页暂存区 | `document-upload.tsx` | URL/文本产出的 `UPLOADED` 文档自动出现在此列表 |

---

## 架构设计

### 总体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                         │
│                                                                 │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │  📁 选择文件  │  │   🔗 输入网址   │  │   📋 粘贴文字    │  │
│  │  (现有)      │  │   (新增)        │  │   (新增)         │  │
│  └──────┬───────┘  └────────┬────────┘  └────────┬─────────┘  │
│         │                   │                    │             │
│         │           POST /fetch-url              │             │
│         │                   │        new File([text], "x.txt") │
│         │                   │                    │             │
│         └───────────────────┴────────────────────┘            │
│                             │                                   │
│                    POST /documents/upload                       │
│                   （现有接口，所有来源统一入口）                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   document_service            │
              │   .upload_document()          │
              │   status = UPLOADED           │
              └───────────────┬───────────────┘
                              │
                   用户在暂存区点击"保存到知识库"
                              │
                    POST /documents/confirm
                              │
                              ▼
              ┌───────────────────────────────┐
              │   document_service            │
              │   .confirm_documents()        │
              │   status: UPLOADED → PENDING  │
              │   创建 DocumentIndex 记录      │
              │   触发 reconcile 任务          │
              └───────────────────────────────┘
```

---

## URL 导入详细设计

### 新增后端接口

**接口**：`POST /api/v1/collections/{collection_id}/documents/fetch-url`

此接口是本次需求唯一新增的后端接口。

**职责**：
1. 接收用户提交的 URL 列表
2. 通过 HTTP 调用内部 `POST /api/v1/web/read` 接口抓取页面内容
3. 将每个 URL 的抓取结果包装为虚拟 `UploadFile` 对象
4. 调用现有 `document_service.upload_document()` 存储到对象存储（status=UPLOADED）
5. 返回创建的 `document_id` 列表，前端将其合并到暂存区

**Request Body**：

```json
{
  "urls": [
    "https://example.com/article1",
    "https://example.com/article2"
  ]
}
```

| 字段 | 类型 | 必填 | 约束 |
|------|------|------|------|
| `urls` | `string[]` | ✅ | 1~10 个，必须是合法 http/https URL |

**Response**（200 OK）：

```json
{
  "documents": [
    {
      "id": "doc_abc123",
      "name": "示例文章标题.md",
      "status": "UPLOADED",
      "size": 8192,
      "url": "https://example.com/article1",
      "fetch_status": "success"
    },
    {
      "id": null,
      "name": null,
      "status": null,
      "url": "https://example.com/article2",
      "fetch_status": "error",
      "error": "页面无法访问（403）"
    }
  ],
  "total": 2,
  "succeeded": 1,
  "failed": 1
}
```

**说明**：
- 接口同步执行（URL 数量限制在 10 个以内，抓取过程在接口请求内完成）
- 部分 URL 失败不影响其他 URL，前端针对失败项目展示错误信息
- 成功的文档处于 `UPLOADED` 状态，出现在前端暂存区供用户确认

### 后端实现逻辑

在 `aperag/views/collections.py` 中新增路由函数（约 60 行）：

```python
@router.post("/collections/{collection_id}/documents/fetch-url", tags=["documents"])
@audit(resource_type="document", api_name="FetchUrlDocument")
async def fetch_url_document_view(
    request: Request,
    collection_id: str,
    body: view_models.FetchUrlRequest,
    user: User = Depends(required_user),
) -> view_models.FetchUrlResponse:
    """
    Fetch web page content from URLs and create UPLOADED documents.

    Internally calls POST /api/v1/web/read to retrieve page content,
    then wraps each result as a virtual UploadFile and calls
    document_service.upload_document() to persist as UPLOADED documents.
    """
    results = []

    # Step 1: Call web/read service layer (via HTTP to /api/v1/web/read)
    web_read_request = WebReadRequest(url_list=body.urls, timeout=30)
    web_read_response = await _call_web_read(web_read_request, user)

    # Step 2: For each result, wrap as UploadFile and call upload_document()
    for item in web_read_response.results:
        if item.status != "success" or not item.content:
            results.append(FetchUrlResultItem(
                url=item.url,
                fetch_status="error",
                error=item.error or "Failed to fetch content",
            ))
            continue

        # Determine filename from page title or URL
        filename = _url_to_filename(item.title, item.url)

        # Wrap Markdown content as a virtual UploadFile
        virtual_file = _make_upload_file(filename, item.content.encode("utf-8"))

        try:
            doc = await document_service.upload_document(
                user=str(user.id),
                collection_id=collection_id,
                file=virtual_file,
                extra_metadata={"source_url": item.url, "source_type": "url"},
            )
            results.append(FetchUrlResultItem(
                url=item.url,
                fetch_status="success",
                document=doc,
            ))
        except Exception as e:
            results.append(FetchUrlResultItem(
                url=item.url,
                fetch_status="error",
                error=str(e),
            ))

    return FetchUrlResponse(
        documents=results,
        total=len(results),
        succeeded=sum(1 for r in results if r.fetch_status == "success"),
        failed=sum(1 for r in results if r.fetch_status == "error"),
    )
```

### 调用 web/read 的方式

采用**调用 Service 层**而非发起内部 HTTP 请求，直接复用 `web.py` 中的 `_read_with_jina_fallback` / `_read_with_trafilatura_only` 私有函数（或将其提取为 `reader_service` 的共享方法）：

```python
async def _call_web_read(request: WebReadRequest, user: User) -> WebReadResponse:
    """Call web read service layer, with JINA + Trafilatura fallback."""
    jina_api_key = await _get_user_jina_api_key(user)
    if jina_api_key:
        return await _read_with_jina_fallback(request, jina_api_key)
    else:
        return await _read_with_trafilatura_only(request)
```

> 这些私有函数已在 `aperag/views/web.py` 中实现，将其提取到 `aperag/websearch/reader/reader_service.py` 的公共方法即可被两处复用，保持模块化边界。

---

## 文本导入详细设计

### 零后端改动

文本导入**不需要新增任何后端接口**。前端在客户端将用户粘贴的文本封装为标准的 `File` 对象，调用现有上传接口：

```typescript
// web/src/app/.../import/text-import.tsx

const handleImport = async () => {
  const filename = title.trim() ? `${title.trim()}.txt` : `note-${Date.now()}.txt`;
  const file = new File([textContent], filename, { type: "text/plain" });

  // Reuse existing upload API — no new backend endpoint needed
  const response = await apiClient.defaultApi.collectionsCollectionIdDocumentsUploadPost({
    collectionId: collection.id,
    file,
  });

  // Add to staging area (same as file upload)
  onDocumentUploaded(response.data);
};
```

这样文本文档与文件上传产出完全相同的结果（`UPLOADED` 状态的 Document），在暂存区中一视同仁。

---

## 前端设计

### 入口 Dialog

在文档列表页的"添加文档"按钮点击后，显示来源选择 Dialog：

```
┌──────────────────────────────────────────────────┐
│  向知识库中添加文档                          [×]  │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌────────────────┐  ┌──────────┐  ┌──────────┐ │
│  │  📁            │  │  🔗      │  │  📋      │ │
│  │  上传文件       │  │  网址    │  │  粘贴文字 │ │
│  └────────────────┘  └──────────┘  └──────────┘ │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 网址导入表单（`url-import.tsx`）

```
┌──────────────────────────────────────────────────┐
│  ← 网址导入                                [×]   │
├──────────────────────────────────────────────────┤
│  粘贴网址，系统将自动抓取页面内容导入知识库。        │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │ https://example.com/article1             │   │
│  │ https://example.com/article2             │   │
│  │                                          │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  • 每行输入一个网址，最多 10 个                    │
│  • 仅支持公开可访问的网页                          │
│  • 需要登录的页面无法抓取                          │
│                                                  │
│                         [取消]  [抓取并添加]       │
└──────────────────────────────────────────────────┘
```

点击"抓取并添加"后：
1. 调用 `POST /collections/{id}/documents/fetch-url`
2. 成功的 URL 对应文档出现在上传暂存区（同文件上传）
3. 失败的 URL 在 Dialog 内以红色错误信息展示
4. Dialog 关闭，用户在暂存区一起 confirm

### 粘贴文字表单（`text-import.tsx`）

```
┌──────────────────────────────────────────────────┐
│  ← 粘贴文字                                [×]   │
├──────────────────────────────────────────────────┤
│  粘贴文字内容，即可将其导入知识库。                 │
│                                                  │
│  标题（可选）                                     │
│  ┌──────────────────────────────────────────┐   │
│  │ 我的笔记                                  │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  内容                                            │
│  ┌──────────────────────────────────────────┐   │
│  │ 在此处粘贴文字…                           │   │
│  │                                          │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│                         [取消]  [添加]            │
└──────────────────────────────────────────────────┘
```

点击"添加"后：
1. 前端创建 `new File([content], "${title}.txt")` 对象
2. 调用现有 `POST /documents/upload` 接口（无感知）
3. 文档出现在上传暂存区
4. Dialog 关闭，用户在暂存区 confirm

### 前端暂存区（扩展现有 `document-upload.tsx`）

现有暂存区已支持展示所有 `UPLOADED` 状态文档。URL/文本产出的文档与文件上传文档合并展示，confirm 操作完全不变：

```
┌─────────────────────────────────────────────────────────────┐
│  暂存区（待确认）                                             │
├───────────────────────────────────────┬──────┬─────────────┤
│ 文档名                                 │ 大小 │ 状态        │
├───────────────────────────────────────┼──────┼─────────────┤
│ 📄 user_manual.pdf                    │ 5 MB │ ✅ 已上传    │
│ 🌐 示例文章标题.md（来自 URL）          │ 8 KB │ ✅ 已上传    │
│ 📝 我的笔记.txt                        │ 2 KB │ ✅ 已上传    │
│ 🌐 example.com/article2（来自 URL）    │  —   │ ❌ 抓取失败  │
└───────────────────────────────────────┴──────┴─────────────┘
                                    [保存到知识库（3 个文档）]
```

### 前端组件结构

```
web/src/app/workspace/collections/[collectionId]/documents/
├── page.tsx                                  # 文档列表页（现有，加"添加"按钮入口）
└── upload/
    ├── page.tsx                              # 上传页（现有）
    ├── document-upload.tsx                   # 暂存区组件（现有，几乎不改）
    └── import/
        ├── import-dialog.tsx                 # 来源选择 Dialog（新增）
        ├── url-import.tsx                    # URL 输入表单（新增）
        └── text-import.tsx                   # 文本粘贴表单（新增）
```

---

## 新增代码量统计

| 位置 | 改动类型 | 估计行数 |
|------|---------|---------|
| `aperag/api/components/schemas/document.yaml` | 新增 `fetchUrlRequest/Response` schema | ~40 行 |
| `aperag/api/paths/collections.yaml` | 新增 `/fetch-url` 路径 | ~30 行 |
| `aperag/views/collections.py` | 新增一个路由函数 | ~60 行 |
| `aperag/websearch/reader/reader_service.py` | 将私有函数提取为公共方法 | ~20 行重构 |
| `web/src/...import/import-dialog.tsx` | 新建组件 | ~60 行 |
| `web/src/...import/url-import.tsx` | 新建组件 | ~80 行 |
| `web/src/...import/text-import.tsx` | 新建组件 | ~70 行 |
| `web/src/.../document-upload.tsx` | 增加"添加来源"入口触发 | ~10 行 |
| i18n 文件 | 新增翻译 key | ~20 行 |
| **合计** | | **~390 行** |

**不需要改动的部分（完全复用）**：
- `document_service.upload_document()` — 无改动
- `document_service.confirm_documents()` — 无改动
- Celery 任务 / 索引构建流程 — 无改动
- 文档列表页、暂存区主逻辑 — 几乎无改动

---

## API Schema 定义

在 `aperag/api/components/schemas/document.yaml` 中新增：

```yaml
fetchUrlRequest:
  type: object
  properties:
    urls:
      type: array
      items:
        type: string
        format: uri
      minItems: 1
      maxItems: 10
      description: List of URLs to fetch content from
      example:
        - "https://example.com/article1"
        - "https://example.com/article2"
  required:
    - urls

fetchUrlResultItem:
  type: object
  properties:
    url:
      type: string
      description: The source URL
    fetch_status:
      type: string
      enum: ["success", "error"]
    document:
      $ref: '#/Document'
      description: Created document (only present on success)
    error:
      type: string
      description: Error message (only present on failure)
  required:
    - url
    - fetch_status

fetchUrlResponse:
  type: object
  properties:
    documents:
      type: array
      items:
        $ref: '#/fetchUrlResultItem'
    total:
      type: integer
    succeeded:
      type: integer
    failed:
      type: integer
  required:
    - documents
    - total
    - succeeded
    - failed
```

---

## 错误处理

| 场景 | 处理位置 | 处理方式 |
|------|---------|---------|
| URL 格式非法（非 http/https） | 后端校验 | 400，跳过该 URL |
| URL 数量 > 10 | 后端校验 | 400，整体拒绝 |
| URL 页面无法访问（4xx/5xx） | web/read 返回 error | 在响应中标记该 URL 失败 |
| 抓取超时（>30s） | web/read 超时 | 同上 |
| 文档名冲突 | `upload_document()` 抛出异常 | 自动追加序号或返回已存在文档（幂等） |
| 配额超限 | `upload_document()` 抛出异常 | 400，停止处理剩余 URL |
| 文本内容为空 | 前端校验 | 禁用"添加"按钮 |

---

## 数据完整性

URL 导入的文档在 `doc_metadata` 中记录来源信息，便于追溯和未来的定时刷新功能：

```json
{
  "source_type": "url",
  "source_url": "https://example.com/article",
  "page_title": "示例文章标题",
  "fetched_at": "2026-03-05T10:00:00Z",
  "object_path": "user-xxx/col_xxx/doc_xxx/original.md"
}
```

文本导入的文档：

```json
{
  "source_type": "text",
  "object_path": "user-xxx/col_xxx/doc_xxx/original.txt"
}
```

---

## 与现有功能对比

| 维度 | 文件上传 | URL 导入 | 文本导入 |
|------|----------|---------|---------|
| 内容获取方式 | 用户本地文件 | 后端调用 web/read 服务抓取 | 前端直接创建 File 对象 |
| 新增后端接口 | — | 1 个（`/fetch-url`） | **0 个** |
| 初始文档状态 | `UPLOADED` | `UPLOADED` | `UPLOADED` |
| 确认流程 | `POST /documents/confirm` | 同左（完全复用） | 同左（完全复用） |
| 索引构建 | 现有 Celery 任务 | 同左（完全复用） | 同左（完全复用） |
| 文件格式 | 各种格式 | `.md`（Markdown） | `.txt` |

---

## 未来扩展

1. **网络搜索导入**：用户输入关键词 → 调用 `web/search` → 获取 URL 列表 → 复用 `/fetch-url` 接口批量抓取。搜索步骤是新增逻辑，抓取和入库完全复用。

2. **URL 定时刷新**：对 `source_type=url` 的文档，基于 `source_url` 定期重新抓取内容并更新索引。

3. **JavaScript 渲染支持**：web/read 服务升级支持 Playwright 后，`/fetch-url` 接口自动受益，无需改动。

---

## 实施路径

### Phase 1：后端（约 2 天）

1. 在 `document.yaml` schema 中新增 `fetchUrlRequest/Response`
2. 在 `collections.yaml` paths 中注册 `/fetch-url` 路由
3. 运行 `make generate-models`
4. 将 `web.py` 中的 `_read_with_jina_fallback` / `_read_with_trafilatura_only` 提取为 `ReaderService` 的公共方法
5. 在 `views/collections.py` 中实现 `fetch_url_document_view` 路由函数
6. 编写单元测试

### Phase 2：前端（约 2 天）

1. 运行 `make generate-frontend-sdk`
2. 实现 `import-dialog.tsx`（来源选择入口）
3. 实现 `url-import.tsx`（URL 输入 + 调用 `/fetch-url`）
4. 实现 `text-import.tsx`（文本粘贴 + 创建 File 对象 + 调用现有上传接口）
5. 在文档列表页或上传页集成"添加来源"按钮触发 Dialog
6. 新增 i18n key（`zh-CN` 和 `en-US`）

### Phase 3：验证（约 0.5 天）

1. E2E 测试：URL 导入 → 暂存区展示 → confirm → 索引完成
2. E2E 测试：文本粘贴 → 暂存区展示 → confirm → 索引完成
3. 错误场景：无效 URL、超时、配额超限

---

## 相关文件索引

### 参考文件

- `aperag/service/document_service.py` — `upload_document()` 实现（核心复用点）
- `aperag/views/web.py` — `_read_with_jina_fallback()` 等私有函数（待提取为公共方法）
- `aperag/websearch/reader/reader_service.py` — ReaderService（JINA/Trafilatura 实现）
- `web/src/app/.../upload/document-upload.tsx` — 前端暂存区（参考现有实现）

### 修改文件

- `aperag/api/components/schemas/document.yaml` — 新增 Schema
- `aperag/api/paths/collections.yaml` — 新增路由
- `aperag/views/collections.py` — 新增 `fetch_url_document_view`
- `aperag/websearch/reader/reader_service.py` — 提取公共方法

### 新建文件

- `web/src/app/.../documents/upload/import/import-dialog.tsx`
- `web/src/app/.../documents/upload/import/url-import.tsx`
- `web/src/app/.../documents/upload/import/text-import.tsx`
- `web/src/i18n/zh-CN/page_documents_import.json`
- `web/src/i18n/en-US/page_documents_import.json`
