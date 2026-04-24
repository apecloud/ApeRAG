---
title: 导出知识库
position: 50
---

# 导出知识库

本指南介绍如何将知识库中的全部解析产物(原始文件 / 转换文件 / 处理后的 Markdown / 分块 / 图片)打包下载为 ZIP 文件。

## 适用场景

- **迁移到其他 RAG 框架**:将解析结果导入 LlamaIndex、Dify 等平台
- **审查解析质量**:检查 PDF / Word 等文件的解析结果是否有截断或格式问题
- **离线分析**:查看分块策略效果,评估检索与 chunk 关系
- **备份与合规**:定期打包知识库内容作为离线归档

## 权限

- 仅**知识库 Owner**(创建者)可以导出
- Marketplace 订阅用户 / 未登录用户无导出权限 — 前端不展示"导出知识库"按钮,后端会拒绝(`403 Forbidden`)

## 操作步骤

### 1. 进入知识库详情页

在知识库列表点击目标知识库进入详情页。页面顶部操作菜单会显示"导出知识库"按钮(仅 Owner 可见)。

### 2. 触发导出

点击"导出知识库"按钮,系统立即:

- 创建一个后台导出任务,分配 `task_id`
- 弹出导出进度对话框,展示当前进度
- 开始扫描对象存储中对应目录并打包

### 3. 监听进度

进度对话框通过**轮询**实时更新,显示四个阶段:

| 阶段 | 含义 |
|---|---|
| `SCANNING` | 枚举对象存储中的文件清单 |
| `PACKING` | 流式打包到 ZIP,写入 manifest.json |
| `UPLOADING` | ZIP 回写到对象存储 `exports/` 前缀 |
| `COMPLETED` | 完成,下载按钮可用 |

若中途失败(网络 / 对象存储错误 / 用户取消),状态会变成 `FAILED`,对话框展示错误信息。MVP 版本**对话框必须保持打开**直到完成或失败 — 关闭页面会丢失当前任务进度引用。

### 4. 下载 ZIP

状态变为 `COMPLETED` 后,点击"下载"按钮即可下载 ZIP 文件。文件命名:

```
{collection_title}_export_{YYYY-MM-DD}.zip
```

例如:`医学文献库_export_2026-04-24.zip`。

## ZIP 内容

解压后目录结构与对象存储一致:

```
{collection_title}_export_{YYYY-MM-DD}.zip
├── manifest.json           ← 元数据(id → 标题映射)
├── {document_id_1}/
│   ├── original.pdf        ← 用户上传的原始文件
│   ├── converted.pdf       ← MinerU 转换后的 PDF
│   ├── processed_content.md ← 解析生成的 Markdown
│   ├── chunks/             ← 分块 JSON
│   │   ├── chunk_0.json
│   │   └── chunk_1.json
│   └── images/             ← 从文档提取的图片
│       ├── page_0.png
│       └── page_1.png
└── {document_id_2}/
    └── ...
```

### manifest.json

文件级元数据,仅作为信息记录(不影响 ZIP 包内容):

```json
{
  "schema_version": "1.0",
  "collection": {
    "id": "colff4f33902752abee",
    "title": "医学文献库",
    "exported_at": "2026-04-24T10:00:00Z"
  },
  "documents": [
    { "id": "doc_xyz789", "title": "高血压诊疗指南", "status": "COMPLETE" }
  ]
}
```

`documents` 数组列出知识库下所有文档的 id、标题、处理状态,便于迁移时做双向映射。

## 关键行为

### 全量导出,无过滤

导出策略对 `user-{user_id}/{collection_id}/` 前缀下所有对象做**全量**打包,不做过滤:

- 不区分文档状态(PENDING / COMPLETE / FAILED 全部包含)
- 不可选择单文档 / 单类型(如"只导出图片")— 本期未实现
- 不排除历史版本残留文件

### 导出产物生命周期

- 导出 ZIP 保存在对象存储的 `exports/user-{user_id}/export_{task_id}.zip`
- 下载链接有效期与导出 ZIP 生命周期同步:**7 天**后由定时任务自动清理
- 用户应在 7 天内完成下载;过期后需要重新触发导出

### 大知识库的预估

- 后端采用**流式打包**(边扫描边写 ZIP),不会在内存里堆积全部对象
- 大知识库(上万文档 / 上百 GB)的导出时间主要受对象存储吞吐限制
- 导出进度的 `files_processed / total_files` 比例可用于预估剩余时间

## 常见问题

### 为什么看不到"导出知识库"按钮?

按钮**仅对 Owner 渲染**。如果你是通过 Marketplace 订阅别人分享的知识库,无法导出 — 需要向 Owner 请求。

### 导出对话框可以关闭再回来看吗?

MVP 版本**不支持后台运行**。关闭对话框等于放弃当前任务的进度引用;后台打包任务仍会继续,但前端无法恢复进度展示。

后续版本计划支持"导出历史"页面,届时可以离开对话框并随时回来查看状态。

### 导出失败怎么办?

重新触发一次导出即可。常见失败原因:

- 对象存储暂时不可达(可稍后重试)
- ZIP 超过平台硬限(罕见,后端会拒绝)
- 任务在等待资源时超时(增加重试次数)

若重试多次仍失败,联系管理员查看后端日志。

### 导出的 ZIP 如何用于其他 RAG 框架?

ZIP 内容是 ApeRAG 完整的文档处理产物:

- **迁移到 LlamaIndex / Dify**:将 `processed_content.md` + `chunks/*.json` 作为输入重建索引,跳过 parse 环节
- **只要最终文本**:解压后取 `processed_content.md` 即可
- **保留结构化分块**:使用 `chunks/*.json` 复用 ApeRAG 的 chunking 策略

## 相关文档

- 查看文档如何被解析与索引:[上传文档](./document-upload.md)
- 了解文档处理的后端架构:see `docs/modularization/architecture.md` 的 `knowledge_base` / `indexing` domain sections
