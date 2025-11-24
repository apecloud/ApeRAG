---
title: 文档上传流程设计
description: 详细说明ApeRAG前端文档上传功能的完整实现，包括三步上传流程、状态管理、并发控制和用户交互设计
keywords: [document upload, file upload, two-phase commit, progress tracking, batch upload, react, next.js]
---

# 文档上传流程设计

## 概述

ApeRAG的文档上传功能采用**三步引导式上传**设计，提供直观的用户体验和可靠的上传机制。

**核心特性**:
- 📤 **三步引导流程**: 选择文件 → 上传到临时存储 → 确认添加到知识库
- 🔄 **智能重复检测**: 基于文件名、大小、修改时间和类型的前端去重
- 📊 **实时进度跟踪**: 每个文件独立显示上传进度和状态
- ⚡ **并发上传控制**: 限制同时上传3个文件，避免浏览器资源耗尽
- 🎯 **批量操作支持**: 支持批量选择、批量删除、批量确认

## 三步上传流程

```
┌─────────────────────────────────────────────────────────────┐
│                     Step 1: 选择文件                         │
│  - 拖拽上传或点击选择文件                                      │
│  - 前端文件验证（类型、大小、重复）                              │
│  - 显示文件列表，状态为 pending                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Step 2: 上传文件                         │
│  - 并发上传到临时存储（最多3个并发）                            │
│  - 实时显示上传进度（0-100%）                                 │
│  - 每个文件独立状态：uploading → success/failed               │
│  - 后端返回 document_id（状态：UPLOADED）                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Step 3: 确认添加                         │
│  - 所有文件上传成功后进入此步骤                                │
│  - 用户可以选择性确认部分文件                                  │
│  - 点击"保存到知识库"触发确认API                               │
│  - 后端开始索引构建，文档状态变为 PENDING                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                  跳转到文档列表页面
```

## 组件架构

### 核心组件: DocumentUpload

**文件路径**: `web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx`

**组件结构**:

```tsx
DocumentUpload
├── FileUpload (文件上传区域)
│   ├── FileUploadDropzone (拖拽上传)
│   └── FileUploadTrigger (点击选择)
│
├── Progress Indicators (进度指示器)
│   ├── Step 1: 选择文件
│   ├── Step 2: 上传文件
│   └── Step 3: 保存到集合
│
├── DataGrid (文件列表表格)
│   ├── Checkbox (批量选择)
│   ├── FileIcon (文件类型图标)
│   ├── Progress Bar (上传进度)
│   └── Actions (操作菜单)
│
└── Action Buttons
    ├── Upload Button (开始上传)
    ├── Stop Upload Button (停止上传)
    ├── Clear All (清空列表)
    └── Save to Collection (保存到知识库)
```

## 数据结构

### DocumentsWithFile 类型

```typescript
type DocumentsWithFile = {
  // 前端文件对象
  file: File;
  
  // 上传进度 (0-100)
  progress: number;
  
  // 上传状态
  progress_status: 'pending' | 'uploading' | 'success' | 'failed';
  
  // 后端返回的数据（上传成功后填充）
  document_id?: string;      // 文档ID
  filename?: string;         // 文件名
  size?: number;             // 文件大小
  status?: UploadDocumentResponseStatusEnum;  // 文档状态（UPLOADED）
};
```

### 状态管理

```typescript
const [documents, setDocuments] = useState<DocumentsWithFile[]>([]);  // 文件列表
const [step, setStep] = useState<number>(1);                          // 当前步骤
const [rowSelection, setRowSelection] = useState({});                 // 选中的行
const [isUploading, setIsUploading] = useState(false);                // 上传中标志
const [pagination, setPagination] = useState({                        // 分页状态
  pageIndex: 0,
  pageSize: 20,
});

// 上传中的文件集合（用于避免重复上传）
const uploadingFilesRef = useRef<Set<string>>(new Set());
```

## 核心功能实现

### 1. 文件选择和验证

**文件验证逻辑**:

```typescript
const onFileValidate = useCallback(
  (file: File): string | null => {
    // 检查是否已存在相同文件
    const doc = documents.some(
      (doc) =>
        doc.file.name === file.name &&
        doc.file.size === file.size &&
        doc.file.lastModified === file.lastModified &&
        doc.file.type === file.type,
    );
    if (doc) {
      return 'File already exists.';
    }
    return null;
  },
  [documents],
);
```

**文件拒绝处理**:

```typescript
const onFileReject = useCallback((file: File, message: string) => {
  toast.error(message, {
    description: `"${file.name.length > 20 ? `${file.name.slice(0, 20)}...` : file.name}" has been rejected`,
  });
}, []);
```

**重复检测策略**:

| 检查项 | 说明 | 用途 |
|--------|------|------|
| `file.name` | 文件名 | 基础去重 |
| `file.size` | 文件大小（字节） | 精确匹配 |
| `file.lastModified` | 最后修改时间戳 | 区分同名文件 |
| `file.type` | MIME类型 | 确保完全一致 |

### 2. 并发上传控制

**使用 async.eachLimit 控制并发**:

```typescript
import async from 'async';

const startUpload = useCallback((docs: DocumentsWithFile[]) => {
  // 1. 过滤出待上传的文件
  const filesToUpload = docs.filter((doc) => {
    const fileKey = `${doc.file.name}-${doc.file.size}-${doc.file.lastModified}`;
    return (
      doc.progress_status === 'pending' &&
      !doc.document_id &&
      !uploadingFilesRef.current.has(fileKey)  // 避免重复上传
    );
  });
  
  // 2. 标记为上传中
  filesToUpload.forEach((doc) => {
    const fileKey = `${doc.file.name}-${doc.file.size}-${doc.file.lastModified}`;
    uploadingFilesRef.current.add(fileKey);
  });
  
  // 3. 创建上传任务
  const tasks: AsyncTask[] = filesToUpload.map((_doc) => async (callback) => {
    // ... 上传逻辑
  });
  
  // 4. 并发执行（最多3个并发）
  async.eachLimit(
    tasks,
    3,  // 并发数
    (task, callback) => {
      if (uploadController?.signal.aborted) {
        callback(new Error('stop upload'));
      } else {
        task(callback);
      }
    },
    (err) => {
      setIsUploading(false);
    },
  );
}, [collection.id]);
```

**并发控制优势**:

- ✅ 限制浏览器同时请求数，避免资源耗尽
- ✅ 避免后端过载
- ✅ 支持中途取消所有上传
- ✅ 更好的进度追踪

### 3. 上传进度追踪

**模拟进度显示**（实际上传 + 进度动画）:

```typescript
const networkSimulation = async () => {
  const totalChunks = 100;
  let uploadedChunks = 0;
  
  for (let i = 0; i < totalChunks; i++) {
    // 每5-10ms更新一次进度
    await new Promise((resolve) =>
      setTimeout(resolve, Math.random() * 5 + 5),
    );
    
    uploadedChunks++;
    const progress = (uploadedChunks / totalChunks) * 99;  // 最多到99%
    
    // 更新特定文件的进度
    setDocuments((docs) => {
      const doc = docs.find((doc) => _.isEqual(doc.file, file));
      if (doc) {
        doc.progress = Number(progress.toFixed(0));
        doc.progress_status = 'uploading';
      }
      return [...docs];
    });
  }
};

// 并行执行上传和进度动画
const [res] = await Promise.all([
  apiClient.defaultApi.collectionsCollectionIdDocumentsUploadPost({
    collectionId: collection.id,
    file: _doc.file,
  }),
  networkSimulation(),  // 进度动画
]);

// 上传成功，进度设为100%
setDocuments((docs) => {
  const doc = docs.find((doc) => _.isEqual(doc.file, file));
  if (doc && res.data.document_id) {
    Object.assign(doc, {
      ...res.data,
      progress: 100,
      progress_status: 'success',
    });
  }
  return [...docs];
});
```

**为什么模拟进度？**

1. HTTP上传无法获取实时进度（浏览器限制）
2. 提供更好的用户体验，避免长时间无反馈
3. 视觉上更流畅，用户感知更好

### 4. 取消上传

**使用 AbortController**:

```typescript
let uploadController: AbortController | undefined;

// 停止上传
const stopUpload = useCallback(() => {
  setIsUploading(false);
  uploadController?.abort();  // 中止所有正在进行的请求
}, []);

// 页面卸载时自动停止
useEffect(() => stopUpload, [stopUpload]);

// 开始上传时创建新的 controller
const startUpload = () => {
  uploadController = new AbortController();
  // ...
};
```

### 5. 确认添加到知识库

**Step 3: 保存到集合**:

```typescript
const handleSaveToCollection = useCallback(async () => {
  if (!collection.id) return;
  
  // 调用确认API
  const res = await apiClient.defaultApi.collectionsCollectionIdDocumentsConfirmPost({
    collectionId: collection.id,
    confirmDocumentsRequest: {
      document_ids: documents
        .map((doc) => doc.document_id || '')
        .filter((id) => !_.isEmpty(id)),
    },
  });
  
  if (res.status === 200) {
    toast.success('Document added successfully');
    // 跳转回文档列表
    router.push(`/workspace/collections/${collection.id}/documents`);
  }
}, [collection.id, documents, router]);
```

## API集成

### 1. 上传文件 API

**接口**: `POST /api/v1/collections/{collectionId}/documents/upload`

**请求**:

```typescript
apiClient.defaultApi.collectionsCollectionIdDocumentsUploadPost({
  collectionId: collection.id,
  file: file,  // File对象
}, {
  timeout: 1000 * 30,  // 30秒超时
});
```

**响应**:

```typescript
{
  document_id: "doc_xyz789",
  filename: "example.pdf",
  size: 2048576,
  status: "UPLOADED"
}
```

### 2. 确认文档 API

**接口**: `POST /api/v1/collections/{collectionId}/documents/confirm`

**请求**:

```typescript
apiClient.defaultApi.collectionsCollectionIdDocumentsConfirmPost({
  collectionId: collection.id,
  confirmDocumentsRequest: {
    document_ids: ["doc_xyz789", "doc_abc123", ...]
  }
});
```

**响应**:

```typescript
{
  confirmed_count: 3,
  failed_count: 1,
  failed_documents: [
    {
      document_id: "doc_fail123",
      name: "corrupted.pdf",
      error: "CONFIRMATION_FAILED"
    }
  ]
}
```

## UI组件详解

### 1. 文件上传区域

```tsx
<FileUpload
  value={documents.map((doc) => doc.file)}
  onValueChange={(files) => {
    const newFilesToUpload: DocumentsWithFile[] = [];
    files.forEach((file) => {
      if (
        !documents.some(
          (doc) =>
            doc.file.name === file.name &&
            doc.file.size === file.size &&
            doc.file.lastModified === file.lastModified &&
            doc.file.type === file.type,
        )
      ) {
        newFilesToUpload.push({
          file,
          progress: 0,
          progress_status: 'pending',
        });
      }
    });
    if (newFilesToUpload.length > 0) {
      setDocuments((docs) => [...docs, ...newFilesToUpload]);
    }
  }}
  onFileReject={onFileReject}
  onFileValidate={onFileValidate}
>
  <FileUploadDropzone className="h-64 w-full">
    <div className="flex flex-col items-center justify-center gap-2">
      <CloudUpload className="size-12 text-muted-foreground" />
      <div className="text-muted-foreground">
        {page_documents('drag_and_drop_files_here')}
      </div>
      <div className="text-muted-foreground text-sm">
        {page_documents('or')}
      </div>
      <FileUploadTrigger asChild>
        <Button variant="outline" size="sm">
          {page_documents('browse_files')}
        </Button>
      </FileUploadTrigger>
    </div>
  </FileUploadDropzone>
</FileUpload>
```

**特性**:
- 支持拖拽上传
- 支持点击选择文件
- 自动文件验证
- 重复文件检测

### 2. 进度指示器

```tsx
<div className="flex flex-row items-center gap-2">
  {/* Step 1 */}
  <div data-active={step === 1} className="...">
    <Bs1CircleFill className="size-6" />
    <div>{page_documents('step1_select_files')}</div>
  </div>
  
  <ChevronRight />
  
  {/* Step 2 */}
  <div data-active={step === 2} className="...">
    <Bs2CircleFill className="size-6" />
    <div>{page_documents('step2_upload_files')}</div>
  </div>
  
  <ChevronRight />
  
  {/* Step 3 */}
  <div data-active={step === 3} className="...">
    <Bs3CircleFill className="size-6" />
    <div>{page_documents('step3_save_to_collection')}</div>
  </div>
</div>
```

**步骤自动切换逻辑**:

```typescript
useEffect(() => {
  if (documents.length === 0) {
    setStep(1);  // 无文件 → Step 1
  } else if (
    documents.filter((doc) => doc.progress_status === 'success').length !==
    documents.length
  ) {
    setStep(2);  // 有未完成上传 → Step 2
  } else {
    setStep(3);  // 全部上传完成 → Step 3
  }
}, [documents]);
```

### 3. 文件列表表格

使用 `@tanstack/react-table` 实现：

```typescript
const columns: ColumnDef<DocumentsWithFile>[] = [
  {
    id: 'select',
    header: ({ table }) => (
      <Checkbox
        checked={table.getIsAllPageRowsSelected()}
        onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
      />
    ),
    cell: ({ row }) => (
      <Checkbox
        checked={row.getIsSelected()}
        onCheckedChange={(value) => row.toggleSelected(!!value)}
      />
    ),
  },
  {
    accessorKey: 'filename',
    header: 'Filename',
    cell: ({ row }) => {
      const file = row.original.file;
      const extension = _.last(file.type.split('/')) || '';
      return (
        <div className="flex items-center gap-2">
          <FileIcon extension={extension} />
          <div>
            <div>{file.name}</div>
            <div className="text-sm">
              {(file.size / 1000).toFixed(0)} KB
            </div>
          </div>
        </div>
      );
    },
  },
  {
    header: 'Upload Progress',
    cell: ({ row }) => (
      <div className="flex flex-col">
        <Progress value={row.original.progress} />
        <div className="flex justify-between text-xs">
          <div>{row.original.progress}%</div>
          <div data-status={row.original.progress_status}>
            {row.original.progress_status}
          </div>
        </div>
      </div>
    ),
  },
  {
    id: 'actions',
    cell: ({ row }) => (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon">
            <EllipsisVertical />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem
            variant="destructive"
            onClick={() => handleRemoveFile(row.original)}
          >
            <Trash /> Remove
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    ),
  },
];
```

**表格特性**:
- ✅ 复选框批量选择
- ✅ 文件类型图标显示
- ✅ 实时进度条
- ✅ 状态颜色标识
- ✅ 分页支持（每页20条）
- ✅ 删除操作

### 4. 操作按钮

```tsx
<div className="flex items-center gap-2">
  {/* 清空所有 */}
  <Button
    variant="outline"
    onClick={() => {
      setDocuments([]);
      setRowSelection({});
    }}
    disabled={documents.length === 0}
  >
    <BrushCleaning /> Clear All
  </Button>
  
  {/* 开始上传 */}
  <Button
    variant="outline"
    onClick={() => startUpload(documents)}
    disabled={isUploading || documents.length === 0}
  >
    {isUploading ? <LoaderCircle className="animate-spin" /> : <Upload />}
    {isUploading ? 'Uploading...' : 'Start Upload'}
  </Button>
  
  {/* 停止上传 */}
  {isUploading && (
    <Button variant="destructive" onClick={stopUpload}>
      Stop Upload
    </Button>
  )}
  
  {/* 保存到集合 */}
  <Button
    onClick={handleSaveToCollection}
    disabled={
      documents.filter((doc) => doc.progress_status === 'success').length === 0
    }
  >
    <Save /> Save to Collection
  </Button>
</div>
```

## 状态管理流程

```
初始状态
├── documents: []
├── step: 1
├── isUploading: false
└── uploadingFilesRef.current: Set()

↓ 用户选择文件

Step 1: 文件选择完成
├── documents: [{file, progress: 0, progress_status: 'pending'}, ...]
├── step: 1
├── isUploading: false
└── uploadingFilesRef.current: Set()

↓ 点击"开始上传"

Step 2: 上传中
├── documents: [{..., progress: 45, progress_status: 'uploading'}, ...]
├── step: 2
├── isUploading: true
└── uploadingFilesRef.current: Set('file1-key', 'file2-key', ...)

↓ 上传完成

Step 3: 等待确认
├── documents: [{..., progress: 100, progress_status: 'success', document_id: 'doc_xyz'}, ...]
├── step: 3
├── isUploading: false
└── uploadingFilesRef.current: Set()

↓ 点击"保存到集合"

跳转到文档列表页面
```

## 错误处理

### 1. 上传失败

```typescript
catch (err) {
  setDocuments((docs) => {
    const doc = docs.find((doc) => _.isEqual(doc.file, file));
    if (doc) {
      Object.assign(doc, {
        progress: 0,
        progress_status: 'failed',
      });
    }
    return [...docs];
  });
}
```

**失败后的操作**:
- 进度重置为0
- 状态标记为 `failed`
- 可以重新点击"开始上传"重试
- 可以删除失败的文件

### 2. 文件验证失败

```typescript
// 在 onFileValidate 中返回错误信息
return 'File already exists.';

// 或在 onFileReject 中处理
onFileReject={(file, message) => {
  toast.error(message, {
    description: `"${file.name}" has been rejected`,
  });
}}
```

### 3. 网络中断

```typescript
// 用户可以点击"停止上传"
const stopUpload = () => {
  uploadController?.abort();  // 中止所有请求
  setIsUploading(false);
};

// 页面卸载时自动停止
useEffect(() => stopUpload, [stopUpload]);
```

## 性能优化

### 1. 防抖和节流

```typescript
// 使用 lodash 进行文件比较（高效）
_.isEqual(doc.file, file)

// 文件key生成（快速查找）
const fileKey = `${file.name}-${file.size}-${file.lastModified}`;
```

### 2. 状态更新优化

```typescript
// 使用函数式更新，避免闭包陷阱
setDocuments((docs) => {
  const doc = docs.find(...);
  // 修改
  return [...docs];  // 返回新数组触发更新
});
```

### 3. 分页显示

```typescript
// 默认每页20条，避免大列表渲染卡顿
const [pagination, setPagination] = useState({
  pageIndex: 0,
  pageSize: 20,
});
```

### 4. 虚拟滚动（未实现，可优化）

对于超大文件列表（1000+），可以使用虚拟滚动：

```typescript
import { useVirtualizer } from '@tanstack/react-virtual';
```

## 用户体验设计

### 1. 即时反馈

- ✅ 拖拽时显示高亮区域
- ✅ 上传中显示动画图标
- ✅ 进度条实时更新
- ✅ 状态用颜色区分（pending/uploading/success/failed）

### 2. 错误提示

- ✅ 文件验证失败：Toast通知
- ✅ 上传失败：状态标红
- ✅ 确认失败：显示具体错误信息

### 3. 操作引导

- ✅ 三步进度指示器
- ✅ 按钮根据状态启用/禁用
- ✅ 空状态提示
- ✅ 操作成功后自动跳转

### 4. 响应式设计

- ✅ 表格在小屏幕自适应
- ✅ 操作按钮在移动端堆叠
- ✅ 文件名过长时截断显示

## 国际化支持

使用 `next-intl` 进行国际化：

```typescript
const page_documents = useTranslations('page_documents');

// 使用
page_documents('filename')
page_documents('upload_progress')
page_documents('drag_and_drop_files_here')
page_documents('step1_select_files')
page_documents('step2_upload_files')
page_documents('step3_save_to_collection')
```

**翻译文件位置**:
- `web/src/locales/en-US/page_documents.json`
- `web/src/locales/zh-CN/page_documents.json`

## 最佳实践

### 1. 文件大小限制

```typescript
// 前端检查（可选）
const MAX_FILE_SIZE = 100 * 1024 * 1024;  // 100MB

if (file.size > MAX_FILE_SIZE) {
  return 'File size exceeds 100MB';
}
```

### 2. 支持的文件类型

前端可以限制文件类型，但最终验证在后端：

```typescript
const ALLOWED_TYPES = [
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
  // ...
];

if (!ALLOWED_TYPES.includes(file.type)) {
  return 'File type not supported';
}
```

### 3. 自动重试机制（未实现，建议）

```typescript
const uploadWithRetry = async (file: File, retries = 3) => {
  for (let i = 0; i < retries; i++) {
    try {
      return await apiClient.upload(file);
    } catch (err) {
      if (i === retries - 1) throw err;
      await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, i)));
    }
  }
};
```

## 相关文件

### 前端组件
- `web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx` - 主上传组件
- `web/src/app/workspace/collections/[collectionId]/documents/upload/page.tsx` - 上传页面
- `web/src/components/ui/file-upload.tsx` - 文件上传UI组件
- `web/src/components/ui/progress.tsx` - 进度条组件
- `web/src/components/data-grid.tsx` - 数据表格组件

### API客户端
- `web/src/lib/api/client.ts` - API客户端配置
- `web/src/api/` - 自动生成的API接口

### 国际化
- `web/src/locales/en-US/page_documents.json` - 英文翻译
- `web/src/locales/zh-CN/page_documents.json` - 中文翻译

## 总结

ApeRAG的文档上传功能通过**三步引导流程**提供了直观且可靠的用户体验：

1. **Step 1 - 选择文件**: 拖拽或点击选择，前端即时验证
2. **Step 2 - 上传文件**: 并发上传到临时存储，实时进度追踪
3. **Step 3 - 确认添加**: 用户选择性确认，触发索引构建

**核心优势**:
- 🎯 **用户友好**: 三步流程清晰，操作引导明确
- ⚡ **性能优化**: 并发控制、分页显示、状态管理优化
- 🔒 **可靠性高**: 重复检测、错误处理、中途取消支持
- 🌍 **国际化**: 完整的多语言支持
- 📱 **响应式**: 适配移动端和桌面端

这种设计在保证功能完整性的同时，提供了出色的用户体验和系统稳定性。


