---
title: Document Upload Flow Design
description: Detailed explanation of ApeRAG frontend document upload functionality, including three-step upload process, state management, concurrency control, and user interaction design
keywords: [document upload, file upload, two-phase commit, progress tracking, batch upload, react, next.js]
---

# Document Upload Flow Design

## Overview

ApeRAG's document upload feature adopts a **three-step guided upload** design, providing intuitive user experience and reliable upload mechanism.

**Core Features**:
- 📤 **Three-step Guided Process**: Select Files → Upload to Temporary Storage → Confirm Addition to Knowledge Base
- 🔄 **Smart Duplicate Detection**: Frontend deduplication based on filename, size, modification time, and type
- 📊 **Real-time Progress Tracking**: Each file displays upload progress and status independently
- ⚡ **Concurrent Upload Control**: Limit to 3 concurrent uploads to avoid browser resource exhaustion
- 🎯 **Batch Operation Support**: Support batch selection, deletion, and confirmation

## Three-Step Upload Process

```
┌─────────────────────────────────────────────────────────────┐
│                  Step 1: Select Files                        │
│  - Drag & drop or click to select files                     │
│  - Frontend file validation (type, size, duplicate)         │
│  - Display file list with pending status                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Step 2: Upload Files                        │
│  - Concurrent upload to temporary storage (max 3)           │
│  - Real-time progress display (0-100%)                      │
│  - Independent status per file: uploading → success/failed  │
│  - Backend returns document_id (status: UPLOADED)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Step 3: Confirm Addition                    │
│  - Enter this step after all files uploaded successfully    │
│  - User can selectively confirm partial files               │
│  - Click "Save to Collection" to trigger confirm API        │
│  - Backend starts index building, document status → PENDING │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
               Navigate to document list page
```

## Component Architecture

### Core Component: DocumentUpload

**File Path**: `web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx`

**Component Structure**:

```tsx
DocumentUpload
├── FileUpload (File upload area)
│   ├── FileUploadDropzone (Drag & drop)
│   └── FileUploadTrigger (Click to select)
│
├── Progress Indicators
│   ├── Step 1: Select Files
│   ├── Step 2: Upload Files
│   └── Step 3: Save to Collection
│
├── DataGrid (File list table)
│   ├── Checkbox (Batch selection)
│   ├── FileIcon (File type icon)
│   ├── Progress Bar (Upload progress)
│   └── Actions (Action menu)
│
└── Action Buttons
    ├── Upload Button (Start upload)
    ├── Stop Upload Button (Cancel upload)
    ├── Clear All (Clear list)
    └── Save to Collection (Confirm addition)
```

## Data Structures

### DocumentsWithFile Type

```typescript
type DocumentsWithFile = {
  // Frontend file object
  file: File;
  
  // Upload progress (0-100)
  progress: number;
  
  // Upload status
  progress_status: 'pending' | 'uploading' | 'success' | 'failed';
  
  // Backend returned data (populated after successful upload)
  document_id?: string;      // Document ID
  filename?: string;         // Filename
  size?: number;             // File size
  status?: UploadDocumentResponseStatusEnum;  // Document status (UPLOADED)
};
```

### State Management

```typescript
const [documents, setDocuments] = useState<DocumentsWithFile[]>([]);  // File list
const [step, setStep] = useState<number>(1);                          // Current step
const [rowSelection, setRowSelection] = useState({});                 // Selected rows
const [isUploading, setIsUploading] = useState(false);                // Uploading flag
const [pagination, setPagination] = useState({                        // Pagination state
  pageIndex: 0,
  pageSize: 20,
});

// Set of files being uploaded (to avoid duplicate uploads)
const uploadingFilesRef = useRef<Set<string>>(new Set());
```

## Core Feature Implementation

### 1. File Selection and Validation

**File Validation Logic**:

```typescript
const onFileValidate = useCallback(
  (file: File): string | null => {
    // Check if same file already exists
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

**File Rejection Handling**:

```typescript
const onFileReject = useCallback((file: File, message: string) => {
  toast.error(message, {
    description: `"${file.name.length > 20 ? `${file.name.slice(0, 20)}...` : file.name}" has been rejected`,
  });
}, []);
```

**Duplicate Detection Strategy**:

| Check Item | Description | Purpose |
|------------|-------------|---------|
| `file.name` | Filename | Basic deduplication |
| `file.size` | File size (bytes) | Exact match |
| `file.lastModified` | Last modified timestamp | Distinguish same-name files |
| `file.type` | MIME type | Ensure complete match |

### 2. Concurrent Upload Control

**Using async.eachLimit to Control Concurrency**:

```typescript
import async from 'async';

const startUpload = useCallback((docs: DocumentsWithFile[]) => {
  // 1. Filter files to upload
  const filesToUpload = docs.filter((doc) => {
    const fileKey = `${doc.file.name}-${doc.file.size}-${doc.file.lastModified}`;
    return (
      doc.progress_status === 'pending' &&
      !doc.document_id &&
      !uploadingFilesRef.current.has(fileKey)  // Avoid duplicate upload
    );
  });
  
  // 2. Mark as uploading
  filesToUpload.forEach((doc) => {
    const fileKey = `${doc.file.name}-${doc.file.size}-${doc.file.lastModified}`;
    uploadingFilesRef.current.add(fileKey);
  });
  
  // 3. Create upload tasks
  const tasks: AsyncTask[] = filesToUpload.map((_doc) => async (callback) => {
    // ... upload logic
  });
  
  // 4. Execute concurrently (max 3 concurrent)
  async.eachLimit(
    tasks,
    3,  // Concurrency limit
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

**Concurrency Control Benefits**:

- ✅ Limit browser simultaneous requests to avoid resource exhaustion
- ✅ Avoid backend overload
- ✅ Support canceling all uploads mid-way
- ✅ Better progress tracking

### 3. Upload Progress Tracking

**Simulated Progress Display** (Actual upload + progress animation):

```typescript
const networkSimulation = async () => {
  const totalChunks = 100;
  let uploadedChunks = 0;
  
  for (let i = 0; i < totalChunks; i++) {
    // Update progress every 5-10ms
    await new Promise((resolve) =>
      setTimeout(resolve, Math.random() * 5 + 5),
    );
    
    uploadedChunks++;
    const progress = (uploadedChunks / totalChunks) * 99;  // Max 99%
    
    // Update specific file's progress
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

// Execute upload and progress animation in parallel
const [res] = await Promise.all([
  apiClient.defaultApi.collectionsCollectionIdDocumentsUploadPost({
    collectionId: collection.id,
    file: _doc.file,
  }),
  networkSimulation(),  // Progress animation
]);

// Upload successful, set progress to 100%
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

**Why Simulate Progress?**

1. HTTP upload cannot get real-time progress (browser limitation)
2. Provide better user experience, avoid long periods without feedback
3. Visually smoother, better user perception

### 4. Cancel Upload

**Using AbortController**:

```typescript
let uploadController: AbortController | undefined;

// Stop upload
const stopUpload = useCallback(() => {
  setIsUploading(false);
  uploadController?.abort();  // Abort all ongoing requests
}, []);

// Auto-stop when page unmounts
useEffect(() => stopUpload, [stopUpload]);

// Create new controller when starting upload
const startUpload = () => {
  uploadController = new AbortController();
  // ...
};
```

### 5. Confirm Addition to Knowledge Base

**Step 3: Save to Collection**:

```typescript
const handleSaveToCollection = useCallback(async () => {
  if (!collection.id) return;
  
  // Call confirm API
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
    // Navigate back to document list
    router.push(`/workspace/collections/${collection.id}/documents`);
  }
}, [collection.id, documents, router]);
```

## API Integration

### 1. Upload File API

**Endpoint**: `POST /api/v1/collections/{collectionId}/documents/upload`

**Request**:

```typescript
apiClient.defaultApi.collectionsCollectionIdDocumentsUploadPost({
  collectionId: collection.id,
  file: file,  // File object
}, {
  timeout: 1000 * 30,  // 30 second timeout
});
```

**Response**:

```typescript
{
  document_id: "doc_xyz789",
  filename: "example.pdf",
  size: 2048576,
  status: "UPLOADED"
}
```

### 2. Confirm Documents API

**Endpoint**: `POST /api/v1/collections/{collectionId}/documents/confirm`

**Request**:

```typescript
apiClient.defaultApi.collectionsCollectionIdDocumentsConfirmPost({
  collectionId: collection.id,
  confirmDocumentsRequest: {
    document_ids: ["doc_xyz789", "doc_abc123", ...]
  }
});
```

**Response**:

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

## UI Component Details

### 1. File Upload Area

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
        Drag and drop files here
      </div>
      <div className="text-muted-foreground text-sm">
        or
      </div>
      <FileUploadTrigger asChild>
        <Button variant="outline" size="sm">
          Browse Files
        </Button>
      </FileUploadTrigger>
    </div>
  </FileUploadDropzone>
</FileUpload>
```

**Features**:
- Support drag & drop upload
- Support click to select files
- Automatic file validation
- Duplicate file detection

### 2. Progress Indicators

```tsx
<div className="flex flex-row items-center gap-2">
  {/* Step 1 */}
  <div data-active={step === 1} className="...">
    <Bs1CircleFill className="size-6" />
    <div>Select Files</div>
  </div>
  
  <ChevronRight />
  
  {/* Step 2 */}
  <div data-active={step === 2} className="...">
    <Bs2CircleFill className="size-6" />
    <div>Upload Files</div>
  </div>
  
  <ChevronRight />
  
  {/* Step 3 */}
  <div data-active={step === 3} className="...">
    <Bs3CircleFill className="size-6" />
    <div>Save to Collection</div>
  </div>
</div>
```

**Step Auto-switching Logic**:

```typescript
useEffect(() => {
  if (documents.length === 0) {
    setStep(1);  // No files → Step 1
  } else if (
    documents.filter((doc) => doc.progress_status === 'success').length !==
    documents.length
  ) {
    setStep(2);  // Has incomplete uploads → Step 2
  } else {
    setStep(3);  // All uploads complete → Step 3
  }
}, [documents]);
```

### 3. File List Table

Implemented using `@tanstack/react-table`:

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

**Table Features**:
- ✅ Checkbox batch selection
- ✅ File type icon display
- ✅ Real-time progress bar
- ✅ Status color coding
- ✅ Pagination support (20 items per page)
- ✅ Delete action

### 4. Action Buttons

```tsx
<div className="flex items-center gap-2">
  {/* Clear All */}
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
  
  {/* Start Upload */}
  <Button
    variant="outline"
    onClick={() => startUpload(documents)}
    disabled={isUploading || documents.length === 0}
  >
    {isUploading ? <LoaderCircle className="animate-spin" /> : <Upload />}
    {isUploading ? 'Uploading...' : 'Start Upload'}
  </Button>
  
  {/* Stop Upload */}
  {isUploading && (
    <Button variant="destructive" onClick={stopUpload}>
      Stop Upload
    </Button>
  )}
  
  {/* Save to Collection */}
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

## State Management Flow

```
Initial State
├── documents: []
├── step: 1
├── isUploading: false
└── uploadingFilesRef.current: Set()

↓ User selects files

Step 1: File Selection Complete
├── documents: [{file, progress: 0, progress_status: 'pending'}, ...]
├── step: 1
├── isUploading: false
└── uploadingFilesRef.current: Set()

↓ Click "Start Upload"

Step 2: Uploading
├── documents: [{..., progress: 45, progress_status: 'uploading'}, ...]
├── step: 2
├── isUploading: true
└── uploadingFilesRef.current: Set('file1-key', 'file2-key', ...)

↓ Upload complete

Step 3: Waiting for Confirmation
├── documents: [{..., progress: 100, progress_status: 'success', document_id: 'doc_xyz'}, ...]
├── step: 3
├── isUploading: false
└── uploadingFilesRef.current: Set()

↓ Click "Save to Collection"

Navigate to document list page
```

## Error Handling

### 1. Upload Failure

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

**Actions After Failure**:
- Reset progress to 0
- Mark status as `failed`
- Can click "Start Upload" again to retry
- Can delete failed files

### 2. File Validation Failure

```typescript
// Return error message in onFileValidate
return 'File already exists.';

// Or handle in onFileReject
onFileReject={(file, message) => {
  toast.error(message, {
    description: `"${file.name}" has been rejected`,
  });
}}
```

### 3. Network Interruption

```typescript
// User can click "Stop Upload"
const stopUpload = () => {
  uploadController?.abort();  // Abort all requests
  setIsUploading(false);
};

// Auto-stop when page unmounts
useEffect(() => stopUpload, [stopUpload]);
```

## Performance Optimization

### 1. Debounce and Throttle

```typescript
// Use lodash for file comparison (efficient)
_.isEqual(doc.file, file)

// File key generation (fast lookup)
const fileKey = `${file.name}-${file.size}-${file.lastModified}`;
```

### 2. State Update Optimization

```typescript
// Use functional update to avoid closure trap
setDocuments((docs) => {
  const doc = docs.find(...);
  // Modify
  return [...docs];  // Return new array to trigger update
});
```

### 3. Pagination Display

```typescript
// Default 20 items per page to avoid large list rendering lag
const [pagination, setPagination] = useState({
  pageIndex: 0,
  pageSize: 20,
});
```

### 4. Virtual Scrolling (Not Implemented, Can Optimize)

For very large file lists (1000+), can use virtual scrolling:

```typescript
import { useVirtualizer } from '@tanstack/react-virtual';
```

## User Experience Design

### 1. Instant Feedback

- ✅ Show highlight area when dragging
- ✅ Show animation icon during upload
- ✅ Real-time progress bar updates
- ✅ Distinguish status by color (pending/uploading/success/failed)

### 2. Error Messages

- ✅ File validation failed: Toast notification
- ✅ Upload failed: Status marked red
- ✅ Confirmation failed: Show specific error message

### 3. Operation Guidance

- ✅ Three-step progress indicator
- ✅ Buttons enabled/disabled based on state
- ✅ Empty state prompt
- ✅ Auto-navigate after successful operation

### 4. Responsive Design

- ✅ Table adapts on small screens
- ✅ Action buttons stack on mobile
- ✅ Long filenames truncated

## Internationalization Support

Using `next-intl` for internationalization:

```typescript
const page_documents = useTranslations('page_documents');

// Usage
page_documents('filename')
page_documents('upload_progress')
page_documents('drag_and_drop_files_here')
page_documents('step1_select_files')
page_documents('step2_upload_files')
page_documents('step3_save_to_collection')
```

**Translation File Locations**:
- `web/src/locales/en-US/page_documents.json`
- `web/src/locales/zh-CN/page_documents.json`

## Best Practices

### 1. File Size Limit

```typescript
// Frontend check (optional)
const MAX_FILE_SIZE = 100 * 1024 * 1024;  // 100MB

if (file.size > MAX_FILE_SIZE) {
  return 'File size exceeds 100MB';
}
```

### 2. Supported File Types

Frontend can limit file types, but final validation is on backend:

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

### 3. Auto-retry Mechanism (Not Implemented, Recommended)

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

## Related Files

### Frontend Components
- `web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx` - Main upload component
- `web/src/app/workspace/collections/[collectionId]/documents/upload/page.tsx` - Upload page
- `web/src/components/ui/file-upload.tsx` - File upload UI component
- `web/src/components/ui/progress.tsx` - Progress bar component
- `web/src/components/data-grid.tsx` - Data table component

### API Client
- `web/src/lib/api/client.ts` - API client configuration
- `web/src/api/` - Auto-generated API interfaces

### Internationalization
- `web/src/locales/en-US/page_documents.json` - English translations
- `web/src/locales/zh-CN/page_documents.json` - Chinese translations

## Summary

ApeRAG's document upload feature provides intuitive and reliable user experience through a **three-step guided process**:

1. **Step 1 - Select Files**: Drag & drop or click to select, instant frontend validation
2. **Step 2 - Upload Files**: Concurrent upload to temporary storage, real-time progress tracking
3. **Step 3 - Confirm Addition**: User selective confirmation, triggers index building

**Core Advantages**:
- 🎯 **User-Friendly**: Clear three-step process, explicit operation guidance
- ⚡ **Performance Optimized**: Concurrency control, pagination display, state management optimization
- 🔒 **High Reliability**: Duplicate detection, error handling, mid-upload cancellation support
- 🌍 **Internationalized**: Complete multi-language support
- 📱 **Responsive**: Adapts to mobile and desktop

This design ensures functional completeness while providing excellent user experience and system stability.

