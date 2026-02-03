# ApeRAG Document Upload Architecture Design

## Overview

This document details the complete architecture design of the document upload module in the ApeRAG project, covering the full pipeline from file upload, temporary storage, document parsing, format conversion to final index construction.

**Core Design Philosophy**: Adopts a **two-phase commit** pattern, separating file upload (temporary storage) from document confirmation (formal addition), providing better user experience and resource management capabilities.

## System Architecture

### Overall Architecture

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
│  - HTTP request handling                                    │
│  - JWT authentication                                       │
│  - Parameter validation                                     │
└────────┬───────────────────────────────────┬────────────────┘
         │                                   │
         │ document_service.upload_document() │ document_service.confirm_documents()
         ▼                                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Service Layer: aperag/service/document_service.py          │
│  - Business logic orchestration                             │
│  - File validation (type, size)                             │
│  - SHA-256 hash deduplication                               │
│  - Quota checking                                           │
│  - Transaction management                                   │
└────────┬───────────────────────────────────┬────────────────┘
         │                                   │
         │ Step 1                            │ Step 2
         ▼                                   ▼
┌────────────────────────┐     ┌────────────────────────────┐
│  1. Create Document    │     │  1. Update Document status │
│     status=UPLOADED    │     │     UPLOADED → PENDING     │
│  2. Save to ObjectStore│     │  2. Create DocumentIndex   │
│  3. Calculate hash     │     │  3. Trigger indexing tasks │
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
│  │ - document_   │  │ - Original files │  │ - Vectors   │ │
│  │   index       │  │ - Converted files│  │             │ │
│  └───────────────┘  └──────────────────┘  └─────────────┘ │
│                                                             │
│  ┌───────────────┐  ┌──────────────────┐                  │
│  │ Elasticsearch │  │   Neo4j/PG       │                  │
│  │               │  │                  │                  │
│  │ - Full-text   │  │ - Knowledge Graph│                  │
│  └───────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
               ┌───────────────────┐
               │  Celery Workers   │
               │                   │
               │  - Doc parsing    │
               │  - Format convert │
               │  - Content extract│
               │  - Doc chunking   │
               │  - Index building │
               └───────────────────┘
```

### Layered Architecture

```
┌─────────────────────────────────────────────┐
│  View Layer (views/collections.py)         │  HTTP handling, auth, validation
└─────────────────┬───────────────────────────┘
                  │ calls
┌─────────────────▼───────────────────────────┐
│  Service Layer (service/document_service.py)│  Business logic, transaction, permission
└─────────────────┬───────────────────────────┘
                  │ calls
┌─────────────────▼───────────────────────────┐
│  Repository Layer (db/ops.py, objectstore/) │  Data access abstraction
└─────────────────┬───────────────────────────┘
                  │ accesses
┌─────────────────▼───────────────────────────┐
│  Storage Layer (PG, S3, Qdrant, ES, Neo4j) │  Data persistence
└─────────────────────────────────────────────┘
```

## Core Process Details

### Phase 0: API Interface Definition

The system provides three main interfaces:

1. **Upload File** (Two-phase mode - Step 1)
   - Endpoint: `POST /api/v1/collections/{collection_id}/documents/upload`
   - Function: Upload file to temporary storage, status `UPLOADED`
   - Returns: `document_id`, `filename`, `size`, `status`

2. **Confirm Documents** (Two-phase mode - Step 2)
   - Endpoint: `POST /api/v1/collections/{collection_id}/documents/confirm`
   - Function: Confirm uploaded documents, trigger index building
   - Parameters: `document_ids` array
   - Returns: `confirmed_count`, `failed_count`, `failed_documents`

3. **One-step Upload** (Legacy mode, backward compatible)
   - Endpoint: `POST /api/v1/collections/{collection_id}/documents`
   - Function: Upload and directly add to knowledge base, status directly to `PENDING`
   - Supports batch upload

### Phase 1: File Upload and Temporary Storage

#### 1.1 Upload Flow

```
User selects files
    │
    ▼
Frontend calls upload API
    │
    ▼
View layer validates identity and params
    │
    ▼
Service layer processes business logic:
    │
    ├─► Verify collection exists and active
    │
    ├─► Validate file type and size
    │
    ├─► Read file content
    │
    ├─► Calculate SHA-256 hash
    │
    └─► Transaction processing:
        │
        ├─► Duplicate detection (by filename + hash)
        │   ├─ Exact match: Return existing doc (idempotent)
        │   ├─ Same name, different content: Throw conflict error
        │   └─ New document: Continue creation
        │
        ├─► Create Document record (status=UPLOADED)
        │
        ├─► Upload to object store
        │   └─ Path: user-{user_id}/{collection_id}/{document_id}/original{suffix}
        │
        └─► Update document metadata (object_path)
```

#### 1.2 File Validation

**Supported File Types**:
- Documents: `.pdf`, `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx`
- Text: `.txt`, `.md`, `.html`, `.json`, `.xml`, `.yaml`, `.yml`, `.csv`
- Images: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.tif`
- Audio: `.mp3`, `.wav`, `.m4a`
- Archives: `.zip`, `.tar`, `.gz`, `.tgz`

**Size Limits**:
- Default: 100 MB (configurable via `MAX_DOCUMENT_SIZE` environment variable)
- Extracted total size: 5 GB (`MAX_EXTRACTED_SIZE`)

#### 1.3 Duplicate Detection Mechanism

Uses **filename + SHA-256 hash** dual detection:

| Scenario | Filename | Hash | System Behavior |
|----------|----------|------|-----------------|
| Exact match | Same | Same | Return existing document (idempotent) |
| Name conflict | Same | Different | Throw `DocumentNameConflictException` |
| New document | Different | - | Create new document record |

**Advantages**:
- ✅ Supports idempotent upload: Network retries won't create duplicates
- ✅ Prevents content conflicts: Same name with different content prompts user
- ✅ Saves storage space: Same content stored only once

### Phase 2: Temporary Storage Configuration

#### 2.1 Object Storage Types

System supports two object storage backends, switchable via environment variables:

**1. Local Storage (Local filesystem)**

Use cases:
- Development and testing environments
- Small-scale deployments
- Single-machine deployments

Configuration:
```bash
# Development environment
OBJECT_STORE_TYPE=local
OBJECT_STORE_LOCAL_ROOT_DIR=.objects

# Docker environment
OBJECT_STORE_TYPE=local
OBJECT_STORE_LOCAL_ROOT_DIR=/shared/objects
```

Storage path example:
```
.objects/
└── user-google-oauth2-123456/
    └── col_abc123/
        └── doc_xyz789/
            ├── original.pdf              # Original file
            ├── converted.pdf             # Converted PDF
            ├── processed_content.md      # Parsed Markdown
            ├── chunks/                   # Chunked data
            │   ├── chunk_0.json
            │   └── chunk_1.json
            └── images/                   # Extracted images
                ├── page_0.png
                └── page_1.png
```

**2. S3 Storage (Compatible with AWS S3/MinIO/OSS, etc.)**

Use cases:
- Production environments
- Large-scale deployments
- Distributed deployments
- High availability and disaster recovery needs

Configuration:
```bash
OBJECT_STORE_TYPE=s3
OBJECT_STORE_S3_ENDPOINT=http://127.0.0.1:9000  # MinIO/S3 address
OBJECT_STORE_S3_REGION=us-east-1                # AWS Region
OBJECT_STORE_S3_ACCESS_KEY=minioadmin           # Access Key
OBJECT_STORE_S3_SECRET_KEY=minioadmin           # Secret Key
OBJECT_STORE_S3_BUCKET=aperag                   # Bucket name
OBJECT_STORE_S3_PREFIX_PATH=dev/                # Optional path prefix
OBJECT_STORE_S3_USE_PATH_STYLE=true             # Set to true for MinIO
```

#### 2.2 Object Storage Path Rules

**Path Format**:
```
{prefix}/user-{user_id}/{collection_id}/{document_id}/{filename}
```

**Components**:
- `prefix`: Optional global prefix (S3 only)
- `user_id`: User ID (`|` replaced with `-`)
- `collection_id`: Collection ID
- `document_id`: Document ID
- `filename`: Filename (e.g., `original.pdf`, `page_0.png`)

**Multi-tenancy Isolation**:
- Each user has an independent namespace
- Each collection has an independent storage directory
- Each document has an independent folder

### Phase 3: Document Confirmation and Index Building

#### 3.1 Confirmation Flow

```
User clicks "Save to Collection"
    │
    ▼
Frontend calls confirm API
    │
    ▼
Service layer processes:
    │
    ├─► Validate collection configuration
    │
    ├─► Check Quota (deduct quota at confirmation stage)
    │
    └─► For each document_id:
        │
        ├─► Verify document status is UPLOADED
        │
        ├─► Update document status: UPLOADED → PENDING
        │
        ├─► Create index records based on collection config:
        │   ├─ VECTOR (Vector index, required)
        │   ├─ FULLTEXT (Full-text index, required)
        │   ├─ GRAPH (Knowledge graph, optional)
        │   ├─ SUMMARY (Document summary, optional)
        │   └─ VISION (Vision index, optional)
        │
        └─► Return confirmation result
    │
    ▼
Trigger Celery task: reconcile_document_indexes
    │
    ▼
Background async index building
```

#### 3.2 Quota Management

**Check Timing**:
- ❌ Not checked during upload phase (temporary storage doesn't consume quota)
- ✅ Checked during confirmation phase (formal addition consumes quota)

**Quota Types**:

1. **User Global Quota**
   - `max_document_count`: Total document count limit per user
   - Default: 1000 (configurable via `MAX_DOCUMENT_COUNT`)

2. **Per-Collection Quota**
   - `max_document_count_per_collection`: Document count limit per collection
   - Excludes `UPLOADED` and `DELETED` status documents

**Quota Exceeded Handling**:
- Throws `QuotaExceededException`
- Returns HTTP 400 error
- Includes current usage and quota limit information

### Phase 4: Document Parsing and Format Conversion

#### 4.1 Parser Architecture

System uses a **multi-parser chain invocation** architecture, where each parser handles specific file types:

```
DocParser (Main Controller)
    │
    ├─► MinerUParser
    │   └─ Function: High-precision PDF parsing (commercial API)
    │   └─ Supports: .pdf
    │
    ├─► DocRayParser
    │   └─ Function: Document layout analysis and content extraction
    │   └─ Supports: .pdf, .docx, .pptx, .xlsx
    │
    ├─► ImageParser
    │   └─ Function: Image content recognition (OCR + vision understanding)
    │   └─ Supports: .jpg, .png, .gif, .bmp, .tiff
    │
    ├─► AudioParser
    │   └─ Function: Audio transcription (Speech-to-Text)
    │   └─ Supports: .mp3, .wav, .m4a
    │
    └─► MarkItDownParser (Fallback)
        └─ Function: Universal document to Markdown conversion
        └─ Supports: Almost all common formats
```

#### 4.2 Parser Configuration

**Configuration Method**: Dynamically controlled via Collection Config

```json
{
  "parser_config": {
    "use_mineru": false,           // Enable MinerU (requires API Token)
    "use_doc_ray": false,          // Enable DocRay
    "use_markitdown": true,        // Enable MarkItDown (default)
    "mineru_api_token": "xxx"      // MinerU API Token (optional)
  }
}
```

**Environment Variable Configuration**:
```bash
USE_MINERU_API=false              # Globally enable MinerU
MINERU_API_TOKEN=your_token       # MinerU API Token
```

#### 4.3 Parsing Flow

```
Celery Worker receives indexing task
    │
    ▼
1. Download original file from object store
    │
    ▼
2. Select Parser based on file extension
    │
    ├─► Try first matching Parser
    │   ├─ Success: Return parsing result
    │   └─ Failure: FallbackError → Try next Parser
    │
    └─► Final fallback: MarkItDownParser
    │
    ▼
3. Parsing result (Parts):
    │
    ├─► MarkdownPart: Text content
    │   └─ Contains: headings, paragraphs, lists, tables, etc.
    │
    ├─► PdfPart: PDF file
    │   └─ For: linearization, page rendering
    │
    └─► AssetBinPart: Binary resources
        └─ Contains: images, embedded files, etc.
    │
    ▼
4. Post-processing:
    │
    ├─► PDF pages to images (required for Vision index)
    │   └─ Each page rendered as PNG image
    │   └─ Saved to {document_path}/images/page_N.png
    │
    ├─► PDF linearization (speed up browser loading)
    │   └─ Use pikepdf to optimize PDF structure
    │   └─ Saved to {document_path}/converted.pdf
    │
    └─► Extract text content (plain text)
        └─ Merge all MarkdownPart content
        └─ Saved to {document_path}/processed_content.md
    │
    ▼
5. Save to object store
```

#### 4.4 Format Conversion Examples

**Example 1: PDF Document**
```
Input: user_manual.pdf (5 MB)
    │
    ▼
Parser selection: DocRayParser / MarkItDownParser
    │
    ▼
Output Parts:
    ├─ MarkdownPart: "# User Manual\n\n## Chapter 1\n..."
    └─ PdfPart: <original PDF data>
    │
    ▼
Post-processing:
    ├─ Render 50 pages to images → images/page_0.png ~ page_49.png
    ├─ Linearize PDF → converted.pdf
    └─ Extract text → processed_content.md
```

**Example 2: Image File**
```
Input: screenshot.png (2 MB)
    │
    ▼
Parser selection: ImageParser
    │
    ▼
Output Parts:
    ├─ MarkdownPart: "[OCR extracted text]"
    └─ AssetBinPart: <original image data> (vision_index=true)
    │
    ▼
Post-processing:
    └─ Save original image copy → images/file.png
```

**Example 3: Audio File**
```
Input: meeting_record.mp3 (50 MB)
    │
    ▼
Parser selection: AudioParser
    │
    ▼
Output Parts:
    └─ MarkdownPart: "[Transcribed meeting content]"
    │
    ▼
Post-processing:
    └─ Save transcription text → processed_content.md
```

### Phase 5: Index Building

#### 5.1 Index Types and Functions

| Index Type | Required | Function Description | Storage Location |
|-----------|----------|---------------------|------------------|
| **VECTOR** | ✅ Required | Vector retrieval, semantic search | Qdrant / Elasticsearch |
| **FULLTEXT** | ✅ Required | Full-text search, keyword search | Elasticsearch |
| **GRAPH** | ❌ Optional | Knowledge graph, entity & relation extraction | Neo4j / PostgreSQL |
| **SUMMARY** | ❌ Optional | Document summary, LLM generated | PostgreSQL (index_data) |
| **VISION** | ❌ Optional | Vision understanding, image content analysis | Qdrant (vectors) + PG (metadata) |

#### 5.2 Index Building Flow

```
Celery Worker: reconcile_document_indexes task
    │
    ▼
1. Scan DocumentIndex table, find indexes needing processing
    │
    ├─► PENDING status + observed_version < version
    │   └─ Need to create or update index
    │
    └─► DELETING status
        └─ Need to delete index
    │
    ▼
2. Group by document, process one by one
    │
    ▼
3. For each document:
    │
    ├─► parse_document (parse document)
    │   ├─ Download original file from object store
    │   ├─ Call DocParser to parse
    │   └─ Return ParsedDocumentData
    │
    └─► For each index type:
        │
        ├─► create_index (create/update index)
        │   │
        │   ├─ VECTOR index:
        │   │   ├─ Document chunking
        │   │   ├─ Generate vectors using Embedding model
        │   │   └─ Write to Qdrant
        │   │
        │   ├─ FULLTEXT index:
        │   │   ├─ Extract plain text content
        │   │   ├─ Chunk by paragraph/section
        │   │   └─ Write to Elasticsearch
        │   │
        │   ├─ GRAPH index:
        │   │   ├─ Extract entities using LightRAG
        │   │   ├─ Extract entity relationships
        │   │   └─ Write to Neo4j/PostgreSQL
        │   │
        │   ├─ SUMMARY index:
        │   │   ├─ Generate summary using LLM
        │   │   └─ Save to DocumentIndex.index_data
        │   │
        │   └─ VISION index:
        │       ├─ Extract image Assets
        │       ├─ Understand image content using Vision LLM
        │       ├─ Generate image description vectors
        │       └─ Write to Qdrant
        │
        └─► Update index status
            ├─ Success: CREATING → ACTIVE
            └─ Failure: CREATING → FAILED
    │
    ▼
4. Update document overall status
    │
    ├─ All indexes ACTIVE → Document.status = COMPLETE
    ├─ Any index FAILED → Document.status = FAILED
    └─ Some indexes still processing → Document.status = RUNNING
```

#### 5.3 Document Chunking

**Chunking Strategy**:
- Recursive character splitting (RecursiveCharacterTextSplitter)
- Prioritize splitting by natural paragraphs and sections
- Maintain context overlap

**Chunking Parameters**:
```json
{
  "chunk_size": 1000,           // Max characters per chunk
  "chunk_overlap": 200,         // Overlap characters
  "separators": ["\n\n", "\n", " ", ""]  // Separator priority
}
```

**Chunking Result Storage**:
```
{document_path}/chunks/
    ├─ chunk_0.json: {"text": "...", "metadata": {...}}
    ├─ chunk_1.json: {"text": "...", "metadata": {...}}
    └─ ...
```

## Database Design

### Table 1: document (Document Metadata)

**Table Structure**:

| Field | Type | Description | Index |
|-------|------|-------------|-------|
| `id` | String(24) | Document ID, primary key, format: `doc{random_id}` | PK |
| `name` | String(1024) | Filename | - |
| `user` | String(256) | User ID (supports multiple IDPs) | ✅ Index |
| `collection_id` | String(24) | Collection ID | ✅ Index |
| `status` | Enum | Document status (see table below) | ✅ Index |
| `size` | BigInteger | File size (bytes) | - |
| `content_hash` | String(64) | SHA-256 hash (for deduplication) | ✅ Index |
| `object_path` | Text | Object store path (deprecated, use doc_metadata) | - |
| `doc_metadata` | Text | Document metadata (JSON string) | - |
| `gmt_created` | DateTime(tz) | Creation time (UTC) | - |
| `gmt_updated` | DateTime(tz) | Update time (UTC) | - |
| `gmt_deleted` | DateTime(tz) | Deletion time (soft delete) | ✅ Index |

**Unique Constraint**:
```sql
UNIQUE INDEX uq_document_collection_name_active
  ON document (collection_id, name)
  WHERE gmt_deleted IS NULL;
```
- Within the same collection, active document names cannot be duplicated
- Deleted documents are excluded from uniqueness check

**Document Status Enum** (`DocumentStatus`):

| Status | Description | When Set | Visibility |
|--------|-------------|----------|------------|
| `UPLOADED` | Uploaded to temporary storage | `upload_document` API | Frontend file selection UI |
| `PENDING` | Waiting for index building | `confirm_documents` API | Document list (processing) |
| `RUNNING` | Index building in progress | Celery task starts processing | Document list (processing) |
| `COMPLETE` | All indexes completed | All indexes become ACTIVE | Document list (available) |
| `FAILED` | Index building failed | Any index fails | Document list (failed) |
| `DELETED` | Deleted | `delete_document` API | Not visible (soft delete) |
| `EXPIRED` | Temporary document expired | Scheduled cleanup task | Not visible |

**Document Metadata Example** (`doc_metadata` JSON field):
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

### Table 2: document_index (Index Status Management)

**Table Structure**:

| Field | Type | Description | Index |
|-------|------|-------------|-------|
| `id` | Integer | Auto-increment ID, primary key | PK |
| `document_id` | String(24) | Related document ID | ✅ Index |
| `index_type` | Enum | Index type (see table below) | ✅ Index |
| `status` | Enum | Index status (see table below) | ✅ Index |
| `version` | Integer | Index version number | - |
| `observed_version` | Integer | Processed version number | - |
| `index_data` | Text | Index data (JSON), e.g., summary content | - |
| `error_message` | Text | Error message (on failure) | - |
| `gmt_created` | DateTime(tz) | Creation time | - |
| `gmt_updated` | DateTime(tz) | Update time | - |
| `gmt_last_reconciled` | DateTime(tz) | Last reconciliation time | - |

**Unique Constraint**:
```sql
UNIQUE CONSTRAINT uq_document_index
  ON document_index (document_id, index_type);
```
- Each document has only one record per index type

**Index Type Enum** (`DocumentIndexType`):

| Type | Value | Description | External Storage |
|------|-------|-------------|------------------|
| `VECTOR` | "VECTOR" | Vector index | Qdrant / Elasticsearch |
| `FULLTEXT` | "FULLTEXT" | Full-text index | Elasticsearch |
| `GRAPH` | "GRAPH" | Knowledge graph | Neo4j / PostgreSQL |
| `SUMMARY` | "SUMMARY" | Document summary | PostgreSQL (index_data) |
| `VISION` | "VISION" | Vision index | Qdrant + PostgreSQL |

**Index Status Enum** (`DocumentIndexStatus`):

| Status | Description | When Set |
|--------|-------------|----------|
| `PENDING` | Waiting for processing | `confirm_documents` creates index record |
| `CREATING` | Creating | Celery Worker starts processing |
| `ACTIVE` | Ready for use | Index building successful |
| `DELETING` | Marked for deletion | `delete_document` API |
| `DELETION_IN_PROGRESS` | Deleting | Celery Worker is deleting |
| `FAILED` | Failed | Index building failed |

**Version Control Mechanism**:
- `version`: Expected index version (incremented on document update)
- `observed_version`: Processed version number
- When `version > observed_version`, triggers index update

**Reconciler**:
```python
# Query indexes needing processing
SELECT * FROM document_index
WHERE status = 'PENDING'
  AND observed_version < version;

# Update after processing
UPDATE document_index
SET status = 'ACTIVE',
    observed_version = version,
    gmt_last_reconciled = NOW()
WHERE id = ?;
```

### Table Relationship Diagram

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
│  collection_id (FK)             │◄──── Unique constraint: (collection_id, name)
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
│  document_id (FK)               │◄──── Unique constraint: (document_id, index_type)
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

## State Machine and Lifecycle

### Document State Transitions

```
         ┌─────────────────────────────────────────────┐
         │                                             │
         │                                             ▼
    [Upload] ──► UPLOADED ──► [Confirm] ──► PENDING ──► RUNNING ──► COMPLETE
                     │                                   │
                     │                                   ▼
                     │                                FAILED
                     │                                   │
                     │                                   ▼
                     └──────► [Delete] ──────────────► DELETED
                                                         │
                     ┌───────────────────────────────────┘
                     │
                     ▼
                  EXPIRED (Scheduled cleanup of unconfirmed docs)
```

**Key Transitions**:
1. **UPLOADED → PENDING**: User clicks "Save to Collection"
2. **PENDING → RUNNING**: Celery Worker starts processing
3. **RUNNING → COMPLETE**: All indexes successful
4. **RUNNING → FAILED**: Any index fails
5. **Any status → DELETED**: User deletes document

### Index State Transitions

```
  [Create index record] ──► PENDING ──► CREATING ──► ACTIVE
                                           │
                                           ▼
                                        FAILED
                                           │
                                           ▼
                             ┌──────────► PENDING (retry)
                             │
    [Delete request] ────────┼──────────► DELETING ──► DELETION_IN_PROGRESS ──► (record deleted)
                             │
                             └──────────► (directly delete record, if PENDING/FAILED)
```

## Async Task Scheduling (Celery)

### Task Definitions

**Main Task**: `reconcile_document_indexes`
- Trigger timing:
  - After `confirm_documents` API call
  - Scheduled task (every 30 seconds)
  - Manual trigger (admin interface)
- Function: Scan `document_index` table, process indexes needing reconciliation

**Sub-tasks**:
- `parse_document_task`: Parse document content
- `create_vector_index_task`: Create vector index
- `create_fulltext_index_task`: Create full-text index
- `create_graph_index_task`: Create knowledge graph index
- `create_summary_index_task`: Create summary index
- `create_vision_index_task`: Create vision index

### Task Scheduling Strategy

**Concurrency Control**:
- Each Worker processes at most N documents simultaneously (default 4)
- Multiple indexes of each document can be built in parallel
- Use Celery's `task_acks_late=True` to ensure tasks aren't lost

**Failure Retry**:
- Maximum 3 retries
- Exponential backoff (1 min → 5 min → 15 min)
- Marked as `FAILED` after 3 failures

**Idempotency**:
- All tasks support repeated execution
- Use `observed_version` mechanism to avoid duplicate processing
- Same input produces same output

## Design Features and Advantages

### 1. Two-Phase Commit Design

**Advantages**:
- ✅ **Better User Experience**: Fast upload response, doesn't block user operations
- ✅ **Selective Addition**: Can selectively confirm partial files after batch upload
- ✅ **Reasonable Resource Control**: Unconfirmed documents don't build indexes, don't consume quota
- ✅ **Failure Recovery Friendly**: Temporary documents can be periodically cleaned up without affecting business

**Status Isolation**:
```
Temporary status (UPLOADED):
  - Not counted in quota
  - Doesn't trigger indexing
  - Can be automatically cleaned up

Formal status (PENDING/RUNNING/COMPLETE):
  - Counted in quota
  - Triggers index building
  - Won't be automatically cleaned up
```

### 2. Idempotency Design

**File-Level Idempotency**:
- SHA-256 hash deduplication
- Same file uploaded multiple times returns same `document_id`
- Avoids storage space waste

**API-Level Idempotency**:
- `upload_document`: Repeated upload returns existing document
- `confirm_documents`: Repeated confirmation doesn't create duplicate indexes
- `delete_document`: Repeated deletion returns success (soft delete)

### 3. Multi-Tenancy Isolation

**Storage Isolation**:
```
user-{user_A}/...  # User A's files
user-{user_B}/...  # User B's files
```

**Database Isolation**:
- All queries filter by `user` field
- Collection-level permission control (`collection.user`)
- Soft delete support (`gmt_deleted`)

### 4. Flexible Storage Backend

**Unified Interface**:
```python
AsyncObjectStore:
  - put(path, data)
  - get(path)
  - delete_objects_by_prefix(prefix)
```

**Runtime Switching**:
- Switch between Local/S3 via environment variables
- No need to modify business code
- Supports custom storage backends (just implement the interface)

### 5. Transaction Consistency

**Two-Phase Commit for Database + Object Store**:
```python
async with transaction:
    # 1. Create database record
    document = create_document_record()
    
    # 2. Upload to object store
    await object_store.put(path, data)
    
    # 3. Update metadata
    document.doc_metadata = json.dumps(metadata)
    
    # All operations succeed to commit, any failure rolls back
```

**Failure Handling**:
- Database record creation fails: Don't upload file
- File upload fails: Rollback database record
- Metadata update fails: Rollback previous operations

### 6. Observability

**Audit Logging**:
- `@audit` decorator records all document operations
- Includes: user, time, operation type, resource ID

**Task Tracking**:
- `gmt_last_reconciled`: Last processing time
- `error_message`: Failure reason
- Celery task ID: Link log tracing

**Monitoring Metrics**:
- Document upload rate
- Index building duration
- Failure rate statistics

## Performance Optimization

### 1. Async Processing

**Upload Doesn't Block**:
- Returns immediately after file upload to object store
- Index building executes asynchronously in Celery
- Frontend gets progress via polling or WebSocket

### 2. Batch Operations

**Batch Confirmation**:
```python
confirm_documents(document_ids=[id1, id2, ..., idN])
```
- Process multiple documents in one transaction
- Batch create index records
- Reduce database round-trips

### 3. Caching Strategy

**Parsing Result Cache**:
- Parsed content saved to `processed_content.md`
- Subsequent index rebuilds can read directly without re-parsing

**Chunking Result Cache**:
- Chunking results saved to `chunks/` directory
- Vector index rebuilds can reuse chunking results

### 4. Parallel Index Building

**Multiple Indexes in Parallel**:
```python
# VECTOR, FULLTEXT, GRAPH can be built in parallel
await asyncio.gather(
    create_vector_index(),
    create_fulltext_index(),
    create_graph_index()
)
```

## Error Handling

### Common Exceptions

| Exception Type | HTTP Status | Trigger Scenario | Handling Suggestion |
|---------------|-------------|------------------|---------------------|
| `ResourceNotFoundException` | 404 | Collection/document doesn't exist | Check if ID is correct |
| `CollectionInactiveException` | 400 | Collection not active | Wait for collection initialization |
| `DocumentNameConflictException` | 409 | Same name, different content | Rename file or delete old document |
| `QuotaExceededException` | 429 | Quota exceeded | Upgrade plan or delete old documents |
| `InvalidFileTypeException` | 400 | Unsupported file type | Check supported file type list |
| `FileSizeTooLargeException` | 413 | File too large | Split file or compress |

### Exception Propagation

```
Service Layer throws exception
    │
    ▼
View Layer catches and converts
    │
    ▼
Exception Handler unified handling
    │
    ▼
Return standard JSON response:
{
  "error_code": "QUOTA_EXCEEDED",
  "message": "Document count limit exceeded",
  "details": {
    "limit": 1000,
    "current": 1000
  }
}
```

## Related Files Index

### Core Implementation

- **View Layer**: `aperag/views/collections.py` - HTTP interface definition
- **Service Layer**: `aperag/service/document_service.py` - Business logic
- **Database Models**: `aperag/db/models.py` - Document, DocumentIndex table definitions
- **Database Operations**: `aperag/db/ops.py` - CRUD operation encapsulation

### Object Storage

- **Interface Definition**: `aperag/objectstore/base.py` - AsyncObjectStore abstract class
- **Local Implementation**: `aperag/objectstore/local.py` - Local filesystem storage
- **S3 Implementation**: `aperag/objectstore/s3.py` - S3-compatible storage

### Document Parsing

- **Main Controller**: `aperag/docparser/doc_parser.py` - DocParser
- **Parser Implementations**:
  - `aperag/docparser/mineru_parser.py` - MinerU PDF parsing
  - `aperag/docparser/docray_parser.py` - DocRay document parsing
  - `aperag/docparser/markitdown_parser.py` - MarkItDown universal parsing
  - `aperag/docparser/image_parser.py` - Image OCR
  - `aperag/docparser/audio_parser.py` - Audio transcription
- **Document Processing**: `aperag/index/document_parser.py` - Parsing flow orchestration

### Index Building

- **Index Management**: `aperag/index/manager.py` - DocumentIndexManager
- **Vector Index**: `aperag/index/vector_index.py` - VectorIndexer
- **Full-text Index**: `aperag/index/fulltext_index.py` - FulltextIndexer
- **Knowledge Graph**: `aperag/index/graph_index.py` - GraphIndexer
- **Document Summary**: `aperag/index/summary_index.py` - SummaryIndexer
- **Vision Index**: `aperag/index/vision_index.py` - VisionIndexer

### Task Scheduling

- **Task Definitions**: `config/celery_tasks.py` - Celery task registration
- **Reconciler**: `aperag/tasks/reconciler.py` - DocumentIndexReconciler
- **Document Tasks**: `aperag/tasks/document.py` - DocumentIndexTask

### Frontend Implementation

- **Document List**: `web/src/app/workspace/collections/[collectionId]/documents/page.tsx`
- **Document Upload**: `web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx`

## Summary

ApeRAG's document upload module adopts a **two-phase commit + multi-parser chain invocation + parallel multi-index building** architecture design:

**Core Features**:
1. ✅ **Two-Phase Commit**: Upload (temporary storage) → Confirm (formal addition), providing better user experience
2. ✅ **SHA-256 Deduplication**: Prevents duplicate documents, supports idempotent upload
3. ✅ **Flexible Storage Backend**: Local/S3 configurable switching, unified interface abstraction
4. ✅ **Multi-Parser Architecture**: Supports MinerU, DocRay, MarkItDown and other parsers
5. ✅ **Automatic Format Conversion**: PDF→images, audio→text, images→OCR text
6. ✅ **Multi-Index Coordination**: Five index types: vector, full-text, graph, summary, vision
7. ✅ **Quota Management**: Quota deducted at confirmation stage, reasonable resource control
8. ✅ **Async Processing**: Celery task queue, doesn't block user operations
9. ✅ **Transaction Consistency**: Two-phase commit for database + object store
10. ✅ **Observability**: Audit logs, task tracking, complete error information recording

This design ensures both high performance and scalability, supports complex document processing scenarios (multi-format, multi-language, multi-modal), while maintaining good fault tolerance and user experience.
