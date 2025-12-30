---
title: Document Upload Architecture Design
description: Detailed explanation of ApeRAG document upload module's complete architecture design, including upload process, temporary storage configuration, document parsing, format conversion, database design, etc.
keywords: [document upload, architecture, object store, parser, index building, two-phase commit]
---

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

For the complete documentation including:
- API Interface definitions
- File upload and temporary storage
- Document confirmation and index building
- Parser architecture and format conversion
- Index building flow
- Database design (document and document_index tables)
- State machine and lifecycle
- Async task scheduling (Celery)
- Design features and advantages
- Performance optimization
- Error handling

Please refer to the main design document at `/docs/design/document_upload_design.md`.

## Quick Reference

### API Endpoints

1. **Upload File**: `POST /api/v1/collections/{collection_id}/documents/upload`
2. **Confirm Documents**: `POST /api/v1/collections/{collection_id}/documents/confirm`
3. **One-step Upload**: `POST /api/v1/collections/{collection_id}/documents`

### Document Status Flow

```
[Upload] → UPLOADED → [Confirm] → PENDING → RUNNING → COMPLETE
                          ↓                     ↓
                       [Delete]              FAILED
                          ↓                     ↓
                       DELETED ←──────────────┘
```

### Object Storage Configuration

**Local Storage**:
```bash
OBJECT_STORE_TYPE=local
OBJECT_STORE_LOCAL_ROOT_DIR=.objects
```

**S3 Storage**:
```bash
OBJECT_STORE_TYPE=s3
OBJECT_STORE_S3_ENDPOINT=http://127.0.0.1:9000
OBJECT_STORE_S3_BUCKET=aperag
OBJECT_STORE_S3_ACCESS_KEY=minioadmin
OBJECT_STORE_S3_SECRET_KEY=minioadmin
```

### Supported Parsers

- **MinerUParser**: High-precision PDF parsing
- **DocRayParser**: Document layout analysis
- **ImageParser**: Image OCR and vision understanding
- **AudioParser**: Audio transcription
- **MarkItDownParser**: Universal fallback parser

### Index Types

| Type | Required | Storage |
|------|----------|---------|
| VECTOR | ✅ | Qdrant |
| FULLTEXT | ✅ | Elasticsearch |
| GRAPH | ❌ | Neo4j/PostgreSQL |
| SUMMARY | ❌ | PostgreSQL |
| VISION | ❌ | Qdrant + PostgreSQL |

## Related Files

### Backend Core
- `aperag/views/collections.py` - View layer
- `aperag/service/document_service.py` - Service layer
- `aperag/db/models.py` - Database models

### Object Storage
- `aperag/objectstore/base.py` - Storage interface
- `aperag/objectstore/local.py` - Local storage
- `aperag/objectstore/s3.py` - S3 storage

### Document Parsing
- `aperag/docparser/doc_parser.py` - Main parser
- `aperag/docparser/mineru_parser.py` - MinerU parser
- `aperag/docparser/docray_parser.py` - DocRay parser
- `aperag/docparser/markitdown_parser.py` - MarkItDown parser
- `aperag/docparser/image_parser.py` - Image parser
- `aperag/docparser/audio_parser.py` - Audio parser

### Index Building
- `aperag/index/vector_index.py` - Vector indexer
- `aperag/index/fulltext_index.py` - Full-text indexer
- `aperag/index/graph_index.py` - Graph indexer
- `aperag/index/summary_index.py` - Summary indexer
- `aperag/index/vision_index.py` - Vision indexer

### Task Scheduling
- `config/celery_tasks.py` - Celery tasks
- `aperag/tasks/reconciler.py` - Index reconciler
- `aperag/tasks/document.py` - Document tasks

### Frontend
- `web/src/app/workspace/collections/[collectionId]/documents/upload/document-upload.tsx` - Upload component

## Summary

ApeRAG's document upload module adopts a **two-phase commit + multi-parser chain invocation + parallel multi-index building** architecture:

**Core Features**:
1. ✅ **Two-Phase Commit**: Upload (temporary) → Confirm (formal), better UX
2. ✅ **SHA-256 Deduplication**: Prevents duplicates, idempotent upload
3. ✅ **Flexible Storage**: Local/S3 configurable, unified interface
4. ✅ **Multi-Parser**: MinerU, DocRay, MarkItDown, and more
5. ✅ **Auto Conversion**: PDF→images, audio→text, image→OCR
6. ✅ **Multi-Index**: Vector, full-text, graph, summary, vision
7. ✅ **Quota Management**: Deducted at confirmation stage
8. ✅ **Async Processing**: Celery task queue, non-blocking
9. ✅ **Transaction Consistency**: Database + object store 2PC
10. ✅ **Observability**: Audit logs, task tracking, error recording

For complete details, please refer to `/docs/design/document_upload_design.md`.
