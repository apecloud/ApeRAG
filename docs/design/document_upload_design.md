# ApeRAG Document Upload Module Data Flow

## Overview

This document details the complete data flow of the document upload module in the ApeRAG project, from frontend file upload to backend storage and index construction.

**Core Concept**: Adopts a **two-phase commit** design, first uploading to temporary state (UPLOADED), then formally adding to the knowledge base after user confirmation (PENDING → index building).

## Core Interfaces

1. **Upload File**: `POST /api/v1/collections/{collection_id}/documents/upload`
2. **Confirm Documents**: `POST /api/v1/collections/{collection_id}/documents/confirm`
3. **One-step Upload** (legacy): `POST /api/v1/collections/{collection_id}/documents`

## Data Flow Diagram

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
│  - upload_document_view()                                   │
│  - confirm_documents_view()                                 │
│  - JWT authentication, parameter validation                 │
└────────┬───────────────────────────────────┬────────────────┘
         │                                   │
         │ document_service.upload_document() │ document_service.confirm_documents()
         ▼                                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Service Layer: aperag/service/document_service.py          │
│  - File validation (type, size)                             │
│  - Duplicate detection (SHA-256 hash)                       │
│  - Quota check                                              │
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
│                    Data Storage Layer                       │
│                                                             │
│  ┌───────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  PostgreSQL   │  │   Object Store   │  │  Vector DB  │ │
│  │               │  │                  │  │             │ │
│  │ - document    │  │ - Local/S3       │  │ - Qdrant    │ │
│  │ - document_   │  │ - Original files │  │ - Vector    │ │
│  │   index       │  │ - Converted files│  │   indexes   │ │
│  │               │  │                  │  │             │ │
│  └───────────────┘  └──────────────────┘  └─────────────┘ │
│                                                             │
│  ┌───────────────┐  ┌──────────────────┐                  │
│  │ Elasticsearch │  │   Neo4j/PG       │                  │
│  │               │  │                  │                  │
│  │ - Fulltext    │  │ - Knowledge      │                  │
│  │   indexes     │  │   graph          │                  │
│  │               │  │                  │                  │
│  └───────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
               ┌───────────────────┐
               │  Celery Workers   │
               │  - Doc parsing    │
               │  - Chunking       │
               │  - Index building │
               └───────────────────┘
```

## Complete Process Details

### Phase 1: File Upload (Temporary Storage)

#### 1.1 View Layer - HTTP Request Handling

**File**: `aperag/views/collections.py`

```python
@router.post("/collections/{collection_id}/documents/upload", tags=["documents"])
@audit(resource_type="document", api_name="UploadDocument")
async def upload_document_view(
    request: Request,
    collection_id: str,
    file: UploadFile = File(...),
    user: User = Depends(required_user),
) -> view_models.UploadDocumentResponse:
    """Upload a single document file to temporary storage"""
    return await document_service.upload_document(str(user.id), collection_id, file)
```

**Responsibilities**:
- Receive multipart/form-data file uploads
- JWT Token authentication
- Extract path parameters (collection_id)
- Call Service layer
- Return `UploadDocumentResponse` (includes document_id, filename, size, status)

#### 1.2 Service Layer - Business Logic Orchestration

**File**: `aperag/service/document_service.py`

```python
async def upload_document(
    self, user_id: str, collection_id: str, file: UploadFile
) -> view_models.UploadDocumentResponse:
    """Upload a single document file to temporary storage with duplicate detection"""
    # 1. Validate collection exists and is active
    collection = await self._validate_collection(user_id, collection_id)
    
    # 2. Validate file type and size
    file_suffix = self._validate_file(file.filename, file.size)
    
    # 3. Read file content
    file_content = await file.read()
    
    # 4. Calculate file hash (SHA-256)
    file_hash = calculate_file_hash(file_content)
    
    # 5. Transaction processing
    async def _upload_document_atomically(session):
        # 5.1 Duplicate detection
        existing_doc = await self._check_duplicate_document(
            user_id, collection.id, file.filename, file_hash
        )
        
        if existing_doc:
            # Idempotent: return existing document
            return view_models.UploadDocumentResponse(
                document_id=existing_doc.id,
                filename=existing_doc.name,
                size=existing_doc.size,
                status=existing_doc.status,
            )
        
        # 5.2 Create new document (UPLOADED status)
        document_instance = await self._create_document_record(
            session=session,
            user=user_id,
            collection_id=collection.id,
            filename=file.filename,
            size=file.size,
            status=db_models.DocumentStatus.UPLOADED,  # Temporary status
            file_suffix=file_suffix,
            file_content=file_content,
            content_hash=file_hash,
        )
        
        return view_models.UploadDocumentResponse(
            document_id=document_instance.id,
            filename=document_instance.name,
            size=document_instance.size,
            status=document_instance.status,
        )
    
    return await self.db_ops.execute_with_transaction(_upload_document_atomically)
```

**Core Validation Logic**:

1. **Collection Validation** (`_validate_collection`)
   - Collection exists
   - Collection is in ACTIVE status

2. **File Validation** (`_validate_file`)
   - File extension is supported
   - File size within limit (default 100MB)

3. **Duplicate Detection** (`_check_duplicate_document`)
   - Query by filename and SHA-256 hash
   - If same name but different hash: throw `DocumentNameConflictException`
   - If same name and hash: return existing document (idempotent)

#### 1.3 Document Creation Logic

**Method**: `_create_document_record`

```python
async def _create_document_record(
    self,
    session: AsyncSession,
    user: str,
    collection_id: str,
    filename: str,
    size: int,
    status: db_models.DocumentStatus,
    file_suffix: str,
    file_content: bytes,
    custom_metadata: dict = None,
    content_hash: str = None,
) -> db_models.Document:
    # 1. Create database record
    document_instance = db_models.Document(
        user=user,
        name=filename,
        status=status,
        size=size,
        collection_id=collection_id,
        content_hash=content_hash,
    )
    session.add(document_instance)
    await session.flush()
    await session.refresh(document_instance)
    
    # 2. Upload to object storage
    async_obj_store = get_async_object_store()
    upload_path = f"{document_instance.object_store_base_path()}/original{file_suffix}"
    await async_obj_store.put(upload_path, file_content)
    
    # 3. Update metadata
    metadata = {"object_path": upload_path}
    if custom_metadata:
        metadata.update(custom_metadata)
    document_instance.doc_metadata = json.dumps(metadata)
    session.add(document_instance)
    await session.flush()
    
    return document_instance
```

**Object Storage Path Generation**:

```python
# Model method: aperag/db/models.py
def object_store_base_path(self) -> str:
    """Generate the base path for object store"""
    user = self.user.replace("|", "-")
    return f"user-{user}/{self.collection_id}/{self.id}"

# Example storage path:
# user-google-oauth2|123456/col_abc123/doc_xyz789/original.pdf
```

### Phase 2: Confirm Documents (Formal Addition)

#### 2.1 View Layer

**File**: `aperag/views/collections.py`

```python
@router.post("/collections/{collection_id}/documents/confirm", tags=["documents"])
@audit(resource_type="document", api_name="ConfirmDocuments")
async def confirm_documents_view(
    request: Request,
    collection_id: str,
    data: view_models.ConfirmDocumentsRequest,
    user: User = Depends(required_user),
) -> view_models.ConfirmDocumentsResponse:
    """Confirm uploaded documents and add them to collection"""
    return await document_service.confirm_documents(
        str(user.id), collection_id, data.document_ids
    )
```

#### 2.2 Service Layer - Confirmation Logic

**Method**: `confirm_documents`

```python
async def confirm_documents(
    self, user_id: str, collection_id: str, document_ids: list[str]
) -> view_models.ConfirmDocumentsResponse:
    """Confirm uploaded documents and trigger indexing"""
    # 1. Validate collection
    collection = await self._validate_collection(user_id, collection_id)
    
    # 2. Get collection configuration
    collection_config = json.loads(collection.config)
    index_types = self._get_index_types_for_collection(collection_config)
    
    confirmed_count = 0
    failed_count = 0
    failed_documents = []
    
    # 3. Transaction processing
    async def _confirm_documents_atomically(session):
        # 3.1 Check quota (deduct quota only at confirmation stage)
        await self._check_document_quotas(session, user_id, collection_id, len(document_ids))
        
        for document_id in document_ids:
            try:
                # 3.2 Validate document status
                stmt = select(db_models.Document).where(
                    db_models.Document.id == document_id,
                    db_models.Document.user == user_id,
                    db_models.Document.collection_id == collection_id,
                    db_models.Document.status == db_models.DocumentStatus.UPLOADED
                )
                result = await session.execute(stmt)
                document = result.scalar_one_or_none()
                
                if not document:
                    # Document doesn't exist or wrong status
                    failed_documents.append(...)
                    failed_count += 1
                    continue
                
                # 3.3 Update document status: UPLOADED → PENDING
                document.status = db_models.DocumentStatus.PENDING
                document.gmt_updated = utc_now()
                session.add(document)
                
                # 3.4 Create index records
                await document_index_manager.create_or_update_document_indexes(
                    document_id=document.id,
                    index_types=index_types,
                    session=session
                )
                
                confirmed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to confirm document {document_id}: {e}")
                failed_documents.append(...)
                failed_count += 1
        
        return confirmed_count, failed_count, failed_documents
    
    # 4. Execute transaction
    await self.db_ops.execute_with_transaction(_confirm_documents_atomically)
    
    # 5. Trigger index reconciliation task
    _trigger_index_reconciliation()
    
    return view_models.ConfirmDocumentsResponse(
        confirmed_count=confirmed_count,
        failed_count=failed_count,
        failed_documents=failed_documents
    )
```

**Index Type Configuration**:

```python
def _get_index_types_for_collection(self, collection_config: dict) -> list:
    """Get the list of index types to create based on collection configuration"""
    index_types = [
        db_models.DocumentIndexType.VECTOR,     # Vector index (required)
        db_models.DocumentIndexType.FULLTEXT,   # Fulltext index (required)
    ]
    
    if collection_config.get("enable_knowledge_graph", False):
        index_types.append(db_models.DocumentIndexType.GRAPH)
    if collection_config.get("enable_summary", False):
        index_types.append(db_models.DocumentIndexType.SUMMARY)
    if collection_config.get("enable_vision", False):
        index_types.append(db_models.DocumentIndexType.VISION)
    
    return index_types
```

### One-step Upload Interface (Legacy Compatibility)

**Interface**: `POST /api/v1/collections/{collection_id}/documents`

```python
@router.post("/collections/{collection_id}/documents", tags=["documents"])
async def create_documents_view(
    request: Request,
    collection_id: str,
    files: List[UploadFile] = File(...),
    user: User = Depends(required_user),
) -> view_models.DocumentList:
    return await document_service.create_documents(str(user.id), collection_id, files)
```

**Core Logic**:

- One-step completion: file upload + confirmation
- Directly create documents with PENDING status
- Immediately create index records
- Support batch upload of multiple files

## Data Storage Layer

### 1. PostgreSQL - Document Metadata

#### 1.1 Document Table

**File**: `aperag/db/models.py`

```python
class Document(Base):
    __tablename__ = "document"
    __table_args__ = (
        UniqueConstraint("collection_id", "name", "gmt_deleted", 
                        name="uq_document_collection_name_deleted"),
    )
    
    id = Column(String(24), primary_key=True, default=lambda: "doc" + random_id())
    name = Column(String(1024), nullable=False)
    user = Column(String(256), nullable=False, index=True)
    collection_id = Column(String(24), nullable=True, index=True)
    status = Column(EnumColumn(DocumentStatus), nullable=False, index=True)
    size = Column(BigInteger, nullable=False)
    content_hash = Column(String(64), nullable=True, index=True)  # SHA-256
    object_path = Column(Text, nullable=True)
    doc_metadata = Column(Text, nullable=True)  # JSON string
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)
```

**Status Enumeration** (`DocumentStatus`):

| Status | Description | When Set |
|--------|-------------|----------|
| `UPLOADED` | Uploaded to temporary storage | upload_document API |
| `PENDING` | Waiting for index building | confirm_documents API |
| `RUNNING` | Index building in progress | Celery task starts |
| `COMPLETE` | All indexes complete | All index statuses become ACTIVE |
| `FAILED` | Index building failed | Any index fails |
| `DELETED` | Deleted | delete_document API |
| `EXPIRED` | Temporary document expired | Scheduled cleanup task (not implemented) |

#### 1.2 DocumentIndex Table

```python
class DocumentIndex(Base):
    __tablename__ = "document_index"
    __table_args__ = (
        UniqueConstraint("document_id", "index_type", name="uq_document_index"),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(24), nullable=False, index=True)
    index_type = Column(EnumColumn(DocumentIndexType), nullable=False, index=True)
    status = Column(EnumColumn(DocumentIndexStatus), nullable=False, 
                   default=DocumentIndexStatus.PENDING, index=True)
    version = Column(Integer, nullable=False, default=1)
    observed_version = Column(Integer, nullable=False, default=0)
    index_data = Column(Text, nullable=True)  # JSON data
    error_message = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_last_reconciled = Column(DateTime(timezone=True), nullable=True)
```

**Index Types** (`DocumentIndexType`):

- `VECTOR`: Vector index (Qdrant, etc.)
- `FULLTEXT`: Fulltext index (Elasticsearch)
- `GRAPH`: Knowledge graph index (Neo4j/PostgreSQL)
- `SUMMARY`: Document summary
- `VISION`: Vision index (image content)

**Index Status** (`DocumentIndexStatus`):

| Status | Description |
|--------|-------------|
| `PENDING` | Waiting for processing |
| `CREATING` | Creating |
| `ACTIVE` | Ready for use |
| `DELETING` | Marked for deletion |
| `DELETION_IN_PROGRESS` | Deleting |
| `FAILED` | Failed |

### 2. Object Store - File Storage

#### 2.1 Storage Backend Configuration

**File**: `aperag/config.py`

```python
class Config(BaseSettings):
    # Object store type: "local" or "s3"
    object_store_type: str = Field("local", alias="OBJECT_STORE_TYPE")
    
    # Local storage config
    object_store_local_config: Optional[LocalObjectStoreConfig] = None
    
    # S3 storage config
    object_store_s3_config: Optional[S3Config] = None
```

**Environment Variable Configuration**:

```bash
# Local storage (default)
OBJECT_STORE_TYPE=local
OBJECT_STORE_LOCAL_ROOT_DIR=.objects

# S3 storage (MinIO/AWS S3)
OBJECT_STORE_TYPE=s3
OBJECT_STORE_S3_ENDPOINT=http://127.0.0.1:9000
OBJECT_STORE_S3_ACCESS_KEY=minioadmin
OBJECT_STORE_S3_SECRET_KEY=minioadmin
OBJECT_STORE_S3_BUCKET=aperag
OBJECT_STORE_S3_REGION=us-east-1
OBJECT_STORE_S3_PREFIX_PATH=
OBJECT_STORE_S3_USE_PATH_STYLE=true
```

#### 2.2 Object Storage Interface

**File**: `aperag/objectstore/base.py`

```python
class AsyncObjectStore(ABC):
    @abstractmethod
    async def put(self, path: str, data: bytes | IO[bytes]):
        """Upload object to storage"""
        ...
    
    @abstractmethod
    async def get(self, path: str) -> IO[bytes] | None:
        """Download object from storage"""
        ...
    
    @abstractmethod
    async def delete_objects_by_prefix(self, path_prefix: str):
        """Delete all objects with given prefix"""
        ...
```

**Factory Method**:

```python
def get_async_object_store() -> AsyncObjectStore:
    """Factory function to get an asynchronous AsyncObjectStore instance"""
    match settings.object_store_type:
        case "local":
            from aperag.objectstore.local import AsyncLocal, LocalConfig
            return AsyncLocal(LocalConfig(**config_dict))
        case "s3":
            from aperag.objectstore.s3 import AsyncS3, S3Config
            return AsyncS3(S3Config(**config_dict))
```

#### 2.3 Local Storage Implementation

**File**: `aperag/objectstore/local.py`

```python
class AsyncLocal(AsyncObjectStore):
    def __init__(self, cfg: LocalConfig):
        self._base_storage_path = Path(cfg.root_dir).resolve()
        self._base_storage_path.mkdir(parents=True, exist_ok=True)
    
    def _resolve_object_path(self, path: str) -> Path:
        """Resolve and validate object path (security check)"""
        path_components = Path(path.lstrip("/")).parts
        if ".." in path_components:
            raise ValueError("Invalid path: '..' not allowed")
        
        prospective_path = self._base_storage_path.joinpath(*path_components)
        normalized_path = Path(os.path.abspath(prospective_path))
        
        if self._base_storage_path not in normalized_path.parents:
            raise ValueError("Path traversal attempt detected")
        
        return prospective_path
    
    async def put(self, path: str, data: bytes | IO[bytes]):
        """Write file to local filesystem"""
        full_path = self._resolve_object_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(full_path, "wb") as f:
            if isinstance(data, bytes):
                await f.write(data)
            else:
                await f.write(data.read())
```

**Storage Path Example**:

```
.objects/
└── user-google-oauth2-123456/
    └── col_abc123/
        └── doc_xyz789/
            ├── original.pdf         # Original file
            ├── converted.pdf        # Converted PDF
            ├── chunks/              # Chunk data
            │   ├── chunk_0.json
            │   └── chunk_1.json
            └── images/              # Extracted images
                ├── image_0.png
                └── image_1.png
```

#### 2.4 S3 Storage Implementation

**File**: `aperag/objectstore/s3.py`

```python
class AsyncS3(AsyncObjectStore):
    def __init__(self, cfg: S3Config):
        self.cfg = cfg
        self._s3_client = None
    
    async def put(self, path: str, data: bytes | IO[bytes]):
        """Upload file to S3"""
        client = await self._get_client()
        path = self._final_path(path)
        
        if isinstance(data, bytes):
            data = BytesIO(data)
        
        await client.upload_fileobj(data, self.cfg.bucket, path)
    
    def _final_path(self, path: str) -> str:
        """Add prefix path if configured"""
        if self.cfg.prefix_path:
            return f"{self.cfg.prefix_path.rstrip('/')}/{path.lstrip('/')}"
        return path.lstrip('/')
```

### 3. Vector Database - Vector Indexes

**Supported Vector Databases**:

- Qdrant (default)
- Elasticsearch
- Other compatible vector databases

**Configuration Example**:

```bash
VECTOR_DB_TYPE=qdrant
VECTOR_DB_CONTEXT='{"url":"http://localhost","port":6333,"distance":"Cosine"}'
```

### 4. Elasticsearch - Fulltext Indexes

**Environment Variables**:

```bash
ES_HOST_NAME=127.0.0.1
ES_PORT=9200
ES_USER=
ES_PASSWORD=
ES_PROTOCOL=http
```

### 5. Knowledge Graph Storage

**Supported Backends**:

- Neo4j (recommended)
- PostgreSQL (simulated graph database)
- NebulaGraph

## Document Duplicate Detection Mechanism

### Detection Logic

**Method**: `_check_duplicate_document`

```python
async def _check_duplicate_document(
    self, user: str, collection_id: str, filename: str, file_hash: str
) -> db_models.Document | None:
    """
    Check if a document with the same name exists in the collection.
    Returns the existing document if found, None otherwise.
    Raises DocumentNameConflictException if same name but different file hash.
    """
    # 1. Query document with same name
    existing_doc = await self.db_ops.query_document_by_name_and_collection(
        user, collection_id, filename
    )
    
    if existing_doc:
        # 2. If no hash (legacy document), skip hash check
        if existing_doc.content_hash is None:
            logger.warning(f"Existing document {existing_doc.id} has no file hash")
            return existing_doc
        
        # 3. Same hash: true duplicate (idempotent)
        if existing_doc.content_hash == file_hash:
            return existing_doc
        
        # 4. Different hash: filename conflict
        raise DocumentNameConflictException(filename, collection_id)
    
    return None
```

### Hash Algorithm

**SHA-256 File Hash Calculation**:

```python
def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    import hashlib
    return hashlib.sha256(file_content).hexdigest()
```

### Duplicate Strategy

| Scenario | Filename | Hash | Behavior |
|----------|----------|------|----------|
| Exact duplicate | Same | Same | Return existing document (idempotent) |
| Filename conflict | Same | Different | Throw `DocumentNameConflictException` |
| New document | Different | - | Create new document |

## Quota Management

### Check Timing

**Check quota at confirmation stage** (not at upload stage), because:

1. Upload stage is only temporary storage, doesn't consume formal quota
2. Confirmation stage actually consumes resources (index building)
3. Allows users to upload first, then selectively confirm

### Quota Types

```python
async def _check_document_quotas(
    self, session: AsyncSession, user: str, collection_id: str, count: int
):
    """Check and consume document quotas"""
    # 1. Check and consume user global quota
    await quota_service.check_and_consume_quota(
        user, "max_document_count", count, session
    )
    
    # 2. Check per-collection quota
    stmt = select(func.count()).select_from(db_models.Document).where(
        db_models.Document.collection_id == collection_id,
        db_models.Document.status != db_models.DocumentStatus.DELETED,
        db_models.Document.status != db_models.DocumentStatus.UPLOADED,  # Don't count temporary documents
    )
    existing_doc_count = await session.scalar(stmt)
    
    # 3. Get quota limit
    stmt = select(UserQuota).where(
        UserQuota.user == user,
        UserQuota.key == "max_document_count_per_collection"
    )
    per_collection_quota = (await session.execute(stmt)).scalars().first()
    
    # 4. Validate not exceeded
    if per_collection_quota and (existing_doc_count + count) > per_collection_quota.quota_limit:
        raise QuotaExceededException(
            "max_document_count_per_collection",
            per_collection_quota.quota_limit,
            existing_doc_count
        )
```

### Default Quotas

**File**: `aperag/config.py`

```python
class Config(BaseSettings):
    max_document_count: int = Field(1000, alias="MAX_DOCUMENT_COUNT")
    max_document_size: int = Field(100 * 1024 * 1024, alias="MAX_DOCUMENT_SIZE")  # 100MB
```

## Asynchronous Task Processing (Celery)

### Index Reconciliation Mechanism

**File**: `aperag/service/document_service.py`

```python
def _trigger_index_reconciliation():
    """Trigger index reconciliation task in background"""
    try:
        from config.celery_tasks import reconcile_document_indexes
        reconcile_document_indexes.apply_async()
    except Exception as e:
        logger.warning(f"Failed to trigger index reconciliation task: {e}")
```

**Celery Task**: `config/celery_tasks.py`

```python
@celery_app.task(name="reconcile_document_indexes")
def reconcile_document_indexes():
    """Reconcile document indexes based on their status"""
    from aperag.index.manager import document_index_manager
    
    # Process PENDING status indexes
    document_index_manager.reconcile_pending_indexes()
    
    # Process DELETING status indexes
    document_index_manager.reconcile_deleting_indexes()
```

### Index Building Process

1. **Document Parsing**: DocParser parses document content
2. **Document Chunking**: Chunking strategy splits document
3. **Vectorization**: Embedding model generates vectors
4. **Vector Index**: Write to vector database
5. **Fulltext Index**: Write to Elasticsearch
6. **Knowledge Graph**: LightRAG extracts entities and relations
7. **Document Summary**: LLM generates summary (optional)
8. **Vision Index**: Extract and analyze images (optional)

## File Validation

### Supported File Types

**File**: `aperag/docparser/doc_parser.py`

```python
class DocParser:
    def supported_extensions(self) -> list:
        return [
            ".txt", ".md", ".html", ".pdf",
            ".docx", ".doc", ".pptx", ".ppt",
            ".xlsx", ".xls", ".csv",
            ".json", ".xml", ".yaml", ".yml",
            ".png", ".jpg", ".jpeg", ".gif", ".bmp",
            ".mp3", ".wav", ".m4a",
            # ... more formats
        ]
```

**Compressed File Support**:

```python
SUPPORTED_COMPRESSED_EXTENSIONS = [".zip", ".tar", ".gz", ".tgz"]
```

### Size Limit

```python
def _validate_file(self, filename: str, size: int) -> str:
    """Validate file extension and size"""
    supported_extensions = DocParser().supported_extensions()
    supported_extensions += SUPPORTED_COMPRESSED_EXTENSIONS
    
    file_suffix = os.path.splitext(filename)[1].lower()
    
    if file_suffix not in supported_extensions:
        raise invalid_param("file_type", f"unsupported file type {file_suffix}")
    
    if size > settings.max_document_size:
        raise invalid_param("file_size", "file size is too large")
    
    return file_suffix
```

## API Response Format

### UploadDocumentResponse

**Schema**: `aperag/api/components/schemas/document.yaml`

```yaml
uploadDocumentResponse:
  type: object
  properties:
    document_id:
      type: string
      description: ID of the uploaded document
    filename:
      type: string
      description: Name of the uploaded file
    size:
      type: integer
      description: Size of the uploaded file in bytes
    status:
      type: string
      enum:
        - UPLOADED
        - PENDING
        - RUNNING  
        - COMPLETE
        - FAILED
        - DELETED
        - EXPIRED
      description: Status of the document
  required:
    - document_id
    - filename
    - size
    - status
```

**Example**:

```json
{
  "document_id": "doc_xyz789abc",
  "filename": "user_manual.pdf",
  "size": 2048576,
  "status": "UPLOADED"
}
```

### ConfirmDocumentsResponse

```yaml
confirmDocumentsResponse:
  type: object
  properties:
    confirmed_count:
      type: integer
      description: Number of documents successfully confirmed
    failed_count:
      type: integer
      description: Number of documents that failed to confirm
    failed_documents:
      type: array
      items:
        type: object
        properties:
          document_id:
            type: string
          name:
            type: string
          error:
            type: string
  required:
    - confirmed_count
    - failed_count
```

**Example**:

```json
{
  "confirmed_count": 3,
  "failed_count": 1,
  "failed_documents": [
    {
      "document_id": "doc_fail123",
      "name": "corrupted.pdf",
      "error": "CONFIRMATION_FAILED"
    }
  ]
}
```

## Design Features

### 1. Two-Phase Commit Design

**Advantages**:

- ✅ Users can upload first then select: batch upload then selectively add
- ✅ Reduce unnecessary resource consumption: unconfirmed documents don't build indexes
- ✅ Better user experience: fast upload response, background async processing
- ✅ More reasonable quota control: only consume quota after confirmation

**Status Transition**:

```
Upload → UPLOADED → (User confirms) → PENDING → (Celery processes) → RUNNING → COMPLETE
                                                                         ↓
                                                                      FAILED
```

### 2. Idempotency Design

**Duplicate Upload Handling**:

- Same name and content (same hash): return existing document
- Same name different content (different hash): throw conflict exception
- Completely new document: create new record

**Benefits**:

- Network retransmission won't create duplicate documents
- Client can safely retry
- Avoid storage space waste

### 3. Multi-tenant Isolation

**Storage Path Isolation**:

```
user-{user_id}/{collection_id}/{document_id}/...
```

**Database Isolation**:

- All queries filter by user field
- Collection-level permission control
- Soft delete support (gmt_deleted)

### 4. Flexible Storage Backends

**Support Local and S3**:

- Local: suitable for development, testing, small-scale deployment
- S3: suitable for production, large-scale deployment
- Unified `AsyncObjectStore` interface
- Runtime configuration switching

### 5. Transaction Consistency

**Core operations within transactions**:

```python
async def _upload_document_atomically(session):
    # 1. Create database record
    # 2. Upload file to object storage
    # 3. Update metadata
    # All operations succeed to commit, any failure rolls back
```

**Benefits**:

- Avoid partially successful dirty data
- Database records and object storage remain consistent
- Automatic cleanup on failure

### 6. Clear Layered Architecture

```
View Layer (views/collections.py)
    ↓ calls
Service Layer (service/document_service.py)
    ↓ calls
Repository Layer (db/ops.py, objectstore/)
    ↓ accesses
Storage Layer (PostgreSQL, S3, Qdrant, ES, Neo4j)
```

**Separation of Concerns**:

- View: HTTP handling, parameter validation, authentication
- Service: business logic, transaction orchestration
- Repository: data access
- Storage: data persistence

## Performance Optimization

### 1. Chunked Upload (Planned, Not Implemented)

```python
# Large file chunked upload support
async def upload_document_chunk(
    document_id: str,
    chunk_index: int,
    chunk_data: bytes,
    total_chunks: int
):
    # Upload single chunk
    # Merge after all chunks complete
    pass
```

### 2. Batch Operations

- `confirm_documents` supports batch confirmation
- `delete_documents` supports batch deletion
- Batch query index status

### 3. Asynchronous Processing

- File upload returns immediately
- Index building executes asynchronously in Celery
- Frontend polls or uses WebSocket for progress

### 4. Object Storage Optimization

- S3 uses multipart upload
- Local uses aiofiles for async writes
- Support Range requests (partial download)

## Error Handling

### Common Exceptions

```python
# 1. Collection doesn't exist or unavailable
raise ResourceNotFoundException("Collection", collection_id)
raise CollectionInactiveException(collection_id)

# 2. File validation failed
raise invalid_param("file_type", f"unsupported file type {file_suffix}")
raise invalid_param("file_size", "file size is too large")

# 3. Duplicate conflict
raise DocumentNameConflictException(filename, collection_id)

# 4. Quota exceeded
raise QuotaExceededException("max_document_count", limit, current)

# 5. Document not found
raise DocumentNotFoundException(f"Document not found: {document_id}")
```

### Exception Handling Hierarchy

**Unified exception handling at View layer**:

```python
# aperag/exception_handlers.py
@app.exception_handler(BusinessException)
async def business_exception_handler(request: Request, exc: BusinessException):
    return JSONResponse(
        status_code=400,
        content={
            "error_code": exc.error_code.name,
            "message": str(exc)
        }
    )
```

## Related Files

### Core Implementation

- `aperag/views/collections.py` - View layer interface
- `aperag/service/document_service.py` - Service layer business logic
- `aperag/source/upload.py` - UploadSource implementation
- `aperag/db/models.py` - Database models
- `aperag/db/ops.py` - Database operations
- `aperag/api/components/schemas/document.yaml` - OpenAPI Schema

### Object Storage

- `aperag/objectstore/base.py` - Storage interface definition
- `aperag/objectstore/local.py` - Local storage implementation
- `aperag/objectstore/s3.py` - S3 storage implementation

### Document Processing

- `aperag/docparser/doc_parser.py` - Document parser
- `aperag/docparser/chunking.py` - Document chunking
- `aperag/index/manager.py` - Index manager
- `aperag/index/vector_index.py` - Vector index
- `aperag/index/fulltext_index.py` - Fulltext index
- `aperag/index/graph_index.py` - Graph index

### Task Queue

- `config/celery_tasks.py` - Celery task definitions
- `aperag/tasks/` - Task implementations

### Frontend Implementation

- `web/src/app/workspace/collections/[collectionId]/documents/page.tsx` - Document list page
- `web/src/components/documents/upload-documents.tsx` - Upload component

## Summary

ApeRAG's document upload module adopts a **two-phase commit + idempotency design + flexible storage** architecture:

1. **Two-Phase Commit**: Upload (UPLOADED) → Confirm (PENDING) → Index building
2. **SHA-256 Hash Deduplication**: Avoid duplicate documents, support idempotent uploads
3. **Flexible Storage Backends**: Local/S3 configurable switching
4. **Quota Management**: Deduct quota only at confirmation stage, reasonable resource control
5. **Multi-Index Coordination**: Vector, fulltext, graph, summary, vision multiple index types
6. **Clear Layered Architecture**: View → Service → Repository → Storage
7. **Celery Async Processing**: Index building doesn't block upload response
8. **Transaction Consistency**: Database and object storage operations are atomic

This design ensures performance while supporting complex document processing scenarios, with good scalability and fault tolerance.


