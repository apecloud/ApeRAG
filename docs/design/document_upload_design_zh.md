# ApeRAG 文档上传模块数据流程

## 概述

本文档详细说明ApeRAG项目中文档上传模块的完整数据流程，从前端文件上传到后端存储、索引构建的全链路实现。

**核心理念**: 采用**两阶段提交**设计，先上传到临时状态（UPLOADED），用户确认后再正式添加到知识库（PENDING → 索引构建）。

## 核心接口

1. **上传文件**: `POST /api/v1/collections/{collection_id}/documents/upload`
2. **确认文档**: `POST /api/v1/collections/{collection_id}/documents/confirm`
3. **一步上传**（旧接口）: `POST /api/v1/collections/{collection_id}/documents`

## 数据流图

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
│  - JWT身份验证、参数验证                                      │
└────────┬───────────────────────────────────┬────────────────┘
         │                                   │
         │ document_service.upload_document() │ document_service.confirm_documents()
         ▼                                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Service Layer: aperag/service/document_service.py          │
│  - 文件验证（类型、大小）                                     │
│  - 重复检测（SHA-256 hash）                                  │
│  - Quota检查                                                │
│  - 事务管理                                                  │
└────────┬───────────────────────────────────┬────────────────┘
         │                                   │
         │ Step 1                            │ Step 2
         ▼                                   ▼
┌────────────────────────┐     ┌────────────────────────────┐
│  1. 创建Document记录    │     │  1. 更新Document状态       │
│     status=UPLOADED    │     │     UPLOADED → PENDING     │
│  2. 保存到ObjectStore  │     │  2. 创建DocumentIndex记录  │
│  3. 计算content_hash   │     │  3. 触发索引构建任务        │
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
│  │ - document_   │  │ - 原始文件        │  │ - 向量索引  │ │
│  │   index       │  │ - 转换后的文件    │  │             │ │
│  │               │  │                  │  │             │ │
│  └───────────────┘  └──────────────────┘  └─────────────┘ │
│                                                             │
│  ┌───────────────┐  ┌──────────────────┐                  │
│  │ Elasticsearch │  │   Neo4j/PG       │                  │
│  │               │  │                  │                  │
│  │ - 全文索引     │  │ - 知识图谱       │                  │
│  │               │  │                  │                  │
│  └───────────────┘  └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
               ┌───────────────────┐
               │  Celery Workers   │
               │  - 文档解析        │
               │  - 分块处理        │
               │  - 索引构建        │
               └───────────────────┘
```

## 完整流程详解

### 阶段1: 文件上传（临时存储）

#### 1.1 View层 - HTTP请求处理

**文件**: `aperag/views/collections.py`

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

**职责**:
- 接收 multipart/form-data 文件上传
- JWT Token身份验证
- 提取路径参数 (collection_id)
- 调用Service层
- 返回`UploadDocumentResponse`（包含document_id、filename、size、status）

#### 1.2 Service层 - 业务逻辑编排

**文件**: `aperag/service/document_service.py`

```python
async def upload_document(
    self, user_id: str, collection_id: str, file: UploadFile
) -> view_models.UploadDocumentResponse:
    """Upload a single document file to temporary storage with duplicate detection"""
    # 1. 验证集合存在且激活
    collection = await self._validate_collection(user_id, collection_id)
    
    # 2. 验证文件类型和大小
    file_suffix = self._validate_file(file.filename, file.size)
    
    # 3. 读取文件内容
    file_content = await file.read()
    
    # 4. 计算文件哈希（SHA-256）
    file_hash = calculate_file_hash(file_content)
    
    # 5. 事务处理
    async def _upload_document_atomically(session):
        # 5.1 重复检测
        existing_doc = await self._check_duplicate_document(
            user_id, collection.id, file.filename, file_hash
        )
        
        if existing_doc:
            # 幂等操作：返回已存在文档
            return view_models.UploadDocumentResponse(
                document_id=existing_doc.id,
                filename=existing_doc.name,
                size=existing_doc.size,
                status=existing_doc.status,
            )
        
        # 5.2 创建新文档（UPLOADED状态）
        document_instance = await self._create_document_record(
            session=session,
            user=user_id,
            collection_id=collection.id,
            filename=file.filename,
            size=file.size,
            status=db_models.DocumentStatus.UPLOADED,  # 临时状态
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

**核心验证逻辑**:

1. **集合验证** (`_validate_collection`)
   - 集合是否存在
   - 集合是否处于ACTIVE状态

2. **文件验证** (`_validate_file`)
   - 文件扩展名是否支持
   - 文件大小是否超过限制（默认100MB）

3. **重复检测** (`_check_duplicate_document`)
   - 按文件名和SHA-256哈希查询
   - 如果文件名相同但哈希不同：抛出`DocumentNameConflictException`
   - 如果文件名和哈希都相同：返回已存在文档（幂等）

#### 1.3 文档创建逻辑

**方法**: `_create_document_record`

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
    # 1. 创建数据库记录
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
    
    # 2. 上传到对象存储
    async_obj_store = get_async_object_store()
    upload_path = f"{document_instance.object_store_base_path()}/original{file_suffix}"
    await async_obj_store.put(upload_path, file_content)
    
    # 3. 更新元数据
    metadata = {"object_path": upload_path}
    if custom_metadata:
        metadata.update(custom_metadata)
    document_instance.doc_metadata = json.dumps(metadata)
    session.add(document_instance)
    await session.flush()
    
    return document_instance
```

**对象存储路径生成**:

```python
# 模型方法：aperag/db/models.py
def object_store_base_path(self) -> str:
    """Generate the base path for object store"""
    user = self.user.replace("|", "-")
    return f"user-{user}/{self.collection_id}/{self.id}"

# 实际存储路径示例：
# user-google-oauth2|123456/col_abc123/doc_xyz789/original.pdf
```

### 阶段2: 确认文档（正式添加）

#### 2.1 View层

**文件**: `aperag/views/collections.py`

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

#### 2.2 Service层 - 确认逻辑

**方法**: `confirm_documents`

```python
async def confirm_documents(
    self, user_id: str, collection_id: str, document_ids: list[str]
) -> view_models.ConfirmDocumentsResponse:
    """Confirm uploaded documents and trigger indexing"""
    # 1. 验证集合
    collection = await self._validate_collection(user_id, collection_id)
    
    # 2. 获取集合配置
    collection_config = json.loads(collection.config)
    index_types = self._get_index_types_for_collection(collection_config)
    
    confirmed_count = 0
    failed_count = 0
    failed_documents = []
    
    # 3. 事务处理
    async def _confirm_documents_atomically(session):
        # 3.1 检查Quota（确认阶段才扣除配额）
        await self._check_document_quotas(session, user_id, collection_id, len(document_ids))
        
        for document_id in document_ids:
            try:
                # 3.2 验证文档状态
                stmt = select(db_models.Document).where(
                    db_models.Document.id == document_id,
                    db_models.Document.user == user_id,
                    db_models.Document.collection_id == collection_id,
                    db_models.Document.status == db_models.DocumentStatus.UPLOADED
                )
                result = await session.execute(stmt)
                document = result.scalar_one_or_none()
                
                if not document:
                    # 文档不存在或状态不对
                    failed_documents.append(...)
                    failed_count += 1
                    continue
                
                # 3.3 更新文档状态：UPLOADED → PENDING
                document.status = db_models.DocumentStatus.PENDING
                document.gmt_updated = utc_now()
                session.add(document)
                
                # 3.4 创建索引记录
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
    
    # 4. 执行事务
    await self.db_ops.execute_with_transaction(_confirm_documents_atomically)
    
    # 5. 触发索引协调任务
    _trigger_index_reconciliation()
    
    return view_models.ConfirmDocumentsResponse(
        confirmed_count=confirmed_count,
        failed_count=failed_count,
        failed_documents=failed_documents
    )
```

**索引类型配置**:

```python
def _get_index_types_for_collection(self, collection_config: dict) -> list:
    """Get the list of index types to create based on collection configuration"""
    index_types = [
        db_models.DocumentIndexType.VECTOR,     # 向量索引（必选）
        db_models.DocumentIndexType.FULLTEXT,   # 全文索引（必选）
    ]
    
    if collection_config.get("enable_knowledge_graph", False):
        index_types.append(db_models.DocumentIndexType.GRAPH)
    if collection_config.get("enable_summary", False):
        index_types.append(db_models.DocumentIndexType.SUMMARY)
    if collection_config.get("enable_vision", False):
        index_types.append(db_models.DocumentIndexType.VISION)
    
    return index_types
```

### 一步上传接口（兼容旧版）

**接口**: `POST /api/v1/collections/{collection_id}/documents`

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

**核心逻辑**:

- 一次性完成：文件上传 + 确认添加
- 直接创建状态为 PENDING 的文档
- 立即创建索引记录
- 支持批量上传多个文件

## 数据存储层

### 1. PostgreSQL - 文档元数据

#### 1.1 Document表

**文件**: `aperag/db/models.py`

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
    doc_metadata = Column(Text, nullable=True)  # JSON字符串
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_deleted = Column(DateTime(timezone=True), nullable=True, index=True)
```

**状态枚举** (`DocumentStatus`):

| 状态 | 说明 | 何时设置 |
|------|------|----------|
| `UPLOADED` | 已上传到临时存储 | upload_document 接口 |
| `PENDING` | 等待索引构建 | confirm_documents 接口 |
| `RUNNING` | 索引构建中 | Celery任务开始处理 |
| `COMPLETE` | 所有索引完成 | 所有索引状态变为ACTIVE |
| `FAILED` | 索引构建失败 | 任一索引失败 |
| `DELETED` | 已删除 | delete_document 接口 |
| `EXPIRED` | 临时文档过期 | 定时清理任务（未实现） |

#### 1.2 DocumentIndex表

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
    index_data = Column(Text, nullable=True)  # JSON数据
    error_message = Column(Text, nullable=True)
    gmt_created = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_updated = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    gmt_last_reconciled = Column(DateTime(timezone=True), nullable=True)
```

**索引类型** (`DocumentIndexType`):

- `VECTOR`: 向量索引（Qdrant等）
- `FULLTEXT`: 全文索引（Elasticsearch）
- `GRAPH`: 知识图谱索引（Neo4j/PostgreSQL）
- `SUMMARY`: 文档摘要
- `VISION`: 视觉索引（图片内容）

**索引状态** (`DocumentIndexStatus`):

| 状态 | 说明 |
|------|------|
| `PENDING` | 等待处理 |
| `CREATING` | 创建中 |
| `ACTIVE` | 就绪可用 |
| `DELETING` | 标记删除 |
| `DELETION_IN_PROGRESS` | 删除中 |
| `FAILED` | 失败 |

### 2. Object Store - 文件存储

#### 2.1 存储后端配置

**文件**: `aperag/config.py`

```python
class Config(BaseSettings):
    # Object store type: "local" or "s3"
    object_store_type: str = Field("local", alias="OBJECT_STORE_TYPE")
    
    # Local storage config
    object_store_local_config: Optional[LocalObjectStoreConfig] = None
    
    # S3 storage config
    object_store_s3_config: Optional[S3Config] = None
```

**环境变量配置**:

```bash
# Local存储（默认）
OBJECT_STORE_TYPE=local
OBJECT_STORE_LOCAL_ROOT_DIR=.objects

# S3存储（MinIO/AWS S3）
OBJECT_STORE_TYPE=s3
OBJECT_STORE_S3_ENDPOINT=http://127.0.0.1:9000
OBJECT_STORE_S3_ACCESS_KEY=minioadmin
OBJECT_STORE_S3_SECRET_KEY=minioadmin
OBJECT_STORE_S3_BUCKET=aperag
OBJECT_STORE_S3_REGION=us-east-1
OBJECT_STORE_S3_PREFIX_PATH=
OBJECT_STORE_S3_USE_PATH_STYLE=true
```

#### 2.2 对象存储接口

**文件**: `aperag/objectstore/base.py`

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

**工厂方法**:

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

#### 2.3 Local存储实现

**文件**: `aperag/objectstore/local.py`

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

**存储路径示例**:

```
.objects/
└── user-google-oauth2-123456/
    └── col_abc123/
        └── doc_xyz789/
            ├── original.pdf         # 原始文件
            ├── converted.pdf        # 转换后的PDF
            ├── chunks/              # 分块数据
            │   ├── chunk_0.json
            │   └── chunk_1.json
            └── images/              # 提取的图片
                ├── image_0.png
                └── image_1.png
```

#### 2.4 S3存储实现

**文件**: `aperag/objectstore/s3.py`

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

### 3. 向量数据库 - 向量索引

**支持的向量数据库**:

- Qdrant（默认）
- Elasticsearch
- 其他兼容接口的向量数据库

**配置示例**:

```bash
VECTOR_DB_TYPE=qdrant
VECTOR_DB_CONTEXT='{"url":"http://localhost","port":6333,"distance":"Cosine"}'
```

### 4. Elasticsearch - 全文索引

**环境变量**:

```bash
ES_HOST_NAME=127.0.0.1
ES_PORT=9200
ES_USER=
ES_PASSWORD=
ES_PROTOCOL=http
```

### 5. 知识图谱存储

**支持的后端**:

- Neo4j（推荐）
- PostgreSQL（自模拟图数据库）
- NebulaGraph

## 文档重复检测机制

### 检测逻辑

**方法**: `_check_duplicate_document`

```python
async def _check_duplicate_document(
    self, user: str, collection_id: str, filename: str, file_hash: str
) -> db_models.Document | None:
    """
    Check if a document with the same name exists in the collection.
    Returns the existing document if found, None otherwise.
    Raises DocumentNameConflictException if same name but different file hash.
    """
    # 1. 查询同名文档
    existing_doc = await self.db_ops.query_document_by_name_and_collection(
        user, collection_id, filename
    )
    
    if existing_doc:
        # 2. 如果没有哈希（旧版文档），跳过哈希检查
        if existing_doc.content_hash is None:
            logger.warning(f"Existing document {existing_doc.id} has no file hash")
            return existing_doc
        
        # 3. 哈希相同：真正的重复（幂等）
        if existing_doc.content_hash == file_hash:
            return existing_doc
        
        # 4. 哈希不同：文件名冲突
        raise DocumentNameConflictException(filename, collection_id)
    
    return None
```

### 哈希算法

**SHA-256文件哈希计算**:

```python
def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content"""
    import hashlib
    return hashlib.sha256(file_content).hexdigest()
```

### 重复策略

| 场景 | 文件名 | 哈希 | 行为 |
|------|-------|------|------|
| 完全相同 | 相同 | 相同 | 返回已存在文档（幂等） |
| 文件名冲突 | 相同 | 不同 | 抛出`DocumentNameConflictException` |
| 新文档 | 不同 | - | 创建新文档 |

## Quota（配额）管理

### 检查时机

**在确认阶段检查配额**（不在上传阶段），因为：

1. 上传阶段只是临时存储，不占用正式配额
2. 确认阶段才真正消耗资源（索引构建）
3. 允许用户先上传后选择性确认

### 配额类型

```python
async def _check_document_quotas(
    self, session: AsyncSession, user: str, collection_id: str, count: int
):
    """Check and consume document quotas"""
    # 1. 检查并消耗用户全局配额
    await quota_service.check_and_consume_quota(
        user, "max_document_count", count, session
    )
    
    # 2. 检查单个集合配额
    stmt = select(func.count()).select_from(db_models.Document).where(
        db_models.Document.collection_id == collection_id,
        db_models.Document.status != db_models.DocumentStatus.DELETED,
        db_models.Document.status != db_models.DocumentStatus.UPLOADED,  # 不计入临时文档
    )
    existing_doc_count = await session.scalar(stmt)
    
    # 3. 获取配额限制
    stmt = select(UserQuota).where(
        UserQuota.user == user,
        UserQuota.key == "max_document_count_per_collection"
    )
    per_collection_quota = (await session.execute(stmt)).scalars().first()
    
    # 4. 验证是否超出
    if per_collection_quota and (existing_doc_count + count) > per_collection_quota.quota_limit:
        raise QuotaExceededException(
            "max_document_count_per_collection",
            per_collection_quota.quota_limit,
            existing_doc_count
        )
```

### 默认配额

**文件**: `aperag/config.py`

```python
class Config(BaseSettings):
    max_document_count: int = Field(1000, alias="MAX_DOCUMENT_COUNT")
    max_document_size: int = Field(100 * 1024 * 1024, alias="MAX_DOCUMENT_SIZE")  # 100MB
```

## 异步任务处理（Celery）

### 索引协调机制

**文件**: `aperag/service/document_service.py`

```python
def _trigger_index_reconciliation():
    """Trigger index reconciliation task in background"""
    try:
        from config.celery_tasks import reconcile_document_indexes
        reconcile_document_indexes.apply_async()
    except Exception as e:
        logger.warning(f"Failed to trigger index reconciliation task: {e}")
```

**Celery任务**: `config/celery_tasks.py`

```python
@celery_app.task(name="reconcile_document_indexes")
def reconcile_document_indexes():
    """Reconcile document indexes based on their status"""
    from aperag.index.manager import document_index_manager
    
    # 处理PENDING状态的索引
    document_index_manager.reconcile_pending_indexes()
    
    # 处理DELETING状态的索引
    document_index_manager.reconcile_deleting_indexes()
```

### 索引构建流程

1. **文档解析**: DocParser解析文档内容
2. **文档分块**: Chunking策略切分文档
3. **向量化**: Embedding模型生成向量
4. **向量索引**: 写入向量数据库
5. **全文索引**: 写入Elasticsearch
6. **知识图谱**: LightRAG提取实体关系
7. **文档摘要**: LLM生成摘要（可选）
8. **视觉索引**: 提取和分析图片（可选）

## 文件验证

### 支持的文件类型

**文件**: `aperag/docparser/doc_parser.py`

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
            # ... 更多格式
        ]
```

**压缩文件支持**:

```python
SUPPORTED_COMPRESSED_EXTENSIONS = [".zip", ".tar", ".gz", ".tgz"]
```

### 大小限制

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

## API响应格式

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

**示例**:

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

**示例**:

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

## 设计特点

### 1. 两阶段提交设计

**优势**:

- ✅ 用户可以先上传后选择：批量上传后选择性添加
- ✅ 减少不必要的资源消耗：未确认的文档不构建索引
- ✅ 更好的用户体验：快速上传响应，后台异步处理
- ✅ 配额控制更合理：只有确认后才消耗配额

**状态转换**:

```
上传 → UPLOADED → (用户确认) → PENDING → (Celery处理) → RUNNING → COMPLETE
                                                              ↓
                                                           FAILED
```

### 2. 幂等性设计

**重复上传处理**:

- 同名同内容（哈希相同）：返回已存在文档
- 同名不同内容（哈希不同）：抛出冲突异常
- 完全新文档：创建新记录

**好处**:

- 网络重传不会创建重复文档
- 客户端可以安全重试
- 避免存储空间浪费

### 3. 多租户隔离

**存储路径隔离**:

```
user-{user_id}/{collection_id}/{document_id}/...
```

**数据库隔离**:

- 所有查询都带 user 字段过滤
- 集合级别的权限控制
- 软删除支持（gmt_deleted）

### 4. 灵活的存储后端

**支持Local和S3**:

- Local: 适合开发测试、小规模部署
- S3: 适合生产环境、大规模部署
- 统一的`AsyncObjectStore`接口
- 运行时配置切换

### 5. 事务一致性

**核心操作都在事务内**:

```python
async def _upload_document_atomically(session):
    # 1. 创建数据库记录
    # 2. 上传文件到对象存储
    # 3. 更新元数据
    # 所有操作成功才提交，任一失败则回滚
```

**好处**:

- 避免部分成功的脏数据
- 数据库记录和对象存储保持一致
- 失败自动清理

### 6. 分层架构清晰

```
View Layer (views/collections.py)
    ↓ 调用
Service Layer (service/document_service.py)
    ↓ 调用
Repository Layer (db/ops.py, objectstore/)
    ↓ 访问
Storage Layer (PostgreSQL, S3, Qdrant, ES, Neo4j)
```

**职责分离**:

- View: HTTP处理、参数验证、认证
- Service: 业务逻辑、事务编排
- Repository: 数据访问
- Storage: 数据持久化

## 性能优化

### 1. 分块上传（未实现，规划中）

```python
# 大文件分块上传支持
async def upload_document_chunk(
    document_id: str,
    chunk_index: int,
    chunk_data: bytes,
    total_chunks: int
):
    # 上传单个分块
    # 所有分块完成后合并
    pass
```

### 2. 批量操作

- `confirm_documents`支持批量确认
- `delete_documents`支持批量删除
- 批量查询索引状态

### 3. 异步处理

- 文件上传后立即返回
- 索引构建在Celery中异步执行
- 前端轮询或WebSocket获取进度

### 4. 对象存储优化

- S3使用分段上传（multipart upload）
- Local使用aiofiles异步写入
- 支持Range请求（部分下载）

## 错误处理

### 常见异常

```python
# 1. 集合不存在或不可用
raise ResourceNotFoundException("Collection", collection_id)
raise CollectionInactiveException(collection_id)

# 2. 文件验证失败
raise invalid_param("file_type", f"unsupported file type {file_suffix}")
raise invalid_param("file_size", "file size is too large")

# 3. 重复冲突
raise DocumentNameConflictException(filename, collection_id)

# 4. 配额超限
raise QuotaExceededException("max_document_count", limit, current)

# 5. 文档不存在
raise DocumentNotFoundException(f"Document not found: {document_id}")
```

### 异常处理层级

**View层统一异常处理**:

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

## 相关文件

### 核心实现

- `aperag/views/collections.py` - View层接口
- `aperag/service/document_service.py` - Service层业务逻辑
- `aperag/source/upload.py` - UploadSource实现
- `aperag/db/models.py` - 数据库模型
- `aperag/db/ops.py` - 数据库操作
- `aperag/api/components/schemas/document.yaml` - OpenAPI Schema

### 对象存储

- `aperag/objectstore/base.py` - 存储接口定义
- `aperag/objectstore/local.py` - Local存储实现
- `aperag/objectstore/s3.py` - S3存储实现

### 文档处理

- `aperag/docparser/doc_parser.py` - 文档解析器
- `aperag/docparser/chunking.py` - 文档分块
- `aperag/index/manager.py` - 索引管理器
- `aperag/index/vector_index.py` - 向量索引
- `aperag/index/fulltext_index.py` - 全文索引
- `aperag/index/graph_index.py` - 图索引

### 任务队列

- `config/celery_tasks.py` - Celery任务定义
- `aperag/tasks/` - 任务实现

### 前端实现

- `web/src/app/workspace/collections/[collectionId]/documents/page.tsx` - 文档列表页面
- `web/src/components/documents/upload-documents.tsx` - 上传组件

## 总结

ApeRAG的文档上传模块采用**两阶段提交 + 幂等设计 + 灵活存储**架构：

1. **两阶段提交**：上传（UPLOADED）→ 确认（PENDING）→ 索引构建
2. **SHA-256哈希去重**：避免重复文档，支持幂等上传
3. **灵活存储后端**：Local/S3可配置切换
4. **配额管理**：确认阶段才扣除配额，合理控制资源
5. **多索引协调**：向量、全文、图谱、摘要、视觉多种索引类型
6. **清晰的分层架构**：View → Service → Repository → Storage
7. **Celery异步处理**：索引构建不阻塞上传响应
8. **事务一致性**：数据库和对象存储操作原子化

这种设计既保证了性能，又支持复杂的文档处理场景，同时具有良好的可扩展性和容错能力。

