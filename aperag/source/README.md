# ApeRAG Source Module

The `aperag/source` module provides a unified interface for connecting to and retrieving documents from various data sources in the ApeRAG system. This module implements a plugin-based architecture that allows seamless integration with different document repositories, cloud storage services, collaboration platforms, and more.

## Overview

The source module is designed around a common abstraction that enables ApeRAG to ingest documents from diverse sources into its RAG pipeline. Each data source implements the same interface, providing consistency across different types of document repositories.

### Key Components

- **Base Classes**: Abstract interfaces and data models for all sources
- **Source Implementations**: Concrete implementations for specific platforms
- **Utility Functions**: Helper functions for document processing and temporary file management
- **Client Libraries**: Platform-specific API clients (Feishu, Tencent)

## Architecture

### Core Abstractions

#### `Source` (Abstract Base Class)
All data sources inherit from the `Source` base class and implement these key methods:

- `scan_documents()` → `Iterator[RemoteDocument]`: Discover available documents
- `prepare_document(name, metadata)` → `LocalDocument`: Download and prepare documents for processing
- `sync_enabled()` → `bool`: Indicates if the source supports synchronization
- `cleanup_document(filepath)`: Clean up temporary files
- `close()`: Release resources

#### Data Models

**`RemoteDocument`**
```python
class RemoteDocument(BaseModel):
    name: str                           # Document identifier
    size: Optional[int] = None          # Size in bytes
    metadata: Dict[str, Any] = {}       # Additional metadata
```

**`LocalDocument`**
```python
class LocalDocument(BaseModel):
    name: str                           # Document identifier
    path: str                           # Local file system path
    size: Optional[int] = None          # Size in bytes
    metadata: Dict[str, Any] = {}       # Additional metadata
```

## Supported Data Sources

### 1. Local File System (`local.py`)
- **Purpose**: Access documents from local directories
- **Configuration**: Requires `path` parameter
- **Features**: Recursive directory scanning, file metadata extraction
- **Sync Support**: ✅ Yes

### 2. Amazon S3 (`s3.py`)
- **Purpose**: Access documents from S3-compatible object storage
- **Configuration**: Requires `access_key_id`, `secret_access_key`, `bucket`, `region`
- **Features**: Multi-bucket support, prefix filtering, connection validation
- **Sync Support**: ✅ Yes

### 3. Alibaba Cloud OSS (`oss.py`)
- **Purpose**: Access documents from Alibaba Cloud Object Storage Service
- **Configuration**: Similar to S3 with OSS-specific endpoints
- **Features**: OSS API integration, bucket management
- **Sync Support**: ✅ Yes

### 4. Feishu/Lark (`feishu/`)
- **Purpose**: Access documents from Feishu (Lark) workspace
- **Configuration**: Requires `space_id`, API credentials
- **Features**: 
  - Support for both API v1 and v2
  - Document type support: `docx`, `doc`
  - Multiple export formats: Markdown, Plain Text, PDF
  - Hierarchical document structure traversal
- **Sync Support**: ✅ Yes
- **Special Features**: BFS traversal, parent title tracking, multi-format export

### 5. Tencent Cloud (`tencent/`)
- **Purpose**: Access documents from Tencent Cloud services
- **Configuration**: Tencent Cloud API credentials
- **Features**: Tencent-specific document formats and APIs
- **Sync Support**: ✅ Yes

### 6. Email (`Email.py`)
- **Purpose**: Access emails from mail servers
- **Configuration**: Mail server settings, credentials
- **Features**:
  - Support for both IMAP and POP3 protocols
  - Automatic protocol detection and fallback
  - HTML to plain text conversion
  - Spam detection (optional, using ML models)
  - Support for multiple character encodings
- **Sync Support**: ✅ Yes

### 7. Web URLs (`url.py`)
- **Purpose**: Crawl and extract content from web pages
- **Configuration**: Target URLs
- **Features**: HTML content extraction, web scraping
- **Sync Support**: ❌ No (one-time crawling)

### 8. GitHub/Git (`github.py`)
- **Purpose**: Access files from Git repositories
- **Configuration**: Repository URL, authentication
- **Features**: Repository cloning, file extraction
- **Sync Support**: ✅ Yes

### 9. FTP (`ftp.py`)
- **Purpose**: Access files from FTP servers
- **Configuration**: FTP server credentials and paths
- **Features**: Directory traversal, file transfer
- **Sync Support**: ✅ Yes

### 10. Upload (`upload.py`)
- **Purpose**: Handle manually uploaded documents
- **Configuration**: Upload directory configuration
- **Features**: Direct file upload processing
- **Sync Support**: ✅ Yes

## Usage

### Factory Pattern
The module uses a factory pattern to create source instances:

```python
from aperag.source.base import get_source
from aperag.schema.view_models import CollectionConfig

# Create a collection configuration
config = CollectionConfig(
    source="s3",
    access_key_id="your_key",
    secret_access_key="your_secret",
    bucket="your_bucket",
    region="us-west-2"
)

# Get the appropriate source instance
source = get_source(config)

# Scan for documents
for document in source.scan_documents():
    print(f"Found: {document.name}")
    
    # Prepare document for processing
    local_doc = source.prepare_document(document.name, document.metadata)
    
    # Process the local document
    process_document(local_doc.path)
    
    # Clean up
    source.cleanup_document(local_doc.path)

# Close the source connection
source.close()
```

### Configuration
Each source type requires specific configuration parameters in the `CollectionConfig`:

```python
# S3 Example
config = CollectionConfig(
    source="s3",
    access_key_id="AWS_ACCESS_KEY",
    secret_access_key="AWS_SECRET_KEY",
    bucket="my-documents",
    region="us-east-1",
    dir="documents/"  # Optional prefix
)

# Feishu Example
config = CollectionConfig(
    source="feishu",
    space_id="space_123",
    # Additional Feishu API credentials in context
)

# Email Example
config = CollectionConfig(
    source="email",
    pop_server="mail.example.com",
    port=993,
    email_address="user@example.com",
    email_password="password"
)
```

## Advanced Features

### Synchronization Support
Most sources support synchronization, which enables:
- Incremental document updates
- Change detection
- Efficient re-processing

### Metadata Handling
Each source can attach relevant metadata to documents:
- File modification times
- Document hierarchies (Feishu)
- Email headers and recipients
- Git commit information

### Error Handling
The module includes comprehensive error handling:
- `CustomSourceInitializationError`: Configuration or connection issues
- Graceful degradation for individual document failures
- Timeout handling for network operations

### Temporary File Management
The module provides utilities for safe temporary file handling:
- Automatic cleanup
- Proper file naming conventions
- Platform-independent paths

## Extension

To add a new data source:

1. Create a new Python file in the `aperag/source/` directory
2. Implement the `Source` abstract class
3. Add the source to the factory function in `base.py`
4. Implement required configuration parameters

Example:

```python
from aperag.source.base import Source, RemoteDocument, LocalDocument

class MyCustomSource(Source):
    def __init__(self, ctx: CollectionConfig):
        super().__init__(ctx)
        # Initialize your source

    def scan_documents(self) -> Iterator[RemoteDocument]:
        # Implement document discovery
        pass

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        # Implement document preparation
        pass

    def sync_enabled(self):
        return True  # or False
```

## Dependencies

The module uses various third-party libraries depending on the source:
- `boto3`: AWS S3 integration
- `imaplib`/`poplib`: Email access
- `beautifulsoup4`: HTML parsing
- `transformers`: ML-based spam detection
- Platform-specific SDKs for Feishu and Tencent

## Security Considerations

- All credentials are handled through configuration objects
- Temporary files are created with appropriate permissions
- Network connections include timeout and retry mechanisms
- Input validation for external data sources 