# ApeRAG 源模块

`aperag/source` 模块提供了一个统一的接口，用于在 ApeRAG 系统中连接和检索来自各种数据源的文档。该模块实现了基于插件的架构，可以与不同的文档存储库、云存储服务、协作平台等无缝集成。

## 概述

源模块围绕一个共同的抽象设计，使 ApeRAG 能够将来自不同源的文档摄取到其 RAG 管道中。每个数据源都实现相同的接口，从而在不同类型的文档存储库之间提供一致性。

---

### 关键组件

-   **基类**：所有源的抽象接口和数据模型
-   **源实现**：特定平台的具体实现
-   **实用函数**：文档处理和临时文件管理的辅助函数
-   **客户端库**：平台特定的 API 客户端（飞书、腾讯）

---

## 架构

---

### 核心抽象

#### `Source`（抽象基类）
所有数据源都继承自 `Source` 基类并实现以下关键方法：

-   `scan_documents()` → `Iterator[RemoteDocument]`：发现可用文档
-   `prepare_document(name, metadata)` → `LocalDocument`：下载并准备文档以进行处理
-   `sync_enabled()` → `bool`：指示源是否支持同步
-   `cleanup_document(filepath)`：清理临时文件
-   `close()`：释放资源

---

#### 数据模型

**`RemoteDocument`**
```python
class RemoteDocument(BaseModel):
    name: str                           # 文档标识符
    size: Optional[int] = None          # 大小（字节）
    metadata: Dict[str, Any] = {}       # 附加元数据
```

**`LocalDocument`**
```python
class LocalDocument(BaseModel):
    name: str                           # 文档标识符
    path: str                           # 本地文件系统路径
    size: Optional[int] = None          # 大小（字节）
    metadata: Dict[str, Any] = {}       # 附加元数据
```

---

## 支持的数据源

### 1. 本地文件系统（`local.py`）
-   **目的**：从本地目录访问文档
-   **配置**：需要 `path` 参数
-   **特性**：递归目录扫描、文件元数据提取
-   **同步支持**：✅ 是

### 2. Amazon S3（`s3.py`）
-   **目的**：从 S3 兼容对象存储访问文档
-   **配置**：需要 `access_key_id`、`secret_access_key`、`bucket`、`region`
-   **特性**：多存储桶支持、前缀过滤、连接验证
-   **同步支持**：✅ 是

### 3. 阿里云 OSS（`oss.py`）
-   **目的**：从阿里云对象存储服务访问文档
-   **配置**：类似于 S3，但有 OSS 特定的端点
-   **特性**：OSS API 集成、存储桶管理
-   **同步支持**：✅ 是

### 4. 飞书/Lark（`feishu/`）
-   **目的**：从飞书 (Lark) 工作区访问文档
-   **配置**：需要 `space_id`、API 凭据
-   **特性**：
    -   支持 API v1 和 v2
    -   文档类型支持：`docx`、`doc`
    -   多种导出格式：Markdown、纯文本、PDF
    -   分层文档结构遍历
-   **同步支持**：✅ 是
-   **特殊特性**：BFS 遍历、父标题跟踪、多格式导出

### 5. 腾讯云（`tencent/`）
-   **目的**：从腾讯云服务访问文档
-   **配置**：腾讯云 API 凭据
-   **特性**：腾讯特定的文档格式和 API
-   **同步支持**：✅ 是

### 6. 电子邮件（`Email.py`）
-   **目的**：从邮件服务器访问电子邮件
-   **配置**：邮件服务器设置、凭据
-   **特性**：
    -   支持 IMAP 和 POP3 协议
    -   自动协议检测和回退
    -   HTML 到纯文本转换
    -   垃圾邮件检测（可选，使用机器学习模型）
    -   支持多种字符编码
-   **同步支持**：✅ 是

### 7. 网页 URL（`url.py`）
-   **目的**：抓取并提取网页内容
-   **配置**：目标 URL
-   **特性**：HTML 内容提取、网页抓取
-   **同步支持**：❌ 否（一次性抓取）

### 8. GitHub/Git（`github.py`）
-   **目的**：从 Git 仓库访问文件
-   **配置**：仓库 URL、认证
-   **特性**：仓库克隆、文件提取
-   **同步支持**：✅ 是

### 9. FTP（`ftp.py`）
-   **目的**：从 FTP 服务器访问文件
-   **配置**：FTP 服务器凭据和路径
-   **特性**：目录遍历、文件传输
-   **同步支持**：✅ 是

### 10. 上传（`upload.py`）
-   **目的**：处理手动上传的文档
-   **配置**：上传目录配置
-   **特性**：直接文件上传处理
-   **同步支持**：✅ 是

---

## 用法

---

### 工厂模式
模块使用工厂模式创建源实例：

```python
from aperag.source.base import get_source
from aperag.schema.view_models import CollectionConfig

# 创建集合配置
config = CollectionConfig(
    source="s3",
    access_key_id="your_key",
    secret_access_key="your_secret",
    bucket="your_bucket",
    region="us-west-2"
)

# 获取相应的源实例
source = get_source(config)

# 扫描文档
for document in source.scan_documents():
    print(f"找到: {document.name}")
    
    # 准备文档进行处理
    local_doc = source.prepare_document(document.name, document.metadata)
    
    # 处理本地文档
    process_document(local_doc.path)
    
    # 清理
    source.cleanup_document(local_doc.path)

# 关闭源连接
source.close()
```

---

### 配置
每种源类型在 `CollectionConfig` 中需要特定的配置参数：

```python
# S3 示例
config = CollectionConfig(
    source="s3",
    access_key_id="AWS_ACCESS_KEY",
    secret_access_key="AWS_SECRET_KEY",
    bucket="my-documents",
    region="us-east-1",
    dir="documents/"  # 可选前缀
)

# 飞书示例
config = CollectionConfig(
    source="feishu",
    space_id="space_123",
    # 上下文中附加的飞书 API 凭据
)

# 电子邮件示例
config = CollectionConfig(
    source="email",
    pop_server="mail.example.com",
    port=993,
    email_address="user@example.com",
    email_password="password"
)
```

---

## 高级特性

---

### 同步支持
大多数源支持同步，这使得：
-   增量文档更新
-   变更检测
-   高效的再处理

### 元数据处理
每个源都可以将相关元数据附加到文档：
-   文件修改时间
-   文档层次结构（飞书）
-   电子邮件标题和收件人
-   Git 提交信息

### 错误处理
模块包含全面的错误处理：
-   `CustomSourceInitializationError`：配置或连接问题
-   单个文档故障的优雅降级
-   网络操作的超时处理

### 临时文件管理
模块提供用于安全临时文件处理的实用程序：
-   自动清理
-   适当的文件命名约定
-   平台无关路径

---

## 扩展

要添加新的数据源：

1.  在 `aperag/source/` 目录中创建一个新的 Python 文件
2.  实现 `Source` 抽象类
3.  将源添加到 `base.py` 中的工厂函数
4.  实现所需的配置参数

示例：

```python
from aperag.source.base import Source, RemoteDocument, LocalDocument

class MyCustomSource(Source):
    def __init__(self, ctx: CollectionConfig):
        super().__init__(ctx)
        # 初始化您的源
        
    def scan_documents(self) -> Iterator[RemoteDocument]:
        # 实现文档发现
        pass

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        # 实现文档准备
        pass

    def sync_enabled(self):
        return True  # 或 False
```

---

## 依赖项

模块根据源使用各种第三方库：
-   `boto3`：AWS S3 集成
-   `imaplib`/`poplib`：电子邮件访问
-   `beautifulsoup4`：HTML 解析
-   `transformers`：基于机器学习的垃圾邮件检测
-   飞书和腾讯的平台特定 SDK

---

## 安全注意事项

-   所有凭据都通过配置对象处理
-   临时文件以适当的权限创建
-   网络连接包括超时和重试机制
-   外部数据源的输入验证