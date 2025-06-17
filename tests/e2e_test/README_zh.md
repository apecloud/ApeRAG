# ApeRAG E2E 测试指南

本目录包含 ApeRAG 项目的端到端（E2E）测试，用于验证系统的整体功能和 API 接口。

## 📁 目录结构

```
tests/e2e_test/
├── .env                    # 环境配置文件（需要创建）
├── .env.template          # 环境配置模板（可选）
├── conftest.py            # pytest fixtures 定义
├── config.py              # 配置管理
├── utils.py               # 工具函数
├── README.md              # 本文档
├── test_*.py              # 测试文件
├── testdata/              # 测试数据
│   ├── basic-flow.yaml    # 基础流程配置
│   └── rag-flow.yaml      # RAG 流程配置
└── evaluation/            # 评估相关
```

## 🚀 快速开始

### 1. 环境准备

确保 ApeRAG 服务正在运行：

```bash
# 启动 ApeRAG 服务
cd /path/to/ApeRAG
make run-backend
make run-celery
```

### 2. 创建环境配置文件

在 `tests/e2e_test/` 目录下创建 `.env` 文件：

```bash
cd tests/e2e_test
cp .env.template .env  # 如果有模板文件
# 或者直接创建
touch .env
```

### 3. 配置环境变量

编辑 `.env` 文件，添加以下配置：

```bash
# API 服务配置
API_BASE_URL=http://localhost:8000
WS_BASE_URL=ws://localhost:8000/api/v1

# Embedding 模型服务配置
EMBEDDING_MODEL_PROVIDER=siliconflow
EMBEDDING_MODEL_PROVIDER_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL_PROVIDER_API_KEY=your_siliconflow_api_key
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_MODEL_CUSTOM_PROVIDER=openai

# 文本生成模型服务配置
COMPLETION_MODEL_PROVIDER=openrouter
COMPLETION_MODEL_PROVIDER_URL=https://openrouter.ai/api/v1
COMPLETION_MODEL_PROVIDER_API_KEY=your_openrouter_api_key
COMPLETION_MODEL_NAME=deepseek/deepseek-r1-distill-qwen-32b:free
COMPLETION_MODEL_CUSTOM_PROVIDER=openrouter

# Rerank 模型服务配置
RERANK_MODEL_PROVIDER=siliconflow
RERANK_MODEL_PROVIDER_URL=https://api.siliconflow.cn/v1
RERANK_MODEL_PROVIDER_API_KEY=your_siliconflow_api_key
RERANK_MODEL_NAME=BAAI/bge-large-zh-1.5
```

### 4. 运行测试

```bash
# 运行所有 e2e 测试
make e2e-test

# 运行特定测试文件
pytest tests/e2e_test/test_chat.py

# 运行特定测试类或方法
pytest tests/e2e_test/test_chat.py::test_chat_message_openai_api_non_streaming

# 显示详细输出
pytest tests/e2e_test/ -v

# 显示实时输出
pytest tests/e2e_test/ -s

# 停在第一个失败的测试
pytest tests/e2e_test/ -x
```

## ⚙️ 配置说明

### 环境变量详解

#### API 服务配置
- `API_BASE_URL`: ApeRAG API 服务的基础 URL（默认: http://localhost:8000）
- `WS_BASE_URL`: WebSocket API 的基础 URL（默认: ws://localhost:8000/api/v1）

#### 模型服务提供商配置

**Embedding 模型**
- `EMBEDDING_MODEL_PROVIDER`: Embedding 模型服务提供商名称
- `EMBEDDING_MODEL_PROVIDER_URL`: 服务提供商的 API URL
- `EMBEDDING_MODEL_PROVIDER_API_KEY`: API 密钥（必填）
- `EMBEDDING_MODEL_NAME`: 使用的 Embedding 模型名称
- `EMBEDDING_MODEL_CUSTOM_PROVIDER`: 自定义提供商类型

**文本生成模型**
- `COMPLETION_MODEL_PROVIDER`: 文本生成模型服务提供商名称
- `COMPLETION_MODEL_PROVIDER_URL`: 服务提供商的 API URL
- `COMPLETION_MODEL_PROVIDER_API_KEY`: API 密钥（必填）
- `COMPLETION_MODEL_NAME`: 使用的文本生成模型名称
- `COMPLETION_MODEL_CUSTOM_PROVIDER`: 自定义提供商类型

**Rerank 模型**
- `RERANK_MODEL_PROVIDER`: Rerank 模型服务提供商名称
- `RERANK_MODEL_PROVIDER_URL`: 服务提供商的 API URL
- `RERANK_MODEL_PROVIDER_API_KEY`: API 密钥（必填）
- `RERANK_MODEL_NAME`: 使用的 Rerank 模型名称

### 推荐配置组合

#### 1. 使用 OpenRouter + SiliconFlow
```bash
COMPLETION_MODEL_PROVIDER=openrouter
COMPLETION_MODEL_NAME=deepseek/deepseek-r1-distill-qwen-32b:free
EMBEDDING_MODEL_PROVIDER=siliconflow
EMBEDDING_MODEL_NAME=BAAI/bge-m3
RERANK_MODEL_PROVIDER=siliconflow
RERANK_MODEL_NAME=BAAI/bge-large-zh-1.5
```

## 🧪 可用的 Fixtures

E2E 测试提供了以下 pytest fixtures，可以在测试中直接使用：

### 认证相关 Fixtures

#### `register_user` (module scope)
自动注册一个测试用户
```python
def test_something(register_user):
    username = register_user["username"]
    email = register_user["email"]
    password = register_user["password"]
```

#### `login_user` (module scope)
登录测试用户并返回认证信息
```python
def test_something(login_user):
    cookies = login_user["cookies"]
    user = login_user["user"]
```

#### `cookie_client` (module scope)
返回带有 Cookie 认证的 httpx.Client
```python
def test_something(cookie_client):
    resp = cookie_client.get("/api/v1/collections")
```

#### `api_key` (module scope)
动态创建 API Key 用于测试，测试完成后自动删除
```python
def test_something(api_key):
    # api_key 是字符串格式的密钥
    headers = {"Authorization": f"Bearer {api_key}"}
```

#### `client`
返回带有 API Key 认证的 httpx.Client
```python
def test_something(client):
    resp = client.get("/api/v1/collections")
```

### 模型服务 Fixtures

#### `setup_model_service_provider` (module scope)
自动配置测试所需的模型服务提供商（completion、embedding、rerank）

### 业务对象 Fixtures

#### `collection`
创建一个测试知识库，测试完成后自动删除
```python
def test_something(client, collection):
    collection_id = collection["id"]
    # collection 包含完整的知识库信息
```

#### `document`
在测试知识库中上传一个测试文档，测试完成后自动删除
```python
def test_something(client, document, collection):
    doc_id = document["id"]
    content = document["content"]
```

#### `bot`
创建一个测试机器人，关联测试知识库
```python
def test_something(client, bot):
    bot_id = bot["id"]
    # bot 包含完整的机器人信息
```

#### 专用 Bot Fixtures
- `knowledge_bot`: 创建知识型机器人
- `basic_bot`: 创建基础型机器人

#### Chat Fixtures
- `knowledge_chat`: 为知识型机器人创建对话
- `basic_chat`: 为基础型机器人创建对话

### 工具类 Fixtures

#### `api_helper`
提供 API 测试的辅助方法
```python
def test_something(api_helper, bot, chat):
    # 测试 OpenAI API 非流式
    api_helper.test_openai_api_non_streaming(
        bot_id=bot["id"], 
        chat_id=chat["id"], 
        message="Hello", 
        test_name="My Test"
    )
    
    # 测试 OpenAI API 流式
    api_helper.test_openai_api_streaming(...)
    
    # 测试前端 API 非流式
    api_helper.test_frontend_api_non_streaming(...)
    
    # 测试前端 API 流式
    api_helper.test_frontend_api_streaming(...)
```

## 📝 编写测试

### 测试文件结构

```python
import pytest
from http import HTTPStatus

def test_my_feature(client, collection):
    """Test description
    
    Args:
        client: Authenticated HTTP client
        collection: Test collection fixture
    """
    # Arrange
    data = {"title": "Test"}
    
    # Act
    resp = client.post("/api/v1/endpoint", json=data)
    
    # Assert
    assert resp.status_code == HTTPStatus.OK
    result = resp.json()
    assert result["title"] == "Test"
```

### 测试参数化

```python
@pytest.mark.parametrize("bot_type,message", [
    ("knowledge", "What is ApeRAG?"),
    ("basic", "Hello, how are you today?"),
])
def test_chat_message(api_helper, bot_type, message, request):
    """Test chat messages for different bot types"""
    bot = request.getfixturevalue(f"{bot_type}_bot")
    chat = request.getfixturevalue(f"{bot_type}_chat")
    
    api_helper.test_openai_api_non_streaming(
        bot_id=bot["id"],
        chat_id=chat["id"],
        message=message,
        test_name=f"Chat {bot_type}"
    )
```

### 工具函数使用

```python
from tests.e2e_test.utils import assert_dict_subset

def test_collection_update(client, collection):
    update_data = {"title": "Updated Title"}
    resp = client.put(f"/api/v1/collections/{collection['id']}", json=update_data)
    
    result = resp.json()
    assert_dict_subset(update_data, result)
```

## 📊 性能测试

```bash
make e2e-performance-test
```

## 💡 最佳实践

### 1. 测试隔离
- 每个测试使用独立的资源（用户、知识库、机器人等）
- 测试完成后自动清理资源
- 使用 fixture 的 scope 控制资源生命周期

### 2. 错误处理
- 验证正常流程和异常流程
- 检查错误响应的格式和内容
- 使用合适的断言方法

### 3. 测试数据管理
- 使用 `testdata/` 目录存放测试配置文件
- 测试数据应该小而精确
- 避免依赖外部数据源

### 4. 可维护性
- 测试命名要清晰明确
- 添加必要的文档注释
- 复用通用的测试逻辑

## 📚 相关文档

- [ApeRAG API 文档](../../docs/)
- [项目架构说明](../../README.md)
- [开发环境搭建](../../docs/HOW-TO-DEBUG-zh.md)

## 🤝 贡献指南

1. 添加新测试时，确保使用合适的 fixtures
2. 测试应该是独立且可重复的
3. 添加必要的文档和注释
4. 运行完整的测试套件确保没有破坏现有功能
5. 遵循项目的代码风格和命名约定

---

如有问题，请参考项目文档或提交 issue。 