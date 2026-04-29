# ApeRAG Pytest E2E 测试指南

本目录现在只保留 Hurl 暂时不适合接管的 pytest 版本产品级 E2E。
新的黑盒 HTTP 覆盖优先放到 `tests/e2e_http/`，这里已经收缩成仍然依赖
pytest fixtures 和辅助函数的小型补充区。

## 📁 目录结构

```
tests/e2e_pytest/
├── .env                    # 环境配置文件（需要创建）
├── conftest.py            # pytest fixtures 定义
├── config.py              # 配置管理
├── utils.py               # 工具函数
├── README.md              # 本文档
├── test_*.py              # 测试文件
```

## 当前范围

- `test_document_download.py`
  保留一个很小的下载补充层；主下载链路已经迁到 Hurl。当前残余里既有少量负路径校验，
  也保留了很窄的 happy-path 断言，用来守下载文件名 / 响应头 / content-type 行为。

本阶段退休：
- legacy websocket chat 的 pytest 覆盖
- 没有明确 owner 的 streaming / websocket pytest 残余

本阶段已迁走：
- available-model 覆盖迁到 `tests/e2e_http/hurl/full/10_provider_llm.hurl`
- provider model CRUD 迁到 `tests/e2e_http/hurl/full/10_provider_llm.hurl`
- bot CRUD 与 agent config 覆盖迁到 `tests/e2e_http/hurl/full/12_bot.hurl`
- 确定性的 chat create/list/get/update/delete 覆盖迁到 `tests/e2e_http/hurl/full/13_chat_http.hurl`
- OpenAI 风格的 `/v1/chat/completions` 契约迁到 `tests/e2e_http/hurl/full/13_chat_http.hurl`

## 🚀 快速开始

### 1. 环境准备

确保 ApeRAG 服务正在运行：

```bash
# 启动 ApeRAG 服务
cd /path/to/ApeRAG
make serve-api
make serve-worker
```

### 2. 创建环境配置文件

在 `tests/e2e_pytest/` 目录下创建 `.env` 文件：

```bash
cd tests/e2e_pytest
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

```

### 4. 运行测试

```bash
# 只运行 pytest 残余 E2E 补充层
make test-e2e

# 运行特定测试文件
pytest tests/e2e_pytest/test_chat.py

# 运行某个保留的 chat 测试
pytest tests/e2e_pytest/test_chat.py::test_chat_message_websocket_api

# 显示详细输出
pytest tests/e2e_pytest/ -v

# 显示实时输出
pytest tests/e2e_pytest/ -s

# 停在第一个失败的测试
pytest tests/e2e_pytest/ -x
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

### 推荐配置组合

#### 1. 使用 OpenRouter + SiliconFlow
```bash
COMPLETION_MODEL_PROVIDER=openrouter
COMPLETION_MODEL_NAME=deepseek/deepseek-r1-distill-qwen-32b:free
EMBEDDING_MODEL_PROVIDER=siliconflow
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

## 生成产物

- 运行时生成的 coverage / benchmark 产物放在 `tests/report/`。
- 这些文件属于测试输出，不属于测试源码，不应提交进 git。

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
    resp = cookie_client.get("/api/v2/collections")
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
    resp = client.get("/api/v2/collections")
```

### 模型服务 Fixtures

#### `setup_model_service_provider` (module scope)
自动配置测试所需的模型服务提供商（completion、embedding）

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
def test_chat_message(bot_type, message, request):
    """Test chat messages for different bot types"""
    bot = request.getfixturevalue(f"{bot_type}_bot")
    chat = request.getfixturevalue(f"{bot_type}_chat")

    assert bot["id"]
    assert chat["id"]
```

### 工具函数使用

```python
from tests.e2e_pytest.utils import assert_dict_subset

def test_collection_update(client, collection):
    update_data = {"title": "Updated Title"}
    resp = client.put(f"/api/v2/collections/{collection['id']}", json=update_data)
    
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
- 只有在多个测试真正共享时才抽出独立测试数据，其他残余 pytest 输入优先就地内联
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
