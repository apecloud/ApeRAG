# ApeRAG E2E Testing Guide

This directory contains end-to-end (E2E) tests for the ApeRAG project, used to validate overall system functionality and API interfaces.

## 📁 Directory Structure

```
tests/e2e_test/
├── .env                    # Environment configuration file (needs to be created)
├── .env.template          # Environment configuration template (optional)
├── conftest.py            # pytest fixtures definition
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── README.md              # This document
├── test_*.py              # Test files
├── testdata/              # Test data
│   ├── basic-flow.yaml    # Basic flow configuration
│   └── rag-flow.yaml      # RAG flow configuration
└── evaluation/            # Evaluation related
```

## 🚀 Quick Start

### 1. Environment Setup

Ensure ApeRAG services are running:

```bash
# Start ApeRAG services
cd /path/to/ApeRAG
make run-backend
make run-celery
```

### 2. Create Environment Configuration File

Create `.env` file in `tests/e2e_test/` directory:

```bash
cd tests/e2e_test
cp .env.template .env  # If template file exists
# Or create directly
touch .env
```

### 3. Configure Environment Variables

Edit the `.env` file and add the following configuration:

```bash
# API Service Configuration
API_BASE_URL=http://localhost:8000
WS_BASE_URL=ws://localhost:8000/api/v1

# Embedding Model Service Configuration
EMBEDDING_MODEL_PROVIDER=siliconflow
EMBEDDING_MODEL_PROVIDER_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL_PROVIDER_API_KEY=your_siliconflow_api_key
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_MODEL_CUSTOM_PROVIDER=openai

# Completion Model Service Configuration
COMPLETION_MODEL_PROVIDER=openrouter
COMPLETION_MODEL_PROVIDER_URL=https://openrouter.ai/api/v1
COMPLETION_MODEL_PROVIDER_API_KEY=your_openrouter_api_key
COMPLETION_MODEL_NAME=deepseek/deepseek-r1-distill-qwen-32b:free
COMPLETION_MODEL_CUSTOM_PROVIDER=openrouter

# Rerank Model Service Configuration
RERANK_MODEL_PROVIDER=siliconflow
RERANK_MODEL_PROVIDER_URL=https://api.siliconflow.cn/v1
RERANK_MODEL_PROVIDER_API_KEY=your_siliconflow_api_key
RERANK_MODEL_NAME=BAAI/bge-large-zh-1.5
```

### 4. Run Tests

```bash
# Run all e2e tests
make e2e-test

# Run specific test file
pytest tests/e2e_test/test_chat.py

# Run specific test class or method
pytest tests/e2e_test/test_chat.py::test_chat_message_openai_api_non_streaming

# Show detailed output
pytest tests/e2e_test/ -v

# Show real-time output
pytest tests/e2e_test/ -s

# Stop at first failure
pytest tests/e2e_test/ -x
```

## ⚙️ Configuration Guide

### Environment Variables Explained

#### API Service Configuration
- `API_BASE_URL`: Base URL for ApeRAG API service (default: http://localhost:8000)
- `WS_BASE_URL`: Base URL for WebSocket API (default: ws://localhost:8000/api/v1)

#### Model Service Provider Configuration

**Embedding Model**
- `EMBEDDING_MODEL_PROVIDER`: Embedding model service provider name
- `EMBEDDING_MODEL_PROVIDER_URL`: Service provider API URL
- `EMBEDDING_MODEL_PROVIDER_API_KEY`: API key (required)
- `EMBEDDING_MODEL_NAME`: Embedding model name to use
- `EMBEDDING_MODEL_CUSTOM_PROVIDER`: Custom provider type

**Completion Model**
- `COMPLETION_MODEL_PROVIDER`: Completion model service provider name
- `COMPLETION_MODEL_PROVIDER_URL`: Service provider API URL
- `COMPLETION_MODEL_PROVIDER_API_KEY`: API key (required)
- `COMPLETION_MODEL_NAME`: Completion model name to use
- `COMPLETION_MODEL_CUSTOM_PROVIDER`: Custom provider type

**Rerank Model**
- `RERANK_MODEL_PROVIDER`: Rerank model service provider name
- `RERANK_MODEL_PROVIDER_URL`: Service provider API URL
- `RERANK_MODEL_PROVIDER_API_KEY`: API key (required)
- `RERANK_MODEL_NAME`: Rerank model name to use

### Recommended Configuration Combinations

#### 1. Using OpenRouter + SiliconFlow
```bash
COMPLETION_MODEL_PROVIDER=openrouter
COMPLETION_MODEL_NAME=deepseek/deepseek-r1-distill-qwen-32b:free
EMBEDDING_MODEL_PROVIDER=siliconflow
EMBEDDING_MODEL_NAME=BAAI/bge-m3
RERANK_MODEL_PROVIDER=siliconflow
RERANK_MODEL_NAME=BAAI/bge-large-zh-1.5
```

## 🧪 Available Fixtures

The E2E tests provide the following pytest fixtures that can be used directly in tests:

### Authentication Related Fixtures

#### `register_user` (module scope)
Automatically register a test user
```python
def test_something(register_user):
    username = register_user["username"]
    email = register_user["email"]
    password = register_user["password"]
```

#### `login_user` (module scope)
Login test user and return authentication information
```python
def test_something(login_user):
    cookies = login_user["cookies"]
    user = login_user["user"]
```

#### `cookie_client` (module scope)
Return httpx.Client with cookie-based authentication
```python
def test_something(cookie_client):
    resp = cookie_client.get("/api/v1/collections")
```

#### `api_key` (module scope)
Dynamically create API Key for testing, automatically delete after tests complete
```python
def test_something(api_key):
    # api_key is a string format key
    headers = {"Authorization": f"Bearer {api_key}"}
```

#### `client`
Return httpx.Client with API Key authentication
```python
def test_something(client):
    resp = client.get("/api/v1/collections")
```

### Model Service Fixtures

#### `setup_model_service_provider` (module scope)
Automatically configure model service providers required for testing (completion, embedding, rerank)

### Business Object Fixtures

#### `collection`
Create a test collection, automatically delete after test completion
```python
def test_something(client, collection):
    collection_id = collection["id"]
    # collection contains complete collection information
```

#### `document`
Upload a test document to the test collection, automatically delete after test completion
```python
def test_something(client, document, collection):
    doc_id = document["id"]
    content = document["content"]
```

#### `bot`
Create a test bot associated with test collection
```python
def test_something(client, bot):
    bot_id = bot["id"]
    # bot contains complete bot information
```

#### Specialized Bot Fixtures
- `knowledge_bot`: Create knowledge-type bot
- `basic_bot`: Create basic-type bot

#### Chat Fixtures
- `knowledge_chat`: Create chat for knowledge-type bot
- `basic_chat`: Create chat for basic-type bot

### Utility Fixtures

#### `api_helper`
Provide helper methods for API testing
```python
def test_something(api_helper, bot, chat):
    # Test OpenAI API non-streaming
    api_helper.test_openai_api_non_streaming(
        bot_id=bot["id"], 
        chat_id=chat["id"], 
        message="Hello", 
        test_name="My Test"
    )
    
    # Test OpenAI API streaming
    api_helper.test_openai_api_streaming(...)
    
    # Test frontend API non-streaming
    api_helper.test_frontend_api_non_streaming(...)
    
    # Test frontend API streaming
    api_helper.test_frontend_api_streaming(...)
```

## 📝 Writing Tests

### Test File Structure

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

### Test Parameterization

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

### Using Utility Functions

```python
from tests.e2e_test.utils import assert_dict_subset

def test_collection_update(client, collection):
    update_data = {"title": "Updated Title"}
    resp = client.put(f"/api/v1/collections/{collection['id']}", json=update_data)
    
    result = resp.json()
    assert_dict_subset(update_data, result)
```

## 📊 Performance Testing

```bash
make e2e-performance-test
```

## 💡 Best Practices

### 1. Test Isolation
- Each test uses independent resources (users, collections, bots, etc.)
- Automatically clean up resources after test completion
- Use fixture scope to control resource lifecycle

### 2. Error Handling
- Validate both normal and exception flows
- Check error response format and content
- Use appropriate assertion methods

### 3. Test Data Management
- Use `testdata/` directory to store test configuration files
- Test data should be small and precise
- Avoid dependencies on external data sources

### 4. Maintainability
- Test naming should be clear and explicit
- Add necessary documentation and comments
- Reuse common test logic

## 📚 Related Documentation

- [ApeRAG API Documentation](../../docs/)
- [Project Architecture Guide](../../README.md)
- [Development Environment Setup](../../docs/HOW-TO-DEBUG.md)

## 🤝 Contributing Guidelines

1. When adding new tests, ensure proper use of fixtures
2. Tests should be independent and repeatable
3. Add necessary documentation and comments
4. Run complete test suite to ensure no existing functionality is broken
5. Follow project code style and naming conventions

---

For questions, please refer to project documentation or submit an issue. 