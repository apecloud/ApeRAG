# ApeRAG Pytest E2E Testing Guide

This directory contains only the remaining pytest-based product E2E tests that
Hurl should not own yet. New black-box HTTP coverage should prefer
`tests/e2e_http/`, while this directory is now the narrow residue for scenarios
that still depend on pytest fixtures and helpers.

## 📁 Directory Structure

```
tests/e2e_pytest/
├── .env                    # Environment configuration file (needs to be created)
├── conftest.py            # pytest fixtures definition
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── README.md              # This document
├── test_*.py              # Test files
```

## Current Scope

- `test_document_download.py`
  keeps a small download supplement while the main download path now lives in
  Hurl. The residue still includes a few negative-path checks plus narrow
  happy-path assertions for filename/header/content-type behavior.

Retired in this phase:
- legacy websocket chat pytest coverage
- speculative streaming/websocket pytest residue without an active owner

Migrated in this phase:
- available-model coverage moved to `tests/e2e_http/hurl/full/10_provider_llm.hurl`
- provider model CRUD moved to `tests/e2e_http/hurl/full/10_provider_llm.hurl`
- bot CRUD and agent-config coverage moved to `tests/e2e_http/hurl/full/12_bot.hurl`
- deterministic chat create/list/get/update/delete moved to `tests/e2e_http/hurl/full/13_chat_http.hurl`
- OpenAI-shaped `/v1/chat/completions` contract moved to `tests/e2e_http/hurl/full/13_chat_http.hurl`

## 🚀 Quick Start

### 1. Environment Setup

Ensure ApeRAG services are running:

```bash
# Start ApeRAG services
cd /path/to/ApeRAG
make serve-api
make serve-worker
```

### 2. Create Environment Configuration File

Create `.env` file in `tests/e2e_pytest/` directory:

```bash
cd tests/e2e_pytest
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

```

### 4. Run Tests

```bash
# Run the residual pytest E2E supplement only
make test-e2e

# Run specific test file
pytest tests/e2e_pytest/test_chat.py

# Run a specific retained chat test
pytest tests/e2e_pytest/test_chat.py::test_chat_message_websocket_api

# Show detailed output
pytest tests/e2e_pytest/ -v

# Show real-time output
pytest tests/e2e_pytest/ -s

# Stop at first failure
pytest tests/e2e_pytest/ -x
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

### Recommended Configuration Combinations

#### 1. Using OpenRouter + SiliconFlow
```bash
COMPLETION_MODEL_PROVIDER=openrouter
COMPLETION_MODEL_NAME=deepseek/deepseek-r1-distill-qwen-32b:free
EMBEDDING_MODEL_PROVIDER=siliconflow
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

## Generated Artifacts

- Runtime coverage and benchmark artifacts belong under `tests/report/`.
- Those files are generated output, not source files, and should stay out of git.

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
    resp = cookie_client.get("/api/v2/collections")
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
    resp = client.get("/api/v2/collections")
```

### Model Service Fixtures

#### `setup_model_service_provider` (module scope)
Automatically configure model service providers required for testing (completion and embedding)

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
def test_chat_message(bot_type, message, request):
    """Test chat messages for different bot types"""
    bot = request.getfixturevalue(f"{bot_type}_bot")
    chat = request.getfixturevalue(f"{bot_type}_chat")

    assert bot["id"]
    assert chat["id"]
```

### Using Utility Functions

```python
from tests.e2e_pytest.utils import assert_dict_subset

def test_collection_update(client, collection):
    update_data = {"title": "Updated Title"}
    resp = client.put(f"/api/v2/collections/{collection['id']}", json=update_data)
    
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
- Keep residual pytest test inputs inline unless multiple tests truly share them
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
