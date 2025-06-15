# Model Test Scripts

This directory contains scripts for testing model availability and functionality in deployed ApeRAG systems.

## test_embedding_model.py

Tests all available embedding models to verify which ones are actually functional after system deployment.

### Purpose

After deploying ApeRAG, some models/providers may not work due to:
- Missing or incorrect API keys
- Provider service unavailability
- Network connectivity issues
- Model configuration problems

This script helps identify which embedding models are actually usable.

### Usage

```bash
# Basic usage (uses default admin/admin credentials)
python tests/model_test/test_embedding_model.py

# With custom configuration
APERAG_API_URL=http://your-server:8000 \
APERAG_USERNAME=your_username \
APERAG_PASSWORD=your_password \
EMBEDDING_TEST_TEXT="Your custom test text" \
python tests/model_test/test_embedding_model.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APERAG_API_URL` | `http://localhost:8000` | Base URL for the ApeRAG API |
| `APERAG_USERNAME` | `admin` | Username for authentication |
| `APERAG_PASSWORD` | `admin` | Password for authentication |
| `EMBEDDING_TEST_TEXT` | Mixed EN/CN text | Custom text for embedding test |

### Output

The script generates:

1. **Console output**: Real-time test progress and results
2. **JSON report**: `embedding_model_test_report.json` with detailed results

### Report Format

```json
{
  "test_summary": {
    "total_models": 10,
    "passed": 7,
    "failed": 3,
    "success_rate": 70.0,
    "test_text": "This is a test sentence...",
    "timestamp": "2025-01-15 14:30:00"
  },
  "results": [
    {
      "provider": "openai",
      "model": "text-embedding-3-small",
      "dimension": 1536,
      "test_pass": true,
      "response_time_seconds": 1.23,
      "error_message": null
    },
    {
      "provider": "invalid_provider",
      "model": "some-model",
      "dimension": null,
      "test_pass": false,
      "response_time_seconds": 0.45,
      "error_message": "HTTP Error: 400 - Provider 'invalid_provider' not found"
    }
  ]
}
```

### Example Output

```
============================================================
ApeRAG Embedding Model Test
============================================================
API Base URL: http://localhost:8000
Test Text: This is a test sentence for embedding generation. 这是一个用于生成嵌入向量的测试句子。
Report File: embedding_model_test_report.json
============================================================
Successfully logged in as admin
Fetching available models...
Successfully fetched models.

Found 5 embedding models to test:
  - openai / text-embedding-3-small
  - siliconflow / BAAI/bge-large-zh-v1.5
  - anthropic / text-embedding-ada-002
  - invalid_provider / some-model
  - openai / text-embedding-3-large

[1/5] Testing: openai / text-embedding-3-small
  Status: ✅ PASSED
  Dimension: 1536
  Response Time: 1.23s
--------------------------------------------------
[2/5] Testing: siliconflow / BAAI/bge-large-zh-v1.5
  Status: ✅ PASSED
  Dimension: 1024
  Response Time: 0.89s
--------------------------------------------------
[3/5] Testing: anthropic / text-embedding-ada-002
  Status: ❌ FAILED
  Error: HTTP Error: 400 - Provider 'anthropic' not found
--------------------------------------------------

============================================================
TEST SUMMARY
============================================================
Total Models Tested: 5
Passed: 3
Failed: 2
Success Rate: 60.0%

Report saved to: /path/to/embedding_model_test_report.json
```

### Dependencies

- `httpx`: For HTTP requests
- `json`: For JSON handling (built-in)
- `os`: For environment variables (built-in)
- `time`: For timing and timestamps (built-in)
- `typing`: For type hints (built-in)

Install dependencies:
```bash
pip install httpx
``` 