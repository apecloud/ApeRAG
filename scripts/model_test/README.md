# Model Audit Scripts

Tests the availability and functionality of models in the deployed ApeRAG system.

## Test Scripts

### embedding_model_audit.py
Tests all available embedding models to verify which ones are actually usable.

### rerank_model_audit.py
Tests all available rerank models to verify the reranking functionality.

### completion_model_audit.py
Tests the specified completion model to verify text generation functionality. Provider, model, and prompts can be manually configured.

## Usage

```bash
# Test embedding models
python scripts/model_test/embedding_model_audit.py

# Test rerank models
python scripts/model_test/rerank_model_audit.py

# Test completion models (requires manual script configuration)
python scripts/model_test/completion_model_audit.py
```

## Environment Variables

| Variable | Default Value | Description |
|---|---|---|
| `APERAG_API_URL` | `http://localhost:8000` | ApeRAG API address |
| `APERAG_USERNAME` | `user@nextmail.com` | Login username |
| `APERAG_PASSWORD` | `123456` | Login password |

## Output

Each script will generate:
- Real-time console output
- Detailed test report in JSON format

## Dependencies

```bash
pip install httpx yaml
```
