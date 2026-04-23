POST {{base_url}}/api/v1/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

POST {{base_url}}/api/v2/collections
Content-Type: application/json
{
  "title": "Evaluation V2 Collection {{run_id}}",
  "description": "HTTP evaluation v2 coverage",
  "type": "document",
  "config": {
    "source": "system",
    "enable_vector": false,
    "enable_fulltext": false,
    "enable_knowledge_graph": false,
    "enable_summary": false,
    "enable_vision": false
  }
}
HTTP 200
[Captures]
collection_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{collection_id}}"
jsonpath "$.config.source" == "system"

POST {{base_url}}/api/v2/bots
Content-Type: application/json
{
  "title": "Evaluation V2 Bot {{run_id}}",
  "description": "HTTP evaluation v2 run coverage",
  "type": "agent"
}
HTTP 200
[Captures]
bot_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{bot_id}}"
jsonpath "$.type" == "agent"

POST {{base_url}}/api/v2/evaluation-datasets
Content-Type: application/json
{
  "name": "Evaluation Dataset {{run_id}}",
  "description": "Evaluation dataset contract coverage",
  "collection_id": "{{collection_id}}",
  "source_type": "manual",
  "schema_hint": {
    "language": "en-US",
    "contract": "evaluation-v2"
  },
  "items": [
    {
      "case_key": "hello-case",
      "input_message": "Say hello in one sentence.",
      "expected_answer": "Hello.",
      "reference_context": "Greet the user briefly.",
      "tags": ["smoke", "greeting"],
      "case_metadata": {
        "locale": "en-US"
      },
      "sort_key": 0
    },
    {
      "case_key": "summary-case",
      "input_message": "Summarize the testing goal.",
      "expected_answer": "Summarize the contract coverage goal.",
      "reference_context": "Focus on test coverage.",
      "tags": ["smoke", "summary"],
      "case_metadata": {
        "locale": "en-US"
      },
      "sort_key": 1
    }
  ]
}
HTTP 200
[Captures]
dataset_id: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{dataset_id}}"
jsonpath "$.collection_id" == "{{collection_id}}"
jsonpath "$.name" == "Evaluation Dataset {{run_id}}"
jsonpath "$.source_type" == "manual"
jsonpath "$.schema_hint.language" == "en-US"
jsonpath "$.item_count" == 2

GET {{base_url}}/api/v2/evaluation-datasets?collection_id={{collection_id}}&page=1&page_size=20
HTTP 200
[Asserts]
jsonpath "$.items[0].id" == "{{dataset_id}}"
jsonpath "$.items[0].collection_id" == "{{collection_id}}"
jsonpath "$.items[0].item_count" == 2
jsonpath "$.pagination.total" == 1
jsonpath "$.pagination.offset" == 0
jsonpath "$.pagination.limit" == 20

GET {{base_url}}/api/v2/evaluation-datasets/{{dataset_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{dataset_id}}"
jsonpath "$.name" == "Evaluation Dataset {{run_id}}"
jsonpath "$.item_count" == 2

PUT {{base_url}}/api/v2/evaluation-datasets/{{dataset_id}}
Content-Type: application/json
{
  "name": "Evaluation Dataset {{run_id}} Updated",
  "description": "Evaluation dataset updated"
}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{dataset_id}}"
jsonpath "$.name" == "Evaluation Dataset {{run_id}} Updated"
jsonpath "$.description" == "Evaluation dataset updated"

GET {{base_url}}/api/v2/evaluation-datasets/{{dataset_id}}/items?page=1&page_size=100
HTTP 200
[Captures]
dataset_item_id: jsonpath "$.items[0].id"
[Asserts]
jsonpath "$.items[0].id" == "{{dataset_item_id}}"
jsonpath "$.items[0].dataset_id" == "{{dataset_id}}"
jsonpath "$.items[0].case_key" == "hello-case"
jsonpath "$.items[0].input_message" == "Say hello in one sentence."
jsonpath "$.items[0].tags[0]" == "smoke"
jsonpath "$.items[1].case_key" == "summary-case"
jsonpath "$.items[1].sort_key" == 1
jsonpath "$.pagination.total" == 2
jsonpath "$.pagination.limit" == 100

GET {{base_url}}/api/v2/evaluation-datasets/{{dataset_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{dataset_id}}"
jsonpath "$.item_count" == 2

POST {{base_url}}/api/v2/evaluation-runs
Content-Type: application/json
{
  "name": "Evaluation Run {{run_id}}",
  "bot_id": "{{bot_id}}",
  "dataset_id": "{{dataset_id}}",
  "judge": {
    "mode": "exact_match"
  },
  "bot_config_snapshot": {
    "type": "agent"
  },
  "model_config_snapshot": {
    "model": "contract-test"
  }
}
HTTP 200
[Captures]
run_id_v2: jsonpath "$.id"
[Asserts]
jsonpath "$.id" == "{{run_id_v2}}"
jsonpath "$.bot_id" == "{{bot_id}}"
jsonpath "$.dataset_id" == "{{dataset_id}}"
jsonpath "$.collection_id" == "{{collection_id}}"
jsonpath "$.name" == "Evaluation Run {{run_id}}"
jsonpath "$.status" == "queued"
jsonpath "$.judge_config.mode" == "exact_match"
jsonpath "$.summary.total" == 2
jsonpath "$.summary.pending" == 2
jsonpath "$.summary.completed" == 0

GET {{base_url}}/api/v2/evaluation-runs?bot_id={{bot_id}}&page=1&page_size=20
HTTP 200
[Asserts]
jsonpath "$.items[0].id" == "{{run_id_v2}}"
jsonpath "$.items[0].status" == "queued"
jsonpath "$.items[0].dataset_id" == "{{dataset_id}}"
jsonpath "$.items[0].summary.total" == 2
jsonpath "$.pagination.total" == 1
jsonpath "$.pagination.limit" == 20

GET {{base_url}}/api/v2/evaluation-runs/{{run_id_v2}}
HTTP 200
[Asserts]
jsonpath "$.run.id" == "{{run_id_v2}}"
jsonpath "$.run.status" == "queued"
jsonpath "$.run.dataset_id" == "{{dataset_id}}"
jsonpath "$.summary.total" == 2
jsonpath "$.summary.pending" == 2
jsonpath "$.progress.percent" == 0

GET {{base_url}}/api/v2/evaluation-runs/{{run_id_v2}}/items?page=1&page_size=100
HTTP 200
[Captures]
run_item_id: jsonpath "$.items[0].id"
[Asserts]
jsonpath "$.items[0].id" == "{{run_item_id}}"
jsonpath "$.items[0].run_id" == "{{run_id_v2}}"
jsonpath "$.items[0].source_dataset_item_id" == "{{dataset_item_id}}"
jsonpath "$.items[0].case_key" == "hello-case"
jsonpath "$.items[0].status" == "pending"
jsonpath "$.items[0].attempt_count" == 0
jsonpath "$.pagination.total" == 2

GET {{base_url}}/api/v2/evaluation-runs/{{run_id_v2}}/items/{{run_item_id}}/attempts
HTTP 200
[Asserts]
body contains "\"items\":[]"

POST {{base_url}}/api/v2/evaluation-runs/{{run_id_v2}}/cancel
HTTP 200
[Asserts]
jsonpath "$.run_id" == "{{run_id_v2}}"
jsonpath "$.status" == "cancelled"

GET {{base_url}}/api/v2/evaluation-runs/{{run_id_v2}}
HTTP 200
[Asserts]
jsonpath "$.run.id" == "{{run_id_v2}}"
jsonpath "$.run.status" == "cancelled"
