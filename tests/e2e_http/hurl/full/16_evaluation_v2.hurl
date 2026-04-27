POST {{base_url}}/api/v2/auth/login
Content-Type: application/json
{
  "username": "{{username}}",
  "password": "{{password}}"
}
HTTP 200

POST {{base_url}}/api/v2/collections
Content-Type: application/json
{
  "title": "Evaluation V3 Collection {{run_id}}",
  "description": "HTTP evaluation v3 coverage",
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

# Title must NOT be the literal "Default Agent Bot" — auth.py:84 already
# auto-creates a bot with that exact title on user registration, and the
# default-bot resolver picks the oldest title match, which would be the
# auto-created bot, not this one.
POST {{base_url}}/api/v2/bots
Content-Type: application/json
{
  "title": "Eval Agent Bot {{run_id}}",
  "description": "HTTP evaluation v3 run coverage",
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
    "contract": "evaluation-v3"
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
jsonpath "$.item_count" == 2
jsonpath "$.schema_hint.language" == "en-US"

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
[Asserts]
jsonpath "$.items[0].case_key" == "hello-case"
jsonpath "$.items[0].input_message" == "Say hello in one sentence."
jsonpath "$.items[0].tags[0]" == "smoke"
jsonpath "$.items[1].case_key" == "summary-case"
jsonpath "$.items[1].sort_key" == 1
jsonpath "$.pagination.total" == 2
jsonpath "$.pagination.limit" == 100

POST {{base_url}}/api/v2/evaluation-datasets/{{dataset_id}}/items
Content-Type: application/json
{
  "items": [
    {
      "case_key": "followup-case",
      "input_message": "How should the bot handle ambiguous input?",
      "expected_answer": "Ask a clarifying question.",
      "sort_key": 2
    }
  ]
}
HTTP 200
[Asserts]
jsonpath "$.items[0].case_key" == "followup-case"
jsonpath "$.items[0].input_message" == "How should the bot handle ambiguous input?"

GET {{base_url}}/api/v2/evaluation-datasets/{{dataset_id}}
HTTP 200
[Asserts]
jsonpath "$.item_count" == 3

POST {{base_url}}/api/v2/evaluation-runs
Content-Type: application/json
{
  "name": "Evaluation Run {{run_id}}",
  "dataset_id": "{{dataset_id}}",
  "bot_id": "{{bot_id}}",
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
jsonpath "$.summary.total" == 3
jsonpath "$.summary.pending" == 3
jsonpath "$.summary.completed" == 0

# POST without bot_id exercises the default-bot resolver; the resolved
# bot_id is NOT {{bot_id}} (that one is titled "Eval Agent Bot ..."),
# it's the registration-auto bot titled "Default Agent Bot". Capture
# and verify the title instead of comparing ids.
POST {{base_url}}/api/v2/evaluation-runs
Content-Type: application/json
{
  "name": "Evaluation Run {{run_id}} Default Bot",
  "dataset_id": "{{dataset_id}}"
}
HTTP 200
[Captures]
run_id_default_bot: jsonpath "$.id"
default_resolved_bot_id: jsonpath "$.bot_id"
[Asserts]
jsonpath "$.id" == "{{run_id_default_bot}}"
jsonpath "$.dataset_id" == "{{dataset_id}}"
jsonpath "$.collection_id" == "{{collection_id}}"
jsonpath "$.status" == "queued"

GET {{base_url}}/api/v2/bots/{{default_resolved_bot_id}}
HTTP 200
[Asserts]
jsonpath "$.id" == "{{default_resolved_bot_id}}"
jsonpath "$.title" == "Default Agent Bot"

GET {{base_url}}/api/v2/evaluation-runs?bot_id={{bot_id}}&page=1&page_size=20
HTTP 200
[Asserts]
# Wave 3 T3.1 chunk 2 + post-pass-7 evaluation cross-loop fix:
# Pattern C in-process dispatch (asyncio.create_task on shared loop)
# starts the worker on the very next event-loop tick, so by the time
# this GET arrives the run has typically left "queued" already. Accept
# any non-terminal-failure in-flight or terminal state — the contract
# this test pins is "the run shows up in the list", not "dispatch is
# slow enough that 'queued' is observable".
jsonpath "$.items[0].status" matches "^(queued|running|completed|cancelled)$"
jsonpath "$.pagination.limit" == 20

GET {{base_url}}/api/v2/evaluation-runs?collection_id={{collection_id}}&page=1&page_size=20
HTTP 200
[Asserts]
jsonpath "$.items[0].collection_id" == "{{collection_id}}"
jsonpath "$.pagination.limit" == 20

GET {{base_url}}/api/v2/evaluation-runs?dataset_id={{dataset_id}}&page=1&page_size=20
HTTP 200
[Asserts]
jsonpath "$.items[0].dataset_id" == "{{dataset_id}}"

GET {{base_url}}/api/v2/evaluation-runs/{{run_id_v2}}
HTTP 200
[Asserts]
jsonpath "$.run.id" == "{{run_id_v2}}"
# Status / summary counts / progress.percent are timing-sensitive
# under Pattern C in-process dispatch (see comment at line ~218);
# accept any in-flight or terminal state. ``summary.total`` is fixed
# by the dataset cardinality so it stays exact.
jsonpath "$.run.status" matches "^(queued|running|completed|cancelled)$"
jsonpath "$.run.dataset_id" == "{{dataset_id}}"
jsonpath "$.summary.total" == 3

GET {{base_url}}/api/v2/evaluation-runs/{{run_id_v2}}/items?page=1&page_size=100
HTTP 200
[Captures]
run_item_id: jsonpath "$.items[0].id"
[Asserts]
jsonpath "$.items[0].id" == "{{run_item_id}}"
jsonpath "$.items[0].run_id" == "{{run_id_v2}}"
jsonpath "$.items[0].case_key" == "hello-case"
# Item status / attempt_count race with the in-process Pattern C
# worker that may have already started this item; accept any in-
# flight or terminal state. Total cardinality remains exact.
jsonpath "$.items[0].status" matches "^(pending|running|completed|failed|cancelled)$"
jsonpath "$.pagination.total" == 3

GET {{base_url}}/api/v2/evaluation-runs/{{run_id_v2}}/items/{{run_item_id}}/attempts
HTTP 200
[Asserts]
# The attempts list may be empty (no execution started yet) or
# populated (worker already attempted this item) under Pattern C
# in-process dispatch; just confirm the envelope shape exists.
jsonpath "$.items" exists

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
