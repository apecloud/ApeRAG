# Full HTTP E2E

This directory is reserved for broader HTTP coverage after the smoke layer is stable.

Current files:
- `10_provider_llm.hurl`
  provider configuration, available models, model CRUD, default models, embeddings, rerank
- `11_document_full.hurl`
  staged upload, confirm, list/detail, download, rebuild indexes
- `12_bot.hurl`
  bot CRUD plus agent config get/update
- `13_chat_http.hurl`
  chat create/list/get/update/delete plus OpenAI-shaped `/v1/chat/completions` backed by Agent Runtime V3 turns
- `14_graph_http.hurl`
  graph labels and graph overview endpoints
- `15_agent_runtime_v3.hurl`
  v2 agent runtime HTTP contract: create turn, idempotency, snapshot, cancel
- `17_chat_collection_flow.hurl`
  provider-aware business flow: collection create, document upload/confirm, bot bind collection, chat create, turn create/snapshot

Supplemental scripted flows executed by `tests/e2e_http/scripts/run_full.sh`:
- `run_chat_collection_flow.sh`
  waits for vector/fulltext indexing, then proves chat returns a non-empty answer artifact and a non-empty reference bundle
- `run_graph_index_flow.sh`
  waits for graph indexing to become `ACTIVE`, then proves graph labels / nodes / edges are non-empty

WebSocket and streaming-specific coverage should remain in a thin supplemental layer rather than being forced into Hurl.
