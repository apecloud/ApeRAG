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
  chat create/list/get/update/delete plus OpenAI-shaped `/v1/chat/completions` contract
- `14_graph_http.hurl`
  graph labels and graph overview endpoints
- `15_agent_runtime_v3.hurl`
  v2 agent runtime HTTP contract: create turn, idempotency, snapshot, cancel

WebSocket and streaming-specific coverage should remain in a thin supplemental layer rather than being forced into Hurl.
