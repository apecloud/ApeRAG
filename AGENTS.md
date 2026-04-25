# Agent Guide

## Observability

ApeRAG's observability entrypoint is `aperag.observability`.

- Default mode is `APERAG_OBSERVABILITY_MODE=local`: no extra observability service is required.
- Logs should stay structured JSON and include trace/span correlation fields.
- Export telemetry through OTLP only (`OTEL_EXPORTER_OTLP_ENDPOINT`) when a deployment needs a backend or collector.
- Do not add backend-specific exporters or deployment profiles for tracing systems.
- Do not log prompts, document bodies, API keys, cookies, authorization headers, database passwords, or raw LLM responses.
- New business instrumentation should use stable low-cardinality names and attributes.

Read the full design before changing observability behavior:

- `docs/zh-CN/deployment/observability.md`
