# Graph Extraction Model Benchmark

This benchmark compares LLMs on ApeRAG knowledge-graph extraction quality,
latency, and cost. It is a manual benchmark, not a default test suite, because
it calls external providers and consumes API credits.

## Run

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
make benchmark-graph-extraction
```

The default run uses the current ApeRAG graph extraction prompt via
`aperag.indexing.llm.render_extraction_prompt` and does not send
`response_format`. That matches current graph indexing behavior and gives a
prompt-only baseline.

To simulate the proposed JSON-mode fix:

```bash
make benchmark-graph-extraction RESPONSE_FORMAT_JSON=1
```

Results are written to:

```text
tests/report/graph-extraction-openrouter.json
```

## Model Selection

The default model list favors Chinese/domestic OpenRouter models with a cost
range suitable for background graph indexing, plus a few western baseline
models for comparison:

- `qwen/qwen-turbo`
- `qwen/qwen3-30b-a3b-instruct-2507`
- `z-ai/glm-4.7-flash`
- `deepseek/deepseek-v4-flash`
- `deepseek/deepseek-v3.2`
- `moonshotai/kimi-k2.5`
- `xiaomi/mimo-v2-flash`
- `stepfun/step-3.5-flash`
- `x-ai/grok-4.1-fast`
- `google/gemini-3-flash-preview`
- `google/gemini-2.5-flash`

The committed `baseline-2026-04-29.json` intentionally includes only a subset
of these candidates (`qwen/qwen3-30b-a3b-instruct-2507`,
`deepseek/deepseek-v4-flash`, `x-ai/grok-4.1-fast`,
`google/gemini-2.5-flash`, and `google/gemini-3-flash-preview`). The other
default candidates remain in the runner for follow-up runs; they were not
included in the first baseline to keep the initial manual run bounded after
network timeouts and long-tail model latency.

Override models with:

```bash
make benchmark-graph-extraction MODELS='qwen/qwen-plus,moonshotai/kimi-k2.6'
```

## Scoring

Each sample has a short hand-written expected entity list and relation endpoint
list. The runner computes:

- JSON parse success
- entity hit rate
- relation endpoint hit rate
- latency
- output tokens per second
- estimated cost from OpenRouter usage/pricing

The scores are directional. They are intended to compare model/prompt changes
against the same sample set, not to be a complete graph-quality judge.
