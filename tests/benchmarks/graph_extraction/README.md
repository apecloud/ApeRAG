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
list. The runner aggregates **per-document** (per task #30 spec § 6.3 +
Planetegg msg=ea7efa7b acceptance criteria) across all windows for that
document/model/window-size combination:

1. `llm_call_count` — total LLM calls per document
2. `input_tokens_total` + `output_tokens_total`
3. `wall_time_s` — sum of per-window latencies
4. `timeout_or_failure_count` — any window that errored or returned non-JSON
5. `entities_count` + `relations_count` — extraction totals (raw)
6. `duplicate_entity_count` + `duplicate_relation_count` — normalized-name dups
7. `source_chunk_ids_valid` / `source_chunk_ids_total` — fraction of records
   whose `source_chunk_ids` are a non-empty subset of the window's chunk_ids
   (per task #30 §3.1.3 hard requirement #2)

Plus the legacy directional scores (`entity_hit_rate`, `relation_hit_rate`,
`json_ok_rate`, `estimated_cost_usd`).

The scores are directional. They are intended to compare model/prompt/window
changes against the same sample set, not to be a complete graph-quality judge.

## Chunk window matrix (task #30 B1)

The harness simulates the production `_GraphChunkWindow` (PR #1918 commit
`3255fa56`) by splitting each sample text into `--pseudo-chunks-per-doc`
(default `4`) pseudo-chunks and grouping them into non-overlapping windows
of size `--chunk-window-size`. The rendered prompt uses the new
`render_extraction_prompt(window_chunks=...)` API (PR #1920 commit
`01b45196`) so each window carries `[[chunk_id=<id> index=<n>]]` boundary
markers.

Single-shape run:

```bash
uv run python tests/benchmarks/graph_extraction/runner.py \
    --models qwen/qwen3-30b-a3b-instruct-2507 \
    --chunk-window-size 3
```

Matrix sweep (one run, many results blocks):

```bash
uv run python tests/benchmarks/graph_extraction/runner.py \
    --models qwen/qwen3-30b-a3b-instruct-2507,google/gemini-2.5-flash \
    --matrix 1,2,3,5
```

Dry-run schema check (no provider call, used by B2 for ingestion verify
per Planetegg msg=cbe84223):

```bash
uv run python tests/benchmarks/graph_extraction/runner.py \
    --dry-run --matrix 1,2,3,5 --output /tmp/b1_dry_run.json
```

For real multi-chunk documents, raise `--pseudo-chunks-per-doc` (B2 may
add larger samples than the current 3 when comparing window>1 effects on
real-world-sized payloads).

The harness structural pieces (chunking, windowing, validity counting,
per-document aggregation) are unit-tested in
`test_runner_units.py` so the matrix output schema stays pinned even
though the benchmark itself runs out-of-CI.
