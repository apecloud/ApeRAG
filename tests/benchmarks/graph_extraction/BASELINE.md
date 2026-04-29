# Graph Extraction Baseline - 2026-04-29

Mode: `prompt_only`

Samples: `asf_cn`, `esd_cn`, `vendor_esd_en`

The runner uses ApeRAG's current `render_extraction_prompt` and omits
`response_format`, matching current graph indexing behavior. Prices are from
OpenRouter catalog at run time. Cost is the average observed cost per sample.

| Model | Input $/M | Output $/M | JSON | Entity Hit | Relation Hit | Entity Desc Chars | Relation Desc Chars | Latency | Tokens/s | Avg Cost | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `google/gemini-2.5-flash` | 0.300 | 2.500 | 3/3 | 91.2% | 68.6% | 37.3 | 37.1 | 8.21s | 240.05 | $0.00515477 | Best quality and fastest in this small baseline, but highest cost among tested models. |
| `qwen/qwen3-30b-a3b-instruct-2507` | 0.090 | 0.300 | 3/3 | 80.7% | 65.7% | 40.7 | 35.7 | 23.68s | 72.86 | $0.00058278 | Best cost-quality tradeoff in the baseline; good Chinese and English extraction. |
| `x-ai/grok-4.1-fast` | 0.200 | 0.500 | 3/3 | 80.7% | 57.1% | 33.1 | 28.4 | 18.58s | 216.91 | $0.00210109 | Fast and stable JSON, relation recall behind Qwen/Gemini 2.5. |
| `deepseek/deepseek-v4-flash` | 0.140 | 0.280 | 3/3 | 78.9% | 48.6% | 36.6 | 29.6 | 139.83s | 40.17 | $0.00133707 | Retest confirmed instability: high latency variance and an English-sample parse failure; do not rank as a recommended candidate yet. |
| `google/gemini-3-flash-preview` | 0.500 | 3.000 | 3/3 | 68.4% | 34.3% | 38.7 | 29.7 | 8.22s | 173.44 | $0.00469524 | Fast, but weaker extraction quality than Gemini 2.5 on these samples. |

## Initial Recommendation

For a production default today, `qwen/qwen3-30b-a3b-instruct-2507` is the most
balanced option in this baseline: it is much cheaper than Gemini, faster than
DeepSeek V4 Flash in the observed OpenRouter path, and close to the best model
on relation extraction.

`google/gemini-2.5-flash` is the quality/speed ceiling among these five, but its
observed per-sample cost is roughly 8.8x Qwen's in this run.

`deepseek/deepseek-v4-flash` should not be selected on price alone. A follow-up
retest reproduced high latency variance and found an English-sample parse
failure, so it needs more investigation before joining the recommended set.

## Caveats

- This is a small three-sample baseline intended for repeatable comparisons, not
  a definitive leaderboard.
- The hit-rate scorer uses fuzzy endpoint matching and does not use an LLM judge.
- OpenRouter latency can vary by upstream routing and time of day; rerun before
  final production changes.
- Issue #1861 tracks adding `response_format={"type":"json_object"}` to the
  graph extraction path. After that lands, rerun with
  `make benchmark-graph-extraction RESPONSE_FORMAT_JSON=1` and compare against
  this prompt-only baseline.

## Extended Sweep

Mode: `prompt_only`, `--timeout 60`, `--attempts 1`, `--concurrency 9`

This sweep was run after the initial five-model baseline to quickly screen the
remaining default candidates. A timeout here means the model did not finish a
sample within 60 seconds; that is treated as an indexing-throughput risk rather
than retried away.

| Model | Input $/M | Output $/M | JSON | Entity Hit | Relation Hit | Latency | Tokens/s | Avg Cost | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen/qwen-turbo` | 0.0325 | 0.130 | 3/3 | 71.9% | 60.0% | 18.84s | 82.28 | $0.00021917 | Cheapest usable candidate; weaker than Qwen3-30B but plausible as a low-cost fallback. |
| `z-ai/glm-4.7-flash` | 0.060 | 0.400 | 2/3 | 92.3%* | 73.9%* | 47.33s | 132.31 | $0.00275474 | Promising successful samples, but one timeout; needs a stability rerun before recommendation. |
| `deepseek/deepseek-v3.2` | 0.252 | 0.378 | 1/3 | 72.2%* | 45.5%* | 24.65s | 36.91 | $0.00056489 | Two timeouts in this sweep; not stable enough for default selection here. |
| `moonshotai/kimi-k2.5` | 0.440 | 2.000 | 1/3 | 94.4%* | 66.7%* | 55.90s | 86.29 | $0.01383782 | Good when it completes, but expensive and timed out on two samples. |
| `xiaomi/mimo-v2-flash` | 0.090 | 0.290 | 3/3 | 78.9% | 48.6% | 15.73s | 104.97 | $0.00056904 | Fast and cheap; relation extraction is weaker than Qwen3-30B. |
| `stepfun/step-3.5-flash` | 0.100 | 0.300 | 0/3 | 0.0% | 0.0% | timeout | 0.00 | $0 | Not recommended. |
| `qwen/qwen3.6-plus` | 0.325 | 1.950 | 0/3 | 0.0% | 0.0% | timeout | 0.00 | $0 | Not recommended from this OpenRouter path. |
| `moonshotai/kimi-k2.6` | 0.7448 | 4.655 | 0/3 | 0.0% | 0.0% | timeout | 0.00 | $0 | Too slow/expensive for collection graph extraction. |
| `xiaomi/mimo-v2-pro` | 1.000 | 3.000 | 3/3 | 84.2% | 62.9% | 40.39s | 76.73 | $0.00957007 | Solid quality, but slower and much more expensive than Qwen3-30B. |

`*` means the rate is computed only over completed samples and should not be
compared directly with 3/3-complete models.

The extended sweep reinforces the initial recommendation: keep
`qwen/qwen3-30b-a3b-instruct-2507` as the default candidate, keep
`google/gemini-2.5-flash` as the quality/speed ceiling when cost is acceptable,
and consider `qwen/qwen-turbo` or `xiaomi/mimo-v2-flash` only as lower-cost
fallbacks.
