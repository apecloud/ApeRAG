#!/usr/bin/env python3
"""Manual benchmark for knowledge-graph extraction model quality.

This is intentionally not part of the default pytest or CI suite: it calls
external LLM providers and consumes API credits. Run it only when comparing
models, prompts, JSON-mode changes, or chunk-window matrices.

task #30 B1 (msg=cecae5ed): supports the graph chunk window matrix
(``--chunk-window-size`` / ``--matrix``). The harness aggregates 7
metrics per-document (per Planetegg msg=ea7efa7b acceptance criteria
echoed in spec § 6.3): LLM call count, input+output tokens, wall time,
timeout/failure rate, entity+relation totals, duplicate rate, and
``source_chunk_ids`` validity rate.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BENCH_DIR / "samples"
DEFAULT_OUTPUT = ROOT / "tests" / "report" / "graph-extraction-openrouter.json"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MAX_REQUEST_ATTEMPTS = 3

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aperag.indexing.llm import render_extraction_prompt  # noqa: E402

DEFAULT_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Document",
    "Disease",
    "Facility",
    "Vehicle",
    "Animal",
    "Material",
    "Standard",
    "Process",
    "Equipment",
    "Requirement",
    "Concept",
    "Location",
]

DEFAULT_MODELS = [
    "qwen/qwen-turbo",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "z-ai/glm-4.7-flash",
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v3.2",
    "moonshotai/kimi-k2.5",
    "xiaomi/mimo-v2-flash",
    "stepfun/step-3.5-flash",
    "x-ai/grok-4.1-fast",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash",
]


@dataclass(frozen=True)
class ModelPrice:
    input_per_token: float = 0.0
    output_per_token: float = 0.0
    context_length: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated OpenRouter model IDs.",
    )
    parser.add_argument(
        "--samples-dir",
        default=str(SAMPLES_DIR),
        help="Directory containing sample JSON files.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Result JSON path.",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=24,
        help="Rendered ApeRAG prompt max_entities value.",
    )
    parser.add_argument(
        "--max-relations",
        type=int,
        default=24,
        help="Rendered ApeRAG prompt max_relations value.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per OpenRouter chat completion timeout in seconds.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent model/sample calls. Keep low to avoid provider rate limits.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Request attempts per OpenRouter call. Use 1 for quick baseline sweeps.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Optional completion token cap. Default 0 omits max_tokens to match ApeRAG graph indexing.",
    )
    parser.add_argument(
        "--response-format-json",
        action="store_true",
        help='Also send response_format={"type":"json_object"}. Current ApeRAG graph indexing does not do this yet.',
    )
    parser.add_argument(
        "--chunk-window-size",
        type=int,
        default=1,
        help=(
            "Number of consecutive pseudo-chunks the harness groups into a "
            "single graph extraction window per LLM call (task #30 §3.1.1). "
            "Default 1 = legacy single-chunk behavior. Cannot be combined "
            "with --matrix."
        ),
    )
    parser.add_argument(
        "--matrix",
        default="",
        help=(
            "Comma-separated list of chunk-window sizes to sweep in one "
            "run (e.g. --matrix 1,2,3,5). Each window size produces an "
            "independent results block tagged in the output JSON. Cannot "
            "be combined with --chunk-window-size."
        ),
    )
    parser.add_argument(
        "--pseudo-chunks-per-doc",
        type=int,
        default=4,
        help=(
            "How many pseudo-chunks each sample text is split into before "
            "being grouped into windows. Default 4 keeps short benchmark "
            "samples interpretable across window sizes 1/2/3/5 — for "
            "real multi-chunk documents B2 should add larger samples "
            "and raise this."
        ),
    )
    parser.add_argument(
        "--few-shot-locale",
        default=None,
        help=(
            "Optional few-shot example locale (zh / en / cross_chunk) "
            "passed through to render_extraction_prompt. Default None = "
            "no extra example (task #30 §3.1.3 default-off)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip provider calls; print the harness output schema with "
            "synthetic placeholder result rows. Used by Planetegg B2 "
            "baseline connectivity check (msg=cbe84223)."
        ),
    )
    return parser.parse_args()


def require_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    return key


def curl_json_request(
    method: str,
    url: str,
    *,
    api_key: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Put the Authorization header in curl's stdin config instead of argv so
    # API keys do not show up in process listings.
    config = [
        "silent",
        "show-error",
        f"max-time = {int(timeout)}",
        f"request = {method}",
        f'url = "{url}"',
        f'header = "Authorization: Bearer {api_key}"',
        'header = "Content-Type: application/json"',
        'header = "HTTP-Referer: https://github.com/apecloud/ApeRAG"',
        'header = "X-Title: ApeRAG graph extraction benchmark"',
        'write-out = "\\n%{http_code}"',
    ]
    payload_file_name = None
    try:
        if payload is not None:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as payload_file:
                json.dump(payload, payload_file, ensure_ascii=False)
                payload_file_name = payload_file.name
            config.append(f'data-binary = "@{payload_file_name}"')
        proc = subprocess.run(
            ["curl", "--config", "-"],
            input="\n".join(config) + "\n",
            text=True,
            capture_output=True,
            timeout=timeout + 5,
            check=False,
        )
    finally:
        if payload_file_name:
            Path(payload_file_name).unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"curl exited {proc.returncode}")
    body, _, status_code_text = proc.stdout.rpartition("\n")
    status_code = int(status_code_text or "0")
    if status_code >= 400:
        raise RuntimeError(f"OpenRouter HTTP {status_code}: {body[:500]}")
    return json.loads(body)


def request_with_retries(
    method: str,
    url: str,
    *,
    api_key: str,
    timeout: float,
    attempts: int,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    max_attempts = max(1, attempts)
    for attempt in range(1, max_attempts + 1):
        try:
            return curl_json_request(method, url, api_key=api_key, timeout=timeout, payload=payload)
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(float(attempt))
    assert last_error is not None
    raise last_error


def load_samples(samples_dir: Path) -> list[dict[str, Any]]:
    samples = []
    for path in sorted(samples_dir.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            sample = json.load(fh)
        required = {"id", "text", "language", "expected_entities", "expected_relations"}
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        samples.append(sample)
    if not samples:
        raise ValueError(f"no sample JSON files under {samples_dir}")
    return samples


def fetch_model_prices(api_key: str, timeout: float, attempts: int) -> dict[str, ModelPrice]:
    payload = request_with_retries(
        "GET",
        f"{OPENROUTER_URL}/models",
        api_key=api_key,
        timeout=timeout,
        attempts=attempts,
    )
    prices: dict[str, ModelPrice] = {}
    for row in payload.get("data", []):
        pricing = row.get("pricing") or {}
        prices[row.get("id")] = ModelPrice(
            input_per_token=float(pricing.get("prompt") or 0),
            output_per_token=float(pricing.get("completion") or 0),
            context_length=row.get("context_length"),
        )
    return prices


def strip_code_fence(raw: str) -> str:
    payload = raw.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload)
    return payload.strip()


def normalize_name(name: Any) -> str:
    return re.sub(r"\s+", "", str(name or "").lower())


def entity_hit(expected: str, actual_names: set[str]) -> bool:
    expected_norm = normalize_name(expected)
    return any(expected_norm in actual or actual in expected_norm for actual in actual_names)


def relation_hit(expected: list[str], actual_relations: list[dict[str, Any]]) -> bool:
    expected_source = normalize_name(expected[0])
    expected_target = normalize_name(expected[1])
    for relation in actual_relations:
        actual_source = normalize_name(relation.get("source"))
        actual_target = normalize_name(relation.get("target"))
        forward = (expected_source in actual_source or actual_source in expected_source) and (
            expected_target in actual_target or actual_target in expected_target
        )
        reverse = (expected_source in actual_target or actual_target in expected_source) and (
            expected_target in actual_source or actual_source in expected_target
        )
        if forward or reverse:
            return True
    return False


def parse_extraction(raw: str) -> tuple[bool, str | None, list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        parsed = json.loads(strip_code_fence(raw))
    except json.JSONDecodeError as exc:
        return False, str(exc), [], []
    if not isinstance(parsed, dict):
        return False, f"expected JSON object, got {type(parsed).__name__}", [], []
    entities = [entity for entity in parsed.get("entities", []) or [] if isinstance(entity, dict)]
    relations = [relation for relation in parsed.get("relations", []) or [] if isinstance(relation, dict)]
    return True, None, entities, relations


def split_into_pseudo_chunks(text: str, k: int, *, sample_id: str) -> list[dict[str, str]]:
    """Split ``text`` into ``k`` evenly-sized pseudo-chunks.

    The graph extractor uses a real chunker (parser-driven) in production;
    for benchmark reproducibility we just slice on character boundaries
    and let the LLM see synthetic boundary markers via
    ``[[chunk_id=<id> index=<n>]]``. ``k`` is clamped to ``1`` for empty
    text and to ``len(text)`` for very short text so each pseudo-chunk
    holds at least one character.
    """
    text = text or ""
    k = max(1, min(int(k), max(1, len(text))))
    if k == 1:
        return [{"chunk_id": f"{sample_id}.c0", "text": text}]
    base = len(text) // k
    remainder = len(text) % k
    chunks: list[dict[str, str]] = []
    cursor = 0
    for i in range(k):
        size = base + (1 if i < remainder else 0)
        chunks.append({"chunk_id": f"{sample_id}.c{i}", "text": text[cursor : cursor + size]})
        cursor += size
    return chunks


def build_windows(chunks: list[dict[str, str]], window_size: int) -> list[list[dict[str, str]]]:
    """Group consecutive ``chunks`` into non-overlapping windows of size
    ``window_size`` (task #30 §3.1.1 #1: ``window_overlap=0`` first
    version). The last window may be smaller than ``window_size`` when
    ``len(chunks) % window_size != 0``.
    """
    n = max(1, int(window_size))
    return [chunks[i : i + n] for i in range(0, len(chunks), n)]


def source_chunk_ids_validity(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    allowed_chunk_ids: set[str],
) -> tuple[int, int]:
    """Count entities + relations whose ``source_chunk_ids`` are a
    non-empty subset of ``allowed_chunk_ids`` (task #30 §3.1.3 #2 +
    spec § 6.3 ``source_chunk_ids`` validity rate).

    Returns ``(valid, total)`` so callers can aggregate ratios across
    multiple windows / samples / models.
    """
    valid = 0
    total = 0
    for record in [*entities, *relations]:
        total += 1
        ids = record.get("source_chunk_ids")
        if not isinstance(ids, list) or not ids:
            continue
        if all(isinstance(cid, str) and cid in allowed_chunk_ids for cid in ids):
            valid += 1
    return valid, total


def score_result(
    sample: dict[str, Any],
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    *,
    allowed_chunk_ids: set[str] | None = None,
) -> dict[str, Any]:
    actual_names = {normalize_name(entity.get("name")) for entity in entities}
    entity_hits = sum(1 for expected in sample["expected_entities"] if entity_hit(expected, actual_names))
    relation_hits = sum(1 for expected in sample["expected_relations"] if relation_hit(expected, relations))
    entity_descriptions = [str(entity.get("description") or "").strip() for entity in entities]
    relation_descriptions = [str(relation.get("description") or "").strip() for relation in relations]
    descriptions = entity_descriptions + relation_descriptions
    duplicate_entity_count = len(entities) - len({normalize_name(entity.get("name")) for entity in entities})
    duplicate_relation_count = len(relations) - len(
        {(normalize_name(relation.get("source")), normalize_name(relation.get("target"))) for relation in relations}
    )
    valid_refs = total_refs = 0
    if allowed_chunk_ids is not None:
        valid_refs, total_refs = source_chunk_ids_validity(entities, relations, allowed_chunk_ids)
    return {
        "entity_hits": entity_hits,
        "entity_total": len(sample["expected_entities"]),
        "relation_hits": relation_hits,
        "relation_total": len(sample["expected_relations"]),
        "entities_count": len(entities),
        "relations_count": len(relations),
        "duplicate_entity_count": max(0, duplicate_entity_count),
        "duplicate_relation_count": max(0, duplicate_relation_count),
        "source_chunk_ids_valid": valid_refs,
        "source_chunk_ids_total": total_refs,
        "avg_entity_description_chars": round(
            sum(len(description) for description in entity_descriptions) / max(1, len(entity_descriptions)),
            1,
        ),
        "avg_relation_description_chars": round(
            sum(len(description) for description in relation_descriptions) / max(1, len(relation_descriptions)),
            1,
        ),
        "empty_description_count": sum(1 for description in descriptions if not description),
    }


def call_openrouter(
    *,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float,
    attempts: int,
    max_tokens: int,
    response_format_json: bool,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    started = time.perf_counter()
    response = request_with_retries(
        "POST",
        f"{OPENROUTER_URL}/chat/completions",
        api_key=api_key,
        payload=payload,
        timeout=timeout,
        attempts=attempts,
    )
    latency = time.perf_counter() - started
    return response, latency


def _scale_for_window(base: int, window_chunks: list[dict[str, str]]) -> int:
    """Scale per-chunk caps to per-window caps, mirroring task #30 §3.1.2
    A2 ``base * len(window_chunks)`` co-scale formula. Used for the
    benchmark prompt rendering only — production code path is owned by
    ``aperag.indexing.graph_extractor`` (PR #1921 commit 9b4770ae).
    """
    return max(1, int(base) * max(1, len(window_chunks)))


def run_window(
    *,
    api_key: str,
    model: str,
    sample: dict[str, Any],
    window_chunks: list[dict[str, str]],
    prices: dict[str, ModelPrice],
    timeout: float,
    attempts: int,
    max_tokens: int,
    max_entities: int,
    max_relations: int,
    response_format_json: bool,
    few_shot_locale: str | None,
) -> dict[str, Any]:
    """Render + dispatch one extraction window for one (model, sample)
    combination. Returns a per-window result row that callers aggregate
    into per-document metrics via :func:`aggregate_sample`.
    """
    prompt = render_extraction_prompt(
        window_chunks=window_chunks,
        entity_types=DEFAULT_ENTITY_TYPES,
        language=sample["language"],
        max_entities=_scale_for_window(max_entities, window_chunks),
        max_relations=_scale_for_window(max_relations, window_chunks),
        few_shot_locale=few_shot_locale,
    )
    allowed_ids = {chunk["chunk_id"] for chunk in window_chunks}
    try:
        response, latency = call_openrouter(
            api_key=api_key,
            model=model,
            prompt=prompt,
            timeout=timeout,
            attempts=attempts,
            max_tokens=max_tokens,
            response_format_json=response_format_json,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "window_chunk_ids": sorted(allowed_ids),
            "entities": [],
            "relations": [],
            "latency_s": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    content = response["choices"][0]["message"].get("content") or ""
    usage = response.get("usage") or {}
    json_ok, parse_error, entities, relations = parse_extraction(content)
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return {
        "ok": True,
        "json_ok": json_ok,
        "parse_error": parse_error,
        "window_chunk_ids": sorted(allowed_ids),
        "entities": entities,
        "relations": relations,
        "latency_s": round(latency, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "raw_preview": content[:200],
    }


def aggregate_sample(
    *,
    model: str,
    sample: dict[str, Any],
    window_size: int,
    pseudo_chunks_per_doc: int,
    window_results: list[dict[str, Any]],
    prices: dict[str, ModelPrice],
) -> dict[str, Any]:
    """Aggregate per-window results into one per-document row with the
    7 metrics required by spec § 6.3 (per Planetegg msg=ea7efa7b):
    LLM call count, input+output tokens, wall time, timeout/failure,
    entity+relation totals, duplicate rate, ``source_chunk_ids``
    validity rate.
    """
    all_entities: list[dict[str, Any]] = []
    all_relations: list[dict[str, Any]] = []
    allowed_ids: set[str] = set()
    timeouts_or_failures = 0
    json_ok_count = 0
    total_latency = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    parse_errors: list[str] = []
    for row in window_results:
        if not row.get("ok"):
            timeouts_or_failures += 1
            continue
        if not row.get("json_ok"):
            timeouts_or_failures += 1
            if row.get("parse_error"):
                parse_errors.append(str(row["parse_error"]))
        else:
            json_ok_count += 1
        all_entities.extend(row.get("entities") or [])
        all_relations.extend(row.get("relations") or [])
        allowed_ids.update(row.get("window_chunk_ids") or [])
        total_latency += float(row.get("latency_s") or 0.0)
        total_input_tokens += int(row.get("input_tokens") or 0)
        total_output_tokens += int(row.get("output_tokens") or 0)

    score = score_result(sample, all_entities, all_relations, allowed_chunk_ids=allowed_ids)
    price = prices.get(model, ModelPrice())
    estimated_cost = total_input_tokens * price.input_per_token + total_output_tokens * price.output_per_token

    llm_call_count = len(window_results)
    sample_ok = timeouts_or_failures == 0 and llm_call_count > 0

    return {
        "model": model,
        "sample": sample["id"],
        "window_size": window_size,
        "pseudo_chunks_per_doc": pseudo_chunks_per_doc,
        "ok": sample_ok,
        "llm_call_count": llm_call_count,
        "json_ok_count": json_ok_count,
        "timeout_or_failure_count": timeouts_or_failures,
        "wall_time_s": round(total_latency, 2),
        "input_tokens_total": total_input_tokens,
        "output_tokens_total": total_output_tokens,
        "estimated_cost_usd": round(float(estimated_cost), 8),
        "parse_errors": parse_errors[:3],
        **score,
    }


def summarize(
    models: list[str],
    window_sizes: list[int],
    prices: dict[str, ModelPrice],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-(model, window_size) summary that surfaces the spec § 6.3
    decision-relevant aggregates: ``llm_call_count_total`` for cost
    framing, hit-rate for quality framing, ``source_chunk_ids_valid_rate``
    for provenance framing.
    """
    summary = []
    for model in models:
        for window_size in window_sizes:
            rows = [row for row in results if row["model"] == model and row.get("window_size") == window_size]
            ok_rows = [row for row in rows if row.get("ok")]
            entity_total = sum(row.get("entity_total", 0) for row in rows)
            relation_total = sum(row.get("relation_total", 0) for row in rows)
            source_ids_valid = sum(row.get("source_chunk_ids_valid", 0) for row in rows)
            source_ids_total = sum(row.get("source_chunk_ids_total", 0) for row in rows)
            price = prices.get(model, ModelPrice())
            summary.append(
                {
                    "model": model,
                    "window_size": window_size,
                    "price_per_m_input": round(price.input_per_token * 1_000_000, 4),
                    "price_per_m_output": round(price.output_per_token * 1_000_000, 4),
                    "context_length": price.context_length,
                    "samples_run": len(rows),
                    "samples_ok": len(ok_rows),
                    "llm_call_count_total": sum(row.get("llm_call_count", 0) for row in rows),
                    "timeout_or_failure_count_total": sum(row.get("timeout_or_failure_count", 0) for row in rows),
                    "input_tokens_total": sum(row.get("input_tokens_total", 0) for row in rows),
                    "output_tokens_total": sum(row.get("output_tokens_total", 0) for row in rows),
                    "wall_time_s_total": round(sum(row.get("wall_time_s", 0) for row in rows), 2),
                    "estimated_cost_usd_total": round(sum(row.get("estimated_cost_usd", 0) for row in rows), 8),
                    "entity_hit_rate": round(
                        sum(row.get("entity_hits", 0) for row in rows) / max(1, entity_total),
                        3,
                    ),
                    "relation_hit_rate": round(
                        sum(row.get("relation_hits", 0) for row in rows) / max(1, relation_total),
                        3,
                    ),
                    "duplicate_entity_count_total": sum(row.get("duplicate_entity_count", 0) for row in rows),
                    "duplicate_relation_count_total": sum(row.get("duplicate_relation_count", 0) for row in rows),
                    "source_chunk_ids_valid_rate": round(source_ids_valid / max(1, source_ids_total), 3),
                    "source_chunk_ids_total": source_ids_total,
                    "json_ok_rate": round(
                        sum(row.get("json_ok_count", 0) for row in rows)
                        / max(1, sum(row.get("llm_call_count", 0) for row in rows)),
                        3,
                    ),
                }
            )
    return summary


def _resolve_window_sizes(args: argparse.Namespace) -> list[int]:
    if args.matrix and args.chunk_window_size != 1:
        raise SystemExit("--matrix and --chunk-window-size are mutually exclusive")
    if args.matrix:
        sizes = [int(token.strip()) for token in args.matrix.split(",") if token.strip()]
        if not sizes:
            raise SystemExit("--matrix produced an empty list")
        return sizes
    return [int(args.chunk_window_size)]


def _dry_run_payload(
    *,
    models: list[str],
    window_sizes: list[int],
    samples: list[dict[str, Any]],
    pseudo_chunks_per_doc: int,
) -> dict[str, Any]:
    """Synthetic payload for ``--dry-run``: same schema as a real run
    but every metric is ``0`` / placeholder. B2 (Planetegg msg=cbe84223)
    uses this to verify their downstream tooling can ingest the matrix
    output before paying for a real provider run.
    """
    placeholder_results = []
    for model in models:
        for window_size in window_sizes:
            for sample in samples:
                placeholder_results.append(
                    {
                        "model": model,
                        "sample": sample["id"],
                        "window_size": window_size,
                        "pseudo_chunks_per_doc": pseudo_chunks_per_doc,
                        "ok": False,
                        "llm_call_count": 0,
                        "json_ok_count": 0,
                        "timeout_or_failure_count": 0,
                        "wall_time_s": 0.0,
                        "input_tokens_total": 0,
                        "output_tokens_total": 0,
                        "estimated_cost_usd": 0.0,
                        "parse_errors": [],
                        "entity_hits": 0,
                        "entity_total": len(sample["expected_entities"]),
                        "relation_hits": 0,
                        "relation_total": len(sample["expected_relations"]),
                        "entities_count": 0,
                        "relations_count": 0,
                        "duplicate_entity_count": 0,
                        "duplicate_relation_count": 0,
                        "source_chunk_ids_valid": 0,
                        "source_chunk_ids_total": 0,
                        "avg_entity_description_chars": 0.0,
                        "avg_relation_description_chars": 0.0,
                        "empty_description_count": 0,
                        "_dry_run": True,
                    }
                )
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provider": "openrouter",
        "mode": "dry_run",
        "models": models,
        "window_sizes": window_sizes,
        "pseudo_chunks_per_doc": pseudo_chunks_per_doc,
        "samples": [sample["id"] for sample in samples],
        "summary": [],
        "results": placeholder_results,
        "note": (
            "Dry-run skeleton — no provider calls were made. Real runs "
            "populate every metric and add a non-empty 'summary' block."
        ),
    }


def main() -> int:
    args = parse_args()
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    window_sizes = _resolve_window_sizes(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = load_samples(Path(args.samples_dir))

    if args.dry_run:
        payload = _dry_run_payload(
            models=models,
            window_sizes=window_sizes,
            samples=samples,
            pseudo_chunks_per_doc=args.pseudo_chunks_per_doc,
        )
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"mode": "dry_run", "results_count": len(payload["results"])}, ensure_ascii=False))
        print(f"wrote {output_path}", file=sys.stderr)
        return 0

    api_key = require_key()
    prices = fetch_model_prices(api_key, args.timeout, args.attempts)

    work_items: list[tuple[str, dict[str, Any], int, list[list[dict[str, str]]]]] = []
    for sample in samples:
        chunks = split_into_pseudo_chunks(sample["text"], args.pseudo_chunks_per_doc, sample_id=sample["id"])
        for window_size in window_sizes:
            windows = build_windows(chunks, window_size)
            for model in models:
                work_items.append((model, sample, window_size, windows))

    results: list[dict[str, Any]] = []
    max_workers = max(1, args.concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map: dict[concurrent.futures.Future[Any], tuple[str, str, int, int]] = {}
        for model, sample, window_size, windows in work_items:
            for window_index, window_chunks in enumerate(windows):
                print(
                    f"running model={model} sample={sample['id']} window_size={window_size} "
                    f"window={window_index + 1}/{len(windows)}",
                    file=sys.stderr,
                )
                future = executor.submit(
                    run_window,
                    api_key=api_key,
                    model=model,
                    sample=sample,
                    window_chunks=window_chunks,
                    prices=prices,
                    timeout=args.timeout,
                    attempts=args.attempts,
                    max_tokens=args.max_tokens,
                    max_entities=args.max_entities,
                    max_relations=args.max_relations,
                    response_format_json=args.response_format_json,
                    few_shot_locale=args.few_shot_locale,
                )
                future_map[future] = (model, sample["id"], window_size, window_index)

        per_sample_buckets: dict[tuple[str, str, int], list[tuple[int, dict[str, Any]]]] = {}
        for future in concurrent.futures.as_completed(future_map):
            model, sample_id, window_size, window_index = future_map[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "ok": False,
                    "error": repr(exc),
                    "window_chunk_ids": [],
                    "entities": [],
                    "relations": [],
                    "latency_s": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            per_sample_buckets.setdefault((model, sample_id, window_size), []).append((window_index, row))

    sample_by_id = {sample["id"]: sample for sample in samples}
    for (model, sample_id, window_size), bucket in per_sample_buckets.items():
        bucket.sort(key=lambda item: item[0])
        window_results = [row for _, row in bucket]
        results.append(
            aggregate_sample(
                model=model,
                sample=sample_by_id[sample_id],
                window_size=window_size,
                pseudo_chunks_per_doc=args.pseudo_chunks_per_doc,
                window_results=window_results,
                prices=prices,
            )
        )

    sample_order = {sample["id"]: index for index, sample in enumerate(samples)}
    model_order = {model: index for index, model in enumerate(models)}
    window_order = {size: index for index, size in enumerate(window_sizes)}
    results.sort(
        key=lambda row: (
            model_order.get(row["model"], 9999),
            window_order.get(row.get("window_size"), 9999),
            sample_order.get(row["sample"], 9999),
        )
    )

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provider": "openrouter",
        "mode": "response_format_json" if args.response_format_json else "prompt_only",
        "note": (
            "Manual benchmark using ApeRAG render_extraction_prompt. "
            "API keys are intentionally excluded from this artifact."
        ),
        "models": models,
        "window_sizes": window_sizes,
        "pseudo_chunks_per_doc": args.pseudo_chunks_per_doc,
        "samples": [sample["id"] for sample in samples],
        "summary": summarize(models, window_sizes, prices, results),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
