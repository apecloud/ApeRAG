#!/usr/bin/env python3
"""Manual benchmark for knowledge-graph extraction model quality.

This is intentionally not part of the default pytest or CI suite: it calls
external LLM providers and consumes API credits. Run it only when comparing
models, prompts, or JSON-mode changes.
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


def score_result(
    sample: dict[str, Any], entities: list[dict[str, Any]], relations: list[dict[str, Any]]
) -> dict[str, Any]:
    actual_names = {normalize_name(entity.get("name")) for entity in entities}
    entity_hits = sum(1 for expected in sample["expected_entities"] if entity_hit(expected, actual_names))
    relation_hits = sum(1 for expected in sample["expected_relations"] if relation_hit(expected, relations))
    entity_descriptions = [str(entity.get("description") or "").strip() for entity in entities]
    relation_descriptions = [str(relation.get("description") or "").strip() for relation in relations]
    descriptions = entity_descriptions + relation_descriptions
    return {
        "entity_hits": entity_hits,
        "entity_total": len(sample["expected_entities"]),
        "relation_hits": relation_hits,
        "relation_total": len(sample["expected_relations"]),
        "entities_count": len(entities),
        "relations_count": len(relations),
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


def run_one(
    *,
    api_key: str,
    model: str,
    sample: dict[str, Any],
    prices: dict[str, ModelPrice],
    timeout: float,
    attempts: int,
    max_tokens: int,
    max_entities: int,
    max_relations: int,
    response_format_json: bool,
) -> dict[str, Any]:
    prompt = render_extraction_prompt(
        input_text=sample["text"],
        entity_types=DEFAULT_ENTITY_TYPES,
        language=sample["language"],
        max_entities=max_entities,
        max_relations=max_relations,
    )
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
            "model": model,
            "sample": sample["id"],
            "ok": False,
            "error": str(exc),
        }

    content = response["choices"][0]["message"].get("content") or ""
    usage = response.get("usage") or {}
    json_ok, parse_error, entities, relations = parse_extraction(content)
    score = score_result(sample, entities, relations)
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    price = prices.get(model, ModelPrice())
    estimated_cost = usage.get("cost")
    if estimated_cost is None:
        estimated_cost = input_tokens * price.input_per_token + output_tokens * price.output_per_token
    return {
        "model": model,
        "sample": sample["id"],
        "ok": True,
        "json_ok": json_ok,
        "parse_error": parse_error,
        **score,
        "latency_s": round(latency, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_s": round(output_tokens / latency, 2) if output_tokens and latency else None,
        "estimated_cost_usd": round(float(estimated_cost), 8),
        "raw_preview": content[:300],
    }


def summarize(models: list[str], prices: dict[str, ModelPrice], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for model in models:
        rows = [row for row in results if row["model"] == model]
        ok_rows = [row for row in rows if row.get("ok")]
        price = prices.get(model, ModelPrice())
        entity_total = sum(row.get("entity_total", 0) for row in ok_rows)
        relation_total = sum(row.get("relation_total", 0) for row in ok_rows)
        speed_rows = [row for row in ok_rows if row.get("tokens_per_s") is not None]
        parsed_rows = [row for row in ok_rows if row.get("json_ok")]
        summary.append(
            {
                "model": model,
                "price_per_m_input": round(price.input_per_token * 1_000_000, 4),
                "price_per_m_output": round(price.output_per_token * 1_000_000, 4),
                "context_length": price.context_length,
                "runs": len(rows),
                "success_runs": len(ok_rows),
                "json_ok_runs": sum(1 for row in ok_rows if row.get("json_ok")),
                "entity_hit_rate": round(
                    sum(row.get("entity_hits", 0) for row in ok_rows) / max(1, entity_total),
                    3,
                ),
                "relation_hit_rate": round(
                    sum(row.get("relation_hits", 0) for row in ok_rows) / max(1, relation_total),
                    3,
                ),
                "avg_latency_s": round(
                    sum(row.get("latency_s", 0) for row in ok_rows) / max(1, len(ok_rows)),
                    2,
                ),
                "avg_tokens_per_s": round(
                    sum(row.get("tokens_per_s") or 0 for row in speed_rows) / max(1, len(speed_rows)),
                    2,
                ),
                "avg_cost_usd": round(
                    sum(row.get("estimated_cost_usd", 0) for row in ok_rows) / max(1, len(ok_rows)),
                    8,
                ),
                "avg_entity_description_chars": round(
                    sum(row.get("avg_entity_description_chars", 0) for row in parsed_rows) / max(1, len(parsed_rows)),
                    1,
                ),
                "avg_relation_description_chars": round(
                    sum(row.get("avg_relation_description_chars", 0) for row in parsed_rows) / max(1, len(parsed_rows)),
                    1,
                ),
                "empty_description_count": sum(row.get("empty_description_count", 0) for row in parsed_rows),
                "errors": [row.get("error") for row in rows if not row.get("ok")],
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    api_key = require_key()
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = load_samples(Path(args.samples_dir))
    prices = fetch_model_prices(api_key, args.timeout, args.attempts)

    work_items = [(model, sample) for model in models for sample in samples]
    results: list[dict[str, Any]] = []
    max_workers = max(1, args.concurrency)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for model, sample in work_items:
            print(f"running model={model} sample={sample['id']}", file=sys.stderr)
            future = executor.submit(
                run_one,
                api_key=api_key,
                model=model,
                sample=sample,
                prices=prices,
                timeout=args.timeout,
                attempts=args.attempts,
                max_tokens=args.max_tokens,
                max_entities=args.max_entities,
                max_relations=args.max_relations,
                response_format_json=args.response_format_json,
            )
            futures[future] = (model, sample["id"])
        for future in concurrent.futures.as_completed(futures):
            model, sample_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "model": model,
                        "sample": sample_id,
                        "ok": False,
                        "error": repr(exc),
                    }
                )

    sample_order = {sample["id"]: index for index, sample in enumerate(samples)}
    model_order = {model: index for index, model in enumerate(models)}
    results.sort(key=lambda row: (model_order.get(row["model"], 9999), sample_order.get(row["sample"], 9999)))

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "provider": "openrouter",
        "mode": "response_format_json" if args.response_format_json else "prompt_only",
        "note": (
            "Manual benchmark using ApeRAG render_extraction_prompt. "
            "API keys are intentionally excluded from this artifact."
        ),
        "models": models,
        "samples": [sample["id"] for sample in samples],
        "summary": summarize(models, prices, results),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
