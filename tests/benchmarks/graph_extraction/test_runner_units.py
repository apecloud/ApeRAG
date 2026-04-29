# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the manual graph-extraction benchmark harness.

The benchmark itself runs against real OpenRouter providers and is gated
out of CI (manual run with ``OPENROUTER_API_KEY``). These unit tests
exercise the **structural** pieces — chunking, windowing, validity
counting, per-document aggregation — so that the matrix harness keeps
producing the schema documented in spec § 6.3 even when the benchmark
itself is not run in CI.

task #30 B1 (msg=cecae5ed). Sister to PR #1909 / PR #1912 (task #32
Phase A) — same testing-lane pattern: fixture-isolated structural test
that pins the contract a downstream consumer (here ``B2`` real-provider
runs) needs without paying provider cost.
"""

from __future__ import annotations

from tests.benchmarks.graph_extraction import runner


def _sample(sample_id: str = "syn_cn", text: str = "AAABBBCCCDDD") -> dict[str, object]:
    return {
        "id": sample_id,
        "language": "Chinese",
        "text": text,
        "expected_entities": ["A", "B", "C", "D"],
        "expected_relations": [["A", "B"], ["C", "D"]],
    }


def test_split_into_pseudo_chunks_returns_k_chunks_with_distinct_ids():
    chunks = runner.split_into_pseudo_chunks("AAAABBBBCCCC", k=3, sample_id="syn_cn")
    assert [chunk["chunk_id"] for chunk in chunks] == ["syn_cn.c0", "syn_cn.c1", "syn_cn.c2"]
    assert "".join(chunk["text"] for chunk in chunks) == "AAAABBBBCCCC"
    assert all(chunk["text"] for chunk in chunks)


def test_split_into_pseudo_chunks_handles_short_text():
    chunks = runner.split_into_pseudo_chunks("ab", k=5, sample_id="syn_cn")
    assert len(chunks) == 2
    assert "".join(chunk["text"] for chunk in chunks) == "ab"


def test_build_windows_non_overlapping():
    chunks = [{"chunk_id": f"c{i}", "text": "x"} for i in range(5)]
    assert runner.build_windows(chunks, 1) == [[chunks[i]] for i in range(5)]
    assert runner.build_windows(chunks, 2) == [chunks[0:2], chunks[2:4], chunks[4:5]]
    assert runner.build_windows(chunks, 3) == [chunks[0:3], chunks[3:5]]
    assert runner.build_windows(chunks, 5) == [chunks[0:5]]


def test_source_chunk_ids_validity_subset_only():
    allowed = {"c0", "c1"}
    entities = [
        {"name": "A", "source_chunk_ids": ["c0"]},
        {"name": "B", "source_chunk_ids": ["c1", "c0"]},
        {"name": "C", "source_chunk_ids": ["c2"]},
        {"name": "D", "source_chunk_ids": []},
        {"name": "E"},
    ]
    relations = [
        {"source": "A", "target": "B", "source_chunk_ids": ["c0", "c1"]},
        {"source": "C", "target": "D", "source_chunk_ids": ["c0", "ghost"]},
    ]
    valid, total = runner.source_chunk_ids_validity(entities, relations, allowed)
    assert total == 7
    assert valid == 3


def test_aggregate_sample_per_document_metrics():
    """Per-window ``source_chunk_ids_valid`` / ``source_chunk_ids_total``
    are produced inside :func:`run_window`; ``aggregate_sample`` only
    sums them. We pre-populate the per-window counts here.
    """
    sample = _sample()
    window_results = [
        {
            "ok": True,
            "json_ok": True,
            "parse_error": None,
            "window_chunk_ids": ["syn_cn.c0", "syn_cn.c1"],
            "entities": [
                {"name": "A", "description": "anchor", "source_chunk_ids": ["syn_cn.c0"]},
                {"name": "B", "description": "neighbor", "source_chunk_ids": ["syn_cn.c1"]},
                {"name": "A", "description": "duplicate", "source_chunk_ids": ["syn_cn.c1"]},
            ],
            "relations": [
                {
                    "source": "A",
                    "target": "B",
                    "description": "links",
                    "source_chunk_ids": ["syn_cn.c0", "syn_cn.c1"],
                },
            ],
            "source_chunk_ids_valid": 4,
            "source_chunk_ids_total": 4,
            "latency_s": 1.5,
            "input_tokens": 100,
            "output_tokens": 50,
        },
        {
            "ok": True,
            "json_ok": True,
            "parse_error": None,
            "window_chunk_ids": ["syn_cn.c2", "syn_cn.c3"],
            "entities": [
                {"name": "C", "description": "third", "source_chunk_ids": ["syn_cn.c2"]},
                {"name": "D", "description": "fourth", "source_chunk_ids": ["bad_id"]},
            ],
            "relations": [],
            "source_chunk_ids_valid": 1,
            "source_chunk_ids_total": 2,
            "latency_s": 1.2,
            "input_tokens": 80,
            "output_tokens": 40,
        },
    ]

    row = runner.aggregate_sample(
        model="m",
        sample=sample,
        window_size=2,
        pseudo_chunks_per_doc=4,
        window_results=window_results,
        prices={},
    )

    assert row["ok"] is True
    assert row["llm_call_count"] == 2
    assert row["json_ok_count"] == 2
    assert row["timeout_or_failure_count"] == 0
    assert row["wall_time_s"] == 2.7
    assert row["input_tokens_total"] == 180
    assert row["output_tokens_total"] == 90
    assert row["entities_count"] == 5
    assert row["relations_count"] == 1
    assert row["duplicate_entity_count"] == 1
    assert row["entity_hits"] == 4
    assert row["relation_hits"] == 1
    assert row["source_chunk_ids_total"] == 6
    assert row["source_chunk_ids_valid"] == 5


def test_aggregate_sample_source_chunk_ids_is_window_scoped_not_union():
    """A record produced in window-0 that references a chunk_id from
    window-1 must NOT be counted as valid by the per-document aggregate.
    Validity is computed inside ``run_window`` against the window's own
    ``allowed_chunk_ids`` (BLOCKER fix per @ziang msg=56912dae +
    @huangzhangshu msg=cda4dc75). ``aggregate_sample`` only sums.
    """
    sample = _sample()
    # Window-0 records all 1 entity that incorrectly references c1 (a
    # chunk_id that only exists in window-1). Per-window scoring marks
    # it as invalid: 0/1. Window-1 has one valid record: 1/1.
    # If aggregate were to compute on the *union* {c0, c1}, the
    # window-0 record would be misjudged valid → 2/2. The window-scoped
    # path correctly returns 1/2 valid_total.
    window_results = [
        {
            "ok": True,
            "json_ok": True,
            "window_chunk_ids": ["c0"],
            "entities": [{"name": "A", "source_chunk_ids": ["c1"]}],  # ghost ref
            "relations": [],
            "source_chunk_ids_valid": 0,  # produced by run_window vs {"c0"}
            "source_chunk_ids_total": 1,
            "latency_s": 0.5,
            "input_tokens": 10,
            "output_tokens": 5,
        },
        {
            "ok": True,
            "json_ok": True,
            "window_chunk_ids": ["c1"],
            "entities": [{"name": "B", "source_chunk_ids": ["c1"]}],
            "relations": [],
            "source_chunk_ids_valid": 1,
            "source_chunk_ids_total": 1,
            "latency_s": 0.5,
            "input_tokens": 10,
            "output_tokens": 5,
        },
    ]

    row = runner.aggregate_sample(
        model="m",
        sample=sample,
        window_size=1,
        pseudo_chunks_per_doc=2,
        window_results=window_results,
        prices={},
    )

    assert row["source_chunk_ids_valid"] == 1
    assert row["source_chunk_ids_total"] == 2


def test_aggregate_sample_marks_failure_when_any_window_errors():
    sample = _sample()
    window_results = [
        {
            "ok": True,
            "json_ok": True,
            "window_chunk_ids": ["syn_cn.c0"],
            "entities": [],
            "relations": [],
            "latency_s": 0.5,
            "input_tokens": 10,
            "output_tokens": 5,
        },
        {
            "ok": False,
            "error": "boom",
            "window_chunk_ids": ["syn_cn.c1"],
            "entities": [],
            "relations": [],
            "latency_s": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    ]
    row = runner.aggregate_sample(
        model="m",
        sample=sample,
        window_size=1,
        pseudo_chunks_per_doc=2,
        window_results=window_results,
        prices={},
    )
    assert row["ok"] is False
    assert row["timeout_or_failure_count"] == 1
    assert row["llm_call_count"] == 2
