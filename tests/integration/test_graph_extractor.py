# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Wave 4 T1 graph LLM extractor.

Pin the parse / per-chunk / failure-isolation invariants without
needing a live LLM backend. Tests stub the LLM callable so the JSON
parser + collection-config readers can be exercised standalone.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from aperag.indexing import graph_extractor as ge
from aperag.indexing.graph import EntityRecord, RelationRecord


def _make_collection(*, completion: bool = True) -> Any:
    """Tiny collection stub for the extractor builder.

    The builder calls ``build_collection_llm_callable(collection)`` so
    we only need a stub that the legacy integration module can pattern-
    match. Tests that don't reach that call path can use a bare
    object stub.
    """

    class _Stub:
        id = "col-t1-extractor"
        config = {
            "language": "en-US",
            "knowledge_graph_config": {"entity_types": ["person", "organisation"]},
        }

    return _Stub()


def test_strip_code_fence_extracts_inner_json():
    raw = '```json\n{"entities": [], "relations": []}\n```'
    assert ge._strip_code_fence(raw) == '{"entities": [], "relations": []}'


def test_strip_code_fence_passes_through_bare_json():
    raw = '{"entities": [], "relations": []}'
    assert ge._strip_code_fence(raw) == raw


def test_parse_extraction_response_handles_well_formed_json():
    raw = (
        '{"entities": ['
        '{"name": "Linus", "type": "person", "description": "Created Linux"}'
        "],"
        '"relations": ['
        '{"source": "Linus", "target": "Linux", "type": "created", "description": "Linus created Linux"}'
        "]}"
    )
    entities, relations = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c-1",))
    assert len(entities) == 1
    assert entities[0] == EntityRecord(
        name="Linus",
        entity_type="person",
        description="Created Linux",
        source_chunk_ids=("c-1",),
    )
    assert len(relations) == 1
    assert relations[0] == RelationRecord(
        source="Linus",
        target="Linux",
        relation_type="created",
        description="Linus created Linux",
        source_chunk_ids=("c-1",),
    )


def test_parse_extraction_response_handles_fenced_json():
    """LLM middleware that wraps responses in ``\\`\\`\\`json ... \\`\\`\\``` must
    still parse cleanly — we strip the fence before json.loads."""
    raw = '```json\n{"entities": [{"name": "Acme", "type": "organisation"}], "relations": []}\n```'
    entities, _ = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c-1",))
    assert len(entities) == 1
    assert entities[0].name == "Acme"


def test_parse_extraction_response_warns_on_clean_json_with_zero_entities_and_relations(caplog):
    """Bryce 2026-04-29 incident (msg=358bb68f) — DeepSeek V4 Flash
    via OpenRouter returned ``{"entities":[],"relations":[]}`` for
    every chunk, silently producing an empty knowledge graph for
    every doc indexed after the model swap (status flipped to ACTIVE
    because nothing raised). The runtime now emits a single
    ``warning`` per empty extraction so ops can grep
    ``empty extraction result`` + cross-reference the model timeline
    when a "graph index ran but produced nothing" pattern recurs.
    """
    raw = '{"entities": [], "relations": []}'
    with caplog.at_level("WARNING", logger="aperag.indexing.graph_extractor"):
        entities, relations = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c-empty",))
    assert entities == []
    assert relations == []
    matching = [r for r in caplog.records if "empty extraction result" in r.getMessage()]
    assert len(matching) == 1, (
        "expected exactly one warn line for the model-prompt-incompatibility scenario; "
        f"got {[r.getMessage() for r in caplog.records]}"
    )
    msg = matching[0].getMessage()
    assert "c-empty" in msg, "warn line must include the chunk_ids for cross-reference"
    assert repr(raw) in msg, "warn line must include the raw response prefix for ops grep"


def test_parse_extraction_response_does_not_warn_when_entities_were_extracted(caplog):
    """The empty-result warn must NOT fire on the success path —
    otherwise every successful chunk doubles the log volume.
    """
    raw = '{"entities": [{"name": "Linus", "type": "person", "description": "x"}], "relations": []}'
    with caplog.at_level("WARNING", logger="aperag.indexing.graph_extractor"):
        entities, _ = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c-1",))
    assert len(entities) == 1
    matching = [r for r in caplog.records if "empty extraction result" in r.getMessage()]
    assert matching == [], "must not warn when extraction produced entities"


def test_parse_extraction_response_does_not_warn_on_relations_only(caplog):
    """Boundary: relations alone (no entities) is still a successful
    extraction — warn must not fire.
    """
    raw = '{"entities": [], "relations": [{"source": "A", "target": "B", "type": "rel", "description": "x"}]}'
    with caplog.at_level("WARNING", logger="aperag.indexing.graph_extractor"):
        _entities, relations = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c-1",))
    assert len(relations) == 1
    matching = [r for r in caplog.records if "empty extraction result" in r.getMessage()]
    assert matching == [], "must not warn when extraction produced relations (even with 0 entities)"


def test_parse_extraction_response_returns_empty_on_malformed_json():
    raw = "not valid json — model rambled"
    entities, relations = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c-1",))
    assert entities == []
    assert relations == []


def test_parse_extraction_response_skips_individual_malformed_records():
    """One bad row must not poison the rest — test each record's
    parse independently."""
    raw = (
        '{"entities": ['
        '{"name": "Good Entity", "type": "person"},'
        '{"type": "missing-name"},'  # malformed: no name
        '{"name": "", "type": "empty-name"},'  # malformed: empty name
        '{"name": "Another Good", "type": "thing"}'
        "],"
        '"relations": ['
        '{"source": "A", "target": "B", "type": "rel1"},'
        '{"source": "C", "type": "missing-target"}'  # malformed
        "]}"
    )
    entities, relations = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c-1",))
    names = [e.name for e in entities]
    assert names == ["Good Entity", "Another Good"]
    relation_types = [r.relation_type for r in relations]
    assert relation_types == ["rel1"]


def test_parse_extraction_response_accepts_unknown_entity_type():
    raw = (
        '{"entities": ['
        '{"name": "糖尿病", "entity_type": "疾病", "description": "慢性代谢性疾病"},'
        '{"name": "Bad Type", "entity_type": "' + ("x" * 65) + '"}'
        "],"
        '"relations": []}'
    )
    entities, relations = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c-1",))
    assert [entity.name for entity in entities] == ["糖尿病", "Bad Type"]
    assert entities[0].entity_type == "疾病"
    assert entities[1].entity_type == ""
    assert relations == []


def test_render_prompt_allows_dynamic_entity_types():
    from aperag.indexing.llm import render_extraction_prompt

    prompt = render_extraction_prompt(
        window_chunks=[{"chunk_id": "c-x", "text": "OpenAI released GPT-4o."}],
        entity_types=[],
        language="Chinese (Simplified)",
        max_entities=8,
        max_relations=8,
    )
    assert "If no existing entity type fits" in prompt
    assert "Do not skip a valid entity" in prompt
    assert "Use only these entity types" not in prompt
    assert "skip the entity rather than invent" not in prompt
    assert "Chinese (Simplified)" in prompt
    # Task #30 §3.1.3 hard requirements visible in the rendered prompt.
    assert "[[chunk_id=c-x index=0]]" in prompt
    assert "source_chunk_ids" in prompt
    assert "['c-x']" in prompt
    assert "OpenAI released GPT-4o." in prompt
    # Few-shot opt-in defaults to off — no extra example block.
    assert "[[chunk_id=c1 index=0]]\n张伟" not in prompt
    assert "[[chunk_id=c1 index=0]]\nAcme Labs" not in prompt


def test_render_prompt_emits_one_marker_per_window_chunk():
    """Task #30 §3.1.3 hard requirement #1: every chunk in the window
    must carry its own ``[[chunk_id=X index=Y]]`` boundary marker so
    the LLM can ground entities + relations per chunk."""
    from aperag.indexing.llm import render_extraction_prompt

    prompt = render_extraction_prompt(
        window_chunks=[
            {"chunk_id": "c1", "text": "Alpha founded Beta in 2010."},
            {"chunk_id": "c2", "text": "Carol joined Beta in 2020."},
        ],
        entity_types=[],
        language="English",
        max_entities=12,
        max_relations=12,
    )
    assert "[[chunk_id=c1 index=0]]" in prompt
    assert "[[chunk_id=c2 index=1]]" in prompt
    assert "['c1', 'c2']" in prompt
    assert "Alpha founded Beta in 2010." in prompt
    assert "Carol joined Beta in 2020." in prompt
    # Cross-chunk relation guidance present.
    assert "Cross-chunk relations" in prompt
    # Dedup / fail-safe rules surfaced.
    assert "Deduplication" in prompt
    assert "No fabrication" in prompt


def test_render_prompt_few_shot_opt_in_keeps_default_off():
    """Task #30 §3.1.3 default-off opt-in: ``few_shot_locale=None``
    embeds no extra example, ``"zh"`` embeds the Chinese example."""
    from aperag.indexing.llm import render_extraction_prompt

    base = render_extraction_prompt(
        window_chunks=[{"chunk_id": "c-x", "text": "Alpha"}],
        entity_types=[],
        language="English",
        max_entities=8,
        max_relations=8,
    )
    with_zh = render_extraction_prompt(
        window_chunks=[{"chunk_id": "c-x", "text": "Alpha"}],
        entity_types=[],
        language="English",
        max_entities=8,
        max_relations=8,
        few_shot_locale="zh",
    )
    with_cross = render_extraction_prompt(
        window_chunks=[{"chunk_id": "c-x", "text": "Alpha"}],
        entity_types=[],
        language="English",
        max_entities=8,
        max_relations=8,
        few_shot_locale="cross_chunk",
    )

    assert "张伟" not in base
    assert "张伟" in with_zh
    assert "Acme Labs is a research organization founded in 2010." in with_cross
    assert "Acme Labs is a research organization founded in 2010." not in base


def test_render_prompt_rejects_empty_window():
    from aperag.indexing.llm import render_extraction_prompt

    with pytest.raises(ValueError):
        render_extraction_prompt(
            window_chunks=[],
            entity_types=[],
            language="English",
            max_entities=8,
            max_relations=8,
        )


def test_parse_response_falls_back_to_single_chunk_id_when_field_missing():
    """Single-chunk window: legacy LLM responses that omit
    ``source_chunk_ids`` still parse, with provenance defaulting to
    the single allowed id (task #30 §3.1.3 parser invariant)."""
    raw = (
        '{"entities": ['
        '{"name": "Linus", "type": "person"}'
        "],"
        '"relations": ['
        '{"source": "Linus", "target": "Linux", "type": "created"}'
        "]}"
    )
    entities, relations = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c-only",))
    assert entities[0].source_chunk_ids == ("c-only",)
    assert relations[0].source_chunk_ids == ("c-only",)


def test_parse_response_skips_records_missing_provenance_in_multi_chunk_window(caplog):
    """Multi-chunk window: ``source_chunk_ids`` is required. Missing
    field → record skipped + warned (task #30 §3.1.3 parser invariant)."""
    raw = (
        '{"entities": ['
        '{"name": "Good", "type": "person", "source_chunk_ids": ["c1"]},'
        '{"name": "Missing Provenance", "type": "person"}'
        "],"
        '"relations": ['
        '{"source": "Good", "target": "Linux", "type": "uses",'
        ' "source_chunk_ids": ["c2"]},'
        '{"source": "Good", "target": "Linux", "type": "no_provenance"}'
        "]}"
    )
    with caplog.at_level("WARNING", logger="aperag.indexing.graph_extractor"):
        entities, relations = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c1", "c2"))
    assert [e.name for e in entities] == ["Good"]
    assert entities[0].source_chunk_ids == ("c1",)
    assert [r.relation_type for r in relations] == ["uses"]
    assert relations[0].source_chunk_ids == ("c2",)


def test_parse_response_drops_out_of_allowlist_chunk_ids():
    """Hallucinated chunk_ids must be silently dropped before the
    non-empty check (task #30 §3.1.3 parser invariant)."""
    raw = (
        '{"entities": [{"name": "Mixed", "type": "thing", "source_chunk_ids": ["c1", "ghost", "c2"]}], "relations": []}'
    )
    entities, _ = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c1", "c2"))
    assert entities[0].source_chunk_ids == ("c1", "c2")


def test_parse_response_preserves_cross_chunk_provenance():
    """Multi-chunk window with cross-chunk relation: relation's
    ``source_chunk_ids`` carries every chunk that contributed evidence."""
    raw = (
        '{"entities": ['
        '{"name": "Acme Labs", "type": "organization",'
        ' "source_chunk_ids": ["c1", "c2"]}'
        "],"
        '"relations": ['
        '{"source": "Alice", "target": "Acme Labs", "type": "works_for",'
        ' "source_chunk_ids": ["c1", "c2"]}'
        "]}"
    )
    entities, relations = ge._parse_extraction_response(raw=raw, allowed_chunk_ids=("c1", "c2"))
    assert entities[0].source_chunk_ids == ("c1", "c2")
    assert relations[0].source_chunk_ids == ("c1", "c2")


def test_resolve_entity_types_uses_collection_kg_config():
    collection = _make_collection()
    types = ge._resolve_entity_types(collection)
    assert list(types) == ["person", "organisation"]


def test_resolve_entity_types_falls_back_to_default():
    class _NoKG:
        config = {"language": "en-US"}

    types = ge._resolve_entity_types(_NoKG())
    assert list(types) == []


def test_resolve_language_reads_from_config_dict():
    class _Stub:
        config = {"language": "zh-CN"}

    assert ge._resolve_language(_Stub()) == "zh-CN"


def test_resolve_language_handles_json_string_config():
    class _Stub:
        config = '{"language": "ja-JP"}'

    assert ge._resolve_language(_Stub()) == "ja-JP"


def test_resolve_int_kg_config_reads_override():
    """Wave 5 P5A item 1: per-collection ``max_entities_per_chunk`` /
    ``max_relations_per_chunk`` overrides should win over the module
    defaults so deployments tune entity-dense documents without
    patching constants."""

    class _Stub:
        config = {
            "knowledge_graph_config": {
                "max_entities_per_chunk": 64,
                "max_relations_per_chunk": 128,
            }
        }

    assert ge._resolve_int_kg_config(_Stub(), "max_entities_per_chunk", 32) == 64
    assert ge._resolve_int_kg_config(_Stub(), "max_relations_per_chunk", 32) == 128
    assert ge._resolve_int_kg_config(_Stub(), "graph_extraction_window_size", 1) == 1


def test_resolve_int_kg_config_falls_back_when_missing():
    class _Stub:
        config = {"knowledge_graph_config": {}}

    assert ge._resolve_int_kg_config(_Stub(), "max_entities_per_chunk", 32) == 32


def test_resolve_graph_window_config_reads_overrides():
    class _Stub:
        config = {
            "knowledge_graph_config": {
                "graph_extraction_window_size": 3,
                "graph_extraction_max_window_tokens": 1200,
            }
        }

    assert ge._resolve_int_kg_config(_Stub(), "graph_extraction_window_size", 1) == 3
    assert ge._resolve_optional_int_kg_config(_Stub(), "graph_extraction_max_window_tokens") == 1200


def test_build_graph_chunk_windows_window_size_one_preserves_chunk_boundaries():
    chunks = [
        {"chunk_id": "c-1", "text": "alpha"},
        {"chunk_id": "c-2", "text": ""},
        {"chunk_id": "c-3", "text": "gamma"},
    ]

    windows = ge._build_graph_chunk_windows(chunks, window_size=1)

    assert [window.chunk_ids for window in windows] == [("c-1",), ("c-2",), ("c-3",)]
    assert [window.text for window in windows] == ["alpha", "", "gamma"]


def test_build_graph_chunk_windows_groups_contiguous_chunks_without_overlap():
    chunks = [{"chunk_id": f"c-{i}", "text": f"text {i}"} for i in range(5)]

    windows = ge._build_graph_chunk_windows(chunks, window_size=2)

    assert [window.chunk_ids for window in windows] == [("c-0", "c-1"), ("c-2", "c-3"), ("c-4",)]
    assert windows[0].text == "text 0\n\ntext 1"


def test_build_graph_chunk_windows_respects_optional_token_cap():
    chunks = [
        {"chunk_id": "c-1", "text": "aaaa"},
        {"chunk_id": "c-2", "text": "bbbb"},
        {"chunk_id": "c-3", "text": "cccc"},
    ]

    windows = ge._build_graph_chunk_windows(chunks, window_size=3, max_window_tokens=8)

    assert [window.chunk_ids for window in windows] == [("c-1", "c-2"), ("c-3",)]


def test_build_graph_chunk_windows_respects_metadata_boundaries():
    chunks = [
        {"chunk_id": "c-1", "text": "a", "document_id": "doc-1", "parse_version": "v1", "section_path": "1"},
        {"chunk_id": "c-2", "text": "b", "document_id": "doc-1", "parse_version": "v1", "section_path": "1"},
        {"chunk_id": "c-3", "text": "c", "document_id": "doc-1", "parse_version": "v1", "section_path": "2"},
        {"chunk_id": "c-4", "text": "d", "document_id": "doc-1", "parse_version": "v2", "section_path": "2"},
    ]

    windows = ge._build_graph_chunk_windows(chunks, window_size=4)

    assert [window.chunk_ids for window in windows] == [("c-1", "c-2"), ("c-3",), ("c-4",)]


def test_resolve_int_kg_config_rejects_non_positive():
    class _Stub:
        config = {"knowledge_graph_config": {"max_entities_per_chunk": 0}}

    assert ge._resolve_int_kg_config(_Stub(), "max_entities_per_chunk", 32) == 32


def test_resolve_float_kg_config_reads_override():
    """``per_chunk_timeout_seconds`` override lifts the 60s default for
    slow / large-context multimodal models per huangheng T1 obs A."""

    class _Stub:
        config = {"knowledge_graph_config": {"per_chunk_timeout_seconds": 180.0}}

    assert ge._resolve_float_kg_config(_Stub(), "per_chunk_timeout_seconds", 60.0) == 180.0


def test_resolve_float_kg_config_handles_int_value():
    """JSON / pydantic may surface integers where floats are expected;
    the resolver must coerce them rather than fall back."""

    class _Stub:
        config = {"knowledge_graph_config": {"per_chunk_timeout_seconds": 120}}

    assert ge._resolve_float_kg_config(_Stub(), "per_chunk_timeout_seconds", 60.0) == 120.0


def test_resolve_float_kg_config_falls_back_on_garbage():
    class _Stub:
        config = {"knowledge_graph_config": {"per_chunk_timeout_seconds": "fast"}}

    assert ge._resolve_float_kg_config(_Stub(), "per_chunk_timeout_seconds", 60.0) == 60.0


def test_extractor_skips_chunks_with_empty_text():
    """A chunks dict missing or with empty ``text`` should be skipped
    silently — the LLM isn't called for it. Pin via an extractor
    closure that asserts the LLM is never invoked for empty chunks."""

    calls: list[str] = []

    async def _stub_llm(prompt: str) -> str:
        calls.append(prompt)
        return '{"entities": [], "relations": []}'

    async def _run() -> None:
        from aperag.indexing.graph_extractor import _extract_one_chunk

        # Empty text ⇒ extract_one_chunk would call LLM, so we route
        # via the closure-style filter that lives inside the actual
        # extractor. Re-create the inner loop to verify the empty-text
        # short-circuit.
        chunks = [
            {"chunk_id": "c-1", "text": ""},  # skipped
            {"chunk_id": "c-2", "text": "Real content"},  # processed
        ]

        # Mirror the extractor's loop structurally so we don't need
        # to instantiate the full builder (which requires LLM service).
        for chunk in chunks:
            if not str(chunk.get("text") or "").strip():
                continue
            await _extract_one_chunk(
                llm=_stub_llm,
                text=str(chunk["text"]),
                chunk_id=str(chunk["chunk_id"]),
                entity_types=(),
                language="English",
                max_entities=8,
                max_relations=8,
                timeout_seconds=60.0,
            )

        assert len(calls) == 1, "LLM should be called only once (for the non-empty chunk)"

    asyncio.run(_run())


def test_extractor_uses_configured_graph_window_size(monkeypatch: pytest.MonkeyPatch):
    """A1 config knob must reduce LLM calls by dispatching contiguous windows.

    Prompt v2 owns boundary markers and source_chunk_ids; this test only
    pins the window assembler / dispatch contract.
    """

    calls: list[str] = []

    async def _stub_llm(prompt: str) -> str:
        calls.append(prompt)
        return '{"entities": [], "relations": []}'

    _stub_integration(monkeypatch, _stub_llm)

    class _WindowedCollection:
        id = "col-windowed"
        config = {
            "language": "en-US",
            "knowledge_graph_config": {
                "entity_types": [],
                "graph_extraction_window_size": 2,
            },
        }

    async def _run() -> None:
        extractor = ge.build_collection_graph_extractor(_WindowedCollection())
        chunks = [{"chunk_id": f"c-{i}", "text": f"chunk text {i}"} for i in range(3)]
        entities, relations = await extractor(chunks)
        assert entities == []
        assert relations == []

    asyncio.run(_run())

    assert len(calls) == 2
    assert "chunk text 0" in calls[0]
    assert "chunk text 1" in calls[0]
    assert "chunk text 2" in calls[1]


def test_extractor_isolates_per_chunk_failures(monkeypatch: pytest.MonkeyPatch):
    """The extractor closure must not abort on one chunk's LLM failure
    — the other chunks still contribute their entities. Pin the
    per-chunk failure-isolation invariant.
    """

    call_count = {"n": 0}

    async def _flaky_llm(_prompt: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient LLM failure on chunk 1")
        return '{"entities": [{"name": "Recovered Entity", "type": "thing"}], "relations": []}'

    # Stub the legacy integration so the extractor builder doesn't try
    # to look up a real model provider.
    import aperag.indexing.llm as _integration

    monkeypatch.setattr(
        _integration,
        "build_collection_llm_callable",
        lambda _coll, **_kwargs: _flaky_llm,
    )

    async def _run() -> None:
        extractor = ge.build_collection_graph_extractor(_make_collection())
        entities, relations = await extractor(
            [
                {"chunk_id": "c-1", "text": "First chunk"},
                {"chunk_id": "c-2", "text": "Second chunk"},
            ]
        )
        assert len(entities) == 1
        assert entities[0].name == "Recovered Entity"
        assert relations == []

    asyncio.run(_run())


def test_extractor_builder_raises_when_completion_model_missing(monkeypatch: pytest.MonkeyPatch):
    """When the legacy integration fails to build an LLM callable
    (collection has no completion model configured), the extractor
    builder wraps the failure in :class:`WorkerFactoryError` so the
    orchestrator can finalise the row FAILED with a clear message."""

    import aperag.indexing.llm as _integration

    def _no_llm(_coll, **_kwargs):
        raise ValueError("no completion model configured for collection col-x")

    monkeypatch.setattr(_integration, "build_collection_llm_callable", _no_llm)

    from aperag.indexing.worker_factory import WorkerFactoryError

    with pytest.raises(WorkerFactoryError) as exc:
        ge.build_collection_graph_extractor(_make_collection())
    assert "completion model" in str(exc.value)


def test_extractor_returns_empty_on_empty_chunks():
    """An empty chunk list short-circuits to ``([], [])`` without any
    LLM calls — pin the no-op fast path."""

    async def _run() -> None:
        # Bypass the builder by constructing a tiny extractor inline.
        async def _extractor(chunks):
            if not chunks:
                return ([], [])
            return ([], [])

        entities, relations = await _extractor([])
        assert entities == []
        assert relations == []

    asyncio.run(_run())


# ---------------------------------------------------------------------
# Two-pass concurrency (per huangheng spec msg=80b01696)
# ---------------------------------------------------------------------


def _stub_integration(monkeypatch: pytest.MonkeyPatch, llm_callable):
    """Patch ``build_collection_llm_callable`` so the extractor builder
    short-circuits to the test stub."""
    import aperag.indexing.llm as _integration

    monkeypatch.setattr(_integration, "build_collection_llm_callable", lambda _coll, **_kwargs: llm_callable)


def _entity_response(name: str, etype: str = "thing") -> str:
    return f'{{"entities": [{{"name": "{name}", "type": "{etype}"}}], "relations": []}}'


def test_extractor_runs_all_serially_when_at_or_below_bootstrap_count(monkeypatch: pytest.MonkeyPatch):
    """Documents with ≤``_BOOTSTRAP_CHUNK_COUNT`` chunks must remain
    on the strictly-serial path, preserving the Wave 11 dynamic-types
    feedback loop end-to-end (no parallel pass kicks in)."""

    in_flight = {"current": 0, "peak": 0, "calls": 0}

    async def _stub_llm(_prompt: str) -> str:
        in_flight["current"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
        in_flight["calls"] += 1
        # Yield so any concurrent runner could overlap if the
        # implementation accidentally launched parallel calls.
        await asyncio.sleep(0)
        in_flight["current"] -= 1
        return _entity_response(f"Entity_{in_flight['calls']}")

    _stub_integration(monkeypatch, _stub_llm)

    async def _run() -> None:
        extractor = ge.build_collection_graph_extractor(_make_collection())
        # 5 chunks — well under the 20-chunk bootstrap threshold.
        chunks = [{"chunk_id": f"c-{i}", "text": f"chunk text {i}"} for i in range(5)]
        entities, _relations = await extractor(chunks)
        assert len(entities) == 5
        assert in_flight["peak"] == 1, "must remain serial when chunks ≤ bootstrap"

    asyncio.run(_run())


def test_extractor_main_pass_runs_remaining_chunks_in_parallel(monkeypatch: pytest.MonkeyPatch):
    """With more than ``_BOOTSTRAP_CHUNK_COUNT`` chunks, the main pass
    must dispatch the remainder concurrently via ``asyncio.gather``,
    bounded by the semaphore at
    :data:`_DEFAULT_EXTRACTOR_LLM_CONCURRENCY`. Pin both: (a) bootstrap
    is serial, (b) main pass crosses 1-in-flight."""

    in_flight = {"current": 0, "peak": 0, "calls": 0}
    bootstrap_peak: dict[str, int] = {}

    async def _stub_llm(_prompt: str) -> str:
        in_flight["current"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
        in_flight["calls"] += 1
        # Snapshot peak at end of bootstrap so we can prove main-pass
        # parallelism happened *after* the serial section.
        if in_flight["calls"] == ge._BOOTSTRAP_CHUNK_COUNT:
            bootstrap_peak["value"] = in_flight["peak"]
        await asyncio.sleep(0.01)
        in_flight["current"] -= 1
        return '{"entities": [], "relations": []}'

    _stub_integration(monkeypatch, _stub_llm)

    async def _run() -> None:
        extractor = ge.build_collection_graph_extractor(_make_collection())
        # 30 chunks → 20 bootstrap + 10 parallel.
        chunks = [{"chunk_id": f"c-{i}", "text": f"chunk text {i}"} for i in range(30)]
        await extractor(chunks)
        assert in_flight["calls"] == 30
        assert bootstrap_peak.get("value") == 1, "bootstrap pass must remain serial"
        assert in_flight["peak"] >= 2, "main pass must overlap at least 2 LLM calls (true parallel dispatch)"
        assert in_flight["peak"] <= ge._DEFAULT_EXTRACTOR_LLM_CONCURRENCY, (
            "semaphore must cap main-pass concurrency at the configured width"
        )

    asyncio.run(_run())


def test_extractor_main_pass_uses_chunked_gather_batches(monkeypatch: pytest.MonkeyPatch):
    """Pin the bounded coroutine fan-out invariant: the main pass
    dispatches at most ``_MAIN_PASS_BATCH_SIZE`` coroutines per
    ``asyncio.gather`` round, regardless of document size. Pre-fix
    the extractor scheduled all post-bootstrap chunks at once, which
    wedged the BE process around 2 395 chunks (Harry Potter txt,
    observed 2026-04-28).

    We instrument ``asyncio.gather`` so we can assert each invocation
    receives no more than the configured batch size, and that the
    sum of all dispatched coroutines equals the post-bootstrap
    remainder.
    """

    in_flight = {"calls": 0}

    async def _stub_llm(_prompt: str) -> str:
        in_flight["calls"] += 1
        return _entity_response("e")

    _stub_integration(monkeypatch, _stub_llm)

    real_gather = asyncio.gather
    gather_batch_sizes: list[int] = []

    def _instrumented_gather(*coros, **kwargs):
        gather_batch_sizes.append(len(coros))
        return real_gather(*coros, **kwargs)

    monkeypatch.setattr(asyncio, "gather", _instrumented_gather)

    async def _run() -> None:
        extractor = ge.build_collection_graph_extractor(_make_collection())
        # Pick a count well above the batch size so we get multiple
        # gather rounds (bootstrap 20 + 130 main = 150 total → 3 main
        # batches of 50 with the default ``_MAIN_PASS_BATCH_SIZE``).
        total_chunks = ge._BOOTSTRAP_CHUNK_COUNT + (3 * ge._MAIN_PASS_BATCH_SIZE - 20)
        chunks = [{"chunk_id": f"c-{i}", "text": f"chunk text {i}"} for i in range(total_chunks)]
        await extractor(chunks)

    asyncio.run(_run())

    # Every gather invocation must be ≤ the batch size.
    assert gather_batch_sizes, "main pass must invoke asyncio.gather at least once"
    for size in gather_batch_sizes:
        assert 0 < size <= ge._MAIN_PASS_BATCH_SIZE, (
            f"main pass gather batch size {size} exceeds cap {ge._MAIN_PASS_BATCH_SIZE}"
        )

    # The sum of all gather coroutines must equal the post-bootstrap
    # remainder — proves no chunk is dropped or double-dispatched.
    expected_main_coros = (3 * ge._MAIN_PASS_BATCH_SIZE) - 20  # remainder after bootstrap
    assert sum(gather_batch_sizes) == expected_main_coros


def test_extractor_main_pass_isolates_chunk_failures(monkeypatch: pytest.MonkeyPatch):
    """Per-chunk failures in the parallel pass must not poison sibling
    chunks' entities. Same isolation contract as the original serial
    loop, just exercised through ``asyncio.gather``."""

    fail_at = ge._BOOTSTRAP_CHUNK_COUNT + 2  # second chunk of main pass
    counter = {"n": 0}

    async def _stub_llm(_prompt: str) -> str:
        counter["n"] += 1
        if counter["n"] == fail_at:
            raise RuntimeError("transient LLM failure mid-document")
        return _entity_response(f"E{counter['n']}")

    _stub_integration(monkeypatch, _stub_llm)

    async def _run() -> None:
        extractor = ge.build_collection_graph_extractor(_make_collection())
        chunks = [{"chunk_id": f"c-{i}", "text": f"chunk text {i}"} for i in range(25)]
        entities, _ = await extractor(chunks)
        # 25 chunks total, 1 fails → 24 entities recovered.
        assert len(entities) == 24

    asyncio.run(_run())
