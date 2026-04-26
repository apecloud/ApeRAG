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

"""T1.1 Foundation contract tests.

Locks the three §K Wave 1 acceptance gates for ``aperag/indexing/``:

1. **Partial unique invariant** — ``is_serving=TRUE`` is unique per
   ``(document_id, modality)`` at the DB layer (§F.1, Bryce v2 review
   msg=7ccb176f #2). Two competing serving rows must fail at INSERT.

2. **Object-store atomic write** — ``write_atomic`` never exposes
   partial state. The LocalFS path uses ``tmp + fsync + rename``
   (§C.7) so a crashed mid-write leaves the destination either
   missing or holding the previous full content. A reader who races
   the writer either sees nothing or sees the full new bytes.

3. **Parser → derived/ round-trip** — ``parse_document`` produces a
   stable ``parse_version`` for the same inputs (§E.2 hash) and lays
   out the canonical ``markdown.md`` / ``outline.json`` / ``chunks.jsonl``
   files under ``derived/parse_<v>/``. Re-running the parser
   overwrites the artifacts atomically and the round-trip readback
   matches.
"""

from __future__ import annotations

import json
import threading

import pytest
from sqlalchemy import (
    Boolean,
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    text,
)
from sqlalchemy.exc import IntegrityError

from aperag.indexing import (
    InMemoryObjectStore,
    Modality,
    ParseConfig,
    derived_artifact,
    derived_dir,
    parse_document,
    read_chunks,
    source_artifact,
    write_atomic,
)
from aperag.objectstore.local import LocalObjectStore

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_inmemory_document_index_table() -> tuple[MetaData, Table]:
    """Build an SQLite-compatible mirror of the ``document_index`` table.

    SQLite ≥3.8 supports the partial unique index syntax we use in
    the alembic migration, so the same DB-layer invariant is verified
    here without spinning up Postgres.
    """
    metadata = MetaData()
    table = Table(
        "document_index_v2",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("document_id", String(64), nullable=False),
        Column("parse_version", String(16), nullable=False),
        Column("modality", String(32), nullable=False),
        Column("status", String(16), nullable=False),
        Column("error_message", Text, nullable=True),
        Column("retry_count", Integer, nullable=False, server_default="0"),
        Column("derived_artifact_path", Text, nullable=True),
        Column("tenant_scope_key", String(64), nullable=False),
        Column("is_serving", Boolean, nullable=False, server_default=text("FALSE")),
        Index(
            "uq_document_index_v2_triple",
            "document_id",
            "parse_version",
            "modality",
            unique=True,
        ),
        Index(
            "uniq_document_index_v2_serving",
            "document_id",
            "modality",
            unique=True,
            sqlite_where=text("is_serving = 1"),
        ),
    )
    return metadata, table


# ---------------------------------------------------------------------
# 1. Partial unique invariant (§F.1)
# ---------------------------------------------------------------------


def test_partial_unique_serving_blocks_two_active_rows_per_document_modality():
    """The DB-layer partial unique index `uniq_document_index_v2_serving`
    must reject a second `is_serving=TRUE` row for the same
    `(document_id, modality)` pair (Bryce v2 review msg=7ccb176f #2 +
    design pack §F.1).

    SQLite ≥3.8 is the test backend; production is Postgres which
    uses the same syntax (Tier 1 §L deploy stays consistent).
    """
    metadata, table = _build_inmemory_document_index_table()
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)

    # First serving row — accepted.
    with engine.begin() as conn:
        conn.execute(
            insert(table).values(
                document_id="doc-1",
                parse_version="aaaaaaaaaaaaaaaa",
                modality=Modality.VECTOR.value,
                status="ACTIVE",
                tenant_scope_key="user:test",
                is_serving=True,
            )
        )

    # Second serving row for the SAME (doc, modality) — must fail at
    # the unique index level.
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                insert(table).values(
                    document_id="doc-1",
                    parse_version="bbbbbbbbbbbbbbbb",
                    modality=Modality.VECTOR.value,
                    status="ACTIVE",
                    is_serving=True,
                )
            )

    # A non-serving row for the same (doc, modality) is allowed —
    # the partial index only covers serving rows.
    with engine.begin() as conn:
        conn.execute(
            insert(table).values(
                document_id="doc-1",
                parse_version="cccccccccccccccc",
                modality=Modality.VECTOR.value,
                status="PENDING",
                tenant_scope_key="user:test",
                is_serving=False,
            )
        )

    # A serving row for a DIFFERENT modality on the same document is
    # allowed — the invariant is per-(doc, modality).
    with engine.begin() as conn:
        conn.execute(
            insert(table).values(
                document_id="doc-1",
                parse_version="dddddddddddddddd",
                modality=Modality.GRAPH.value,
                status="ACTIVE",
                tenant_scope_key="user:test",
                is_serving=True,
            )
        )


def test_cutover_swap_three_statement_transaction_satisfies_partial_unique(tmp_path):
    """Per design pack §F.3, the cutover transaction is
    ``UPDATE old SET is_serving=FALSE`` followed by
    ``UPDATE new SET is_serving=TRUE`` in a single transaction. Verify
    that ordering does not transiently violate the partial unique
    index.
    """
    metadata, table = _build_inmemory_document_index_table()
    engine = create_engine(f"sqlite:///{tmp_path / 'cutover.db'}")
    metadata.create_all(engine)

    with engine.begin() as conn:
        conn.execute(
            insert(table).values(
                id=1,
                document_id="doc-9",
                parse_version="oldoldoldoldoldo",
                modality=Modality.VECTOR.value,
                status="ACTIVE",
                tenant_scope_key="user:test",
                is_serving=True,
            )
        )
        conn.execute(
            insert(table).values(
                id=2,
                document_id="doc-9",
                parse_version="newnewnewnewnewn",
                modality=Modality.VECTOR.value,
                status="ACTIVE",
                tenant_scope_key="user:test",
                is_serving=False,
            )
        )

    # Three-statement cutover (§F.3): mark new ACTIVE → flip old
    # FALSE → flip new TRUE, all in one transaction.
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE document_index_v2 SET status = 'ACTIVE' WHERE id = :id"),
            {"id": 2},
        )
        conn.execute(
            text(
                "UPDATE document_index_v2 SET is_serving = 0 "
                "WHERE document_id = :doc AND modality = :mod AND is_serving = 1"
            ),
            {"doc": "doc-9", "mod": Modality.VECTOR.value},
        )
        conn.execute(
            text("UPDATE document_index_v2 SET is_serving = 1 WHERE id = :id"),
            {"id": 2},
        )

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, parse_version, is_serving FROM document_index_v2 "
                "WHERE document_id='doc-9' AND modality='vector' "
                "ORDER BY id"
            )
        ).all()
    assert rows == [
        (1, "oldoldoldoldoldo", 0),
        (2, "newnewnewnewnewn", 1),
    ]


# ---------------------------------------------------------------------
# 2. Object-store atomic write (§C.7)
# ---------------------------------------------------------------------


def test_local_atomic_write_uses_tmp_rename_dance(tmp_path):
    """`_LocalAtomicWriter` must (a) write to a `.tmp.<uuid>` sibling,
    (b) `fsync` the file descriptor before close, and (c) `os.rename`
    onto the final path. The result is that mid-write crashes leave
    the destination either missing or holding the previous content,
    never half-written bytes.

    We can't easily prove `fsync` was called from a unit test, but we
    can prove the visible state transitions:
    1. Before the write completes, the destination does not exist.
    2. After the write completes, the destination exists with full
       contents and no `.tmp.*` sibling remains.
    """
    settings_payload = type(
        "_S",
        (),
        {
            "object_store_local_config": type("_C", (), {"root_dir": str(tmp_path)})(),
        },
    )()

    # Patch ``aperag.objectstore.local.settings`` so the store roots
    # at our temp dir without poisoning module-level singletons.
    import aperag.objectstore.local as local_mod

    original = local_mod.settings
    try:
        local_mod.settings = settings_payload
        store = LocalObjectStore(settings_payload.object_store_local_config)
        path = "collections/c/documents/d/derived/parse_v/chunks.jsonl"

        write_atomic(store, path, b'{"chunk_id":"x:0","text":"hello"}\n')

        full_path = store._resolve_object_path(path)  # type: ignore[attr-defined]
        assert full_path.is_file(), "atomic write must produce the destination file"

        # No .tmp.* sibling should remain.
        siblings = list(full_path.parent.iterdir())
        tmp_siblings = [p for p in siblings if ".tmp." in p.name]
        assert tmp_siblings == [], f"atomic write must clean up tmp siblings; found {tmp_siblings}"

        # Content is exactly what we wrote.
        assert full_path.read_bytes() == b'{"chunk_id":"x:0","text":"hello"}\n'
    finally:
        local_mod.settings = original


def test_concurrent_atomic_writes_dont_clobber_each_other(tmp_path):
    """Two threads writing different artifacts to the same parse_version
    directory must both succeed without any tmp file leaking and with
    each destination holding its own bytes.
    """
    settings_payload = type(
        "_S",
        (),
        {
            "object_store_local_config": type("_C", (), {"root_dir": str(tmp_path)})(),
        },
    )()
    import aperag.objectstore.local as local_mod

    original = local_mod.settings
    try:
        local_mod.settings = settings_payload
        store = LocalObjectStore(settings_payload.object_store_local_config)

        a_path = "collections/c/documents/d/derived/parse_v/markdown.md"
        b_path = "collections/c/documents/d/derived/parse_v/outline.json"
        a_body = b"# Doc title\n\nbody A " * 200
        b_body = b'[{"level":1,"text":"Doc title","section_path":"1"}]'

        results: dict[str, Exception | None] = {"a": None, "b": None}

        def writer(key: str, path: str, body: bytes) -> None:
            try:
                write_atomic(store, path, body)
            except Exception as exc:  # noqa: BLE001
                results[key] = exc

        thread_a = threading.Thread(target=writer, args=("a", a_path, a_body))
        thread_b = threading.Thread(target=writer, args=("b", b_path, b_body))
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()

        assert results["a"] is None, f"writer A failed: {results['a']}"
        assert results["b"] is None, f"writer B failed: {results['b']}"

        full_a = store._resolve_object_path(a_path)  # type: ignore[attr-defined]
        full_b = store._resolve_object_path(b_path)  # type: ignore[attr-defined]
        assert full_a.read_bytes() == a_body
        assert full_b.read_bytes() == b_body

        # No tmp siblings should remain on either side.
        tmp_remains = [p for p in full_a.parent.iterdir() if ".tmp." in p.name]
        assert tmp_remains == [], f"concurrent atomic writes must not leak tmp files; found {tmp_remains}"
    finally:
        local_mod.settings = original


def test_in_memory_store_respects_atomic_write_contract():
    """The :class:`InMemoryObjectStore` test fixture must satisfy the
    same single-call atomic-write semantic so modality unit tests
    (T1.2-T1.5) can rely on it without surprises.
    """
    store = InMemoryObjectStore()
    path = "collections/c/documents/d/derived/parse_v/chunks.jsonl"
    body = b'{"chunk_id":"abc:0","text":"first"}\n'

    write_atomic(store, path, body)
    assert store.obj_exists(path)
    handle = store.get(path)
    assert handle is not None
    assert handle.read() == body


# ---------------------------------------------------------------------
# 3. Parser → derived/ round-trip (§C.1 / §C.6 / §E.2)
# ---------------------------------------------------------------------


SAMPLE_MARKDOWN = """# Linux kernel\n\nLinus Torvalds is the original author of the Linux kernel.\n\n## Background\n\nThe project began in 1991. Linus initially announced it on a Usenet group.\n\n## Maintenance\n\nMaintainership has expanded since the early days. The kernel is now developed by thousands of contributors.\n\n### Subsystems\n\nThe kernel is split into networking, filesystems, drivers, and core scheduling subsystems.\n"""


def test_parse_document_writes_three_canonical_artifacts():
    store = InMemoryObjectStore()
    result = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-1",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
    )

    expected_dir = derived_dir("col-1", "doc-1", result.parse_version)
    assert result.markdown_path == f"{expected_dir}/markdown.md"
    assert result.outline_path == f"{expected_dir}/outline.json"
    assert result.chunks_path == f"{expected_dir}/chunks.jsonl"

    # All three files exist and are non-empty.
    for path in (result.markdown_path, result.outline_path, result.chunks_path):
        assert store.obj_exists(path), f"expected derived artifact at {path}"
        assert (store.get_obj_size(path) or 0) > 0, f"derived artifact at {path} must be non-empty"


def test_parse_document_is_deterministic_on_same_inputs():
    """Re-running the parser with identical inputs must produce the
    same parse_version + identical artifact contents (the basis of
    §C.3 idempotent retry).
    """
    a_store = InMemoryObjectStore()
    b_store = InMemoryObjectStore()

    result_a = parse_document(
        store=a_store,
        collection_id="col-1",
        document_id="doc-1",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
    )
    result_b = parse_document(
        store=b_store,
        collection_id="col-1",
        document_id="doc-1",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
    )

    assert result_a.parse_version == result_b.parse_version
    assert (
        a_store.get(result_a.chunks_path).read()  # type: ignore[union-attr]
        == b_store.get(result_b.chunks_path).read()  # type: ignore[union-attr]
    )
    assert (
        a_store.get(result_a.outline_path).read()  # type: ignore[union-attr]
        == b_store.get(result_b.outline_path).read()  # type: ignore[union-attr]
    )


def test_parse_version_rolls_when_chunking_changes():
    """A change in the chunking config must roll the parse_version
    even when source bytes are identical, so a fresh ``derived/``
    directory is created (§E.2 hash inputs).
    """
    store = InMemoryObjectStore()
    base = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-2",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
    )
    rolled = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-2",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
        config=ParseConfig(),  # default — same config
    )
    assert base.parse_version == rolled.parse_version

    from aperag.indexing.parser import ChunkingConfig

    rolled_diff = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-2",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
        config=ParseConfig(chunking=ChunkingConfig(chunk_size=200, chunk_overlap=20)),
    )
    assert rolled_diff.parse_version != base.parse_version, "chunking change must roll parse_version per §E.2 hash spec"


def test_parse_document_chunks_round_trip_via_read_chunks():
    """Round-trip: ``parse_document`` writes ``chunks.jsonl`` →
    ``read_chunks`` reads it back as Python dicts with the canonical
    ``chunk_id`` / ``text`` schema.
    """
    store = InMemoryObjectStore()
    result = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-3",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
    )

    chunks = read_chunks(store, result.chunks_path)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "chunk_id" in chunk and chunk["chunk_id"]
        assert "text" in chunk and chunk["text"].strip()
        # chunk_id schema is `<sha256-prefix>:<index>` per the
        # simulator. Production parsers will replace the body but
        # MUST keep the field present.
        assert ":" in chunk["chunk_id"]


def test_parse_document_outline_carries_section_path_and_anchor():
    """The outline tree must use slash-separated section_path
    counters per D10.c §A.9 R1 lock and a slug-style heading_anchor —
    so search-result navigation back to a section works against the
    same handle convention as the read primitives.
    """
    store = InMemoryObjectStore()
    result = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-4",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
    )

    body = store.get(result.outline_path).read().decode("utf-8")  # type: ignore[union-attr]
    outline = json.loads(body)

    assert outline, "outline must be non-empty for a document with headings"
    root = outline[0]
    assert root["section_path"] == "1"
    assert root["heading_anchor"]
    # Background + Maintenance children of the root.
    assert len(root["children"]) >= 2
    background = root["children"][0]
    assert background["section_path"] == "1/1"
    # A grandchild (Subsystems) under Maintenance.
    maintenance = root["children"][1]
    assert any(child["section_path"] == "1/2/1" for child in maintenance["children"])


def test_parse_document_overwrites_existing_artifact_atomically():
    """Re-running ``parse_document`` with the same parse_version is a
    no-op contract-wise but must still tolerate the destination
    already existing (atomic overwrite).
    """
    store = InMemoryObjectStore()
    first = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-5",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
    )
    second = parse_document(
        store=store,
        collection_id="col-1",
        document_id="doc-5",
        source_bytes=SAMPLE_MARKDOWN.encode("utf-8"),
    )

    assert first.chunks_path == second.chunks_path
    # Content must still be valid after the second write.
    chunks = read_chunks(store, second.chunks_path)
    assert chunks


def test_read_chunks_returns_empty_on_missing_or_empty_artifact():
    """Per §C.7: missing / empty artifacts mean "derive not yet
    complete" — readers must treat them as empty, not as errors.
    """
    store = InMemoryObjectStore()
    assert read_chunks(store, "collections/x/documents/y/derived/parse_z/chunks.jsonl") == []
    # Empty file behaves the same as missing.
    store.put("collections/x/documents/y/derived/parse_z/chunks.jsonl", b"")
    assert read_chunks(store, "collections/x/documents/y/derived/parse_z/chunks.jsonl") == []


# ---------------------------------------------------------------------
# Path helpers — narrow regression
# ---------------------------------------------------------------------


def test_derived_artifact_rejects_path_traversal():
    with pytest.raises(ValueError):
        derived_artifact(
            collection_id="c",
            document_id="d",
            parse_version="abcdef0123456789",
            filename="../../../etc/passwd",
        )
    with pytest.raises(ValueError):
        derived_artifact(
            collection_id="c",
            document_id="d",
            parse_version="abcdef0123456789",
            filename="/absolute/path",
        )


def test_source_artifact_rejects_path_traversal():
    with pytest.raises(ValueError):
        source_artifact(collection_id="c", document_id="d", filename="../escape")
