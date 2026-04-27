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

"""Fulltext writer/retrieval field-shape roundtrip — celery T3.1
post-pass-9 closing fix.

Per architect msg=69df0779: the original Wave 1 ``FulltextModality.
sync`` wrote chunk records with the field name ``text``, while the
production retrieval path
(``aperag/domains/retrieval/pipeline.py:_fulltext_search``) queries
``content`` / ``title`` / ``collection_id`` (the schema the legacy
``aperag/domains/indexing/fulltext_index.py`` wrote pre Wave 3
hard-cut). Result: every fulltext search after Wave 3 returned 0
hits silently — the chat_collection_flow business test exited 1 on
``jq -e items.length > 0``.

This test pins the post-fix invariant: the documents the
:class:`FulltextModality` writes carry exactly the field names the
retrieval pipeline filters / matches on.

Specifically:

* ``content`` — the field ``_fulltext_search`` ``match``-queries.
* ``collection_id`` — the field ``_fulltext_search`` ``term``-filters
  on. Constructor must thread the collection id through.
* ``chunk_id`` / ``document_id`` / ``parse_version`` —
  invariants the cleanup / cutover paths depend on; kept by the
  shape change.
* ``text`` — kept as alias of ``content`` so the existing in-memory
  contract tests + Wave 1 simulator tests do not regress.

A regression that drops ``content`` or ``collection_id`` from the
write shape trips this test on the first assertion.
"""

from __future__ import annotations

from aperag.indexing import (
    FulltextModality,
    InMemoryFulltextBackend,
    InMemoryObjectStore,
    parse_document,
)

COLLECTION_ID = "col-fulltext-roundtrip"
DOCUMENT_ID = "doc-fulltext-roundtrip"

SOURCE_MARKDOWN = b"""# Fulltext Roundtrip

The retrieval pipeline searches the ``content`` field; the writer
must populate it (alongside the canonical ``collection_id`` filter)
so end-to-end search returns hits instead of zero.
"""


def test_fulltext_sync_writes_content_and_collection_id_fields():
    """Run :func:`parse_document` then :meth:`FulltextModality.sync`
    against the in-memory fulltext backend and assert every indexed
    chunk carries the field names ``_fulltext_search`` queries on.
    """

    import asyncio

    async def _run() -> None:
        store = InMemoryObjectStore()
        parsed = parse_document(
            store=store,
            collection_id=COLLECTION_ID,
            document_id=DOCUMENT_ID,
            source_bytes=SOURCE_MARKDOWN,
        )

        backend = InMemoryFulltextBackend()
        modality = FulltextModality(
            backend=backend,
            store=store,
            collection_id=COLLECTION_ID,
        )
        await modality.sync(
            document_id=DOCUMENT_ID,
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )

        documents = backend.documents_for_document(DOCUMENT_ID, parsed.parse_version)
        assert documents, "fulltext sync wrote zero documents — parse_document or sync regressed"
        for doc in documents:
            # The two field names ``_fulltext_search`` actually
            # depends on. Without these the production search
            # returns 0 hits (silent broken).
            assert "content" in doc, f"missing 'content' field in {doc!r}"
            assert doc["content"], "'content' field is empty"
            assert doc.get("collection_id") == COLLECTION_ID, (
                f"collection_id field missing or wrong; got {doc.get('collection_id')!r}"
            )
            # Existing alias kept so legacy in-memory tests do not
            # regress.
            assert doc.get("text") == doc["content"]
            # Invariants the cleanup / cutover paths still need.
            assert doc.get("document_id") == DOCUMENT_ID
            assert doc.get("parse_version") == parsed.parse_version
            assert doc.get("chunk_id")

    asyncio.run(_run())


def test_fulltext_sync_omits_collection_id_when_constructor_omitted():
    """Backward-compat: tests / fixtures that build
    :class:`FulltextModality` without the ``collection_id`` argument
    (e.g. the existing T1.3 contract tests) keep working — the
    written documents simply do not carry a ``collection_id`` field.
    The production worker_factory always passes the id.
    """

    import asyncio

    async def _run() -> None:
        store = InMemoryObjectStore()
        parsed = parse_document(
            store=store,
            collection_id=COLLECTION_ID,
            document_id=DOCUMENT_ID,
            source_bytes=SOURCE_MARKDOWN,
        )
        backend = InMemoryFulltextBackend()
        modality = FulltextModality(backend=backend, store=store)
        await modality.sync(
            document_id=DOCUMENT_ID,
            parse_version=parsed.parse_version,
            derived_artifact_path=parsed.chunks_path,
        )

        documents = backend.documents_for_document(DOCUMENT_ID, parsed.parse_version)
        assert documents
        for doc in documents:
            assert "content" in doc and doc["content"]
            # No collection_id without explicit constructor arg.
            assert "collection_id" not in doc

    asyncio.run(_run())
