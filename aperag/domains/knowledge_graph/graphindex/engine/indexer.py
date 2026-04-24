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

"""The index-a-document pipeline: chunk → extract → store.

Three stages, chained by the module's single public function
``index_document``:

1. ``chunking.chunk_document`` turns raw text into ``Chunk`` DTOs.
2. ``extraction.extract_from_chunk`` makes one LLM call per chunk,
   producing ``(entities, relations)`` DTOs.
3. ``GraphStore.upsert_*`` persists the chunks, entities, and relations
   atomically (within each batch — see the note below).

**Batching & atomicity**: we insert all chunks first, then the extracted
entities, then the relations. If extraction crashes mid-way through a
document, partial chunks may exist in the store — deliberate, so a retry
can resume without re-running every LLM call. Rebuild idempotency is
handled one level up in ``GraphIndexService.index_document`` (which runs
``delete_document_rows`` before invoking this engine), so the engine
itself does not deduplicate against existing rows.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Iterable, Optional

from aperag.domains.knowledge_graph.graphindex.config import GraphIndexConfig
from aperag.domains.knowledge_graph.graphindex.dto import Entity, IndexDocumentResult, Relation
from aperag.domains.knowledge_graph.graphindex.engine.chunking import chunk_document
from aperag.domains.knowledge_graph.graphindex.engine.extraction import extract_from_chunk
from aperag.domains.knowledge_graph.graphindex.storage.base import GraphStore

logger = logging.getLogger(__name__)

LLMCall = Callable[[str], Awaitable[str]]
Tokenize = Callable[[str], Iterable[str]]


async def index_document(
    *,
    store: GraphStore,
    llm: LLMCall,
    config: GraphIndexConfig,
    collection_id: str,
    doc_id: str,
    content: str,
    file_path: str = "",
    tokenize: Optional[Tokenize] = None,
) -> IndexDocumentResult:
    """End-to-end: take raw document text, produce graph rows.

    Empty content returns a zero-filled result without touching storage.
    ``tokenize`` is optional — production leaves it ``None`` to pick up
    the tiktoken default; tests pass ``str.split`` to keep the pipeline
    offline.
    """
    chunks = chunk_document(
        collection_id=collection_id,
        doc_id=doc_id,
        content=content,
        file_path=file_path,
        chunk_token_size=config.chunk_token_size,
        chunk_overlap_token_size=config.chunk_overlap_token_size,
        tokenize=tokenize,
    )
    if not chunks:
        return IndexDocumentResult(
            doc_id=doc_id,
            chunks_created=0,
            entities_extracted=0,
            relations_extracted=0,
        )

    # Persist chunks first so source_chunk_ids on entities/relations
    # point at rows that already exist (not strictly required for
    # correctness — we don't FK-constraint chunk_ids — but helpful for
    # anyone debugging a partial failure).
    await store.upsert_chunks(collection_id=collection_id, chunks=chunks)

    # Extraction pipeline. We issue chunks in parallel up to
    # ``max_chunks_per_batch`` to amortise LLM request overhead, but
    # each chunk is its own LLM call (prompt template expects a single
    # chunk). This gives us a neat failure unit: one chunk's
    # malformed response doesn't poison the rest.
    all_entities: list[Entity] = []
    all_relations: list[Relation] = []
    batch_size = max(1, int(config.max_chunks_per_batch))

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        batch_results = await asyncio.gather(
            *(extract_from_chunk(chunk=c, config=config, llm=llm) for c in batch),
            return_exceptions=True,
        )
        for chunk, result in zip(batch, batch_results):
            if isinstance(result, BaseException):
                logger.warning(
                    "graphindex.indexer: extraction failed on chunk %s (%s: %s); skipping this chunk",
                    chunk.chunk_id,
                    type(result).__name__,
                    result,
                )
                continue
            entities, relations = result
            all_entities.extend(entities)
            all_relations.extend(relations)

    if all_entities:
        await store.upsert_entities(collection_id=collection_id, entities=all_entities)
    if all_relations:
        await store.upsert_relations(collection_id=collection_id, relations=all_relations)

    logger.info(
        "graphindex.indexer: doc %s → %d chunks, %d entities, %d relations",
        doc_id,
        len(chunks),
        len(all_entities),
        len(all_relations),
    )
    return IndexDocumentResult(
        doc_id=doc_id,
        chunks_created=len(chunks),
        entities_extracted=len(all_entities),
        relations_extracted=len(all_relations),
    )


__all__ = ["index_document"]
