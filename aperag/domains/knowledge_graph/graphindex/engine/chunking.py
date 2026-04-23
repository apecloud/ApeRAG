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

"""Document chunking for graphindex v2.

Simple token-window chunker with overlap. No fancy boundary detection,
no semantic splitting — extraction quality does not benefit from it at
current LLM context sizes, and more logic = more failure modes.

The tokenizer is pluggable (``tokenize`` callable) so tests can pass
``str.split`` for deterministic expectations without pulling in tiktoken.
Production wires ``aperag.utils.tokenizer.get_default_tokenizer``.
"""

from __future__ import annotations

import uuid
from typing import Callable, Iterable, List

from aperag.domains.knowledge_graph.graphindex.dto import Chunk


def _default_tokenize(text: str) -> list[str]:
    """Lazy default so unit tests don't have to import tiktoken.

    ``get_default_tokenizer()`` returns the tiktoken ``encode`` callable
    directly (``Callable[[str], List[int]]``), not an Encoding object.
    Stringify the token ids so the chunker stays tokenizer-agnostic at
    the type level.
    """
    from aperag.utils.tokenizer import get_default_tokenizer

    encode = get_default_tokenizer()
    return [str(tok) for tok in encode(text)]


def chunk_document(
    *,
    collection_id: str,
    doc_id: str,
    content: str,
    file_path: str = "",
    chunk_token_size: int = 1200,
    chunk_overlap_token_size: int = 100,
    tokenize: Callable[[str], Iterable[str]] | None = None,
) -> List[Chunk]:
    """Split ``content`` into overlapping chunks.

    Contract:
    * Stable ordering: ``order_in_doc`` starts at 0 and monotonically
      increases. Callers that stream chunks (e.g. indexer) rely on this.
    * Deterministic chunk **count** given the same inputs (important for
      incremental re-ingest).
    * Chunk **ids** are UUID4, intentionally non-deterministic — repeat
      runs for the same document produce different chunk ids. That is
      safe because ``GraphIndexService.index_document`` runs
      ``delete_document_rows`` before extraction, so stale chunks from
      a previous run never survive alongside the new ones.

    Returns an empty list when ``content`` is empty / all whitespace.
    The caller is expected to handle "no chunks" as a valid outcome
    (images-only documents, etc.).
    """
    if not content or not content.strip():
        return []
    if chunk_overlap_token_size >= chunk_token_size:
        raise ValueError("chunk_overlap_token_size must be < chunk_token_size")

    tok = tokenize or _default_tokenize
    tokens = list(tok(content))
    if not tokens:
        return []

    # Walk the token stream in windows of ``chunk_token_size`` with a
    # ``chunk_overlap_token_size`` overlap. The loop terminates when the
    # next window would be strictly smaller than the overlap — otherwise
    # we'd emit infinitely many one-token chunks on the tail.
    stride = chunk_token_size - chunk_overlap_token_size
    chunks: List[Chunk] = []
    order = 0
    # Reconstitute text from token boundaries using character offsets; for
    # the default (tiktoken-encoded) tokenizer we cannot directly join the
    # string-ified token ids, so we fall back to slicing the original
    # content by token index via the tokenizer's decode when available.
    #
    # For simplicity + zero additional dependency, we split `content`
    # proportionally by token count — exact token-level slicing is a
    # nice-to-have but not required for extraction accuracy at LLM
    # contexts >= 2k tokens.
    total_tokens = len(tokens)
    total_chars = len(content)
    if total_tokens == 0:
        return []

    start = 0
    while start < total_tokens:
        end = min(start + chunk_token_size, total_tokens)
        # Proportional character slice. Good enough since we never need
        # to reconstruct exact tokens from the chunk text — the LLM
        # re-tokenizes on its end.
        char_start = (start * total_chars) // total_tokens
        char_end = (end * total_chars) // total_tokens
        chunk_text = content[char_start:char_end]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    collection_id=collection_id,
                    order_in_doc=order,
                    text=chunk_text,
                    file_path=file_path,
                )
            )
            order += 1
        if end == total_tokens:
            break
        start += stride

    return chunks


__all__ = ["chunk_document"]
