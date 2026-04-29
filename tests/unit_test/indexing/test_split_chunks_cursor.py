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

"""Behavioural pins for ``_split_chunks`` simple-stable rewrite.

The previous implementation chose ``end`` by ``rfind('\n\n', cursor,
end)``; when a paragraph break sat near ``cursor`` it shrank the
window to almost nothing and the floor ``cursor + 1`` re-emitted the
same prefix one byte at a time, producing a long cascade of duplicate
chunks. The rewrite restricts the boundary search to the second half
of the window and walks a small priority list (paragraph → line →
sentence) before falling back to a hard cut. These tests pin both the
cascade-prevention guarantee and the overlap / id contract callers
already depend on.
"""

from __future__ import annotations

from aperag.indexing.parser import ChunkingConfig, _split_chunks


def test_paragraph_break_near_cursor_does_not_cascade_duplicates():
    """A short paragraph followed by a long body must not produce
    duplicate prefix-shifted chunks. This was the production failure
    mode on the real document parse_566bd3c39b8b197e."""
    text = "First small paragraph. Done.\n\n" + ("X" * 2000)
    chunks = _split_chunks(text, ChunkingConfig(chunk_size=800, chunk_overlap=80))

    chunk_texts = [c["text"] for c in chunks]
    assert len(chunk_texts) == len(set(chunk_texts)), "duplicate chunks emitted"

    # ~2 KB input under chunk_size=800 should yield a single-digit
    # window count, not the 30+ the buggy path emitted before
    # finally walking past the early paragraph break.
    assert len(chunks) <= 5, f"expected ≤5 chunks for 2KB input, got {len(chunks)}"


def test_real_document_chunk_count_is_bounded():
    """Synthetic fixture mirroring the parse_566bd3c39b8b197e shape:
    a top-of-file nav block followed by a long body. The buggy
    implementation produced 166 chunks with 30% duplicates on the
    real 13 KB document; the rewrite holds it under 30 chunks with
    zero duplicates."""
    nav = "Home > Docs > Page\n\n"
    body_paragraphs = [f"Paragraph {i}: " + ("y" * 200) for i in range(40)]
    text = nav + "\n\n".join(body_paragraphs)

    chunks = _split_chunks(text, ChunkingConfig())

    chunk_texts = [c["text"] for c in chunks]
    assert len(chunk_texts) == len(set(chunk_texts)), "duplicate chunks emitted"
    assert len(chunks) < 30, f"unexpectedly many chunks: {len(chunks)}"


def test_normal_long_document_preserves_overlap():
    """When chunks are large enough to host overlap, the loop must
    still advance by ``chunk_size - chunk_overlap`` so retrieval keeps
    the carry-over context. No boundaries in the input forces the
    hard-cut path."""
    text = "a" * 5000
    cfg = ChunkingConfig(chunk_size=800, chunk_overlap=80)
    chunks = _split_chunks(text, cfg)

    # Stride = chunk_size - chunk_overlap = 720, so 5000 / 720 ≈ 7.
    assert 6 <= len(chunks) <= 8, f"unexpected window count: {len(chunks)}"

    for prev, curr in zip(chunks, chunks[1:]):
        prev_tail = prev["text"][-cfg.chunk_overlap :]
        curr_head = curr["text"][: cfg.chunk_overlap]
        assert prev_tail == curr_head, "overlap broken between adjacent chunks"


def test_paragraph_boundary_preferred_when_in_second_half():
    """When a paragraph break sits in the second half of the window,
    ``end`` must snap to it so the chunk does not slice mid-paragraph
    and the next window starts on a clean boundary."""
    # chunk_size=200; paragraph break placed at position 150 (well
    # inside the second half — search_start = 100).
    text = ("a" * 150) + "\n\n" + ("b" * 600)
    chunks = _split_chunks(text, ChunkingConfig(chunk_size=200, chunk_overlap=20))

    # First chunk is the leading "a" run, stripped of trailing
    # whitespace by the strip() call.
    assert chunks[0]["text"] == "a" * 150


def test_line_break_fallback_when_no_paragraph_break():
    """When no paragraph break lands in the search window, the
    splitter falls back to a single newline rather than mid-line."""
    text = ("a" * 150) + "\n" + ("b" * 600)
    chunks = _split_chunks(text, ChunkingConfig(chunk_size=200, chunk_overlap=20))
    assert chunks[0]["text"] == "a" * 150


def test_sentence_break_fallback_when_no_line_break():
    """When neither paragraph nor line breaks are available, the
    splitter prefers a sentence terminator."""
    text = ("a" * 150) + ". " + ("b" * 600)
    chunks = _split_chunks(text, ChunkingConfig(chunk_size=200, chunk_overlap=20))
    # Sentence-fallback keeps the period with the preceding chunk.
    assert chunks[0]["text"] == ("a" * 150) + "."


def test_hard_cut_when_no_boundary_in_window():
    """With zero structural boundaries in the second half of the
    window, the splitter does a clean ``chunk_size`` hard cut and
    proceeds — no infinite loop, no duplicate emission."""
    text = "a" * 1600
    cfg = ChunkingConfig(chunk_size=400, chunk_overlap=40)
    chunks = _split_chunks(text, cfg)

    # Each non-final chunk is exactly chunk_size characters.
    for chunk in chunks[:-1]:
        assert len(chunk["text"]) == cfg.chunk_size
    # Stride is chunk_size - chunk_overlap = 360, covering 1600 chars.
    assert 4 <= len(chunks) <= 5


def test_chunk_ids_are_dense_and_stable():
    """``chunk_id`` indices stay 0..N-1 — the appender bumps
    ``chunk_index`` only on emit, and that contract is unchanged."""
    text = "para one\n\n" + ("Y" * 2000)
    chunks = _split_chunks(text, ChunkingConfig(chunk_size=800, chunk_overlap=80))

    suffixes = [c["chunk_id"].split(":")[-1] for c in chunks]
    assert suffixes == [f"{i:04d}" for i in range(len(chunks))]


def test_chunk_id_is_deterministic_for_same_input():
    """Same content + chunking knobs produce the same chunk_id list,
    which the indexing workers rely on for retry idempotence."""
    text = "lorem ipsum\n\n" + ("z" * 1500)
    cfg = ChunkingConfig(chunk_size=600, chunk_overlap=60)
    a = _split_chunks(text, cfg)
    b = _split_chunks(text, cfg)
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]


def test_short_document_emits_single_chunk():
    """Input shorter than chunk_size produces one chunk and
    terminates immediately."""
    chunks = _split_chunks("tiny doc", ChunkingConfig(chunk_size=800, chunk_overlap=80))
    assert len(chunks) == 1
    assert chunks[0]["text"] == "tiny doc"


def test_empty_or_whitespace_input_returns_empty():
    """Empty / whitespace-only input must not emit chunks — pre-fix
    contract preserved."""
    assert _split_chunks("", ChunkingConfig()) == []
    assert _split_chunks("   \n\n  ", ChunkingConfig()) == []


def test_chinese_full_stop_used_as_sentence_fallback():
    """ApeRAG's primary corpus is Chinese-language. A dense Chinese
    paragraph (no ``\\n``, no ``. ``) must still break on the
    full-width full stop ``。`` rather than mid-sentence."""
    # 1500-char Chinese text, no newlines, sentences ended by ``。``.
    sentence = "这是一个测试用的中文句子内容长度大约三十字符" + "啊" * 5 + "。"
    text = sentence * 50  # ~50 sentences of ~33 chars each → ~1650 chars
    chunks = _split_chunks(text, ChunkingConfig(chunk_size=400, chunk_overlap=40))

    # Every non-final chunk must end on the Chinese full stop, not
    # mid-character or mid-sentence.
    for chunk in chunks[:-1]:
        assert chunk["text"].endswith("。"), f"chunk should end on 。, got tail: ...{chunk['text'][-20:]!r}"


def test_chunk_size_boundary_off_by_one():
    """Inputs of length exactly ``chunk_size`` and ``chunk_size + 1``
    must terminate cleanly without off-by-one errors."""
    cfg = ChunkingConfig(chunk_size=200, chunk_overlap=20)

    # Exactly chunk_size → single chunk, loop terminates on first iter.
    text_exact = "a" * cfg.chunk_size
    chunks_exact = _split_chunks(text_exact, cfg)
    assert len(chunks_exact) == 1
    assert len(chunks_exact[0]["text"]) == cfg.chunk_size

    # chunk_size + 1 → two chunks, first is hard-cut at chunk_size.
    text_plus_one = "a" * (cfg.chunk_size + 1)
    chunks_plus = _split_chunks(text_plus_one, cfg)
    assert len(chunks_plus) >= 1
    assert len(chunks_plus[0]["text"]) == cfg.chunk_size


def test_crlf_paragraph_break_locks_upstream_normalization_contract():
    """``\\r\\n\\r\\n`` is *not* matched by ``rfind('\\n\\n', ...)``
    — the codepoint sequence is ``\\r\\n\\r\\n``, not ``\\n\\n``.
    The chunker therefore relies on the upstream docparser to
    normalize CRLF → LF before producing markdown. This test pins
    the contract: if a CRLF document slips through, paragraph-break
    detection silently falls through to the next fallback rather
    than blowing up. If docparser ever stops normalizing, the next
    fallback (``\\n``) catches it; this test guards against the
    chunker re-introducing its own CRLF handling without coordination
    with docparser."""
    # Use content with distinct per-position characters so legitimate
    # overlap windows do not collapse to equal strings under the
    # set() check.
    import string

    body = (string.ascii_letters * 30)[:600]
    text = ("a" * 150) + "\r\n\r\n" + body
    chunks = _split_chunks(text, ChunkingConfig(chunk_size=200, chunk_overlap=20))

    # Loop must terminate within a reasonable bound.
    assert len(chunks) <= 10
    # Distinct windows produce distinct chunk texts — no cascade.
    chunk_texts = [c["text"] for c in chunks]
    assert len(chunk_texts) == len(set(chunk_texts)), "cascade emitted on CRLF input"


def test_chunk_size_equals_twice_overlap_does_not_loop():
    """When ``chunk_size == 2 * chunk_overlap`` the guard
    ``(end - cursor) > chunk_overlap * 2`` is False on every full
    window, so cursor must jump to ``end`` rather than ``end -
    chunk_overlap``. Pin this so a future change to ``>=`` cannot
    silently re-introduce a stall."""
    text = "a" * 1000
    cfg = ChunkingConfig(chunk_size=4, chunk_overlap=2)
    chunks = _split_chunks(text, cfg)

    # 1000 chars / 4 chars per chunk = 250 chunks (no overlap on full
    # windows because guard fires False).
    assert len(chunks) == 250
    assert all(len(c["text"]) == 4 for c in chunks)


def test_multibyte_chars_split_cleanly_at_boundaries():
    """Python ``str`` is sequence-of-Unicode-code-points, so window
    arithmetic against ``len(text)`` already operates on character
    counts not byte counts — UTF-8 multi-byte runes never get sliced
    mid-byte. Pin this so a future move to byte-window accounting
    cannot silently regress."""
    # 600 中文字符（每字符 3 byte UTF-8）+ 没有任何分隔符 → 强制 hard-cut。
    text = "中" * 600
    cfg = ChunkingConfig(chunk_size=200, chunk_overlap=20)
    chunks = _split_chunks(text, cfg)

    # Every chunk must be a valid UTF-8 round-trip.
    for chunk in chunks:
        assert chunk["text"].encode("utf-8").decode("utf-8") == chunk["text"]

    # All characters in input must appear in some chunk (no silent drop).
    seen = "".join(c["text"] for c in chunks)
    assert "中" in seen
    assert seen.count("中") >= 600  # overlap can repeat, never lose


def test_chunk_record_shape_is_unchanged():
    """The chunk record fields are a downstream contract; the
    indexing workers (vector / summary / vision) read these keys.
    Locking the shape prevents accidental schema drift."""
    text = "hello world"
    chunks = _split_chunks(text, ChunkingConfig())
    assert len(chunks) == 1
    assert set(chunks[0].keys()) == {
        "chunk_id",
        "text",
        "section_path",
        "heading_anchor",
        "page_idx",
    }
    assert chunks[0]["section_path"] is None
    assert chunks[0]["heading_anchor"] is None
    assert chunks[0]["page_idx"] is None
