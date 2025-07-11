from typing import List

from aperag.docparser.base import Part, TitlePart
from aperag.docparser.chunking import Group, Rechunker, SimpleSemanticSplitter


def mock_tokenizer(text: str) -> List[int]:
    # Simple tokenizer for testing, splits by spaces and returns length of each word
    return [len(word) for word in text.split()]


def mock_char_tokenizer(text: str) -> List[int]:
    # Simple tokenizer for testing, returns ordinal of each char
    return [ord(char) for char in text]


def test_group_creation():
    parts = [
        Part(content="Text 1", metadata={"nesting": 0}),
        TitlePart(content="# Title 1", metadata={"nesting": 0}, level=1),
        Part(content="Text 2", metadata={"nesting": 0}),
        Part(content="Text 3", metadata={"nesting": 1}),
        TitlePart(content="## Title 2", metadata={"nesting": 0}, level=2),
        Part(content="Text 4", metadata={"nesting": 0}),
    ]

    rechunker = Rechunker(chunk_size=100, chunk_overlap=0, tokenizer=mock_tokenizer)
    groups = rechunker._to_groups(parts)

    assert len(groups) == 3
    assert groups[0].title == ""
    assert len(groups[0].items) == 1
    assert groups[1].title == "# Title 1"
    assert len(groups[1].items) == 3
    assert groups[2].title == "## Title 2"
    assert len(groups[2].items) == 2


def test_rechunking_no_merge():
    parts = [
        Part(content="This is a short text.", metadata={}),
        Part(content="This is another short text.", metadata={}),
        Part(content="A longer text that will not fit in one chunk.", metadata={}),
    ]

    rechunker = Rechunker(chunk_size=8, chunk_overlap=0, tokenizer=mock_tokenizer)
    rechunked_parts = rechunker(parts)

    assert len(rechunked_parts) == 4
    assert rechunked_parts[0].content == "This is a short text."
    assert rechunked_parts[1].content == "This is another short text."
    assert "A longer text" in rechunked_parts[2].content
    assert "one chunk" in rechunked_parts[3].content


def test_rechunking_with_merge():
    parts = [
        Part(content="Short text 1.", metadata={}),
        Part(content="Short text 2.", metadata={}),
        Part(content="This is a slightly longer text 3.", metadata={}),
    ]

    rechunker = Rechunker(chunk_size=10, chunk_overlap=0, tokenizer=mock_tokenizer)
    rechunked_parts = rechunker(parts)

    assert len(rechunked_parts) == 2
    assert "Short text 1." in rechunked_parts[0].content
    assert "Short text 2." in rechunked_parts[0].content
    assert rechunked_parts[1].content == "This is a slightly longer text 3."


def test_rechunking_with_title_merge_prevention():
    parts = [
        Part(content="Intro text", metadata={}),
        TitlePart(content="# Main Title", level=1),
        Part(content="Content under main title", metadata={}),
        TitlePart(content="## Subtitle 1", level=2),
        Part(content="Content under subtitle 1", metadata={}),
        TitlePart(content="## Subtitle 2", level=2),
        Part(content="Content under subtitle 2", metadata={}),
        TitlePart(content="# Main Title 2", level=1),
        Part(content="Content under main title 2", metadata={}),
    ]

    rechunker = Rechunker(chunk_size=6, chunk_overlap=0, tokenizer=mock_tokenizer)
    rechunked_parts = rechunker(parts)

    print(rechunked_parts)
    assert len(rechunked_parts) == 9

    assert rechunked_parts[0].content == "Intro text"
    assert rechunked_parts[0].metadata.get("titles") is None

    assert rechunked_parts[1].content == "# Main Title"
    assert rechunked_parts[1].metadata.get("titles") == ["# Main Title"]

    assert rechunked_parts[2].content == "Content under main title"
    assert rechunked_parts[2].metadata.get("titles") == ["# Main Title"]

    assert rechunked_parts[3].content == "## Subtitle 1"
    assert rechunked_parts[3].metadata.get("titles") == ["# Main Title", "## Subtitle 1"]

    assert rechunked_parts[4].content == "Content under subtitle 1"
    assert rechunked_parts[4].metadata.get("titles") == ["# Main Title", "## Subtitle 1"]

    assert rechunked_parts[5].content == "## Subtitle 2"
    assert rechunked_parts[5].metadata.get("titles") == ["# Main Title", "## Subtitle 2"]

    assert rechunked_parts[6].content == "Content under subtitle 2"
    assert rechunked_parts[6].metadata.get("titles") == ["# Main Title", "## Subtitle 2"]

    assert rechunked_parts[7].content == "# Main Title 2"
    assert rechunked_parts[7].metadata.get("titles") == ["# Main Title 2"]

    assert rechunked_parts[8].content == "Content under main title 2"
    assert rechunked_parts[8].metadata.get("titles") == ["# Main Title 2"]


def test_append_group_to_part():
    rechunker = Rechunker(chunk_size=100, chunk_overlap=0, tokenizer=mock_tokenizer)
    group = Group(
        title_level=1,
        title="# Test Group",
        items=[
            Part(content="Part 1 of group.", metadata={}),
            Part(content="Part 2 of group.", metadata={}),
        ],
    )

    dest_part = Part(content="Initial content.", metadata={})
    result_part = rechunker._append_group_to_part(group, dest_part, [])

    assert "Initial content." in result_part.content
    assert "Part 1 of group." in result_part.content
    assert "Part 2 of group." in result_part.content


def test_append_part_to_part():
    rechunker = Rechunker(chunk_size=100, chunk_overlap=0, tokenizer=mock_tokenizer)
    part = Part(
        content="Content to append.",
        metadata={
            "md_source_map": [10, 20],
            "pdf_source_map": [{"page_idx": 1, "bbox": [0, 0, 100, 100]}, {"page_idx": 100, "bbox": [0, 0, 200, 200]}],
        },
    )

    dest_part = Part(
        content="Destination content.",
        metadata={
            "md_source_map": [0, 5],
            "pdf_source_map": [{"page_idx": 0, "bbox": [0, 0, 200, 200]}, {"page_idx": 100, "bbox": [0, 0, 200, 200]}],
        },
    )
    result_part = rechunker._append_part_to_part(part, dest_part, [])

    assert "Destination content." in result_part.content
    assert "Content to append." in result_part.content
    assert result_part.metadata["md_source_map"] == [0, 20]
    assert len(result_part.metadata["pdf_source_map"]) == 3
    assert {"page_idx": 0, "bbox": [0, 0, 200, 200]} in result_part.metadata["pdf_source_map"]
    assert {"page_idx": 1, "bbox": [0, 0, 100, 100]} in result_part.metadata["pdf_source_map"]
    assert {"page_idx": 100, "bbox": [0, 0, 200, 200]} in result_part.metadata["pdf_source_map"]


def test_merge_metadata():
    rechunker = Rechunker(chunk_size=100, chunk_overlap=0, tokenizer=mock_tokenizer)
    dest_part = Part(content="Destination", metadata={})
    src_part = Part(
        content="Source",
        metadata={"md_source_map": [5, 10], "pdf_source_map": [{"page_idx": 1, "bbox": [0, 0, 100, 100]}]},
    )

    rechunker._merge_md_source_map(dest_part, src_part)
    assert dest_part.metadata["md_source_map"] == [5, 10]

    rechunker._merge_pdf_source_map(dest_part, src_part)
    assert dest_part.metadata["pdf_source_map"] == [{"page_idx": 1, "bbox": [0, 0, 100, 100]}]

    src_part2 = Part(
        content="Source 2",
        metadata={
            "md_source_map": [0, 7],
            "pdf_source_map": [{"page_idx": 0, "bbox": [0, 0, 200, 200]}, {"page_idx": 1, "bbox": [0, 0, 100, 100]}],
        },
    )
    rechunker._merge_md_source_map(dest_part, src_part2)
    assert dest_part.metadata["md_source_map"] == [0, 10]

    rechunker._merge_pdf_source_map(dest_part, src_part2)
    assert {"page_idx": 0, "bbox": [0, 0, 200, 200]} in dest_part.metadata["pdf_source_map"]
    assert {"page_idx": 1, "bbox": [0, 0, 100, 100]} in dest_part.metadata["pdf_source_map"]


def test_count_tokens_part():
    rechunker = Rechunker(chunk_size=100, chunk_overlap=0, tokenizer=mock_tokenizer)
    part = Part(content="A text with some words.", metadata={})
    tokens = rechunker._count_tokens(part)
    assert tokens == 5
    assert part.metadata["tokens"] == 5  # Check if tokens are cached


def test_count_tokens_group():
    rechunker = Rechunker(chunk_size=100, chunk_overlap=0, tokenizer=mock_tokenizer)
    group = Group(
        title_level=1,
        title="Test Group",
        items=[
            Part(content="First part.", metadata={}),
            Part(content="Second part with more words.", metadata={}),
        ],
    )
    tokens = rechunker._count_tokens(group)
    assert tokens == 7
    assert group.tokens == 7  # Check if tokens are cached


def test_simple_semantic_splitter_fit():
    splitter = SimpleSemanticSplitter(tokenizer=mock_tokenizer)
    assert splitter._fit("Short text", 10)
    assert not splitter._fit("This is a longer text", 4)


def test_simple_semantic_splitter_recursive_split():
    splitter = SimpleSemanticSplitter(tokenizer=mock_tokenizer)
    text = "This is a sentence. Another sentence here!"
    chunks = splitter._recursive_split(text, 6, 2, 0)
    assert len(chunks) > 1
    assert all(len(mock_tokenizer(chunk)) <= 6 for chunk in chunks)
    assert chunks[0] == "This is a sentence."
    assert chunks[1] == " Another sentence here!"


def test_simple_semantic_splitter_cut_right_side():
    splitter = SimpleSemanticSplitter(tokenizer=mock_tokenizer)
    text = "A long phrase that needs cutting"
    cut_text = splitter._cut_right_side(text, 5)
    assert len(mock_tokenizer(cut_text)) <= 5
    assert cut_text in text


def test_simple_semantic_splitter_merge_small_chunks():
    splitter = SimpleSemanticSplitter(tokenizer=mock_tokenizer)
    chunks = ["small chunk 1", "small chunk 2", "slightly larger chunk"]
    merged_chunks = splitter._merge_small_chunks(chunks, 15)
    assert len(merged_chunks) <= len(chunks)
    assert all(len(mock_tokenizer(chunk)) <= 15 for chunk in merged_chunks)


def test_simple_semantic_splitter_split():
    splitter = SimpleSemanticSplitter(tokenizer=mock_tokenizer)
    text = "This is a test with multiple sentences. Some are short. Others are longer and more complex."
    chunks = splitter.split(text, 15, 3)
    assert len(chunks) > 1
    assert all(len(mock_tokenizer(chunk)) <= 15 for chunk in chunks)


def test_rechunker_with_empty_parts_and_groups():
    parts = [
        Part(content="", metadata={}),  # Empty content
        TitlePart(content="", level=1),  # Empty title
        Part(content="Valid Content", metadata={}),
    ]

    rechunker = Rechunker(chunk_size=10, chunk_overlap=0, tokenizer=mock_tokenizer)
    rechunked_parts = rechunker(parts)

    assert len(rechunked_parts) == 1
    assert rechunked_parts[0].content == "Valid Content"


def test_rechunker_edge_case_large_title():
    parts = [
        TitlePart(content="# " + "A" * 40, level=1),  # Large title
        Part(content="Normal Content", metadata={}),
    ]

    rechunker = Rechunker(chunk_size=35, chunk_overlap=0, tokenizer=mock_char_tokenizer)
    rechunked_parts = rechunker(parts)

    assert len(rechunked_parts) == 2
    assert "# AAAAA" in rechunked_parts[0].content
    assert "AAAAA" in rechunked_parts[1].content
    assert "Normal Content" in rechunked_parts[1].content


def test_splitter_with_long_text_no_separators():
    splitter = SimpleSemanticSplitter(tokenizer=mock_char_tokenizer)
    long_text = "A" * 100  # Very long text with no separators
    chunks = splitter.split(long_text, 20, 5)

    assert len(chunks) > 1
    assert all(len(mock_char_tokenizer(chunk)) <= 20 for chunk in chunks)


def test_rechunker_with_overlapping_chunks():
    parts = [Part(content="SentenceOneSentenceTwoSentenceThree", metadata={})]

    rechunker = Rechunker(chunk_size=20, chunk_overlap=5, tokenizer=mock_char_tokenizer)
    rechunked_parts = rechunker(parts)

    assert len(rechunked_parts) == 3
    # Check overlapped
    assert len("".join([part.content for part in rechunked_parts])) > len(parts[0].content)
