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

"""Contract tests for D8 §2.5 Anthropic-shape citations adapter (Phase 8 #75)."""

from __future__ import annotations

from aperag.domains.agent_runtime.tools.citations import (
    build_citation,
    transform_reference_bundle_items,
)


def test_build_citation_with_char_indices_returns_char_location():
    citation = build_citation(
        cited_text="result snippet",
        location_metadata={"start_char_index": 10, "end_char_index": 50, "title": "Doc"},
    )
    assert citation.location.type == "char_location"
    assert citation.location.start_char == 10
    assert citation.location.end_char == 50
    assert citation.location.doc_title == "Doc"


def test_build_citation_with_page_number_returns_page_location():
    citation = build_citation(
        cited_text="page text",
        location_metadata={"page_number": 3, "title": "Doc.pdf"},
    )
    assert citation.location.type == "page_location"
    assert citation.location.page_index == 3


def test_build_citation_with_block_index_returns_content_block_location():
    citation = build_citation(
        cited_text="block",
        location_metadata={"content_block_index": 7},
    )
    assert citation.location.type == "content_block_location"
    assert citation.location.block_index == 7


def test_build_citation_with_url_returns_url_citation():
    citation = build_citation(
        cited_text="web result",
        location_metadata={"url": "https://example.com/x", "title": "Web"},
    )
    assert citation.location.type == "url_citation"
    assert citation.location.url == "https://example.com/x"
    assert citation.location.title == "Web"


def test_build_citation_falls_back_to_url_citation_with_empty_url():
    citation = build_citation(cited_text="orphan", location_metadata={})
    assert citation.location.type == "url_citation"
    assert citation.location.url == ""


def test_transform_reference_bundle_items_skips_blank_snippets():
    items = [
        {"snippet": "valid", "metadata": {"url": "https://x"}},
        {"snippet": "", "metadata": {"url": "https://y"}},
        {"content": "also valid", "metadata": {"page_number": 2}},
    ]
    citations = transform_reference_bundle_items(items)
    assert len(citations) == 2
    assert citations[0].cited_text == "valid"
    assert citations[1].cited_text == "also valid"


def test_transform_promotes_top_level_uri_into_metadata_url():
    items = [{"snippet": "text", "uri": "https://promoted"}]
    citations = transform_reference_bundle_items(items)
    assert citations[0].location.url == "https://promoted"


def test_transform_skips_non_dict_items():
    items = [{"snippet": "ok"}, "garbage", None]
    citations = transform_reference_bundle_items(items)
    assert len(citations) == 1


def test_char_location_takes_precedence_over_page_when_both_present():
    citation = build_citation(
        cited_text="t",
        location_metadata={
            "start_char_index": 0,
            "end_char_index": 10,
            "page_number": 2,
        },
    )
    assert citation.location.type == "char_location"
