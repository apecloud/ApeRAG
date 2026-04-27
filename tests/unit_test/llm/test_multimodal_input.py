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

"""Unit tests for Wave 6 task #39: provider-specific multimodal input
payload dispatcher (`build_multimodal_input_payload`).
"""

from __future__ import annotations

import pytest

from aperag.llm.embed.multimodal_input import build_multimodal_input_payload

_DATA_URL = "data:image/jpeg;base64,AAAA"
_ALT = "two cats on the sofa"


@pytest.mark.parametrize("provider", ["voyage_ai", "voyageai", "voyage", "VOYAGE", " Voyage "])
def test_voyage_payload_uses_content_envelope_with_image_base64_part(provider):
    """Voyage AI multimodal embedding wraps each input in a
    ``{"content": [parts]}`` envelope — image part must use
    ``"image_base64"`` discriminator + carry the data URL inline.
    """

    payload = build_multimodal_input_payload(provider=provider, image_data_url=_DATA_URL, alt_text=_ALT)

    assert isinstance(payload, list) and len(payload) == 1
    item = payload[0]
    assert "content" in item
    parts = item["content"]
    image_parts = [p for p in parts if p.get("type") == "image_base64"]
    text_parts = [p for p in parts if p.get("type") == "text"]
    assert len(image_parts) == 1 and image_parts[0]["image_base64"] == _DATA_URL
    assert len(text_parts) == 1 and text_parts[0]["text"] == _ALT


def test_voyage_payload_omits_text_part_when_alt_text_empty():
    payload = build_multimodal_input_payload(provider="voyage_ai", image_data_url=_DATA_URL, alt_text="")
    parts = payload[0]["content"]
    assert all(p.get("type") != "text" for p in parts), "empty alt_text must not produce a text part"


@pytest.mark.parametrize("provider", ["jina_ai", "jinaai", "jina"])
def test_jina_payload_uses_flat_image_and_text_items(provider):
    """Jina (clip-v2 / embeddings-v4) accepts a flat list of single-key
    dicts: ``{"image": ...}`` for images, ``{"text": ...}`` for text.
    """

    payload = build_multimodal_input_payload(provider=provider, image_data_url=_DATA_URL, alt_text=_ALT)

    assert payload == [{"image": _DATA_URL}, {"text": _ALT}]


def test_jina_payload_omits_text_when_alt_text_empty():
    payload = build_multimodal_input_payload(provider="jina_ai", image_data_url=_DATA_URL, alt_text="   ")
    assert payload == [{"image": _DATA_URL}]


def test_cohere_payload_uses_image_then_text_items():
    payload = build_multimodal_input_payload(provider="cohere", image_data_url=_DATA_URL, alt_text=_ALT)
    assert payload == [{"image": _DATA_URL}, {"text": _ALT}]


def test_openai_payload_uses_chat_multimodal_envelope():
    """OpenAI's documented multimodal request shape uses
    ``{"type": "image_url", "image_url": {"url": ...}}`` parts.
    """

    payload = build_multimodal_input_payload(provider="openai", image_data_url=_DATA_URL, alt_text=_ALT)

    assert payload == [
        {"type": "image_url", "image_url": {"url": _DATA_URL}},
        {"type": "text", "text": _ALT},
    ]


def test_openai_payload_alias_openai_multimodal_resolves_same_shape():
    a = build_multimodal_input_payload(provider="openai_multimodal", image_data_url=_DATA_URL, alt_text="")
    b = build_multimodal_input_payload(provider="openai", image_data_url=_DATA_URL, alt_text="")
    assert a == b


def test_unknown_provider_falls_back_to_litellm_default_shape():
    """Unmapped providers preserve the Wave 5 P2 LiteLLM-documented
    default shape so the prior behaviour is unchanged.
    """

    payload = build_multimodal_input_payload(provider="some-new-provider", image_data_url=_DATA_URL, alt_text=_ALT)
    assert payload == [
        {"image_url": {"url": _DATA_URL}},
        {"text": _ALT},
    ]


def test_none_provider_resolves_to_default():
    payload = build_multimodal_input_payload(provider=None, image_data_url=_DATA_URL, alt_text="x")
    assert payload[0] == {"image_url": {"url": _DATA_URL}}


def test_alt_text_whitespace_treated_as_empty_across_providers():
    """A whitespace-only ``alt_text`` must not produce a text part on
    any provider — pairing the image with " " would change the cache
    key + may confuse the embedder.
    """

    for provider in ("voyage_ai", "jina_ai", "cohere", "openai", "unknown"):
        payload = build_multimodal_input_payload(provider=provider, image_data_url=_DATA_URL, alt_text="   ")
        flat = payload[0].get("content", payload)
        text_present = any(
            ("text" in p and p["text"]) or (p.get("type") == "text")
            for p in (flat if isinstance(flat, list) else [flat])
        )
        assert not text_present, f"{provider}: whitespace alt_text must not produce a text part"
