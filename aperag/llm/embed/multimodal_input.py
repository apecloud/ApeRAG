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

"""Provider-specific multimodal embedding input payload builders.

Wave 6 task #39 per `docs/modularization/indexing-redesign-design-pack.md`
§G.2.5.1: the Wave 5 P2 chunk 1 :func:`EmbeddingService._embed_image_via_litellm`
shipped a single LiteLLM-style ``input=[{"image_url": {"url": "data:..."}}, {"text":...}]``
shape that mirrors the OpenAI multimodal *chat completion* request.
That shape is **not** the canonical *embedding* request shape for the
real multimodal-capable providers (Voyage AI, Jina, Cohere, etc.) —
LiteLLM may translate it transparently for some providers but ships
no guarantee. This module dispatches a per-provider payload so the
operator-configured embedder receives the input shape it actually
documents.

The dispatcher matches the ``embedding_provider`` string used by
:class:`EmbeddingService` (LiteLLM provider keyword) and is purposely
**hard-cut** — there is no shim or fall-back to a "compatibility"
shape; unknown providers fall through to the LiteLLM-documented
default that the rest of the embedding stack already uses.
"""

from __future__ import annotations

from typing import Any

# Canonical provider keywords (matches LiteLLM ``custom_llm_provider`` /
# the provider prefix on ``model="<provider>/<model>"`` calls). The
# accepted-aliases tuple lets operators name the provider in any
# common way (LiteLLM itself accepts ``"voyage_ai"`` and ``"voyage"``).
_VOYAGE_ALIASES = ("voyage_ai", "voyageai", "voyage")
_JINA_ALIASES = ("jina_ai", "jinaai", "jina")
_OPENAI_MULTIMODAL_ALIASES = ("openai_multimodal", "openai")
_COHERE_ALIASES = ("cohere",)


def build_multimodal_input_payload(
    *,
    provider: str | None,
    image_data_url: str,
    alt_text: str,
) -> list[dict[str, Any]]:
    """Return the ``input=`` payload for ``litellm.embedding(...)``.

    ``image_data_url`` must already be a base64 data URL
    (``data:image/<kind>;base64,<...>``) — the caller is responsible
    for MIME detection. ``alt_text`` may be empty; providers that
    accept paired text+image inputs use it, others ignore it.

    Returns a ``list[dict]`` because every supported provider's
    embedding wire shape is ``input: [...]`` even for a single image.
    """

    p = (provider or "").strip().lower()
    if p in _VOYAGE_ALIASES:
        return _voyage_payload(image_data_url, alt_text)
    if p in _JINA_ALIASES:
        return _jina_payload(image_data_url, alt_text)
    if p in _COHERE_ALIASES:
        return _cohere_payload(image_data_url, alt_text)
    if p in _OPENAI_MULTIMODAL_ALIASES:
        return _openai_payload(image_data_url, alt_text)
    return _default_payload(image_data_url, alt_text)


def _voyage_payload(image_data_url: str, alt_text: str) -> list[dict[str, Any]]:
    """Voyage AI ``voyage-multimodal-3`` input shape.

    Voyage's multimodal embedding API expects each input to be a
    ``{"content": [...]}`` envelope listing one or more parts; image
    parts use ``{"type": "image_base64", "image_base64": "<data url>"}``
    and text parts use ``{"type": "text", "text": "..."}``. The text
    part is omitted when the caller didn't pass an ``alt_text``.
    """

    parts: list[dict[str, Any]] = [{"type": "image_base64", "image_base64": image_data_url}]
    if alt_text and alt_text.strip():
        parts.append({"type": "text", "text": alt_text.strip()})
    return [{"content": parts}]


def _jina_payload(image_data_url: str, alt_text: str) -> list[dict[str, Any]]:
    """Jina (``jina-clip-v2`` / ``jina-embeddings-v4``) input shape.

    Jina's multimodal embedding endpoint accepts a list of single-key
    dicts: ``{"image": "<data url>"}`` for images and
    ``{"text": "..."}`` for text. They are embedded jointly so paired
    image + alt-text returns a single fused vector.
    """

    items: list[dict[str, Any]] = [{"image": image_data_url}]
    if alt_text and alt_text.strip():
        items.append({"text": alt_text.strip()})
    return items


def _cohere_payload(image_data_url: str, alt_text: str) -> list[dict[str, Any]]:
    """Cohere multimodal embedding (``embed-*-v3`` with image input).

    Cohere's image embedding uses ``{"image": "<data url>"}`` per item;
    text is sent as a separate string entry in the same ``texts``
    array. Cohere does not return a fused vector for paired
    text+image, so we keep both items independent — the caller can
    choose which vector to consume.
    """

    items: list[dict[str, Any]] = [{"image": image_data_url}]
    if alt_text and alt_text.strip():
        items.append({"text": alt_text.strip()})
    return items


def _openai_payload(image_data_url: str, alt_text: str) -> list[dict[str, Any]]:
    """OpenAI multimodal embedding input shape (LiteLLM-mapped).

    OpenAI's standard ``text-embedding-3-*`` models do not accept
    image input — operators that flip ``Model.supports_multimodal_embedding``
    on an OpenAI text embedder will hit a runtime error from the
    provider. This builder formats the same multipart envelope used
    by the OpenAI multimodal *chat* request (the closest documented
    shape) so the failure surfaces as a provider-side 4xx with a
    clear message instead of a silently-truncated text-only embed.
    """

    parts: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": image_data_url}}]
    if alt_text and alt_text.strip():
        parts.append({"type": "text", "text": alt_text.strip()})
    return parts


def _default_payload(image_data_url: str, alt_text: str) -> list[dict[str, Any]]:
    """Fallback to the Wave 5 P2 LiteLLM-documented default shape.

    Used when ``embedding_provider`` is unset or matches no canonical
    alias above. Mirrors the Wave 5 P2 chunk 1 baseline so an
    unmapped provider keeps the prior behaviour rather than raising —
    operators see the same error path they would have seen pre-#39.
    """

    payload: list[dict[str, Any]] = [{"image_url": {"url": image_data_url}}]
    if alt_text and alt_text.strip():
        payload.append({"text": alt_text.strip()})
    return payload


__all__ = ["build_multimodal_input_payload"]
