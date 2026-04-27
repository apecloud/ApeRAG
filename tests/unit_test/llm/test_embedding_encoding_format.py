"""Focused regression test for task #15 — Alibaba ``encoding_format`` adaptation.

Root cause recap (from #11 Phase B run 24845074156):

    litellm.BadRequestError: OpenAIException - Error code: 400 -
    {'error': {'message': "'encoding_format' only support with [float, base64]",
               'type': 'invalid_request_error'},
     'request_id': 'e707ed9e-...'}

LiteLLM was calling Alibaba DashScope's ``compatible-mode/v1/embeddings``
without an explicit ``encoding_format`` and DashScope rejected the default
LiteLLM forwarded. The fix pins ``encoding_format="float"`` in
``aperag/llm/embed/embedding_service.py::_embed_batch``. This test nails
that down so the pin cannot silently drift back to the default.

We mock ``litellm.embedding`` at the module level so the test requires no
provider credentials and runs in the plain unit_test suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from aperag.llm.embed import embedding_service as embedding_module
from aperag.llm.embed.embedding_service import EmbeddingService


def _fake_litellm_response(batch_size: int, dim: int = 4):
    """Minimal response shape the service consumes: ``response["data"][i]["embedding"]``."""
    return {"data": [{"embedding": [0.0] * dim} for _ in range(batch_size)]}


def _build_service() -> EmbeddingService:
    return EmbeddingService(
        embedding_provider="alibabacloud",
        embedding_model="text-embedding-v3",
        embedding_service_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_service_api_key="sk-test-not-used",
        embedding_max_chunks_in_batch=8,
        caching=False,
    )


def test_embed_batch_pins_encoding_format_to_float():
    """The single-batch invocation must pass ``encoding_format='float'``.

    Alibaba DashScope returns HTTP 400 if ``encoding_format`` is anything
    outside ``{'float', 'base64'}`` (including the value LiteLLM defaulted
    to before #15). Pinning ``'float'`` is what unblocks the provider
    path. This assertion catches any future refactor that drops the kwarg.
    """

    service = _build_service()
    batch = ["ApeRAG is a retrieval augmented generation platform.", "Embeddings convert text into vectors."]

    with patch.object(embedding_module.litellm, "embedding", return_value=_fake_litellm_response(len(batch))) as m:
        result = service._embed_batch(batch)

    assert len(result) == len(batch)
    assert m.call_count == 1
    kwargs = m.call_args.kwargs
    assert kwargs.get("encoding_format") == "float", (
        "encoding_format must be pinned to 'float' so Alibaba DashScope "
        "compatible-mode/v1/embeddings does not 400 with 'encoding_format' "
        "only support with [float, base64]. Task #15 / #11 Phase B."
    )
    # Keep the rest of the call shape stable so the fix is not mistaken for
    # a broader invocation refactor.
    assert kwargs.get("custom_llm_provider") == "alibabacloud"
    assert kwargs.get("model") == "text-embedding-v3"
    assert kwargs.get("input") == batch


def test_embed_batch_passes_encoding_format_for_every_provider():
    """Not Alibaba-specific: pin applies to every provider the service talks to.

    Per the #15 design note, ``encoding_format='float'`` is the OpenAI
    baseline and every OpenAI-compatible provider in our surface
    (alibabacloud / openrouter / jina / etc) accepts it, so there is no
    reason to branch per-provider. If this test loses its Alibaba cousin
    (via provider-selection refactor), we still want the pin to survive.
    """

    for provider in ("openrouter", "jina_ai", "alibabacloud", "openai"):
        service = EmbeddingService(
            embedding_provider=provider,
            embedding_model="text-embedding-3-small",
            embedding_service_url="https://example.invalid",
            embedding_service_api_key="sk-test-not-used",
            embedding_max_chunks_in_batch=8,
            caching=False,
        )
        with patch.object(embedding_module.litellm, "embedding", return_value=_fake_litellm_response(1)) as m:
            service._embed_batch(["hello"])
        assert m.call_args.kwargs.get("encoding_format") == "float", (
            f"provider {provider!r} lost the encoding_format='float' pin"
        )


def test_embedding_service_source_pins_encoding_format_float():
    """Static guard: the literal ``encoding_format="float"`` must live in
    ``embedding_service.py`` next to the ``litellm.embedding(`` call.

    Complements the behavioural test above by catching a refactor that
    keeps the argument exposed as an instance/class attribute or reads it
    from a config object — both of which could accidentally pass through
    an empty value. The failure message gives the reviewer a concrete
    pointer back to #15 without re-reading history.
    """

    source = Path(embedding_module.__file__).read_text(encoding="utf-8")
    assert 'encoding_format="float"' in source, (
        'embedding_service.py must keep the literal encoding_format="float" '
        "pin in the litellm.embedding(...) call. See task #15."
    )
