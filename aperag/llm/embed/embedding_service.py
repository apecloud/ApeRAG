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

from __future__ import annotations

import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple

import litellm

from aperag.cache import NAMESPACE_EMBEDDING, application_cache_policy, get_sync_application_cache
from aperag.llm.llm_error_types import (
    BatchProcessingError,
    EmbeddingError,
    EmptyTextError,
    wrap_litellm_error,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self,
        embedding_provider: str,
        embedding_model: str,
        embedding_service_url: str,
        embedding_service_api_key: str,
        embedding_max_chunks_in_batch: int,
        embedding_max_workers: int = 1,
        multimodal: bool = False,
        caching: bool = True,
    ):
        self.embedding_provider = embedding_provider
        self.model = embedding_model
        self.api_base = embedding_service_url
        self.api_key = embedding_service_api_key
        self.max_chunks = embedding_max_chunks_in_batch
        self.max_workers = max(1, embedding_max_workers)
        self.multimodal = multimodal
        self.caching = caching

    def embed_documents(self, contents: List[str]) -> List[List[float]]:
        """
        Embed multiple documents in parallel batches.

        Args:
            contents: List of documents (texts or base64-encoded images) to embed

        Returns:
            List of embedding vectors in the same order as input contents
        """
        # Validate inputs
        if not contents:
            raise EmptyTextError(0)

        # Check for empty contents
        empty_indices = [i for i, text in enumerate(contents) if not text or not text.strip()]
        if empty_indices:
            logger.warning(f"Found {len(empty_indices)} empty content at indices: {empty_indices}")
            if len(empty_indices) == len(contents):
                raise EmptyTextError(len(empty_indices))

        try:
            # Clean contents by replacing newlines with spaces
            clean_contents = [t.replace("\n", " ") if t and t.strip() else " " for t in contents]
            # Determine batch size (use max_chunks or process all at once if not set)
            batch_size = self.max_chunks or len(clean_contents)

            # Store results with original indices to ensure correct ordering
            results_dict: Dict[int, List[float]] = {}

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = []

                # Submit batches for processing with their starting indices
                for start in range(0, len(clean_contents), batch_size):
                    batch = clean_contents[start : start + batch_size]
                    # Pass both the batch and starting index to track position
                    future = pool.submit(self._embed_batch_with_indices, batch, start)
                    futures.append(future)

                # Process completed futures and store results by index
                failed_batches = []
                for future in as_completed(futures):
                    try:
                        # Get results with their original indices
                        batch_results = future.result()
                        for idx, embedding in batch_results:
                            results_dict[idx] = embedding
                    except Exception as e:
                        failed_batches.append(str(e))
                        logger.error(f"Batch processing failed: {e}")

                if failed_batches:
                    raise BatchProcessingError(
                        batch_size=batch_size,
                        reason=f"Failed to process {len(failed_batches)} batches: {failed_batches[:3]} "
                        f"contents: {contents}",
                    )

            # Reconstruct the result list in the original order
            results = [results_dict[i] for i in range(len(clean_contents))]
            return results
        except (EmptyTextError, BatchProcessingError, EmbeddingError):
            # Re-raise our custom embedding errors
            raise
        except Exception as e:
            logger.error(f"Document embedding failed: {str(e)}")
            raise wrap_litellm_error(e, "embedding", self.embedding_provider, self.model) from e

    async def aembed_documents(self, contents: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, contents)

    def embed_query(self, content: str) -> List[float]:
        """
        Embed a single query content.

        Args:
            content: content to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not content or not content.strip():
            raise EmptyTextError(1)

        try:
            return self.embed_documents([content])[0]
        except (EmptyTextError, EmbeddingError):
            # Re-raise our custom embedding errors
            raise
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise wrap_litellm_error(e, "embedding", self.embedding_provider, self.model) from e

    async def aembed_query(self, content: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, content)

    def is_multimodal(self) -> bool:
        return self.multimodal

    def embed_image(self, image_bytes: bytes, alt_text: str = "") -> List[float]:
        """Embed image bytes into a vector via the configured multimodal model.

        Wave 5 Phase 2 T7 item 1 per §G.2.5.1 amendment: the canonical
        multimodal embedding API surface that the new vision modality
        consumes (replacing the Wave 3 placeholder ``embed_query(f"{image_id}|
        {alt_text}")`` string-concat pattern). Operators wire a real
        multimodal embedder (CLIP / GPT-4V / etc.) on the collection's
        embedder spec and set ``multimodal=True``; the chunk 4b vision
        gate (``is_multimodal()`` check) then self-disables and the
        worker_factory ``_build_vision_worker._embed`` callsite invokes
        this method with real image bytes.

        Encodes the image as a base64 data URL and forwards it to the
        underlying LiteLLM-shaped embedding call via ``input=[{
        "image_url": {"url": "data:image/...;base64,..."}}]`` — providers
        that support multimodal embedding (Voyage AI, Jina v3, OpenAI
        multimodal embeddings, etc.) accept this shape natively. If
        ``multimodal=False`` (operator opted into vision without
        configuring a multimodal embedder), raises ``EmbeddingError`` —
        the worker_factory chunk 4b gate already prevents this state but
        the runtime check is a defense-in-depth that surfaces clear
        diagnostics if the gate is bypassed.

        ``alt_text`` is appended as a textual hint for embedders that
        accept paired text+image inputs (improves retrieval recall on
        images with meaningful captions / OCR'd alt-text). Embedders
        that ignore it (image-only embedders) silently drop it.

        Wave 6 follow-up: provider-specific multimodal embedding
        configuration (per-provider input format, MIME-type detection,
        size limits) is staged in the cross-cutting refactor batch (per
        §K.10 Wave 6 backlog). Until then, deployments that configure a
        multimodal embedder must verify their provider accepts the
        ``input=[{"image_url": {"url": "data:..."}}]`` shape (LiteLLM-
        documented multimodal-capable providers).
        """
        if not self.multimodal:
            raise EmbeddingError(
                "embed_image called on a non-multimodal EmbeddingService — "
                "set multimodal=True on the collection's embedder spec or "
                "set collection.config.enable_vision=false until a real "
                "multimodal embedder is configured (Wave 5 P2 §G.2.5.1)"
            )
        if not image_bytes:
            raise EmptyTextError(1)

        def _compute() -> List[float]:
            try:
                return self._embed_image_via_litellm(image_bytes=image_bytes, alt_text=alt_text)
            except (EmptyTextError, EmbeddingError):
                raise
            except Exception as e:
                logger.error(f"Image embedding failed: {str(e)}")
                raise wrap_litellm_error(e, "embedding", self.embedding_provider, self.model) from e

        if not self.caching:
            return _compute()

        cache = get_sync_application_cache()
        return cache.get_or_compute(
            namespace=NAMESPACE_EMBEDDING,
            key_data=self._cache_key_for_image(image_bytes, alt_text),
            compute=_compute,
            policy=application_cache_policy(NAMESPACE_EMBEDDING),
        )

    async def aembed_image(self, image_bytes: bytes, alt_text: str = "") -> List[float]:
        return await asyncio.to_thread(self.embed_image, image_bytes, alt_text)

    def _embed_image_via_litellm(self, *, image_bytes: bytes, alt_text: str) -> List[float]:
        """Underlying LiteLLM multimodal embedding call.

        Encodes the image as a base64 data URL and dispatches to a
        provider-specific input payload shape via
        :func:`build_multimodal_input_payload` (Wave 6 task #39 per
        §G.2.5.1). Voyage / Jina / Cohere / OpenAI all document
        different multimodal embedding wire shapes; the dispatcher
        emits the canonical shape for the configured provider and
        falls back to the Wave 5 P2 LiteLLM-documented default for
        unknown providers.
        """
        import base64
        import imghdr

        from litellm import embedding as litellm_embedding

        from aperag.llm.embed.multimodal_input import build_multimodal_input_payload

        # Detect MIME type from the image bytes header (avoids relying
        # on caller-provided alt_text format hints). Falls back to
        # image/jpeg if detection fails — most providers tolerate
        # mismatched MIME on data-URL inputs.
        kind = imghdr.what(None, h=image_bytes) or "jpeg"
        mime = f"image/{kind}"
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        input_payload = build_multimodal_input_payload(
            provider=self.embedding_provider,
            image_data_url=data_url,
            alt_text=alt_text,
        )

        response = litellm_embedding(
            model=f"{self.embedding_provider}/{self.model}" if self.embedding_provider else self.model,
            input=input_payload,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        # LiteLLM normalises response shape to OpenAI-style; pull the
        # embedding from the first (and only) data element.
        data = getattr(response, "data", None) or response.get("data")  # type: ignore[union-attr]
        if not data:
            raise EmbeddingError("multimodal embedding response missing data field")
        first = data[0]
        embedding = getattr(first, "embedding", None)
        if embedding is None and isinstance(first, dict):
            embedding = first.get("embedding")
        if embedding is None:
            raise EmbeddingError("multimodal embedding response missing embedding field")
        return list(embedding)

    def _embed_batch_with_indices(self, batch: Sequence[str], start_idx: int) -> List[Tuple[int, List[float]]]:
        """Process a batch of texts and return embeddings with their original indices."""
        try:
            embeddings = self._embed_batch(batch)
            # Return each embedding with its corresponding index in the original list
            return [(start_idx + i, embedding) for i, embedding in enumerate(embeddings)]
        except Exception as e:
            logger.error(f"Batch embedding with indices failed: {str(e)}")
            # Convert litellm errors for batch processing
            raise wrap_litellm_error(e, "embedding", self.embedding_provider, self.model) from e

    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        if not self.caching:
            return self._embed_batch_uncached(batch)

        cache = get_sync_application_cache()

        def compute_missing(missing_items: list[str]) -> dict[str, List[float]]:
            embeddings = self._embed_batch_uncached(missing_items)
            return dict(zip(missing_items, embeddings))

        return cache.get_many_or_compute_missing(
            namespace=NAMESPACE_EMBEDDING,
            items=list(batch),
            key_data_for_item=self._cache_key_for_input,
            compute_missing=compute_missing,
            policy=application_cache_policy(NAMESPACE_EMBEDDING),
        )

    def _embed_batch_uncached(self, batch: Sequence[str]) -> List[List[float]]:
        """
        Embed a batch of contents using litellm.

        Args:
            batch: Sequence of contents to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding fails
        """

        try:
            response = litellm.embedding(
                custom_llm_provider=self.embedding_provider,
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                input=list(batch),
                caching=False,
                # Pin ``encoding_format="float"`` so LiteLLM does not forward
                # an OpenAI-default that Alibaba DashScope's
                # ``compatible-mode/v1/embeddings`` rejects with
                # ``'encoding_format' only support with [float, base64]``
                # (observed as ``litellm.BadRequestError`` 400 in task #11
                # Phase B). ``"float"`` is accepted by every OpenAI-compat
                # provider we call here and matches the wire shape
                # ``response["data"][i]["embedding"]`` we already consume.
                # See task #15.
                encoding_format="float",
            )

            if not response or "data" not in response:
                raise EmbeddingError(
                    "Invalid response format from embedding API",
                    {"provider": self.embedding_provider, "model": self.model, "batch_size": len(batch)},
                )

            embeddings = [item["embedding"] for item in response["data"]]

            # Validate embedding dimensions
            if embeddings and len(set(len(emb) for emb in embeddings)) > 1:
                dimensions = [len(emb) for emb in embeddings]
                logger.warning(f"Inconsistent embedding dimensions: {set(dimensions)}")

            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding API call failed: {str(e)}")
            # Convert litellm errors to our custom types
            raise wrap_litellm_error(e, "embedding", self.embedding_provider, self.model) from e

    def _cache_key_for_input(self, text: str) -> dict:
        return {
            "provider": self.embedding_provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_hash": hashlib.sha256((self.api_key or "").encode("utf-8")).hexdigest(),
            "input": text,
            "multimodal": self.multimodal,
            "encoding_format": "float",
        }

    def _cache_key_for_image(self, image_bytes: bytes, alt_text: str) -> dict:
        """Cache key shape for ``embed_image`` (Wave 6 task #37).

        Per cache README the namespaced key never embeds raw bytes — the
        image is identified by ``sha256(image_bytes)`` so the Redis key
        stays bounded. ``alt_text`` is part of the key because providers
        that accept paired text+image inputs return a different vector
        when the textual hint changes; ``alt_text=""`` collapses to the
        same key for image-only callers.
        """

        return {
            "kind": "image",
            "provider": self.embedding_provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_hash": hashlib.sha256((self.api_key or "").encode("utf-8")).hexdigest(),
            "file_hash": hashlib.sha256(image_bytes).hexdigest(),
            "alt_text": alt_text or "",
            "multimodal": True,
        }
