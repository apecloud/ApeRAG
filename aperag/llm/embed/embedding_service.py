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
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple
from urllib.parse import urlparse

import litellm
import requests
from litellm.utils import is_base64_encoded

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
        multimodal: bool = False,
        caching: bool = True,
    ):
        self.embedding_provider = embedding_provider
        self.model = embedding_model
        self.api_base = embedding_service_url
        self.api_key = embedding_service_api_key
        self.max_chunks = embedding_max_chunks_in_batch
        self.max_workers = 8
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
                        reason=f"Failed to process {len(failed_batches)} batches: {failed_batches[:3]}",
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

    def _is_valid_url(self, url: str) -> bool:
        """Check if a string is a valid URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

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
        """
        Embed a batch of contents using litellm.

        Args:
            batch: Sequence of contents to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding fails
        """

        # HACK: litellm currently doesn't support jina's multimodal embedding API,
        #       it doesn't send the image content in the required format. Working around it for now.
        if self.multimodal and self.embedding_provider == "jina_ai":
            if any([is_base64_encoded(item) for item in batch]):
                return self._jina_ai_multimodal_embedding(batch)

        try:
            response = litellm.embedding(
                custom_llm_provider=self.embedding_provider,
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                input=list(batch),
                caching=self.caching,
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

    def _jina_ai_multimodal_embedding(self, batch: Sequence[str]) -> List[List[float]]:
        """
        Embed a batch of images using Jina AI multimodal embedding API.
        """
        api_url = "https://api.jina.ai/v1/embeddings"
        if self.api_base:
            api_url = self.api_base + "/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        input_list = []
        for item in batch:
            is_url = self._is_valid_url(item)
            base64_encoded = is_base64_encoded(item)
            if is_url or base64_encoded:
                if base64_encoded:
                    item = item.split(",")[1]
                input_list.append({"image": item})
            else:
                input_list.append({"text": item})
        payload = {
            "model": self.model,
            "input": input_list,
        }

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=180)
            response.raise_for_status()
            response_json = response.json()

            if not response_json or "data" not in response_json:
                raise EmbeddingError(
                    "Invalid response format from Jina AI embedding API",
                    {"provider": self.embedding_provider, "model": self.model, "batch_size": len(batch)},
                )

            embeddings = [item["embedding"] for item in response_json["data"]]
            return embeddings
        except requests.exceptions.RequestException as e:
            logger.error(f"Jina AI API request failed: {e}")
            raise EmbeddingError(
                f"Jina AI API request failed: {e}",
                {"provider": self.embedding_provider, "model": self.model},
            ) from e
