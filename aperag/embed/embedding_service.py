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

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Sequence, Tuple

import litellm
from langchain.embeddings.base import Embeddings
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingService(Embeddings):
    def __init__(
        self,
        embedding_provider: str,
        embedding_model: str,
        embedding_service_url: str,
        embedding_service_api_key: str,
        embedding_max_chunks_in_batch: int,
    ):
        self.embedding_provider = embedding_provider
        self.model = f"{embedding_model}"
        self.api_base = embedding_service_url
        self.api_key = embedding_service_api_key
        self.max_chunks = embedding_max_chunks_in_batch
        self.max_workers = 8

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents in parallel batches.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors in the same order as input texts
        """
        # Clean texts by replacing newlines with spaces
        clean_texts = [t.replace("\n", " ") for t in texts]
        # Determine batch size (use max_chunks or process all at once if not set)
        chunk_size = self.max_chunks or len(clean_texts)

        # Store results with original indices to ensure correct ordering
        results_dict: Dict[int, List[float]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = []

            # Submit batches for processing with their starting indices
            for start in range(0, len(clean_texts), chunk_size):
                batch = clean_texts[start : start + chunk_size]
                # Pass both the batch and starting index to track position
                future = pool.submit(self._embed_batch_with_indices, batch, start)
                futures.append(future)

            # Process completed futures and store results by index
            for future in as_completed(futures):
                # Get results with their original indices
                batch_results = future.result()
                for idx, embedding in batch_results:
                    results_dict[idx] = embedding

        # Reconstruct the result list in the original order
        results = [results_dict[i] for i in range(len(clean_texts))]
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def _embed_batch_with_indices(self, batch: Sequence[str], start_idx: int) -> List[Tuple[int, List[float]]]:
        embeddings = self._embed_batch(batch)
        # Return each embedding with its corresponding index in the original list
        return [(start_idx + i, embedding) for i, embedding in enumerate(embeddings)]

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _embed_batch(self, batch: Sequence[str]) -> List[List[float]]:
        response = litellm.embedding(
            custom_llm_provider=self.embedding_provider,
            model=self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            input=list(batch),
        )
        return [item["embedding"] for item in response["data"]]
