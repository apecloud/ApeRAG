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

from typing import List

import litellm

from aperag.query.query import DocumentWithScore


class RerankService:
    """
    Rerank service that supports dynamic model configuration.
    Similar to EmbeddingService, this service accepts provider and model information
    at runtime instead of relying on environment variables.
    """

    def __init__(
        self,
        rerank_provider: str,
        rerank_model: str,
        rerank_service_url: str,
        rerank_service_api_key: str,
    ):
        """
        Initialize the rerank service with dynamic configuration.

        Args:
            rerank_provider: The custom LLM provider (e.g., 'jina_ai', 'openai')
            rerank_model: The model name to use for reranking
            rerank_service_url: The API base URL for the rerank service
            rerank_service_api_key: The API key for authentication
        """
        self.rerank_provider = rerank_provider
        self.model = rerank_model
        self.api_base = rerank_service_url
        self.api_key = rerank_service_api_key

    async def rank(self, query: str, results: List[DocumentWithScore]) -> List[DocumentWithScore]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query
            results: List of documents with scores to rerank

        Returns:
            List of documents reordered by relevance
        """
        try:
            if not results:
                return []

            documents = [d.text for d in results]

            resp = await litellm.arerank(
                custom_llm_provider=self.rerank_provider,
                model=self.model,
                query=query,
                documents=documents,
                api_key=self.api_key,
                api_base=self.api_base,
                return_documents=False,
            )

            # Reorder documents based on the returned indices
            indices = [item["index"] for item in resp["results"]]
            return [results[i] for i in indices]
        except Exception as e:
            raise RuntimeError(
                f"Failed to rerank documents: {e}. "
                f"Provider: {self.rerank_provider}, Model: {self.model}, "
                f"API Base: {self.api_base}"
            ) from e
