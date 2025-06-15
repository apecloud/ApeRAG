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

import logging
from typing import List

import litellm

from aperag.llm.llm_error_types import (
    InvalidDocumentError,
    RerankError,
    TooManyDocumentsError,
    wrap_litellm_error,
)
from aperag.query.query import DocumentWithScore

logger = logging.getLogger(__name__)


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

        # Set document limit based on provider
        self.max_documents = 1000

    async def rank(self, query: str, results: List[DocumentWithScore]) -> List[DocumentWithScore]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query
            results: List of documents with scores to rerank

        Returns:
            List of documents reordered by relevance

        Raises:
            InvalidDocumentError: If documents are invalid
            TooManyDocumentsError: If too many documents provided
            RerankError: If reranking fails
        """
        try:
            # Validate inputs
            if not query or not query.strip():
                raise InvalidDocumentError("Query cannot be empty")

            if not results:
                logger.info("No documents to rerank, returning empty list")
                return []

            # Check document count limits
            if len(results) > self.max_documents:
                raise TooManyDocumentsError(
                    document_count=len(results), max_documents=self.max_documents, model_name=self.model
                )

            # Validate documents
            documents = []
            invalid_indices = []
            for i, doc in enumerate(results):
                if not doc or not hasattr(doc, "text") or not doc.text or not doc.text.strip():
                    invalid_indices.append(i)
                    documents.append(" ")  # Use placeholder for empty docs
                else:
                    documents.append(doc.text)

            if invalid_indices:
                logger.warning(f"Found {len(invalid_indices)} invalid documents at indices: {invalid_indices}")
                if len(invalid_indices) == len(results):
                    raise InvalidDocumentError("All documents are empty or invalid", document_count=len(results))

            # Call litellm rerank API
            resp = await litellm.arerank(
                custom_llm_provider=self.rerank_provider,
                model=self.model,
                query=query,
                documents=documents,
                api_key=self.api_key,
                api_base=self.api_base,
                return_documents=False,
            )

            # Validate response
            if not resp or "results" not in resp:
                raise RerankError(
                    "Invalid response format from rerank API",
                    {"provider": self.rerank_provider, "model": self.model, "document_count": len(documents)},
                )

            # Reorder documents based on the returned indices
            try:
                indices = [item["index"] for item in resp["results"]]

                # Validate indices
                if len(indices) != len(results):
                    logger.warning(f"Rerank returned {len(indices)} indices for {len(results)} documents")

                # Check for invalid indices
                invalid_rerank_indices = [idx for idx in indices if idx < 0 or idx >= len(results)]
                if invalid_rerank_indices:
                    raise RerankError(
                        f"Invalid rerank indices: {invalid_rerank_indices}",
                        {
                            "provider": self.rerank_provider,
                            "model": self.model,
                            "invalid_indices": invalid_rerank_indices,
                        },
                    )

                # Reorder results
                reranked_results = [results[i] for i in indices if 0 <= i < len(results)]

                logger.info(f"Successfully reranked {len(reranked_results)} documents")
                return reranked_results

            except (KeyError, IndexError, TypeError) as e:
                raise RerankError(
                    f"Failed to parse rerank response: {str(e)}",
                    {
                        "provider": self.rerank_provider,
                        "model": self.model,
                        "response_keys": list(resp.keys()) if isinstance(resp, dict) else "non-dict",
                    },
                ) from e

        except (InvalidDocumentError, TooManyDocumentsError, RerankError):
            # Re-raise our custom rerank errors
            raise
        except Exception as e:
            logger.error(f"Rerank operation failed: {str(e)}")
            # Convert litellm errors to our custom types
            raise wrap_litellm_error(e, "rerank", self.rerank_provider, self.model) from e

    def validate_configuration(self) -> None:
        """
        Validate the rerank service configuration.

        Raises:
            InvalidConfigurationError: If configuration is invalid
        """
        from aperag.llm.llm_error_types import InvalidConfigurationError

        if not self.rerank_provider:
            raise InvalidConfigurationError("rerank_provider", self.rerank_provider, "Provider cannot be empty")

        if not self.model:
            raise InvalidConfigurationError("model", self.model, "Model name cannot be empty")

        if not self.api_key:
            raise InvalidConfigurationError("api_key", None, "API key cannot be empty")

        if not self.api_base:
            raise InvalidConfigurationError("api_base", self.api_base, "API base URL cannot be empty")
