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
# Export all LLM error types for easy access
# Export services
from aperag.llm.completion.completion_service import CompletionService
from aperag.llm.embed.embedding_service import EmbeddingService
from aperag.llm.llm_error_types import (
    # API errors
    AuthenticationError,
    BatchProcessingError,
    # Service-specific errors
    CompletionError,
    DimensionMismatchError,
    EmbeddingError,
    EmptyTextError,
    InvalidConfigurationError,
    InvalidDocumentError,
    InvalidPromptError,
    LLMAPIError,
    LLMConfigurationError,
    # Base error classes
    LLMError,
    ModelNotFoundError,
    # Configuration errors
    ProviderNotFoundError,
    QuotaExceededError,
    RateLimitError,
    RerankError,
    ResponseParsingError,
    ScoreOutOfRangeError,
    ServerError,
    TextTooLongError,
    TimeoutError,
    ToolCallError,
    TooManyDocumentsError,
    is_retryable_error,
    # Utility functions
    wrap_litellm_error,
)
from aperag.llm.rerank.rerank_service import RerankService

__all__ = [
    # Error classes
    "LLMError",
    "LLMConfigurationError",
    "LLMAPIError",
    "ProviderNotFoundError",
    "ModelNotFoundError",
    "InvalidConfigurationError",
    "AuthenticationError",
    "RateLimitError",
    "TimeoutError",
    "QuotaExceededError",
    "ServerError",
    "CompletionError",
    "InvalidPromptError",
    "ResponseParsingError",
    "ToolCallError",
    "EmbeddingError",
    "TextTooLongError",
    "EmptyTextError",
    "DimensionMismatchError",
    "BatchProcessingError",
    "RerankError",
    "InvalidDocumentError",
    "TooManyDocumentsError",
    "ScoreOutOfRangeError",
    "wrap_litellm_error",
    "is_retryable_error",
    # Services
    "CompletionService",
    "EmbeddingService",
    "RerankService",
]

# Initialize cache when module is imported
from aperag.llm import litellm_cache, litellm_logging

litellm_cache.setup_litellm_cache()
litellm_logging.setup_litellm_logging()
