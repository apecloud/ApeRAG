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
"""LLM runtime package.

Keep package import side effects minimal. Individual runners import the
legacy LiteLLM-backed services only when that runner is selected.
"""

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
]
