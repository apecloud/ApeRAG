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

"""
LLM Error Types

This module defines a comprehensive hierarchy of exceptions for LLM operations,
including completion, embedding, and rerank services.
"""

from typing import Any, Dict, Optional


class LLMError(Exception):
    """Base exception for all LLM-related errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


# Configuration-related errors
class LLMConfigurationError(LLMError):
    """Base class for configuration-related errors"""

    pass


class ProviderNotFoundError(LLMConfigurationError):
    """Raised when the specified LLM provider is not found or not configured"""

    def __init__(self, provider_name: str, service_type: str = "LLM"):
        message = f"{service_type} provider '{provider_name}' not found or not configured"
        super().__init__(message, {"provider_name": provider_name, "service_type": service_type})
        self.provider_name = provider_name
        self.service_type = service_type


class ModelNotFoundError(LLMConfigurationError):
    """Raised when the specified model is not found or not available"""

    def __init__(self, model_name: str, provider_name: str = None, service_type: str = "LLM"):
        message = f"{service_type} model '{model_name}' not found"
        if provider_name:
            message += f" for provider '{provider_name}'"
        super().__init__(
            message, {"model_name": model_name, "provider_name": provider_name, "service_type": service_type}
        )
        self.model_name = model_name
        self.provider_name = provider_name
        self.service_type = service_type


class InvalidConfigurationError(LLMConfigurationError):
    """Raised when configuration parameters are invalid"""

    def __init__(self, config_field: str, config_value: Any = None, reason: str = "Invalid configuration"):
        message = f"Invalid configuration for '{config_field}': {reason}"
        super().__init__(message, {"config_field": config_field, "config_value": config_value, "reason": reason})
        self.config_field = config_field
        self.config_value = config_value
        self.reason = reason


# API-related errors
class LLMAPIError(LLMError):
    """Base class for API-related errors"""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.status_code = status_code


class AuthenticationError(LLMAPIError):
    """Raised when API authentication fails"""

    def __init__(self, provider_name: str = None, details: Optional[Dict[str, Any]] = None):
        message = "Authentication failed"
        if provider_name:
            message += f" for provider '{provider_name}'"
        super().__init__(message, 401, details)
        self.provider_name = provider_name


class RateLimitError(LLMAPIError):
    """Raised when API rate limit is exceeded"""

    def __init__(
        self, provider_name: str = None, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None
    ):
        message = "API rate limit exceeded"
        if provider_name:
            message += f" for provider '{provider_name}'"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, 429, details)
        self.provider_name = provider_name
        self.retry_after = retry_after


class TimeoutError(LLMAPIError):
    """Raised when API request times out"""

    def __init__(self, timeout_seconds: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        message = "API request timed out"
        if timeout_seconds:
            message += f" after {timeout_seconds} seconds"
        super().__init__(message, 408, details)
        self.timeout_seconds = timeout_seconds


class QuotaExceededError(LLMAPIError):
    """Raised when API quota is exceeded"""

    def __init__(
        self, provider_name: str = None, quota_type: str = "requests", details: Optional[Dict[str, Any]] = None
    ):
        message = f"API {quota_type} quota exceeded"
        if provider_name:
            message += f" for provider '{provider_name}'"
        super().__init__(message, 429, details)
        self.provider_name = provider_name
        self.quota_type = quota_type


class ServerError(LLMAPIError):
    """Raised when the API server returns an error"""

    def __init__(self, status_code: int, provider_name: str = None, details: Optional[Dict[str, Any]] = None):
        message = f"Server error (HTTP {status_code})"
        if provider_name:
            message += f" from provider '{provider_name}'"
        super().__init__(message, status_code, details)
        self.provider_name = provider_name


# Completion-specific errors
class CompletionError(LLMError):
    """Base class for completion-specific errors"""

    pass


class InvalidPromptError(CompletionError):
    """Raised when the prompt is invalid"""

    def __init__(self, reason: str = "Invalid prompt format or content", prompt_preview: str = None):
        message = f"Invalid prompt: {reason}"
        details = {"reason": reason}
        if prompt_preview:
            details["prompt_preview"] = prompt_preview[:200] + "..." if len(prompt_preview) > 200 else prompt_preview
        super().__init__(message, details)
        self.reason = reason


class ResponseParsingError(CompletionError):
    """Raised when the completion response cannot be parsed"""

    def __init__(self, reason: str = "Failed to parse completion response", response_preview: str = None):
        message = f"Response parsing error: {reason}"
        details = {"reason": reason}
        if response_preview:
            details["response_preview"] = (
                response_preview[:200] + "..." if len(response_preview) > 200 else response_preview
            )
        super().__init__(message, details)
        self.reason = reason


class ToolCallError(CompletionError):
    """Raised when tool calling fails"""

    def __init__(self, tool_name: str = None, reason: str = "Tool call failed"):
        message = f"Tool call error: {reason}"
        if tool_name:
            message += f" for tool '{tool_name}'"
        super().__init__(message, {"tool_name": tool_name, "reason": reason})
        self.tool_name = tool_name
        self.reason = reason


# Embedding-specific errors
class EmbeddingError(LLMError):
    """Base class for embedding-specific errors"""

    pass


class TextTooLongError(EmbeddingError):
    """Raised when input text exceeds the model's token limit"""

    def __init__(self, text_length: Optional[int] = None, max_length: Optional[int] = None, model_name: str = None):
        if text_length is not None and max_length is not None:
            message = f"Text too long: {text_length} tokens exceeds maximum {max_length} tokens"
        else:
            message = "Text too long: input text exceeds model's token limit"
        if model_name:
            message += f" for model '{model_name}'"
        super().__init__(message, {"text_length": text_length, "max_length": max_length, "model_name": model_name})
        self.text_length = text_length
        self.max_length = max_length
        self.model_name = model_name


class EmptyTextError(EmbeddingError):
    """Raised when trying to embed empty or whitespace-only text"""

    def __init__(self, text_count: int = 1):
        message = "Cannot embed empty text"
        if text_count > 1:
            message += f" (found {text_count} empty texts)"
        super().__init__(message, {"text_count": text_count})
        self.text_count = text_count


class DimensionMismatchError(EmbeddingError):
    """Raised when embedding dimensions don't match expected values"""

    def __init__(self, expected_dim: int, actual_dim: int, model_name: str = None):
        message = f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
        if model_name:
            message += f" for model '{model_name}'"
        super().__init__(message, {"expected_dim": expected_dim, "actual_dim": actual_dim, "model_name": model_name})
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim
        self.model_name = model_name


class BatchProcessingError(EmbeddingError):
    """Raised when batch processing of embeddings fails"""

    def __init__(self, batch_size: int, failed_indices: list = None, reason: str = "Batch processing failed"):
        message = f"Batch processing error (batch size: {batch_size}): {reason}"
        details = {"batch_size": batch_size, "reason": reason}
        if failed_indices:
            details["failed_indices"] = failed_indices
        super().__init__(message, details)
        self.batch_size = batch_size
        self.failed_indices = failed_indices or []
        self.reason = reason


# Rerank-specific errors
class RerankError(LLMError):
    """Base class for rerank-specific errors"""

    pass


class InvalidDocumentError(RerankError):
    """Raised when documents for reranking are invalid"""

    def __init__(self, reason: str = "Invalid document format", document_count: int = None):
        message = f"Invalid documents for reranking: {reason}"
        if document_count is not None:
            message += f" (document count: {document_count})"
        super().__init__(message, {"reason": reason, "document_count": document_count})
        self.reason = reason
        self.document_count = document_count


class TooManyDocumentsError(RerankError):
    """Raised when too many documents are provided for reranking"""

    def __init__(
        self, document_count: Optional[int] = None, max_documents: Optional[int] = None, model_name: str = None
    ):
        if document_count is not None and max_documents is not None:
            message = f"Too many documents for reranking: {document_count} exceeds maximum {max_documents}"
        else:
            message = "Too many documents for reranking: document count exceeds model's limit"
        if model_name:
            message += f" for model '{model_name}'"
        super().__init__(
            message, {"document_count": document_count, "max_documents": max_documents, "model_name": model_name}
        )
        self.document_count = document_count
        self.max_documents = max_documents
        self.model_name = model_name


class ScoreOutOfRangeError(RerankError):
    """Raised when rerank scores are out of expected range"""

    def __init__(self, score: float, expected_range: tuple = (0.0, 1.0), model_name: str = None):
        message = f"Rerank score {score} is out of expected range {expected_range}"
        if model_name:
            message += f" for model '{model_name}'"
        super().__init__(message, {"score": score, "expected_range": expected_range, "model_name": model_name})
        self.score = score
        self.expected_range = expected_range
        self.model_name = model_name


# Utility functions for error handling
def wrap_litellm_error(
    e: Exception, service_type: str = "LLM", provider_name: str = None, model_name: str = None
) -> LLMError:
    """
    Convert litellm exceptions to our custom exception types

    Args:
        e: The original exception from litellm
        service_type: Type of service ("completion", "embedding", "rerank", etc.)
        provider_name: Name of the LLM provider
        model_name: Name of the model

    Returns:
        LLMError: Appropriate custom exception
    """
    error_msg = str(e).lower()

    # Authentication errors
    if any(keyword in error_msg for keyword in ["unauthorized", "authentication", "invalid api key", "api key"]):
        return AuthenticationError(provider_name, {"original_error": str(e)})

    # Rate limiting errors
    if any(keyword in error_msg for keyword in ["rate limit", "too many requests", "quota"]):
        if "quota" in error_msg:
            return QuotaExceededError(provider_name, details={"original_error": str(e)})
        return RateLimitError(provider_name, details={"original_error": str(e)})

    # Timeout errors
    if any(keyword in error_msg for keyword in ["timeout", "timed out", "connection timeout"]):
        return TimeoutError(details={"original_error": str(e)})

    # Model/Provider not found errors
    if any(keyword in error_msg for keyword in ["model not found", "invalid model", "model does not exist"]):
        return ModelNotFoundError(model_name or "unknown", provider_name, service_type)

    if any(keyword in error_msg for keyword in ["provider not found", "invalid provider", "unknown provider"]):
        return ProviderNotFoundError(provider_name or "unknown", service_type)

    # Server errors (5xx status codes)
    if any(keyword in error_msg for keyword in ["internal server error", "server error", "503", "502", "500"]):
        status_code = 500
        for code in [500, 502, 503, 504]:
            if str(code) in error_msg:
                status_code = code
                break
        return ServerError(status_code, provider_name, {"original_error": str(e)})

    # Service-specific errors
    if service_type == "embedding":
        if any(keyword in error_msg for keyword in ["text too long", "token limit", "input too large"]):
            return TextTooLongError(None, None, model_name)  # Pass None when exact lengths are unavailable
        if any(keyword in error_msg for keyword in ["empty text", "no input"]):
            return EmptyTextError()

    elif service_type == "rerank":
        if any(keyword in error_msg for keyword in ["too many documents", "document limit"]):
            return TooManyDocumentsError(None, None, model_name)  # Pass None when exact counts are unavailable
        if any(keyword in error_msg for keyword in ["invalid document", "document format"]):
            return InvalidDocumentError("Invalid document format detected")

    elif service_type == "completion":
        if any(keyword in error_msg for keyword in ["invalid prompt", "prompt format"]):
            return InvalidPromptError("Invalid prompt format detected")

    # Default to base API error
    return LLMAPIError(f"{service_type.title()} API error: {str(e)}", details={"original_error": str(e)})


def is_retryable_error(error: LLMError) -> bool:
    """
    Determine if an error is retryable

    Args:
        error: The LLM error to check

    Returns:
        bool: True if the error is retryable, False otherwise
    """
    # Configuration errors are not retryable
    if isinstance(error, LLMConfigurationError):
        return False

    # Input validation errors are not retryable
    if isinstance(
        error, (InvalidPromptError, TextTooLongError, EmptyTextError, InvalidDocumentError, TooManyDocumentsError)
    ):
        return False

    # API errors that might be temporary
    if isinstance(error, (RateLimitError, TimeoutError, ServerError)):
        return True

    # Authentication and quota errors are not retryable
    if isinstance(error, (AuthenticationError, QuotaExceededError)):
        return False

    # Default to not retryable
    return False
