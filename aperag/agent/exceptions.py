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

"""Agent-specific exceptions and error handling utilities."""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from aperag.exceptions import BusinessException, ErrorCode

logger = logging.getLogger(__name__)

# Type variable for retry decorator
T = TypeVar("T")


class AgentError(BusinessException):
    """Base exception for all agent-related errors."""

    def __init__(
        self,
        error_code: ErrorCode = ErrorCode.AGENT_ERROR,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ):
        super().__init__(error_code, message, details)
        self.retryable = retryable


class MCPConnectionError(AgentError):
    """MCP server connection error with automatic retry support."""

    def __init__(
        self,
        server_name: str = "unknown",
        url: Optional[str] = None,
        underlying_error: Optional[Exception] = None,
    ):
        details = {"server_name": server_name}
        if url:
            details["url"] = url
        if underlying_error:
            details["underlying_error"] = str(underlying_error)

        message = f"Failed to connect to MCP server '{server_name}'"
        if url:
            message += f" at {url}"

        super().__init__(
            error_code=ErrorCode.MCP_CONNECTION_ERROR,
            message=message,
            details=details,
            retryable=True,
        )


class MCPAppInitializationError(AgentError):
    """MCP application initialization error."""

    def __init__(
        self,
        reason: str,
        config_details: Optional[Dict[str, Any]] = None,
        underlying_error: Optional[Exception] = None,
    ):
        details = {"reason": reason}
        if config_details:
            details["config"] = config_details
        if underlying_error:
            details["underlying_error"] = str(underlying_error)

        super().__init__(
            error_code=ErrorCode.MCP_APP_INIT_ERROR,
            message=f"MCP application initialization failed: {reason}",
            details=details,
            retryable=False,
        )


class ToolExecutionError(AgentError):
    """Tool execution error with context information."""

    def __init__(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        underlying_error: Optional[Exception] = None,
        retryable: bool = True,
    ):
        details = {"tool_name": tool_name}
        if tool_args:
            details["tool_args"] = tool_args
        if underlying_error:
            details["underlying_error"] = str(underlying_error)

        message = f"Tool execution failed: {tool_name}"
        if error_message:
            message += f" - {error_message}"

        super().__init__(
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            message=message,
            details=details,
            retryable=retryable,
        )


class EventListenerError(AgentError):
    """Event listener error."""

    def __init__(
        self,
        event_type: str,
        error_message: str,
        event_data: Optional[Dict[str, Any]] = None,
        underlying_error: Optional[Exception] = None,
    ):
        details = {"event_type": event_type, "error_message": error_message}
        if event_data:
            details["event_data"] = event_data
        if underlying_error:
            details["underlying_error"] = str(underlying_error)

        super().__init__(
            error_code=ErrorCode.EVENT_LISTENER_ERROR,
            message=f"Event listener error for {event_type}: {error_message}",
            details=details,
            retryable=False,
        )


class StreamFormattingError(AgentError):
    """Stream formatting error."""

    def __init__(
        self,
        formatter_type: str,
        content: Optional[Any] = None,
        underlying_error: Optional[Exception] = None,
    ):
        details = {"formatter_type": formatter_type}
        if content is not None:
            details["content"] = str(content)[:500]  # Limit content size in error details
        if underlying_error:
            details["underlying_error"] = str(underlying_error)

        super().__init__(
            error_code=ErrorCode.STREAM_FORMATTING_ERROR,
            message=f"Stream formatting failed: {formatter_type}",
            details=details,
            retryable=False,
        )


class AgentConfigurationError(AgentError):
    """Agent configuration error."""

    def __init__(
        self,
        config_key: str,
        reason: str,
        provided_value: Optional[Any] = None,
    ):
        details = {"config_key": config_key, "reason": reason}
        if provided_value is not None:
            details["provided_value"] = str(provided_value)

        super().__init__(
            error_code=ErrorCode.AGENT_CONFIG_ERROR,
            message=f"Invalid agent configuration for '{config_key}': {reason}",
            details=details,
            retryable=False,
        )


# Alias for backward compatibility
AgentConfigError = AgentConfigurationError


class ToolReferenceExtractionError(AgentError):
    """Tool reference extraction error."""

    def __init__(
        self,
        extraction_step: str,
        tool_call_id: Optional[str] = None,
        underlying_error: Optional[Exception] = None,
    ):
        details = {"extraction_step": extraction_step}
        if tool_call_id:
            details["tool_call_id"] = tool_call_id
        if underlying_error:
            details["underlying_error"] = str(underlying_error)

        super().__init__(
            error_code=ErrorCode.TOOL_REFERENCE_EXTRACTION_ERROR,
            message=f"Tool reference extraction failed at step: {extraction_step}",
            details=details,
            retryable=False,
        )


class JSONParsingError(AgentError):
    """JSON parsing error with enhanced context."""

    def __init__(
        self,
        content: str,
        parsing_context: str,
        underlying_error: Optional[Exception] = None,
    ):
        details = {
            "parsing_context": parsing_context,
            "content_preview": content[:200] if content else "None",  # First 200 chars
            "content_length": len(content) if content else 0,
        }
        if underlying_error:
            details["underlying_error"] = str(underlying_error)

        super().__init__(
            error_code=ErrorCode.JSON_PARSING_ERROR,
            message=f"JSON parsing failed in {parsing_context}",
            details=details,
            retryable=False,
        )


class AgentTimeoutError(AgentError):
    """Agent operation timeout error."""

    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        elapsed_seconds: Optional[float] = None,
    ):
        details = {"operation": operation, "timeout_seconds": timeout_seconds}
        if elapsed_seconds is not None:
            details["elapsed_seconds"] = elapsed_seconds

        super().__init__(
            error_code=ErrorCode.AGENT_TIMEOUT_ERROR,
            message=f"Agent operation '{operation}' timed out after {timeout_seconds}s",
            details=details,
            retryable=True,
        )


# Utility functions for error handling and retry logic


def safe_json_parse(content: str, context: str = "unknown") -> Dict[str, Any]:
    """
    Safely parse JSON content with enhanced error handling.

    Args:
        content: JSON string content to parse
        context: Context description for error reporting

    Returns:
        Parsed JSON data as dictionary

    Raises:
        JSONParsingError: If parsing fails
    """
    if not content:
        raise JSONParsingError("", context, ValueError("Empty content"))

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise JSONParsingError(content, context, e) from e
    except Exception as e:
        raise JSONParsingError(content, context, e) from e


def extract_tool_result_data(tool_result: Union[str, Dict[str, Any]], context: str = "tool_result") -> Dict[str, Any]:
    """
    Extract and parse tool result data with common patterns handling.

    Args:
        tool_result: Tool result data (string or dict)
        context: Context for error reporting

    Returns:
        Parsed tool result data

    Raises:
        JSONParsingError: If parsing fails
        ToolReferenceExtractionError: If extraction fails
    """
    try:
        # Handle string input
        if isinstance(tool_result, str):
            result_data = safe_json_parse(tool_result, f"{context}_string_parse")
        else:
            result_data = tool_result

        # Handle array format where data is in first element's text field
        if isinstance(result_data, list) and len(result_data) > 0:
            first_item = result_data[0]
            if isinstance(first_item, dict) and "text" in first_item:
                try:
                    return safe_json_parse(first_item["text"], f"{context}_text_field_parse")
                except JSONParsingError:
                    logger.warning(f"Failed to parse text field as JSON in {context}, using raw text")
                    return {"raw_text": first_item["text"]}

        return result_data

    except JSONParsingError:
        raise  # Re-raise JSON parsing errors as-is
    except Exception as e:
        raise ToolReferenceExtractionError("data_extraction", context, e) from e


async def with_retry(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (MCPConnectionError, AgentTimeoutError),
    *args,
    **kwargs,
) -> T:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exception types that should trigger retry
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Function execution result

    Raises:
        Last exception if all retries are exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                break

            # Calculate delay with exponential backoff
            delay = min(base_delay * (exponential_base**attempt), max_delay)
            logger.warning(
                f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            if asyncio.iscoroutinefunction(func):
                await asyncio.sleep(delay)
            else:
                import time

                time.sleep(delay)

        except Exception as e:
            # Non-retryable exception, fail immediately
            logger.error(f"Function {func.__name__} failed with non-retryable error: {e}")
            raise

    # If we get here, all retries were exhausted
    raise last_exception


def handle_agent_error(
    operation: str,
    default_return: Optional[Any] = None,
    log_level: str = "error",
    reraise: bool = True,
):
    """
    Decorator for handling agent errors with consistent logging and optional default returns.

    Args:
        operation: Description of the operation being performed
        default_return: Default value to return if error occurs and reraise=False
        log_level: Logging level for error messages
        reraise: Whether to re-raise the exception after logging

    Returns:
        Decorator function
    """

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except AgentError as e:
                getattr(logger, log_level)(
                    f"Agent error in {operation}: {e.message}",
                    extra={"error_code": e.error_code.error_name, "details": e.details},
                )
                if reraise:
                    raise
                return default_return
            except Exception as e:
                getattr(logger, log_level)(f"Unexpected error in {operation}: {e}")
                if reraise:
                    raise AgentError(message=f"Unexpected error in {operation}: {str(e)}")
                return default_return

        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AgentError as e:
                getattr(logger, log_level)(
                    f"Agent error in {operation}: {e.message}",
                    extra={"error_code": e.error_code.error_name, "details": e.details},
                )
                if reraise:
                    raise
                return default_return
            except Exception as e:
                getattr(logger, log_level)(f"Unexpected error in {operation}: {e}")
                if reraise:
                    raise AgentError(message=f"Unexpected error in {operation}: {str(e)}")
                return default_return

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Convenience functions for creating common agent exceptions


def mcp_connection_failed(server_name: str, url: Optional[str] = None, error: Optional[Exception] = None):
    """Create MCP connection error."""
    return MCPConnectionError(server_name, url, error)


def mcp_init_failed(reason: str, config: Optional[Dict[str, Any]] = None, error: Optional[Exception] = None):
    """Create MCP initialization error."""
    return MCPAppInitializationError(reason, config, error)


def tool_execution_failed(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
    error: Optional[Exception] = None,
    retryable: bool = True,
):
    """Create tool execution error."""
    return ToolExecutionError(tool_name, args, message, error, retryable)


def agent_config_invalid(key: str, reason: str, value: Optional[Any] = None):
    """Create agent configuration error."""
    return AgentConfigError(key, reason, value)


def agent_timeout(operation: str, timeout: float, elapsed: Optional[float] = None):
    """Create agent timeout error."""
    return AgentTimeoutError(operation, timeout, elapsed)
