# -*- coding: utf-8 -*-
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

from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, Optional


class ErrorCode(Enum):
    """Business error codes with HTTP status mapping"""

    # Generic errors (1000-1099)
    UNKNOWN_ERROR = ("UNKNOWN_ERROR", 1000, HTTPStatus.INTERNAL_SERVER_ERROR)
    INVALID_PARAMETER = ("INVALID_PARAMETER", 1001, HTTPStatus.BAD_REQUEST)
    RESOURCE_NOT_FOUND = ("RESOURCE_NOT_FOUND", 1002, HTTPStatus.NOT_FOUND)
    UNAUTHORIZED = ("UNAUTHORIZED", 1003, HTTPStatus.UNAUTHORIZED)
    FORBIDDEN = ("FORBIDDEN", 1004, HTTPStatus.FORBIDDEN)
    CONFLICT = ("CONFLICT", 1005, HTTPStatus.CONFLICT)
    DATABASE_ERROR = ("DATABASE_ERROR", 1050, HTTPStatus.INTERNAL_SERVER_ERROR)
    VALIDATION_ERROR = ("VALIDATION_ERROR", 1051, HTTPStatus.BAD_REQUEST)

    # Collection errors (1100-1199)
    COLLECTION_NOT_FOUND = ("COLLECTION_NOT_FOUND", 1101, HTTPStatus.NOT_FOUND)
    COLLECTION_INACTIVE = ("COLLECTION_INACTIVE", 1102, HTTPStatus.BAD_REQUEST)
    COLLECTION_QUOTA_EXCEEDED = ("COLLECTION_QUOTA_EXCEEDED", 1103, HTTPStatus.FORBIDDEN)
    COLLECTION_HAS_RELATED_BOTS = ("COLLECTION_HAS_RELATED_BOTS", 1104, HTTPStatus.BAD_REQUEST)
    INVALID_SOURCE_CONFIG = ("INVALID_SOURCE_CONFIG", 1105, HTTPStatus.BAD_REQUEST)

    # Bot errors (1200-1299)
    BOT_NOT_FOUND = ("BOT_NOT_FOUND", 1201, HTTPStatus.NOT_FOUND)
    BOT_QUOTA_EXCEEDED = ("BOT_QUOTA_EXCEEDED", 1202, HTTPStatus.FORBIDDEN)
    BOT_CONFIG_INVALID = ("BOT_CONFIG_INVALID", 1203, HTTPStatus.BAD_REQUEST)
    LLM_PROVIDER_NOT_FOUND = ("LLM_PROVIDER_NOT_FOUND", 1204, HTTPStatus.BAD_REQUEST)
    API_KEY_NOT_FOUND = ("API_KEY_NOT_FOUND", 1205, HTTPStatus.BAD_REQUEST)

    # Document errors (1300-1399)
    DOCUMENT_NOT_FOUND = ("DOCUMENT_NOT_FOUND", 1301, HTTPStatus.NOT_FOUND)
    DOCUMENT_QUOTA_EXCEEDED = ("DOCUMENT_QUOTA_EXCEEDED", 1302, HTTPStatus.FORBIDDEN)
    UNSUPPORTED_FILE_TYPE = ("UNSUPPORTED_FILE_TYPE", 1303, HTTPStatus.BAD_REQUEST)
    FILE_SIZE_TOO_LARGE = ("FILE_SIZE_TOO_LARGE", 1304, HTTPStatus.BAD_REQUEST)
    TOO_MANY_DOCUMENTS = ("TOO_MANY_DOCUMENTS", 1305, HTTPStatus.BAD_REQUEST)
    INVALID_DOCUMENT_CONFIG = ("INVALID_DOCUMENT_CONFIG", 1306, HTTPStatus.BAD_REQUEST)

    # Chat errors (1400-1499)
    CHAT_NOT_FOUND = ("CHAT_NOT_FOUND", 1401, HTTPStatus.NOT_FOUND)
    MESSAGE_NOT_FOUND = ("MESSAGE_NOT_FOUND", 1402, HTTPStatus.NOT_FOUND)

    # Flow errors (1500-1599)
    FLOW_CONFIG_NOT_FOUND = ("FLOW_CONFIG_NOT_FOUND", 1501, HTTPStatus.BAD_REQUEST)
    FLOW_EXECUTION_FAILED = ("FLOW_EXECUTION_FAILED", 1502, HTTPStatus.INTERNAL_SERVER_ERROR)

    # Auth errors (1600-1699)
    AUTH_TOKEN_INVALID = ("AUTH_TOKEN_INVALID", 1601, HTTPStatus.UNAUTHORIZED)
    AUTH_TOKEN_EXPIRED = ("AUTH_TOKEN_EXPIRED", 1602, HTTPStatus.UNAUTHORIZED)

    # Search errors (1700-1799)
    SEARCH_NOT_FOUND = ("SEARCH_NOT_FOUND", 1701, HTTPStatus.NOT_FOUND)

    # LLM Provider errors (1800-1899)
    LLM_PROVIDER_ALREADY_EXISTS = ("LLM_PROVIDER_ALREADY_EXISTS", 1801, HTTPStatus.CONFLICT)
    LLM_MODEL_NOT_FOUND = ("LLM_MODEL_NOT_FOUND", 1802, HTTPStatus.NOT_FOUND)
    PROVIDER_NOT_PUBLIC = ("PROVIDER_NOT_PUBLIC", 1803, HTTPStatus.BAD_REQUEST)

    # Graph errors (1900-1999)
    GRAPH_SERVICE_ERROR = ("GRAPH_SERVICE_ERROR", 1901, HTTPStatus.INTERNAL_SERVER_ERROR)

    # Agent errors (2000-2099)
    AGENT_ERROR = ("AGENT_ERROR", 2000, HTTPStatus.INTERNAL_SERVER_ERROR)
    MCP_CONNECTION_ERROR = ("MCP_CONNECTION_ERROR", 2001, HTTPStatus.SERVICE_UNAVAILABLE)
    MCP_APP_INIT_ERROR = ("MCP_APP_INIT_ERROR", 2002, HTTPStatus.INTERNAL_SERVER_ERROR)
    TOOL_EXECUTION_ERROR = ("TOOL_EXECUTION_ERROR", 2003, HTTPStatus.BAD_REQUEST)
    EVENT_LISTENER_ERROR = ("EVENT_LISTENER_ERROR", 2004, HTTPStatus.INTERNAL_SERVER_ERROR)
    STREAM_FORMATTING_ERROR = ("STREAM_FORMATTING_ERROR", 2005, HTTPStatus.INTERNAL_SERVER_ERROR)
    AGENT_CONFIG_ERROR = ("AGENT_CONFIG_ERROR", 2006, HTTPStatus.BAD_REQUEST)
    TOOL_REFERENCE_EXTRACTION_ERROR = ("TOOL_REFERENCE_EXTRACTION_ERROR", 2007, HTTPStatus.INTERNAL_SERVER_ERROR)
    JSON_PARSING_ERROR = ("JSON_PARSING_ERROR", 2008, HTTPStatus.BAD_REQUEST)
    AGENT_TIMEOUT_ERROR = ("AGENT_TIMEOUT_ERROR", 2009, HTTPStatus.REQUEST_TIMEOUT)

    # Future resources can use ranges:
    # 2100-2199: Reserved for future resource type 3
    # ... and so on

    def __init__(self, name: str, code: int, http_status: HTTPStatus):
        self.error_name = name
        self.code = code
        self.http_status = http_status


class BusinessException(Exception):
    """Base business exception class"""

    def __init__(self, error_code: ErrorCode, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.error_code = error_code
        self.message = message or error_code.error_name.replace("_", " ").title()
        self.details = details or {}
        super().__init__(self.message)

    @property
    def http_status(self) -> HTTPStatus:
        return self.error_code.http_status

    @property
    def code(self) -> int:
        return self.error_code.code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code.error_name,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


# Specific exception classes for better type safety and usage
class ResourceNotFoundException(BusinessException):
    """Resource not found exception"""

    def __init__(self, resource_type: str, resource_id: str = None):
        message = f"{resource_type} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(ErrorCode.RESOURCE_NOT_FOUND, message)


class CollectionNotFoundException(BusinessException):
    """Collection not found exception"""

    def __init__(self, collection_id: str):
        super().__init__(ErrorCode.COLLECTION_NOT_FOUND, f"Collection not found: {collection_id}")


class CollectionInactiveException(BusinessException):
    """Collection inactive exception"""

    def __init__(self, collection_id: str):
        super().__init__(ErrorCode.COLLECTION_INACTIVE, f"Collection is inactive: {collection_id}")


class QuotaExceededException(BusinessException):
    """Quota exceeded exception"""

    def __init__(self, resource_type: str, limit: int, current_usage: int = None):
        # Map resource types to appropriate error codes
        error_code_map = {
            "max_collection_count": ErrorCode.COLLECTION_QUOTA_EXCEEDED,
            "max_document_count": ErrorCode.DOCUMENT_QUOTA_EXCEEDED,
            "max_document_count_per_collection": ErrorCode.DOCUMENT_QUOTA_EXCEEDED,
            "max_bot_count": ErrorCode.BOT_QUOTA_EXCEEDED,
        }

        error_code = error_code_map.get(resource_type, ErrorCode.COLLECTION_QUOTA_EXCEEDED)

        # Create a more user-friendly message
        if current_usage is not None:
            message = f"已达到{resource_type}的配额限制。当前使用量: {current_usage}/{limit}"
        else:
            message = f"已达到{resource_type}的配额限制 ({limit})"

        # Add details for better error handling
        details = {
            "quota_type": resource_type,
            "quota_limit": limit,
            "current_usage": current_usage,
            "quota_exceeded": True,
        }

        super().__init__(error_code, message, details)


class BotNotFoundException(BusinessException):
    """Bot not found exception"""

    def __init__(self, bot_id: str):
        super().__init__(ErrorCode.BOT_NOT_FOUND, f"Bot not found: {bot_id}")


class DocumentNotFoundException(BusinessException):
    """Document not found exception"""

    def __init__(self, document_id: str):
        super().__init__(ErrorCode.DOCUMENT_NOT_FOUND, f"Document not found: {document_id}")


class ChatNotFoundException(BusinessException):
    """Chat not found exception"""

    def __init__(self, chat_id: str):
        super().__init__(ErrorCode.CHAT_NOT_FOUND, f"Chat not found: {chat_id}")


class GraphServiceError(BusinessException):
    """Graph service operation error"""

    def __init__(self, message: str):
        super().__init__(ErrorCode.GRAPH_SERVICE_ERROR, message)


class InvalidParameterException(BusinessException):
    """Invalid parameter exception"""

    def __init__(self, parameter_name: str, reason: str = None):
        message = f"Invalid parameter: {parameter_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(ErrorCode.INVALID_PARAMETER, message)


class ValidationException(BusinessException):
    """Validation error exception"""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(ErrorCode.INVALID_PARAMETER, message, details)


class PermissionDeniedError(BusinessException):
    """Permission denied exception"""

    def __init__(self, message: str = None):
        super().__init__(ErrorCode.FORBIDDEN, message or "Permission denied")


class NotFoundException(BusinessException):
    """Not found exception"""

    def __init__(self, message: str):
        super().__init__(ErrorCode.RESOURCE_NOT_FOUND, message)


# Marketplace related exceptions
class MarketplaceError(BusinessException):
    """Base class for marketplace-related errors"""

    pass


class CollectionNotPublishedError(MarketplaceError):
    """Exception raised when trying to access an unpublished collection"""

    def __init__(self, collection_id: str = None):
        message = "Collection is not published to marketplace"
        if collection_id:
            message += f" (ID: {collection_id})"
        super().__init__(ErrorCode.INVALID_PARAMETER, message)


class AlreadySubscribedError(MarketplaceError):
    """Exception raised when user tries to subscribe to an already subscribed collection"""

    def __init__(self, collection_id: str = None):
        message = "Already subscribed to this collection"
        if collection_id:
            message += f" (ID: {collection_id})"
        super().__init__(ErrorCode.INVALID_PARAMETER, message)


class SubscriptionNotFoundError(MarketplaceError):
    """Exception raised when subscription is not found"""

    def __init__(self, collection_id: str = None):
        message = "Subscription not found"
        if collection_id:
            message += f" for collection (ID: {collection_id})"
        super().__init__(ErrorCode.NOT_FOUND, message)


class SelfSubscriptionError(MarketplaceError):
    """Exception raised when user tries to subscribe to their own collection"""

    def __init__(self, collection_id: str = None):
        message = "Cannot subscribe to your own collection"
        if collection_id:
            message += f" (ID: {collection_id})"
        super().__init__(ErrorCode.INVALID_PARAMETER, message)


class CollectionMarketplaceAccessDeniedError(MarketplaceError):
    """Exception raised when user doesn't have subscription access to marketplace collection"""

    def __init__(self, collection_id: str = None, reason: str = None):
        message = "Access denied to marketplace collection"
        if collection_id:
            message += f" (ID: {collection_id})"
        if reason:
            message += f" - {reason}"
        super().__init__(ErrorCode.FORBIDDEN, message)


# Convenience functions for common exceptions
def not_found(resource_type: str, resource_id: str = None) -> ResourceNotFoundException:
    """Create a resource not found exception"""
    return ResourceNotFoundException(resource_type, resource_id)


def invalid_param(parameter_name: str, reason: str = None) -> InvalidParameterException:
    """Create an invalid parameter exception"""
    return InvalidParameterException(parameter_name, reason)


def quota_exceeded(resource_type: str, limit: int, current_usage: int = None) -> QuotaExceededException:
    """Create a quota exceeded exception"""
    return QuotaExceededException(resource_type, limit, current_usage)


def validation_error(message: str, details: Dict[str, Any] = None) -> ValidationException:
    """Create a validation error exception"""
    return ValidationException(message, details)
