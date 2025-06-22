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

import functools
import logging
import time
from typing import Any, Dict, Optional

from fastapi import Request

from aperag.service.audit_service import audit_service

logger = logging.getLogger(__name__)


def _extract_response_data(response: Any) -> Optional[Dict[str, Any]]:
    """Extract response data from the returned response object"""
    try:
        # If response is already a dict (common for JSON APIs)
        if isinstance(response, dict):
            return response

        # If response has a dict() method (Pydantic models)
        elif hasattr(response, "dict"):
            return response.dict()

        # If response has a model_dump() method (Pydantic v2)
        elif hasattr(response, "model_dump"):
            return response.model_dump()

        # If response is a list of dicts or models
        elif isinstance(response, list):
            result = []
            for item in response:
                if isinstance(item, dict):
                    result.append(item)
                elif hasattr(item, "dict"):
                    result.append(item.dict())
                elif hasattr(item, "model_dump"):
                    result.append(item.model_dump())
                else:
                    result.append(str(item))
            return {"items": result}

        # For other types, try to convert to string
        else:
            return {"response": str(response)}

    except Exception as e:
        logger.debug(f"Failed to extract response data: {e}")
        return {"status": "success", "type": type(response).__name__}


def _clean_data_for_audit(data):
    """Clean data for audit logging - remove null values and sensitive information"""
    if data is None:
        return None

    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Skip null/None values
            if value is None:
                continue

            # Filter out sensitive fields
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in ["password", "secret", "token", "key"]):
                cleaned[key] = "***FILTERED***"
            else:
                # Recursively clean nested data
                cleaned_value = _clean_data_for_audit(value)
                if cleaned_value is not None or isinstance(cleaned_value, (bool, int, float, str)):
                    # Keep non-null values and primitive types (including False, 0, empty string)
                    if not (isinstance(cleaned_value, dict) and len(cleaned_value) == 0):
                        # Don't add empty dicts
                        cleaned[key] = cleaned_value

        return cleaned if cleaned else None

    elif isinstance(data, list):
        cleaned = []
        for item in data:
            cleaned_item = _clean_data_for_audit(item)
            if cleaned_item is not None:
                cleaned.append(cleaned_item)
        return cleaned if cleaned else None

    else:
        # For primitive types, return as-is
        return data


def _extract_request_data_from_args(request: Request, kwargs: dict) -> Optional[Dict[str, Any]]:
    """Extract request data from function arguments (after FastAPI parsing)"""
    try:
        # Extract parsed data from function arguments
        # FastAPI injects parsed JSON data as function parameters
        parsed_data = {}
        for key, value in kwargs.items():
            # Skip the request object itself
            if isinstance(value, Request):
                continue

            # Skip User objects and other database model objects
            if hasattr(value, "__tablename__"):  # SQLAlchemy model
                continue

            # Try to serialize the value
            try:
                if hasattr(value, "dict"):  # Pydantic model
                    serialized = value.dict()
                    # Clean up the serialized data - remove null values and filter sensitive data
                    cleaned_data = _clean_data_for_audit(serialized)
                    if cleaned_data:  # Only add if there's actual data
                        parsed_data[key] = cleaned_data
                elif hasattr(value, "model_dump"):  # Pydantic v2
                    serialized = value.model_dump()
                    # Clean up the serialized data
                    cleaned_data = _clean_data_for_audit(serialized)
                    if cleaned_data:  # Only add if there's actual data
                        parsed_data[key] = cleaned_data
                elif isinstance(value, (dict, list, str, int, float, bool)):
                    # For basic types, also clean the data
                    cleaned_data = _clean_data_for_audit(value)
                    if cleaned_data is not None:  # Allow False, 0, empty string but not None
                        parsed_data[key] = cleaned_data
                else:
                    # For other types, convert to string but skip if it looks like an object
                    str_value = str(value)
                    if " object at 0x" not in str_value:  # Skip object representations
                        parsed_data[key] = str_value
            except Exception:
                # Skip problematic values
                continue

        # Return the actual data directly, not wrapped in any structure
        # If there's only one main data object, return it directly
        if len(parsed_data) == 1:
            return list(parsed_data.values())[0]
        elif len(parsed_data) > 1:
            return parsed_data
        else:
            return None

    except Exception as e:
        logger.warning(f"Failed to extract request data from args: {e}")
        return None


def _extract_client_info(request) -> tuple[Optional[str], Optional[str]]:
    """Extract client IP and User-Agent from request"""
    try:
        # Get IP address
        ip_address = None
        if hasattr(request, "client") and request.client:
            ip_address = request.client.host

        # Check for forwarded headers
        if hasattr(request, "headers"):
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                ip_address = forwarded_for.split(",")[0].strip()
            elif request.headers.get("X-Real-IP"):
                ip_address = request.headers.get("X-Real-IP")

        # Get User-Agent
        user_agent = None
        if hasattr(request, "headers"):
            user_agent = request.headers.get("User-Agent")

        return ip_address, user_agent
    except Exception as e:
        logger.warning(f"Failed to extract client info: {e}")
        return None, None


async def _log_audit_async(
    request: Request,
    resource_type: str,
    api_name: str,
    start_time_ms: int,
    end_time_ms: int,
    status_code: int,
    request_data: dict,
    response_data: dict,
    error_message: str = None,
):
    """Log audit information asynchronously"""
    try:
        # Get user info from request state
        user_id = getattr(request.state, "user_id", None)
        username = getattr(request.state, "username", None)

        # Extract client info
        ip_address, user_agent = _extract_client_info(request)

        # Log audit in background
        import asyncio

        asyncio.create_task(
            audit_service.log_audit(
                user_id=user_id,
                username=username,
                resource_type=resource_type,
                api_name=api_name,
                http_method=request.method,
                path=request.url.path,
                status_code=status_code,
                start_time=start_time_ms,
                end_time=end_time_ms,
                request_data=request_data,
                response_data=response_data,
                error_message=error_message,
                ip_address=ip_address,
                user_agent=user_agent,
            )
        )
    except Exception as audit_error:
        logger.error(f"Failed to log audit: {audit_error}")


def audit(resource_type: str, api_name: str = None):
    """
    Decorator for API endpoints to enable automatic audit logging

    Args:
        resource_type: The resource type for audit (e.g., 'collection', 'user', etc.)
        api_name: Optional API name override (defaults to function name)
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the request object in the arguments
            request = None
            for v in kwargs.values():
                if isinstance(v, Request):
                    request = v
                    break

            if not request:
                # If no request found, just call the original function
                return await func(*args, **kwargs)

            # Skip GET requests - only audit change operations
            if request.method.upper() == "GET":
                return await func(*args, **kwargs)

            # Record start time
            start_time_ms = int(time.time() * 1000)
            actual_api_name = api_name or func.__name__

            try:
                # Call the original function first to get the parsed data
                response = await func(*args, **kwargs)

                # Record end time
                end_time_ms = int(time.time() * 1000)

                # Extract request data from function arguments (after parsing)
                request_data = _extract_request_data_from_args(request, kwargs)

                # Extract response data
                response_data = _extract_response_data(response)

                # Log audit asynchronously
                await _log_audit_async(
                    request=request,
                    resource_type=resource_type,
                    api_name=actual_api_name,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    status_code=200,  # Success
                    request_data=request_data,
                    response_data=response_data,
                    error_message=None,
                )

                return response

            except Exception as e:
                # Record end time for error case
                end_time_ms = int(time.time() * 1000)

                # Extract request data if possible
                try:
                    request_data = _extract_request_data_from_args(request, kwargs)
                except Exception:
                    request_data = {"method": request.method, "path": request.url.path}

                # Log audit for error case
                await _log_audit_async(
                    request=request,
                    resource_type=resource_type,
                    api_name=actual_api_name,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    status_code=500,  # Error
                    request_data=request_data,
                    response_data={"error": str(e)},
                    error_message=str(e),
                )

                # Re-raise the exception
                raise

        return wrapper

    return decorator
