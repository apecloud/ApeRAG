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
import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import Request

from aperag.service.audit_service import audit_service

logger = logging.getLogger(__name__)


def audit_api(resource_type: str, api_name: str = None):
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
                # Extract request data
                request_data = await _extract_request_data(request)
                
                # Call the original function
                response = await func(*args, **kwargs)
                
                # Record end time
                end_time_ms = int(time.time() * 1000)
                
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
                    error_message=None
                )
                
                return response
                
            except Exception as e:
                # Record end time for error case
                end_time_ms = int(time.time() * 1000)
                
                # Extract request data if not already done
                try:
                    request_data = await _extract_request_data(request)
                except:
                    request_data = None
                
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
                    error_message=str(e)
                )
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator


async def _extract_request_data(request: Request) -> Optional[Dict[str, Any]]:
    """Extract request data safely without consuming the body"""
    try:
        # Try to get the body if it hasn't been consumed yet
        # In FastAPI, we need to be careful not to consume the body
        # that's needed by the actual endpoint
        
        # Get query parameters (safe to read multiple times)
        if request.query_params:
            return dict(request.query_params)
        
        # For now, let's just extract safe data to avoid body consumption issues
        # We can enhance this later if needed
        return {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params) if request.query_params else None,
            "headers": dict(request.headers) if hasattr(request, 'headers') else None
        }
            
    except Exception as e:
        logger.warning(f"Failed to extract request data: {e}")
        
    return None


def _extract_response_data(response: Any) -> Optional[Dict[str, Any]]:
    """Extract response data from the returned response object"""
    try:
        # If response is already a dict (common for JSON APIs)
        if isinstance(response, dict):
            return response
        
        # If response has a dict() method (Pydantic models)
        elif hasattr(response, 'dict'):
            return response.dict()
        
        # If response has a model_dump() method (Pydantic v2)
        elif hasattr(response, 'model_dump'):
            return response.model_dump()
        
        # If response is a list of dicts or models
        elif isinstance(response, list):
            result = []
            for item in response:
                if isinstance(item, dict):
                    result.append(item)
                elif hasattr(item, 'dict'):
                    result.append(item.dict())
                elif hasattr(item, 'model_dump'):
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


async def _log_audit_async(request: Request, resource_type: str, api_name: str,
                          start_time_ms: int, end_time_ms: int, status_code: int,
                          request_data: dict, response_data: dict, error_message: str = None):
    """Log audit information asynchronously"""
    try:
        # Get user info from request state
        user_id = getattr(request.state, 'user_id', None)
        username = getattr(request.state, 'username', None)
        
        # Extract client info
        ip_address, user_agent = audit_service._extract_client_info(request)
        
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
                user_agent=user_agent
            )
        )
    except Exception as audit_error:
        logger.error(f"Failed to log audit: {audit_error}") 