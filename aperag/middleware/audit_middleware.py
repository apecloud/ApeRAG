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

import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from aperag.service.audit_service import audit_service

logger = logging.getLogger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic audit logging"""

    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.exclude_paths = [
            "/docs", "/openapi.json", "/redoc", "/static",
            "/health", "/favicon.ico", "/metrics"
        ]

    def _should_audit(self, path: str, method: str) -> bool:
        """Check if the request should be audited"""
        if not self.enabled:
            return False
        
        # Skip GET requests - only audit change operations
        if method.upper() == "GET":
            return False
        
        # Skip excluded paths
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return False
        
        # Only audit API endpoints
        if not path.startswith("/api/"):
            return False
            
        return True

    def _get_audit_info_from_route(self, request: Request) -> Tuple[Optional[str], Optional[str]]:
        """Get API name and resource type from route name and tags"""
        try:
            if hasattr(request, 'scope') and 'route' in request.scope:
                route = request.scope['route']
                
                # Get API name from route name
                api_name = None
                if hasattr(route, 'name') and route.name:
                    api_name = route.name
                elif hasattr(route, 'endpoint') and hasattr(route.endpoint, '__name__'):
                    api_name = route.endpoint.__name__
                
                # Get resource type from tags
                resource_type = None
                if hasattr(route, 'tags') and route.tags:
                    resource_type = audit_service.get_resource_type_from_tags(route.tags)
                
                # Debug logging
                logger.debug(f"Route info - Path: {request.url.path}, API name: {api_name}, Tags: {getattr(route, 'tags', None)}, Resource type: {resource_type}")
                
                # Both API name and resource type are required
                if api_name and resource_type:
                    return api_name, resource_type
                elif api_name or resource_type:
                    # Log when we have partial info to help debugging
                    logger.warning(f"Partial audit info - Path: {request.url.path}, API name: {api_name}, Resource type: {resource_type}, Tags: {getattr(route, 'tags', None)}")
                        
        except Exception as e:
            logger.warning(f"Failed to get audit info from route: {e}")
            
        return None, None

    async def _extract_request_data(self, request: Request) -> Optional[Dict[str, Any]]:
        """Extract request data safely"""
        try:
            # Get JSON body if available
            if request.headers.get("content-type", "").startswith("application/json"):
                body = await request.body()
                if body:
                    return json.loads(body.decode())
            
            # Get form data if available
            elif request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
                form_data = await request.form()
                return dict(form_data)
            
            # Get query parameters
            if request.query_params:
                return dict(request.query_params)
                
        except Exception as e:
            logger.warning(f"Failed to extract request data: {e}")
            
        return None

    async def dispatch(self, request: Request, call_next):
        # Check if audit is needed
        if not self._should_audit(request.url.path, request.method):
            return await call_next(request)

        # Record start time in milliseconds
        start_time_ms = int(time.time() * 1000)
        request_data = None
        response_data = None
        error_message = None
        status_code = 200
        end_time_ms = None

        try:
            # Extract request data before calling next
            request_data = await self._extract_request_data(request)
            
            # Call the actual endpoint
            response = await call_next(request)
            status_code = response.status_code
            
            # Record end time
            end_time_ms = int(time.time() * 1000)
            
            # Get audit info from route (now available after call_next)
            api_name, resource_type = self._get_audit_info_from_route(request)
            
            # Only proceed with audit if we have both api_name and resource_type
            if api_name and resource_type:
                # Try to extract response data for non-streaming responses
                if hasattr(response, 'body'):
                    try:
                        response_body = response.body.decode() if response.body else None
                        if response_body:
                            response_data = json.loads(response_body)
                    except:
                        pass
                
                # Log audit asynchronously
                try:
                    # Get user info from request state (set by auth middleware)
                    user_id = getattr(request.state, 'user_id', None)
                    username = getattr(request.state, 'username', None)
                    
                    # Extract client info
                    ip_address, user_agent = audit_service._extract_client_info(request)
                    
                    # Log audit in background (don't await to avoid blocking)
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
                    
        except Exception as e:
            error_message = str(e)
            status_code = 500
            end_time_ms = int(time.time() * 1000)
            
            # Still try to log the error audit if route info is available
            try:
                api_name, resource_type = self._get_audit_info_from_route(request)
                if api_name and resource_type:
                    # Get user info from request state (set by auth middleware)
                    user_id = getattr(request.state, 'user_id', None)
                    username = getattr(request.state, 'username', None)
                    
                    # Extract client info
                    ip_address, user_agent = audit_service._extract_client_info(request)
                    
                    # Log audit in background (don't await to avoid blocking)
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
                logger.error(f"Failed to log error audit: {audit_error}")
            
            # Re-raise for normal error handling
            raise

        return response 