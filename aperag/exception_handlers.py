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

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from aperag.exceptions import BusinessException

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI):
    """Register global exception handlers for the FastAPI application"""

    @app.exception_handler(BusinessException)
    async def business_exception_handler(request: Request, exc: BusinessException) -> JSONResponse:
        """Handle business exceptions and convert to proper HTTP responses"""
        logger.warning(
            f"Business exception: {exc.error_code.error_name} - {exc.message}",
            extra={
                "error_code": exc.error_code.error_name,
                "code": exc.code,
                "details": exc.details,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return JSONResponse(
            status_code=exc.http_status.value,
            content={
                "success": False,
                "error_code": exc.error_code.error_name,
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "path": request.url.path,
            },
        )

    @app.exception_handler(SQLAlchemyError)
    async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        """Handle SQLAlchemy database errors"""
        logger.error(
            f"Database error: {str(exc)}",
            extra={"exception_type": type(exc).__name__, "path": request.url.path, "method": request.method},
            exc_info=True,
        )

        # Don't expose internal database errors to clients
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "DATABASE_ERROR",
                "code": 1050,
                "message": "A database error occurred",
                "details": {},
                "path": request.url.path,
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle ValueError exceptions as validation errors"""
        logger.warning(f"Validation error: {str(exc)}", extra={"path": request.url.path, "method": request.method})

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_code": "VALIDATION_ERROR",
                "code": 1051,
                "message": str(exc),
                "details": {},
                "path": request.url.path,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all other unhandled exceptions"""
        logger.error(
            f"Unhandled exception: {str(exc)}",
            extra={"exception_type": type(exc).__name__, "path": request.url.path, "method": request.method},
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "INTERNAL_ERROR",
                "code": 1000,
                "message": "An unexpected error occurred",
                "details": {},
                "path": request.url.path,
            },
        )
