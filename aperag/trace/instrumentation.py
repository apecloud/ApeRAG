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
OpenTelemetry instrumentation for FastAPI and SQLAlchemy.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import optional instrumentors
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FASTAPI_INSTRUMENTATION_AVAILABLE = True
except ImportError:
    FASTAPI_INSTRUMENTATION_AVAILABLE = False

try:
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    SQLALCHEMY_INSTRUMENTATION_AVAILABLE = True
except ImportError:
    SQLALCHEMY_INSTRUMENTATION_AVAILABLE = False


def init_fastapi_instrumentation(app: Any = None) -> bool:
    """
    Initialize FastAPI instrumentation for automatic HTTP request tracing.

    Args:
        app: FastAPI application instance (optional, can be called later)

    Returns:
        True if instrumentation was configured, False otherwise
    """
    if not FASTAPI_INSTRUMENTATION_AVAILABLE:
        logger.warning("FastAPI instrumentation not available - skipping")
        return False

    try:
        if app is not None:
            # Instrument specific app
            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI app instrumentation enabled")
        else:
            # Global instrumentation - will apply to all FastAPI apps
            FastAPIInstrumentor().instrument()
            logger.info("FastAPI global instrumentation enabled")
        return True
    except Exception as e:
        logger.warning(f"Failed to instrument FastAPI: {e}")
        return False


def init_sqlalchemy_instrumentation() -> bool:
    """
    Initialize SQLAlchemy instrumentation for automatic database query tracing.

    Returns:
        True if instrumentation was configured, False otherwise
    """
    if not SQLALCHEMY_INSTRUMENTATION_AVAILABLE:
        logger.warning("SQLAlchemy instrumentation not available - skipping")
        return False

    try:
        # Global SQLAlchemy instrumentation
        SQLAlchemyInstrumentor().instrument()
        logger.info("SQLAlchemy instrumentation enabled")
        return True
    except Exception as e:
        logger.warning(f"Failed to instrument SQLAlchemy: {e}")
        return False


def is_fastapi_instrumentation_available() -> bool:
    """Check if FastAPI instrumentation is available."""
    return FASTAPI_INSTRUMENTATION_AVAILABLE


def is_sqlalchemy_instrumentation_available() -> bool:
    """Check if SQLAlchemy instrumentation is available."""
    return SQLALCHEMY_INSTRUMENTATION_AVAILABLE
