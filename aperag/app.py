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

import os

from aperag.config import settings

# Initialize OpenTelemetry FIRST - before any other imports
from aperag.trace import init_tracing

# Initialize tracing with configuration
if settings.otel_enabled:
    init_tracing(
        service_name=settings.otel_service_name,
        service_version=settings.otel_service_version,
        jaeger_endpoint=settings.jaeger_endpoint if settings.jaeger_enabled else None,
        enable_console=settings.otel_console_enabled,
        enable_fastapi=settings.otel_fastapi_enabled,
        enable_sqlalchemy=settings.otel_sqlalchemy_enabled,
        enable_mcp=settings.otel_mcp_enabled,
    )

from fastapi import FastAPI  # noqa: E402

from aperag.agent.agent_event_listener import agent_event_listener  # noqa: E402
from aperag.agent.agent_session_manager_lifecycle import agent_session_manager_lifespan  # noqa: E402
from aperag.exception_handlers import register_exception_handlers
from aperag.llm.litellm_track import register_custom_llm_track
from aperag.mcp import mcp_server
from aperag.views.api_key import router as api_key_router
from aperag.views.audit import router as audit_router
from aperag.views.auth import router as auth_router
from aperag.views.chat import router as chat_router
from aperag.views.openai import router as openai_router
from aperag.views.bot import router as bot_router
from aperag.views.collections import router as collections_router
from aperag.views.config import router as config_router
from aperag.views.evaluation import router as evaluation_router
from aperag.views.flow import router as flow_router
from aperag.views.graph import router as graph_router
from aperag.views.llm import router as llm_router
from aperag.views.main import router as main_router
from aperag.views.marketplace import router as marketplace_router
from aperag.views.marketplace_collections import router as marketplace_collections_router
from aperag.views.settings import router as settings_router
from aperag.views.web import router as web_router

# Initialize MCP server integration with stateless HTTP to fix OpenAI tool call sequence issues
mcp_app = mcp_server.http_app(path="/", stateless_http=True)


# Combined lifespan function for both MCP and Agent session management
async def combined_lifespan(app: FastAPI):
    """Combined lifespan manager for MCP and Agent sessions."""
    # Initialize the global proxy listener at startup
    await agent_event_listener.initialize()

    # Start MCP server first
    async with mcp_app.lifespan(app):
        # Then start Agent session manager
        async with agent_session_manager_lifespan(app):
            yield


# Create the main FastAPI app with combined lifespan
app = FastAPI(
    title="ApeRAG API",
    description="Knowledge management and retrieval system",
    version="1.0.0",
    lifespan=combined_lifespan,  # Combined lifecycle management
)

# Register global exception handlers
register_exception_handlers(app)

register_custom_llm_track()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint for container health monitoring"""
    return {"status": "healthy", "service": "aperag-api"}


app.include_router(auth_router, prefix="/api/v1")
app.include_router(main_router, prefix="/api/v1")
app.include_router(collections_router, prefix="/api/v1")  # Add collections router
app.include_router(api_key_router, prefix="/api/v1")
app.include_router(audit_router, prefix="/api/v1")  # Add audit router
app.include_router(flow_router, prefix="/api/v1")
app.include_router(llm_router, prefix="/api/v1")
app.include_router(graph_router, prefix="/api/v1")
app.include_router(marketplace_router, prefix="/api/v1")  # Add marketplace router
app.include_router(marketplace_collections_router, prefix="/api/v1")  # Add marketplace collections router
app.include_router(settings_router, prefix="/api/v1")
app.include_router(web_router, prefix="/api/v1")  # Add web search router
app.include_router(evaluation_router, prefix="/api/v1")
app.include_router(bot_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(openai_router, prefix="/v1")
app.include_router(config_router, prefix="/api/v1/config")

# Only include test router in dev mode
if os.environ.get("DEPLOYMENT_MODE") == "dev":
    from aperag.views.test import router as test_router

    app.include_router(test_router, prefix="/api/v1")

# Mount the MCP server at /mcp path
app.mount("/mcp", mcp_app)
