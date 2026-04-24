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

from aperag.domains.knowledge_base.service.collection_service import (
    set_marketplace_collection_ops as _kb_set_marketplace_collection_ops,
)
from aperag.domains.knowledge_base.service.collection_service import (
    set_marketplace_ops as _kb_set_marketplace_ops,
)
from aperag.domains.knowledge_base.service.collection_service import (
    set_quota_ops as _kb_set_quota_ops,
)
from aperag.domains.knowledge_base.service.collection_service import (
    set_search_pipeline_ops as _kb_set_search_pipeline_ops,
)
from aperag.domains.knowledge_graph.api.routes import router as knowledge_graph_router
from aperag.domains.retrieval.api.routes import router as retrieval_router
from aperag.domains.web_access.api.routes import router as web_access_router
from aperag.exception_handlers import register_exception_handlers
from aperag.llm.litellm_track import register_custom_llm_track
from aperag.mcp import mcp_server
from aperag.openapi_spec import custom_generate_unique_id
from aperag.service.marketplace_collection_service import (
    marketplace_collection_service as _legacy_marketplace_collection_service,
)
from aperag.service.marketplace_service import marketplace_service as _legacy_marketplace_service
from aperag.service.quota_service import quota_service as _legacy_quota_service
from aperag.service.search_pipeline_service import search_pipeline_service as _legacy_search_pipeline_service
from aperag.views.agent_runtime import router as agent_runtime_router
from aperag.views.api_key import router as api_key_router
from aperag.views.audit import router as audit_router
from aperag.views.auth import router as auth_router
from aperag.views.bots_v2 import router as bots_v2_router
from aperag.views.chat import router as chat_router
from aperag.views.collections import router as collections_router
from aperag.views.collections_v2 import router as collections_v2_router
from aperag.views.config import router as config_router
from aperag.views.documents_v2 import router as documents_v2_router
from aperag.views.evaluation_v2 import router as evaluation_v2_router
from aperag.views.export import router as export_router
from aperag.views.llm import router as llm_router
from aperag.views.main import router as main_router
from aperag.views.marketplace import router as marketplace_router
from aperag.views.marketplace_collections import router as marketplace_collections_router
from aperag.views.openai import router as openai_router
from aperag.views.prompts import router as prompts_router
from aperag.views.providers_v2 import router as providers_v2_router
from aperag.views.settings import router as settings_router


# Wire the knowledge_base domain's consumer-owned Protocol DI slots
# (Phase 3 Step 5b2c). The legacy service singletons structurally
# satisfy ``MarketplaceOps`` / ``SearchPipelineOps`` / ``QuotaOps``
# directly; ``MarketplaceCollectionOps.check_marketplace_access`` uses
# the public method name from msg=6ab7d211 Q2 while the legacy service
# still exposes the underscore-prefixed original, so a thin adapter
# bridges the two until the Phase 4 marketplace_collection_service move
# drops the ``_`` at its canonical location.
class _MarketplaceCollectionOpsAdapter:
    async def check_marketplace_access(self, user_id: str, collection_id: str) -> dict:
        return await _legacy_marketplace_collection_service._check_marketplace_access(user_id, collection_id)


_kb_set_marketplace_ops(_legacy_marketplace_service)
_kb_set_marketplace_collection_ops(_MarketplaceCollectionOpsAdapter())
_kb_set_search_pipeline_ops(_legacy_search_pipeline_service)
_kb_set_quota_ops(_legacy_quota_service)


# Initialize MCP server integration with stateless HTTP to fix OpenAI tool call sequence issues
mcp_app = mcp_server.http_app(path="/", stateless_http=True)


async def combined_lifespan(app: FastAPI):
    """Combined lifespan manager for the API and MCP server."""
    async with mcp_app.lifespan(app):
        yield


# Create the main FastAPI app with combined lifespan
app = FastAPI(
    title="ApeRAG API",
    description="Knowledge management and retrieval system",
    version="1.0.0",
    lifespan=combined_lifespan,  # Combined lifecycle management
    generate_unique_id_function=custom_generate_unique_id,
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
app.include_router(export_router, prefix="/api/v1")  # Add export router
app.include_router(api_key_router, prefix="/api/v1")
app.include_router(audit_router, prefix="/api/v1")  # Add audit router
app.include_router(llm_router, prefix="/api/v1")
app.include_router(marketplace_router, prefix="/api/v1")  # Add marketplace router
app.include_router(marketplace_collections_router, prefix="/api/v1")  # Add marketplace collections router
app.include_router(settings_router, prefix="/api/v1")
app.include_router(prompts_router, prefix="/api/v1")  # Add prompts router
app.include_router(web_access_router, prefix="/api/v2", tags=["web_access"])  # Add web_access domain router
app.include_router(retrieval_router, prefix="/api/v2", tags=["retrieval"])  # Add retrieval domain router
app.include_router(
    knowledge_graph_router, prefix="/api/v2", tags=["knowledge_graph"]
)  # Add knowledge_graph domain router
app.include_router(chat_router, prefix="/api/v1")
app.include_router(openai_router, prefix="/v1")
app.include_router(config_router, prefix="/api/v1/config")
app.include_router(agent_runtime_router, prefix="/api/v2")
app.include_router(bots_v2_router, prefix="/api/v2")
app.include_router(evaluation_v2_router, prefix="/api/v2")
app.include_router(providers_v2_router, prefix="/api/v2")
app.include_router(collections_v2_router, prefix="/api/v2")
app.include_router(documents_v2_router, prefix="/api/v2")

# Only include test router in dev mode
if os.environ.get("DEPLOYMENT_MODE") == "dev":
    from aperag.views.test import router as test_router

    app.include_router(test_router, prefix="/api/v1")

# Mount the MCP server at /mcp path
app.mount("/mcp", mcp_app)
