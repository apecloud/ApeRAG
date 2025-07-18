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
import os
from typing import Optional

from mcp_agent.app import MCPApp
from mcp_agent.config import LoggerSettings, MCPServerSettings, MCPSettings, OpenAISettings, Settings

from aperag.db.ops import AsyncDatabaseOps
from aperag.schema.view_models import ModelSpec

from .exceptions import (
    MCPAppInitializationError,
    MCPConnectionError,
    agent_config_invalid,
    handle_agent_error,
    mcp_init_failed,
)

logger = logging.getLogger(__name__)


class MCPAppFactory:
    """Factory class for creating MCP applications."""

    @staticmethod
    @handle_agent_error("mcp_app_creation")
    def create_mcp_app(
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        aperag_api_key: Optional[str] = None,
        aperag_url: Optional[str] = None,
    ) -> MCPApp:
        """Create MCPApp instance with the specified parameters"""
        # Validate required parameters
        if not provider:
            raise agent_config_invalid("provider", "Provider name is required")
        if not model:
            raise agent_config_invalid("model", "Model name is required")
        if not base_url:
            raise agent_config_invalid("base_url", "Base URL is required")
        if not api_key:
            raise agent_config_invalid("api_key", "API key is required")

        if not MCPApp:
            raise mcp_init_failed("MCP components not available", {"provider": provider, "model": model})

        # Use provided aperag settings or fall back to environment variables
        aperag_api_key = aperag_api_key or os.getenv("APERAG_API_KEY", "sk-test")
        aperag_url = aperag_url or os.getenv("APERAG_URL", "http://localhost:8000/mcp/")

        if not aperag_api_key:
            raise agent_config_invalid("aperag_api_key", "ApeRAG API key is required")
        if not aperag_url:
            raise agent_config_invalid("aperag_url", "ApeRAG URL is required")

        try:
            # Create settings using the new API structure
            settings = Settings(
                execution_engine="asyncio",
                logger=LoggerSettings(type="console", level="info"),
                mcp=MCPSettings(
                    servers={
                        "aperag": MCPServerSettings(
                            transport="streamable_http",
                            url=aperag_url,
                            headers={
                                "Authorization": f"Bearer {aperag_api_key}",
                                "Content-Type": "application/json",
                            },
                            http_timeout_seconds=30,
                            read_timeout_seconds=120,
                            description="ApeRAG knowledge base server",
                        )
                    }
                ),
                openai=OpenAISettings(
                    api_key=api_key,
                    base_url=base_url,
                    default_model=model,
                    temperature=0.7,
                    max_tokens=60000,
                ),
            )

            mcp_app = MCPApp(name="aperag_agent", settings=settings)
            logger.info(f"Successfully created MCP app for provider: {provider}, model: {model}")
            return mcp_app

        except Exception as e:
            raise mcp_init_failed(
                f"Failed to create MCPApp settings: {str(e)}",
                {"provider": provider, "model": model, "base_url": base_url},
                e,
            )

    @staticmethod
    @handle_agent_error("mcp_app_from_model_spec_creation")
    async def create_mcp_app_from_model_spec(
        model_spec: ModelSpec,
        user_id: Optional[str] = None,
        aperag_api_key: Optional[str] = None,
        aperag_url: Optional[str] = None,
    ) -> MCPApp:
        """Create MCPApp instance from ModelSpec by resolving provider information from database"""
        # Validate input parameters
        if not model_spec:
            raise agent_config_invalid("model_spec", "ModelSpec is required")
        if not model_spec.model:
            raise agent_config_invalid("model_spec.model", "Model name is required in ModelSpec")
        if not model_spec.model_service_provider:
            raise agent_config_invalid(
                "model_spec.model_service_provider", "Model service provider is required in ModelSpec"
            )

        try:
            # Get database operations instance
            db_ops = AsyncDatabaseOps()

            # Query provider details and API key
            provider_info = await db_ops.query_llm_provider_by_name(model_spec.model_service_provider)
            if not provider_info:
                raise mcp_init_failed(
                    f"Provider '{model_spec.model_service_provider}' not found in database",
                    {"provider_name": model_spec.model_service_provider, "user_id": user_id},
                )

            api_key = await db_ops.query_provider_api_key(
                model_spec.model_service_provider, user_id=user_id, need_public=True
            )
            if not api_key:
                raise mcp_init_failed(
                    f"No API key available for provider '{model_spec.model_service_provider}'",
                    {"provider_name": model_spec.model_service_provider, "user_id": user_id},
                )

            # Create MCP app using the resolved information
            return MCPAppFactory.create_mcp_app(
                provider=provider_info.name,
                model=model_spec.model,
                base_url=provider_info.base_url,
                api_key=api_key,
                aperag_api_key=aperag_api_key,
                aperag_url=aperag_url,
            )

        except (MCPAppInitializationError, MCPConnectionError):
            # Re-raise agent-specific exceptions as-is
            raise
        except Exception as e:
            raise mcp_init_failed(
                f"Unexpected error while creating MCPApp from ModelSpec: {str(e)}",
                {
                    "model_spec": {
                        "model": model_spec.model,
                        "provider": model_spec.model_service_provider,
                    },
                    "user_id": user_id,
                },
                e,
            )
