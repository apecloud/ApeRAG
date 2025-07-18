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
from typing import Optional

from mcp_agent import MCPApp, MCPSettings
from mcp_agent.llm import AperagAPISettings, OpenAISettings

from aperag.db.ops import AsyncDatabaseOps
from aperag.schema.view_models import ModelSpec
from aperag.utils.logger import logger


class MCPAppFactory:
    """Factory class for creating MCP applications."""

    @staticmethod
    def create_mcp_app(
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        aperag_api_key: Optional[str] = None,
        aperag_url: Optional[str] = None,
    ) -> Optional[MCPApp]:
        """Create MCPApp instance with the specified parameters"""
        if not MCPApp:
            logger.error("MCP components not available")
            return None

        # Use provided aperag settings or fall back to environment variables
        aperag_api_key = aperag_api_key or os.getenv("APERAG_API_KEY", "sk-test")
        aperag_url = aperag_url or os.getenv("APERAG_URL", "http://localhost:8000/mcp/")

        try:
            # Create ApeRAG API settings
            aperag_settings = AperagAPISettings(api_key=aperag_api_key, base_url=aperag_url)

            # Create OpenAI settings
            openai_settings = OpenAISettings(
                api_key=api_key,
                base_url=base_url,
                default_model=model,
                temperature=0.7,
                max_tokens=2000,
            )

            # Create MCP settings
            mcp_settings = MCPSettings(aperag_api=aperag_settings)

            return MCPApp(name="aperag_agent", settings=mcp_settings, llm_settings=openai_settings)

        except Exception as e:
            logger.error(f"Failed to create MCPApp: {e}")
            return None

    @staticmethod
    async def create_mcp_app_from_model_spec(
        model_spec: ModelSpec,
        user_id: Optional[str] = None,
        aperag_api_key: Optional[str] = None,
        aperag_url: Optional[str] = None,
    ) -> Optional[MCPApp]:
        """Create MCPApp instance from ModelSpec by resolving provider information from database"""
        if not model_spec or not model_spec.model:
            logger.error("ModelSpec or model name is required")
            return None

        if not model_spec.model_service_provider:
            logger.error("ModelSpec must specify model_service_provider")
            return None

        try:
            # Get database operations instance
            db_ops = AsyncDatabaseOps()

            # Query provider details and API key
            provider_info = await db_ops.query_llm_provider_by_name(model_spec.model_service_provider)
            api_key = await db_ops.query_provider_api_key(
                model_spec.model_service_provider, user_id=user_id, need_public=True
            )

            if not provider_info or not api_key:
                logger.error(f"Provider {model_spec.model_service_provider} not found or no API key available")
                return None

            return MCPAppFactory.create_mcp_app(
                provider=provider_info.name,
                model=model_spec.model,
                base_url=provider_info.base_url,
                api_key=api_key,
                aperag_api_key=aperag_api_key,
                aperag_url=aperag_url,
            )

        except Exception as e:
            logger.error(f"Failed to create MCPApp from ModelSpec: {e}")
            return None
