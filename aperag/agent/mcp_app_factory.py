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

from mcp_agent.app import MCPApp
from mcp_agent.config import LoggerSettings, MCPServerSettings, MCPSettings, OpenAISettings, Settings

from .agent_config import AgentConfig
from .exceptions import agent_config_invalid, mcp_init_failed

logger = logging.getLogger(__name__)


class MCPAppFactory:
    """Factory class for creating MCP applications."""

    @staticmethod
    def create_mcp_app(
        model: str,
        llm_provider_name: str,
        base_url: str,
        api_key: str,
        aperag_api_key: str,
        aperag_mcp_url: str,
        # Configurable LLM parameters
        temperature: float = 0.7,
        max_tokens: int = 60000,
    ) -> MCPApp:
        """Create MCPApp instance with the specified parameters."""
        # Validate required parameters
        required_params = {
            "model": model,
            "llm_provider_name": llm_provider_name,
            "base_url": base_url,
            "api_key": api_key,
            "aperag_api_key": aperag_api_key,
            "aperag_mcp_url": aperag_mcp_url,
        }

        for param_name, value in required_params.items():
            if not value:
                raise agent_config_invalid(param_name, f"{param_name} is required")

        try:
            settings = Settings(
                execution_engine="asyncio",
                logger=LoggerSettings(type="console", level="info"),
                mcp=MCPSettings(
                    servers={
                        "aperag": MCPServerSettings(
                            transport="streamable_http",
                            url=aperag_mcp_url,
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
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
            )

            mcp_app = MCPApp(name="aperag_agent", settings=settings)
            logger.info(f"Created MCP app for {llm_provider_name}:{model}")
            return mcp_app

        except Exception as e:
            logger.error(f"Failed to create MCP app: {e}")
            raise mcp_init_failed(f"MCP app creation failed: {str(e)}")

    @staticmethod
    def create_mcp_app_from_config(config: AgentConfig) -> MCPApp:
        """Create MCPApp instance using AgentConfig object."""
        return MCPAppFactory.create_mcp_app(
            model=config.default_model,
            llm_provider_name=config.provider_name,
            base_url=config.base_url,
            api_key=config.api_key,
            aperag_api_key=config.aperag_api_key,
            aperag_mcp_url=config.aperag_mcp_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
