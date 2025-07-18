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

"""Agent configuration management for session creation."""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class AgentConfig:
    """
    Configuration for agent session creation.

    This centralizes all agent-related configuration parameters to make
    the session creation more flexible and maintainable.
    """

    # Basic agent info
    user_id: str
    provider_name: str
    api_key: str
    base_url: str
    default_model: str

    # Agent behavior configuration
    language: str = "en-US"
    instruction: str = ""
    server_names: List[str] = None

    # MCP configuration
    aperag_api_key: str = None
    aperag_url: str = None

    def __post_init__(self):
        """Set defaults for optional parameters."""
        if self.server_names is None:
            self.server_names = ["aperag"]

        # todo delete os.getenv
        if self.aperag_api_key is None:
            self.aperag_api_key = os.getenv("APERAG_API_KEY", "sk-test")

        if self.aperag_url is None:
            self.aperag_url = os.getenv("APERAG_URL", "http://localhost:8000/mcp/")

    def get_session_key(self) -> str:
        """Generate session key based on user and provider."""
        return f"{self.user_id}:{self.provider_name}"

    @classmethod
    def create_from_agent_message(
        cls,
        user_id: str,
        provider_name: str,
        api_key: str,
        base_url: str,
        default_model: str,
        language: str = "en-US",
        **kwargs,
    ) -> "AgentConfig":
        """Create config from agent message parameters."""
        return cls(
            user_id=user_id,
            provider_name=provider_name,
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            language=language,
            **kwargs,
        )
