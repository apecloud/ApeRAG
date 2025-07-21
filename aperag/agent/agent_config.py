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
    chat_id: str

    # LLM Settings
    provider_name: str
    api_key: str
    base_url: str
    default_model: str
    temperature: float = 0.7
    max_tokens: int = 60000

    # MCP configuration
    aperag_api_key: str = None
    aperag_mcp_url: str = None

    # Agent behavior configuration
    language: str = "en-US"
    instruction: str = ""
    server_names: List[str] = None

    def get_session_key(self) -> str:
        """Generate session key based on user, chat, and provider."""
        return f"{self.user_id}:{self.chat_id}:{self.provider_name}"
