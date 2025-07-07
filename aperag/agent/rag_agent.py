#!/usr/bin/env python3
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
Simple ApeRAG Agent - Programmatic Configuration

A minimal RAG agent that connects to ApeRAG via MCP protocol using programmatic configuration.
Usage: python rag_agent2.py
"""

import asyncio
import os

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.config import LoggerSettings, MCPServerSettings, MCPSettings, OpenAISettings, Settings
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

os.environ["APERAG_API_KEY"] = "sk-test"
os.environ["OPENAI_API_KEY"] = "sk-test"
os.environ["APERAG_URL"] = "http://localhost:8000/mcp/"
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["DEFAULT_MODEL"] = "gpt-4o-mini"


class SimpleRAGAgent:
    def __init__(self):
        self.aperag_api_key = os.getenv("APERAG_API_KEY", "sk-test")
        self.aperag_url = os.getenv("APERAG_URL", "http://localhost:8000/mcp/")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "sk-test")
        self.default_model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

        self.settings = self._create_settings()

        self.app = MCPApp(name="rag_agent", settings=self.settings)

    def _create_settings(self) -> Settings:
        """Create mcp-agent settings programmatically"""

        return Settings(
            execution_engine="asyncio",
            logger=LoggerSettings(type="console", level="info"),
            mcp=MCPSettings(
                servers={
                    "aperag": MCPServerSettings(
                        transport="streamable_http",
                        url=self.aperag_url,
                        headers={"Authorization": f"Bearer {self.aperag_api_key}", "Content-Type": "application/json"},
                        http_timeout_seconds=30,
                        read_timeout_seconds=120,
                        description="ApeRAG knowledge base server",
                    )
                }
            ),
            openai=OpenAISettings(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
                default_model=self.default_model,
                temperature=0.7,
                max_tokens=2000,
            ),
        )

    async def interactive_chat(self):
        print("🤖 (Type 'exit' to exit)")
        print("=" * 40)

        async with self.app.run() as running_app:
            agent = Agent(name="assistant", instruction="Test Agent", server_names=["aperag"])

            tools = await agent.list_tools()
            tool_names = [t.name for t in tools.tools]

            # Test if server is in registry
            if "aperag" in running_app.server_registry.registry:
                print("\n✓ Server 'aperag' found in registry!")
                server_config = running_app.server_registry.get_server_config("aperag")
                print(f"Server config: {server_config}")
            else:
                print("\n✗ Server 'aperag' NOT found in registry!")

            if tool_names:
                print(f"✅ Found Tools: {tool_names}")
                return True
            else:
                print("❌ Didn't Found Any Tools")
                return False

            async with agent:
                await asyncio.sleep(2)

                llm = await agent.attach_llm(OpenAIAugmentedLLM)

                while True:
                    try:
                        question = input("\n❓ Question: ").strip()

                        if question.lower() in ["exit", "quit", "q"]:
                            print("👋 Goodbye!")
                            break

                        if not question:
                            continue

                        print("🔍 Thinking...")
                        response = await llm.generate_str(question)
                        print(f"🤖 Answer: {response}")

                    except KeyboardInterrupt:
                        print("\n👋 Goodbye!")
                        break
                    except Exception as e:
                        print(f"❌ Error: {e}")


async def main():
    try:
        print("=" * 40)

        agent = SimpleRAGAgent()
        await agent.interactive_chat()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
