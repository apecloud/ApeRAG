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
Simplified ApeRAG Agent with Orchestrator - Multi-Agent RAG System

A simplified RAG agent using the Orchestrator pattern with specialized worker agents.
Usage: python test_mcp_agent.py
"""

import asyncio
import logging
import os
import warnings

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.config import LoggerSettings, MCPServerSettings, MCPSettings, OpenAISettings, Settings
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.orchestrator.orchestrator import Orchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress httpcore warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*async generator ignored GeneratorExit.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Attempted to exit cancel scope.*")

# Set environment variables if not already set
if not os.getenv("APERAG_API_KEY"):
    os.environ["APERAG_API_KEY"] = "sk-test"
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-test"
if not os.getenv("APERAG_MCP_URL"):
    os.environ["APERAG_MCP_URL"] = "http://localhost:8000/mcp/"
if not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
if not os.getenv("DEFAULT_MODEL"):
    os.environ["DEFAULT_MODEL"] = "openai/gpt-4o-mini"  # Use a more basic model for testing

print("ENV=")
print(os.environ["DEFAULT_MODEL"])


def create_mcp_app() -> MCPApp:
    """Create MCPApp with ApeRAG server configuration"""
    settings = Settings(
        execution_engine="asyncio",
        logger=LoggerSettings(type="console", level="info"),
        mcp=MCPSettings(
            servers={
                "aperag": MCPServerSettings(
                    transport="streamable_http",
                    url=os.getenv("APERAG_MCP_URL", "http://localhost:8000/mcp/"),
                    headers={
                        "Authorization": f"Bearer {os.getenv('APERAG_API_KEY', 'sk-test')}",
                        "Content-Type": "application/json",
                    },
                    http_timeout_seconds=30,
                    read_timeout_seconds=120,
                    description="ApeRAG knowledge base server",
                    env={"APERAG_API_KEY": os.getenv("APERAG_API_KEY", "sk-test")},
                )
            }
        ),
        openai=OpenAISettings(
            api_key=os.getenv("OPENAI_API_KEY", "sk-test"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            default_model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
            temperature=0.7,
            max_tokens=2000,
        ),
    )
    return MCPApp(name="rag_orchestrator", settings=settings)


async def interactive_chat():
    """Interactive chat using Orchestrator pattern"""
    print("🤖 ApeRAG Orchestrator Agent Ready!")
    print("💡 Multi-agent system for intelligent knowledge retrieval")
    print("🔍 Ask me anything and I'll coordinate agents to find the best answers")
    print("💬 (Type 'exit' to exit)")
    print("=" * 60)

    app = create_mcp_app()

    try:
        async with app.run() as context:
            # Verify connection first
            if "aperag" not in context.server_registry.registry:
                print("\n❌ ApeRAG MCP Server Connection Failed!")
                return False

            print("\n✓ ApeRAG MCP Server Connected!")
            server_config = context.server_registry.get_server_config("aperag")
            print(f"📡 Server endpoint: {server_config.url}")

            # Create specialized worker agents
            searcher_agent = Agent(
                name="searcher",
                instruction="""You are a knowledge searcher. Your job is to:
1. List available collections using list_collections() when asked about collections
2. For each candidate collection, run the relevant D10 split-search tools
   directly: vector_search() for semantic similarity, fulltext_search()
   for keyword precision, and graph_search() for entity / relationship
   recall. Compose multiple split tools when one mode is insufficient —
   the legacy search_collection() omnibus is no longer available
   post-D10.h cutover.
3. Each search result item carries chunk_id / section_path /
   heading_anchor on its metadata; use them with read_document_chunk()
   or read_document_section() to fetch the underlying evidence.
4. Return search results with proper formatting and source attribution.""",
                server_names=["aperag"],
            )

            analyzer_agent = Agent(
                name="analyzer",
                instruction="""You are a knowledge analyzer. Your job is to:
1. Analyze and synthesize information from search results
2. Provide comprehensive, well-structured answers
3. Cite sources clearly and provide context
4. Identify gaps in information and suggest follow-up actions""",
                server_names=["aperag"],
            )

            # Simplified: Only use searcher and analyzer first
            # web_agent can be added later if needed

            # Get available tools from searcher
            tools = await searcher_agent.list_tools()
            tool_names = [t.name for t in tools.tools]

            if tool_names:
                print(f"🛠️  Available Tools: {', '.join(tool_names)}")
            else:
                print("❌ No tools available - check MCP server connection")
                return False

            # Create orchestrator with worker agents
            try:
                # First, let's try a single agent without orchestrator for debugging
                async with searcher_agent:
                    llm = await searcher_agent.attach_llm(OpenAIAugmentedLLM)

                    print("\n🧠 Single Agent initialized successfully!")

                    print("\n🚀 Ready to answer your questions!")
                    print("💭 Example questions:")
                    print("   • 'What collections do I have available?'")
                    print("   • 'Find information about API authentication'")
                    print("   • 'Search for recent updates on machine learning'")

                    while True:
                        try:
                            question = input("\n❓ Your Question: ").strip()

                            if question.lower() in ["exit", "quit", "q"]:
                                print("👋 Thank you for using ApeRAG Agent!")
                                break

                            if not question:
                                continue

                            print("🔍 Searching with single agent...")

                            try:
                                # Use simple agent directly
                                response = await asyncio.wait_for(llm.generate_str(question), timeout=60.0)
                                print("\n🤖 **Answer:**")
                                print(response)

                            except asyncio.TimeoutError:
                                print("⏱️ Request timed out. Please try again with a simpler question.")

                            except Exception as e:
                                logger.error(f"Error processing question: {e}")

                                # More specific error handling
                                if "401" in str(e) or "403" in str(e):
                                    print("❌ Authentication error: Please check your OpenAI API key")
                                    print("💡 Set OPENAI_API_KEY environment variable with a valid key")
                                    print("💡 Or check your OpenRouter account balance and permissions")
                                else:
                                    print(f"❌ Error processing question: {e}")
                                    print("💡 Tip: Try rephrasing your question or check your connection")

                        except KeyboardInterrupt:
                            print("\n👋 Thank you for using ApeRAG Agent!")
                            break
                        except Exception as e:
                            logger.error(f"Unexpected error: {e}")
                            print(f"❌ Error: {e}")
                            break

                # This code below won't run due to return in the single agent section
                return True

                orchestrator = Orchestrator(
                    worker_agents=[searcher_agent, analyzer_agent],
                    llm_factory=OpenAIAugmentedLLM,
                    plan_type="iterative",  # Use iterative for dynamic planning
                )

                # Test the orchestrator with a simple initialization
                print("\n🧠 Orchestrator initialized successfully!")

            except Exception as e:
                logger.error(f"Failed to create orchestrator: {e}")
                print(f"❌ Failed to create orchestrator: {e}")
                return False

            print("\n🚀 Ready to answer your questions!")
            print("💭 Example questions:")
            print("   • 'What collections do I have available?'")
            print("   • 'Find information about API authentication'")
            print("   • 'Search for recent updates on machine learning'")

            while True:
                try:
                    question = input("\n❓ Your Question: ").strip()

                    if question.lower() in ["exit", "quit", "q"]:
                        print("👋 Thank you for using ApeRAG Orchestrator Agent!")
                        break

                    if not question:
                        continue

                    print("🔍 Orchestrating multi-agent search...")

                    try:
                        orchestrator.post_tool_call()

                        # Use orchestrator to coordinate agents
                        response = await asyncio.wait_for(
                            orchestrator.generate_str(question),
                            timeout=180.0,  # 3 minutes timeout for complex orchestration
                        )
                        print("\n🤖 **Orchestrated Answer:**")
                        print(response)

                    except asyncio.TimeoutError:
                        print("⏱️ Request timed out. Please try again with a simpler question.")

                    except Exception as e:
                        logger.error(f"Error processing question: {e}")

                        # More specific error handling
                        if "401" in str(e) or "403" in str(e):
                            print("❌ Authentication error: Please check your OpenAI API key")
                            print("💡 Set OPENAI_API_KEY environment variable with a valid key")
                            print("💡 Or check your OpenRouter account balance and permissions")
                        elif "AssertionError" in str(e):
                            print("❌ Orchestrator error: There might be an issue with the agent configuration")
                            print("💡 Try a simpler question or restart the agent")
                        else:
                            print(f"❌ Error processing question: {e}")
                            print("💡 Tip: Try rephrasing your question or check your connection")

                except KeyboardInterrupt:
                    print("\n👋 Thank you for using ApeRAG Orchestrator Agent!")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    print(f"❌ Error: {e}")
                    print("💡 Tip: Try rephrasing your question or check your API key")

    except Exception as e:
        logger.error(f"Failed to initialize MCP connection: {e}")
        print(f"❌ Failed to initialize MCP connection: {e}")
        print("💡 Please check your environment variables and try again")
        return False

    return True


async def main():
    try:
        print("=" * 40)
        print("🚀 Starting ApeRAG Orchestrator Agent...")
        print("=" * 40)

        # Check API key before starting
        openai_key = os.getenv("OPENAI_API_KEY", "sk-test")
        if openai_key == "sk-test":
            print("⚠️  Warning: Using default API key. Set OPENAI_API_KEY for production use.")
        elif openai_key.startswith("sk-or-"):
            print("✓ Using OpenRouter API key")
        else:
            print("✓ Using OpenAI API key")

        success = await interactive_chat()

        if not success:
            print("❌ Agent initialization failed")
            return 1

    except KeyboardInterrupt:
        print("\n👋 ApeRAG Orchestrator Agent shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Allow time for async cleanup
        await asyncio.sleep(0.1)

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
