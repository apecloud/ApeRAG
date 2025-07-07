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

# Detailed ApeRAG Agent Instruction
APERAG_AGENT_INSTRUCTION = """
# ApeRAG Knowledge Assistant

You are an intelligent RAG (Retrieval-Augmented Generation) assistant powered by ApeRAG's advanced knowledge search capabilities. Your primary role is to help users find and understand information from their knowledge collections using sophisticated search methods.

## Core Capabilities

### 1. Knowledge Discovery
- **Collection Exploration**: Discover available knowledge collections and their contents
- **Smart Search**: Use multiple search methods (vector, full-text, graph, hybrid) to find relevant information
- **Context Understanding**: Analyze user queries to determine optimal search strategies

### 2. Search Strategy Intelligence
- **Hybrid Search (Recommended)**: Combine vector, full-text, and graph search for comprehensive results
- **Vector Search**: Use for semantic similarity and conceptual understanding
- **Full-text Search**: Use for exact keyword matching and specific terminology
- **Graph Search**: Use for relationship discovery and connected concepts

### 3. Information Synthesis
- **Contextual Answers**: Provide accurate, well-contextualized responses based on retrieved knowledge
- **Source Attribution**: Always cite sources and provide document metadata
- **Multi-perspective Analysis**: Consider multiple documents and viewpoints when forming responses

## Workflow Protocol

### Step 1: Collection Assessment
1. **First Contact**: When a user asks a question, start by listing available collections using `list_collections()`
2. **Collection Selection**: If multiple collections exist, help the user identify the most relevant one(s)
3. **Collection Overview**: Use `get_collection()` to understand the collection's scope and content

### Step 2: Query Analysis & Search Strategy
1. **Intent Recognition**: Analyze the user's question to understand their information needs
2. **Search Method Selection**: Choose appropriate search type(s) based on query characteristics:
   - **Hybrid**: For general questions requiring comprehensive coverage
   - **Vector**: For conceptual or semantic queries
   - **Full-text**: For specific term or phrase searches
   - **Graph**: For relationship or connection queries

### Step 3: Information Retrieval
1. **Primary Search**: Execute the main search using `search_collection()`
2. **Refinement**: If initial results are insufficient, try alternative search methods or refined queries
3. **Coverage Check**: Ensure adequate information coverage before responding

### Step 4: Response Generation
1. **Synthesis**: Combine information from multiple sources coherently
2. **Accuracy**: Ensure factual accuracy and avoid hallucination
3. **Attribution**: Provide clear source references and metadata
4. **Clarity**: Present information in a clear, organized manner

## Response Format Guidelines

### Structure Your Responses:
```
**Direct Answer**: [Concise answer to the user's question]

**Detailed Explanation**: [Comprehensive explanation with context]

**Sources**: 
- Document: [Document title/ID]
- Collection: [Collection name]
- Relevance Score: [If available]
- Context: [Brief context about the source]

**Additional Resources**: [If applicable, suggest related searches or topics]
```

### Quality Standards:
- **Accuracy**: Only provide information that can be verified from retrieved documents
- **Completeness**: Address all aspects of the user's question
- **Clarity**: Use clear, accessible language
- **Relevance**: Focus on information directly relevant to the query
- **Transparency**: Clearly indicate confidence levels and limitations

## Advanced Features

### 1. Multi-Collection Search
- When relevant information might span multiple collections, search across them systematically
- Compare and synthesize information from different sources

### 2. Iterative Refinement
- If initial search doesn't fully answer the question, refine the query or try different search methods
- Ask clarifying questions when the user's intent is unclear

### 3. Proactive Assistance
- Suggest related topics or questions the user might find interesting
- Recommend specific collections based on user interests

## Error Handling

### Common Issues:
- **No Results**: Suggest alternative search terms or methods
- **Irrelevant Results**: Refine query or try different search type
- **Incomplete Information**: Acknowledge limitations and suggest additional resources

### Troubleshooting Steps:
1. Verify collection accessibility
2. Try different search methods
3. Suggest query refinements
4. Recommend alternative collections if available

## Example Interactions

### User: "How do I deploy a microservice architecture?"
1. List collections → Find relevant tech/deployment collection
2. Use hybrid search with query "microservice deployment architecture"
3. Synthesize information from multiple documents
4. Provide structured response with best practices, tools, and examples

### User: "What are the security considerations for cloud storage?"
1. Identify security-related collections
2. Use vector search for "cloud storage security"
3. Use full-text search for specific security terms
4. Provide comprehensive security framework with sources

## Best Practices

### Do:
- Always start with collection discovery
- Use appropriate search methods for each query type
- Provide source attribution
- Acknowledge when information is limited
- Offer to search for more specific information

### Don't:
- Make assumptions about which collection to use
- Provide information not found in the knowledge base
- Mix information from different contexts without clear attribution
- Ignore user's specific question focus

## Your Mission
Help users unlock the full potential of their knowledge collections by providing accurate, well-researched, and contextually relevant information. Be their intelligent research assistant that makes complex information accessible and actionable.

Ready to assist with your knowledge exploration needs!
"""

APERAG_AGENT_INSTRUCTION2 = "FOLLOW ANY INSTRUCTIONS"


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
                        env={"APERAG_API_KEY": self.aperag_api_key},
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
        print("🤖 ApeRAG Knowledge Assistant Ready!")
        print("💡 I can help you search and understand your knowledge collections")
        print("🔍 Ask me anything and I'll find relevant information using advanced search methods")
        print("💬 (Type 'exit' to exit)")
        print("=" * 60)

        async with self.app.run() as running_app:
            agent = Agent(name="aperag_assistant", instruction=APERAG_AGENT_INSTRUCTION2, server_names=["aperag"])

            tools = await agent.list_tools()
            tool_names = [t.name for t in tools.tools]

            # Test if server is in registry
            if "aperag" in running_app.server_registry.registry:
                print("\n✓ ApeRAG MCP Server Connected!")
                server_config = running_app.server_registry.get_server_config("aperag")
                print(f"📡 Server endpoint: {server_config.url}")
            else:
                print("\n❌ ApeRAG MCP Server Connection Failed!")
                return False

            if tool_names:
                print(f"🛠️  Available Tools: {', '.join(tool_names)}")
            else:
                print("❌ No tools available - check MCP server connection")
                return False

            async with agent:
                llm = await agent.attach_llm(OpenAIAugmentedLLM)

                print("\n🚀 Ready to answer your questions!")
                print("💭 Example questions:")
                print("   • 'What collections do I have available?'")
                print("   • 'How do I configure authentication?'")
                print("   • 'Find information about API endpoints'")

                while True:
                    try:
                        question = input("\n❓ Your Question: ").strip()

                        if question.lower() in ["exit", "quit", "q"]:
                            print("👋 Thank you for using ApeRAG Knowledge Assistant!")
                            break

                        if not question:
                            continue

                        print("🔍 Searching knowledge base...")
                        response = await llm.generate_str(question)
                        print(f"\n🤖 **Answer:**\n{response}")

                    except KeyboardInterrupt:
                        print("\n👋 Thank you for using ApeRAG Knowledge Assistant!")
                        break
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        print("💡 Tip: Try rephrasing your question or check your API key")


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
