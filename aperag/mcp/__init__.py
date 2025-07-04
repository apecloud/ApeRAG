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
ApeRAG MCP (Model Context Protocol) Integration

This module provides MCP server functionality for ApeRAG, allowing
MCP clients to interact with ApeRAG's search and collection management
capabilities.

Features:
- Hybrid search (vector + fulltext + graph)
- Collection management
- API key authentication
- Resource and prompt providers
"""

from .server import mcp_server

__all__ = ["mcp_server"]
