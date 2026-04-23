"""
Search Providers

Different search engine implementations.
"""

from .duckduckgo_search_provider import DuckDuckGoProvider
from .jina_search_provider import JinaSearchProvider

__all__ = [
    "DuckDuckGoProvider",
    "JinaSearchProvider",
]
