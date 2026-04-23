"""
Web Search Module

Provides web search capabilities through different search engines.
"""

from .base_search import BaseSearchProvider
from .search_service import SearchService

__all__ = [
    "SearchService",
    "BaseSearchProvider",
]
