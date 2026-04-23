"""
Reader Providers

Different web content extraction implementations.
"""

from .jina_read_provider import JinaReaderProvider
from .trafilatura_read_provider import TrafilaturaProvider

__all__ = [
    "TrafilaturaProvider",
    "JinaReaderProvider",
]
