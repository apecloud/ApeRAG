"""
Web Reader Module

Provides web content reading and extraction capabilities.
"""

from .base_reader import BaseReaderProvider
from .reader_service import ReaderService

__all__ = [
    "ReaderService",
    "BaseReaderProvider",
]
