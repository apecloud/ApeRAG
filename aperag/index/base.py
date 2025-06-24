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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class IndexType(Enum):
    """Index type enumeration"""

    VECTOR = "VECTOR"
    FULLTEXT = "FULLTEXT"
    GRAPH = "GRAPH"


class IndexStatus(Enum):
    """Index status enumeration"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class IndexResult:
    """Standard index operation result"""

    success: bool
    index_type: IndexType
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "index_type": self.index_type.value,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseIndexer(ABC):
    """Abstract base class for all indexers"""

    def __init__(self, index_type: IndexType):
        self.index_type = index_type

    @abstractmethod
    def create_index(self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """
        Create index for document

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of index creation
        """
        pass

    @abstractmethod
    def update_index(self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """
        Update existing index for document

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of index update
        """
        pass

    @abstractmethod
    def delete_index(self, document_id: int, collection, **kwargs) -> IndexResult:
        """
        Delete index for document

        Args:
            document_id: Document ID
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of index deletion
        """
        pass

    @abstractmethod
    def is_enabled(self, collection) -> bool:
        """
        Check if this index type is enabled for the collection

        Args:
            collection: Collection object

        Returns:
            bool: True if enabled
        """
        pass


class AsyncIndexer(BaseIndexer):
    """Base class for asynchronous indexers (like graph indexing)"""

    @abstractmethod
    def create_index_async(
        self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs
    ) -> IndexResult:
        """
        Create index asynchronously
        Returns immediately with status=RUNNING
        """
        pass

    @abstractmethod
    def update_index_async(
        self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs
    ) -> IndexResult:
        """
        Update index asynchronously
        Returns immediately with status=RUNNING
        """
        pass
