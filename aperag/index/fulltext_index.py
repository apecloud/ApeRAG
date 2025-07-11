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

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from elasticsearch import AsyncElasticsearch, Elasticsearch, NotFoundError

from aperag.config import settings
from aperag.db.ops import db_ops
from aperag.index.base import BaseIndexer, IndexResult, IndexType
from aperag.query.query import DocumentWithScore
from aperag.utils.utils import generate_fulltext_index_name

logger = logging.getLogger(__name__)


class FulltextIndexer(BaseIndexer):
    """Fulltext index implementation"""

    def __init__(self, es_host: str = None):
        super().__init__(IndexType.FULLTEXT)
        self.es_host = es_host if es_host else settings.es_host
        # Add timeout configuration for sync ES client
        self.es = Elasticsearch(
            self.es_host,
            request_timeout=settings.es_timeout,  # Use timeout from settings
            max_retries=settings.es_max_retries,  # Use max retries from settings
            retry_on_timeout=True,  # Retry on timeout errors
        )
        # Add timeout configuration for async ES client using settings
        self.async_es = AsyncElasticsearch(
            self.es_host,
            request_timeout=settings.es_timeout,  # Use timeout from settings
            max_retries=settings.es_max_retries,  # Use max retries from settings
            retry_on_timeout=True,  # Retry on timeout errors
        )

    def is_enabled(self, collection) -> bool:
        """Fulltext indexing is always enabled"""
        return True

    def create_index(self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """
        Create fulltext index for document

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of fulltext index creation
        """
        try:
            # Only create fulltext index when there is content
            if not content or not content.strip():
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "No content to index", "status": "skipped"},
                )

            # Get document for name
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")

            # Insert into fulltext index
            index_name = generate_fulltext_index_name(collection.id)
            self.insert_document(index_name, document_id, document.name, content)

            logger.info(f"Fulltext index created for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"index_name": index_name, "document_name": document.name},
                metadata={"content_length": len(content), "content_words": len(content.split()) if content else 0},
            )

        except Exception as e:
            logger.error(f"Fulltext index creation failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Fulltext index creation failed: {str(e)}"
            )

    def update_index(self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """
        Update fulltext index for document

        Args:
            document_id: Document ID
            content: Document content
            doc_parts: Parsed document parts
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of fulltext index update
        """
        try:
            # Get document for name
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")

            index_name = generate_fulltext_index_name(collection.id)

            # Remove old index
            try:
                self.remove_document(index_name, document_id)
                logger.debug(f"Removed old fulltext index for document {document_id}")
            except Exception as e:
                logger.warning(f"Failed to remove old fulltext index for document {document_id}: {str(e)}")

            # Create new index if there is content
            if content and content.strip():
                self.insert_document(index_name, document_id, document.name, content)
                logger.info(f"Fulltext index updated for document {document_id}")

                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    data={"index_name": index_name, "document_name": document.name},
                    metadata={
                        "content_length": len(content),
                        "content_words": len(content.split()),
                        "operation": "updated",
                    },
                )
            else:
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "No content to index", "status": "skipped"},
                )

        except Exception as e:
            logger.error(f"Fulltext index update failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Fulltext index update failed: {str(e)}"
            )

    def delete_index(self, document_id: int, collection, **kwargs) -> IndexResult:
        """
        Delete fulltext index for document

        Args:
            document_id: Document ID
            collection: Collection object
            **kwargs: Additional parameters

        Returns:
            IndexResult: Result of fulltext index deletion
        """
        try:
            index_name = generate_fulltext_index_name(collection.id)
            self.remove_document(index_name, document_id)

            logger.info(f"Fulltext index deleted for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"index_name": index_name},
                metadata={"operation": "deleted"},
            )

        except Exception as e:
            logger.error(f"Fulltext index deletion failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Fulltext index deletion failed: {str(e)}"
            )

    def remove_document(self, index, doc_id):
        if self.es.indices.exists(index=index).body:
            try:
                self.es.delete(index=index, id=f"{doc_id}")
            except NotFoundError:
                logger.warning("document %s not found in index %s", doc_id, index)
        else:
            logger.warning("index %s not exists", index)

    def insert_document(self, index, doc_id, doc_name, content):
        if self.es.indices.exists(index=index).body:
            doc = {
                "name": doc_name,
                "content": content,
            }
            self.es.index(index=index, id=f"{doc_id}", document=doc)
        else:
            logger.warning("index %s not exists", index)

    async def search_document(self, index: str, keywords: List[str], topk=3) -> List[DocumentWithScore]:
        try:
            resp = await self.async_es.indices.exists(index=index)
            if not resp.body:
                return []

            if not keywords:
                return []

            query = {
                "bool": {
                    "should": [{"match": {"content": keyword}} for keyword in keywords],
                    "minimum_should_match": "80%",
                    # "minimum_should_match": "-1",
                },
            }
            sort = [{"_score": {"order": "desc"}}]
            resp = await self.async_es.search(index=index, query=query, sort=sort, size=topk)
            hits = resp.body["hits"]
            result = []
            for hit in hits["hits"]:
                result.append(
                    DocumentWithScore(
                        text=hit["_source"]["content"],
                        score=hit["_score"],
                        metadata={
                            "source": hit["_source"]["name"],
                        },
                    )
                )
            return result
        except Exception as e:
            logger.error(f"Failed to search documents in index {index}: {str(e)}")
            # Return empty list on error to allow the flow to continue
            return []


# Global instance
fulltext_indexer = FulltextIndexer()


class KeywordExtractor(object):
    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx

    async def extract(self, text):
        raise NotImplementedError


class IKExtractor(KeywordExtractor):
    """
    Extract keywords from text using IK
    """

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        # Add timeout configuration for ES client using settings
        self.client = AsyncElasticsearch(
            ctx.get("es_host", "http://127.0.0.1:9200"),
            request_timeout=ctx.get("es_timeout", settings.es_timeout),  # Use timeout from context or settings
            max_retries=ctx.get("es_max_retries", settings.es_max_retries),  # Use max retries from context or settings
            retry_on_timeout=True,  # Retry on timeout errors
        )
        self.index_name = ctx["index_name"]
        stop_words_path = Path(__file__).parent / "misc" / "stopwords.txt"
        if os.path.exists(stop_words_path):
            with open(stop_words_path) as f:
                self.stop_words = set(f.read().splitlines())
        else:
            self.stop_words = set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()

    async def extract(self, text):
        try:
            resp = await self.client.indices.exists(index=self.index_name)
            if not resp.body:
                logger.warning("index %s not exists", self.index_name)
                return []

            resp = await self.client.indices.analyze(index=self.index_name, body={"text": text, "analyzer": "ik_smart"})
            tokens = {}
            for item in resp.body["tokens"]:
                token = item["token"]
                if token in self.stop_words:
                    continue
                tokens[token] = True
            return tokens.keys()
        except Exception as e:
            logger.error(f"Failed to extract keywords for index {self.index_name}: {str(e)}")
            # Return empty list on error to allow the flow to continue
            return []


es = Elasticsearch(
    settings.es_host,
    request_timeout=settings.es_timeout,  # Use timeout from settings
    max_retries=settings.es_max_retries,  # Use max retries from settings
    retry_on_timeout=True,  # Retry on timeout errors
)


def create_index(index):
    if not es.indices.exists(index=index).body:
        mapping = {
            "properties": {"content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"}}
        }
        es.indices.create(index=index, body={"mappings": mapping})
    else:
        logger.warning("index %s already exists", index)


def delete_index(index):
    if es.indices.exists(index=index).body:
        es.indices.delete(index=index)
