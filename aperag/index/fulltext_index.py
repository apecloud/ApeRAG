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
from typing import Any, Dict, List, Tuple

from elasticsearch import AsyncElasticsearch, Elasticsearch

from aperag.config import settings
from aperag.db.ops import db_ops
from aperag.docparser.chunking import rechunk
from aperag.index.base import BaseIndexer, IndexResult, IndexType
from aperag.query.query import DocumentWithScore
from aperag.utils.tokenizer import get_default_tokenizer
from aperag.utils.utils import generate_fulltext_index_name

logger = logging.getLogger(__name__)


def _create_es_client_config() -> Dict[str, Any]:
    """Create common ES client configuration"""
    return {
        "request_timeout": settings.es_timeout,
        "max_retries": settings.es_max_retries,
        "retry_on_timeout": True,
    }


class FulltextIndexer(BaseIndexer):
    """Fulltext index implementation"""

    def __init__(self, es_host: str = None):
        super().__init__(IndexType.FULLTEXT)
        self.es_host = es_host if es_host else settings.es_host
        config = _create_es_client_config()
        self.es = Elasticsearch(self.es_host, **config)
        self.async_es = AsyncElasticsearch(self.es_host, **config)

    def is_enabled(self, collection) -> bool:
        """Fulltext indexing is always enabled"""
        return True

    def _extract_chunk_data(self, part) -> Tuple[str, str, Dict[str, Any]]:
        """Extract chunk content, title and metadata from a document part"""
        if not hasattr(part, 'content') or not part.content or not part.content.strip():
            return "", "", {}

        chunk_content = part.content.strip()
        chunk_metadata = part.metadata.copy() if hasattr(part, 'metadata') and part.metadata else {}
        titles = chunk_metadata.get('titles', [])
        title_text = " > ".join(titles) if titles else ""

        return chunk_content, title_text, chunk_metadata

    def _process_chunks(self, document_id: int, doc_parts: List[Any], document_name: str, index_name: str) -> Tuple[int, int]:
        """Process and insert all chunks for a document. Returns (chunk_count, total_content_length)"""
        chunk_count = 0
        total_content_length = 0

        chunk_size = settings.chunk_size
        chunk_overlap_size = settings.chunk_overlap_size
        tokenizer = get_default_tokenizer()

        # Rechunk the document parts (resulting in text parts)
        # After rechunk(), parts only contains TextPart
        chunked_parts = rechunk(doc_parts, chunk_size, chunk_overlap_size, tokenizer)

        for chunk_idx, part in enumerate(chunked_parts):
            chunk_content, title_text, chunk_metadata = self._extract_chunk_data(part)
            if not chunk_content:
                continue

            chunk_id = f"{document_id}_{chunk_idx}"
            self._insert_chunk(index_name, chunk_id, document_id, document_name, chunk_content, title_text, chunk_metadata)
            chunk_count += 1
            total_content_length += len(chunk_content)

        return chunk_count, total_content_length

    def _create_success_result(self, index_name: str, document_name: str, chunk_count: int, total_content_length: int, operation: str = "created") -> IndexResult:
        """Create a success IndexResult with chunk statistics"""
        return IndexResult(
            success=True,
            index_type=self.index_type,
            data={"index_name": index_name, "document_name": document_name, "chunk_count": chunk_count},
            metadata={
                "total_content_length": total_content_length,
                "chunk_count": chunk_count,
                "avg_chunk_length": total_content_length // chunk_count if chunk_count > 0 else 0,
                "operation": operation,
            },
        )

    def create_index(self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """Create fulltext index for document chunks"""
        try:
            if not doc_parts:
                logger.info(f"No doc_parts to index for document {document_id}")
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "No doc_parts to index", "status": "skipped"},
                )

            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")

            index_name = generate_fulltext_index_name(collection.id)
            chunk_count, total_content_length = self._process_chunks(document_id, doc_parts, document.name, index_name)

            logger.info(f"Fulltext index created for document {document_id} with {chunk_count} chunks")
            return self._create_success_result(index_name, document.name, chunk_count, total_content_length, "created")

        except Exception as e:
            logger.error(f"Fulltext index creation failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Fulltext index creation failed: {str(e)}"
            )

    def update_index(self, document_id: int, content: str, doc_parts: List[Any], collection, **kwargs) -> IndexResult:
        """Update fulltext index for document chunks"""
        try:
            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")

            index_name = generate_fulltext_index_name(collection.id)

            # Remove old chunks for this document
            try:
                self._remove_document_chunks(index_name, document_id)
                logger.debug(f"Removed old fulltext chunks for document {document_id}")
            except Exception as e:
                logger.warning(f"Failed to remove old fulltext chunks for document {document_id}: {str(e)}")

            # Create new chunks if there are doc_parts
            if doc_parts:
                chunk_count, total_content_length = self._process_chunks(document_id, doc_parts, document.name, index_name)
                logger.info(f"Fulltext index updated for document {document_id} with {chunk_count} chunks")
                return self._create_success_result(index_name, document.name, chunk_count, total_content_length, "updated")
            else:
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "No doc_parts to index", "status": "skipped"},
                )

        except Exception as e:
            logger.error(f"Fulltext index update failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Fulltext index update failed: {str(e)}"
            )

    def delete_index(self, document_id: int, collection, **kwargs) -> IndexResult:
        """Delete fulltext index for document chunks"""
        try:
            index_name = generate_fulltext_index_name(collection.id)
            deleted_count = self._remove_document_chunks(index_name, document_id)

            logger.info(f"Fulltext index deleted for document {document_id}, removed {deleted_count} chunks")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"index_name": index_name, "deleted_chunks": deleted_count},
                metadata={"operation": "deleted", "deleted_chunks": deleted_count},
            )

        except Exception as e:
            logger.error(f"Fulltext index deletion failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Fulltext index deletion failed: {str(e)}"
            )

    def _remove_document_chunks(self, index: str, doc_id: int) -> int:
        """Remove all chunks for a specific document"""
        if not self.es.indices.exists(index=index).body:
            logger.warning("index %s not exists", index)
            return 0

        try:
            query = {
                "query": {
                    "term": {
                        "document_id": doc_id
                    }
                }
            }
            response = self.es.delete_by_query(index=index, body=query)
            deleted_count = response.get('deleted', 0)
            logger.info(f"Deleted {deleted_count} chunks for document {doc_id} from index {index}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to remove chunks for document {doc_id} from index {index}: {str(e)}")
            return 0

    def _insert_chunk(self, index: str, chunk_id: str, doc_id: int, doc_name: str, content: str, title_text: str = "", metadata: Dict[str, Any] = None):
        """Insert a document chunk into the fulltext index"""
        if not self.es.indices.exists(index=index).body:
            logger.warning("index %s not exists", index)
            return

        doc = {
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "name": doc_name,
            "content": content,
            "title": title_text,
            "metadata": metadata or {}
        }
        self.es.index(index=index, id=chunk_id, document=doc)

    async def search_document(self, index: str, keywords: List[str], topk=3) -> List[DocumentWithScore]:
        try:
            resp = await self.async_es.indices.exists(index=index)
            if not resp.body:
                return []

            if not keywords:
                return []

            # Search in both content and title fields
            query = {
                "bool": {
                    "should": [
                        {"match": {"content": keyword}} for keyword in keywords
                    ] + [
                        {"match": {"title": keyword}} for keyword in keywords
                    ],
                    "minimum_should_match": "80%",
                },
            }
            sort = [{"_score": {"order": "desc"}}]
            resp = await self.async_es.search(index=index, query=query, sort=sort, size=topk)
            hits = resp.body["hits"]
            result = []
            for hit in hits["hits"]:
                source = hit["_source"]
                metadata = {
                    "source": source.get("name", ""),
                    "document_id": source.get("document_id"),
                    "chunk_id": source.get("chunk_id"),
                }

                # Add title if available
                if source.get("title"):
                    metadata["title"] = source["title"]

                # Add chunk metadata if available
                if source.get("metadata"):
                    metadata.update(source["metadata"])

                result.append(
                    DocumentWithScore(
                        text=source["content"],
                        score=hit["_score"],
                        metadata=metadata,
                    )
                )
            return result
        except Exception as e:
            logger.error(f"Failed to search documents in index {index}: {str(e)}")
            # Return empty list on error to allow the flow to continue
            return []


# Global instance
fulltext_indexer = FulltextIndexer()


class KeywordExtractor:
    """Base class for keyword extraction"""

    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx

    async def extract(self, text: str) -> List[str]:
        raise NotImplementedError


class IKExtractor(KeywordExtractor):
    """Extract keywords from text using IK analyzer"""

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        config = _create_es_client_config()
        config.update({
            "request_timeout": ctx.get("es_timeout", settings.es_timeout),
            "max_retries": ctx.get("es_max_retries", settings.es_max_retries),
        })

        self.client = AsyncElasticsearch(
            ctx.get("es_host", settings.es_host),
            **config
        )
        self.index_name = ctx["index_name"]
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self) -> set:
        """Load stop words from file"""
        stop_words_path = Path(__file__).parent / "misc" / "stopwords.txt"
        if os.path.exists(stop_words_path):
            with open(stop_words_path) as f:
                return set(f.read().splitlines())
        return set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()

    async def extract(self, text: str) -> List[str]:
        try:
            resp = await self.client.indices.exists(index=self.index_name)
            if not resp.body:
                logger.warning("index %s not exists", self.index_name)
                return []

            resp = await self.client.indices.analyze(
                index=self.index_name,
                body={"text": text, "analyzer": "ik_smart"}
            )

            tokens = set()
            for item in resp.body["tokens"]:
                token = item["token"]
                if token not in self.stop_words:
                    tokens.add(token)
            return list(tokens)

        except Exception as e:
            logger.error(f"Failed to extract keywords for index {self.index_name}: {str(e)}")
            return []


def create_index(index: str):
    """Create ES index with proper mapping for chunks"""
    config = _create_es_client_config()
    es = Elasticsearch(settings.es_host, **config)

    if not es.indices.exists(index=index).body:
        mapping = {
            "properties": {
                "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                "title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                "document_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "name": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": False}
            }
        }
        es.indices.create(index=index, body={"mappings": mapping})
    else:
        logger.warning("index %s already exists", index)


def delete_index(index: str):
    """Delete ES index"""
    config = _create_es_client_config()
    es = Elasticsearch(settings.es_host, **config)

    if es.indices.exists(index=index).body:
        es.indices.delete(index=index)
