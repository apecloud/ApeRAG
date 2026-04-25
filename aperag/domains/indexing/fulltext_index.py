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

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from elasticsearch import AsyncElasticsearch, Elasticsearch

from aperag.config import settings
from aperag.db.ops import db_ops
from aperag.docparser.chunking import rechunk
from aperag.domains.indexing.base import BaseIndexer, IndexResult, IndexType
from aperag.llm.completion.completion_service import CompletionService
from aperag.platform.query.query import DocumentWithScore
from aperag.schema.utils import parseCollectionConfig
from aperag.utils.tokenizer import get_default_tokenizer
from aperag.utils.utils import (
    generate_fulltext_index_alias,
    generate_fulltext_index_name,
    generate_fulltext_physical_index_name,
)

logger = logging.getLogger(__name__)


def _create_es_client_config() -> Dict[str, Any]:
    """Create common ES client configuration"""
    return {
        "request_timeout": settings.es_timeout,
        "max_retries": settings.es_max_retries,
        "retry_on_timeout": True,
    }


def _fulltext_mapping() -> Dict[str, Any]:
    return {
        "properties": {
            "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
            "title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
            "collection_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "chat_id": {"type": "keyword"},
            "name": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": False},
        }
    }


def _fulltext_settings() -> Dict[str, Any]:
    return {
        "number_of_shards": settings.es_fulltext_number_of_shards,
        "number_of_replicas": settings.es_fulltext_number_of_replicas,
    }


def _get_sync_es() -> Elasticsearch:
    return Elasticsearch(settings.es_host, **_create_es_client_config())


def list_alias_targets(alias: Optional[str] = None, es: Optional[Elasticsearch] = None) -> List[str]:
    alias = alias or generate_fulltext_index_alias()
    es = es or _get_sync_es()
    if not es.indices.exists_alias(name=alias).body:
        return []
    return list(es.indices.get_alias(name=alias).body.keys())


def resolve_alias_target(alias: Optional[str] = None, es: Optional[Elasticsearch] = None) -> Optional[str]:
    alias = alias or generate_fulltext_index_alias()
    targets = list_alias_targets(alias, es=es)
    if not targets:
        return None
    if len(targets) > 1:
        raise ValueError(f"fulltext alias {alias} unexpectedly points to multiple indices: {targets}")
    return targets[0]


def ensure_physical_index_exists(
    physical_index: Optional[str] = None,
    es: Optional[Elasticsearch] = None,
) -> Dict[str, Any]:
    physical_index = physical_index or generate_fulltext_physical_index_name()
    es = es or _get_sync_es()

    created_physical = False
    if not es.indices.exists(index=physical_index).body:
        es.indices.create(
            index=physical_index, body={"settings": _fulltext_settings(), "mappings": _fulltext_mapping()}
        )
        created_physical = True

    return {"physical_index": physical_index, "created_physical": created_physical}


def ensure_shared_index(
    alias: Optional[str] = None,
    physical_index: Optional[str] = None,
    es: Optional[Elasticsearch] = None,
) -> Dict[str, Any]:
    alias = alias or generate_fulltext_index_alias()
    es = es or _get_sync_es()

    current_target = resolve_alias_target(alias, es=es)
    if current_target is not None:
        if physical_index is not None:
            ensure_physical_index_exists(physical_index=physical_index, es=es)
        return {
            "alias": alias,
            "physical_index": current_target,
            "created_physical": False,
            "targets": [current_target],
        }

    ensured = ensure_physical_index_exists(physical_index=physical_index, es=es)
    es.indices.put_alias(index=ensured["physical_index"], name=alias)
    return {
        "alias": alias,
        "physical_index": ensured["physical_index"],
        "created_physical": ensured["created_physical"],
        "targets": [ensured["physical_index"]],
    }


def switch_shared_index_alias(
    target_index: str,
    alias: Optional[str] = None,
    es: Optional[Elasticsearch] = None,
) -> Dict[str, Any]:
    alias = alias or generate_fulltext_index_alias()
    es = es or _get_sync_es()
    ensure_physical_index_exists(physical_index=target_index, es=es)

    current_targets = list_alias_targets(alias, es=es)
    if current_targets == [target_index]:
        return {"alias": alias, "target_index": target_index, "previous_targets": current_targets}

    if current_targets:
        actions = [{"remove": {"index": current, "alias": alias}} for current in current_targets]
        actions.append({"add": {"index": target_index, "alias": alias}})
        es.indices.update_aliases(body={"actions": actions})
    else:
        es.indices.put_alias(index=target_index, name=alias)
    return {"alias": alias, "target_index": target_index, "previous_targets": current_targets}


def count_documents(index: str, collection_id: Optional[str] = None, es: Optional[Elasticsearch] = None) -> int:
    es = es or _get_sync_es()
    if not es.indices.exists(index=index).body:
        return 0
    if collection_id is None:
        return int(es.count(index=index).body["count"])
    return int(
        es.count(
            index=index,
            query={"term": {"collection_id": str(collection_id)}},
            routing=str(collection_id),
        ).body["count"]
    )


def delete_collection_documents(
    collection_id: str,
    index: Optional[str] = None,
    es: Optional[Elasticsearch] = None,
) -> int:
    es = es or _get_sync_es()
    index = index or generate_fulltext_index_name()
    if not es.indices.exists(index=index).body:
        return 0
    response = es.delete_by_query(
        index=index,
        body={"query": {"term": {"collection_id": str(collection_id)}}},
        conflicts="proceed",
        refresh=True,
        routing=str(collection_id),
    )
    return int(response.get("deleted", 0))


def build_legacy_reindex_body(
    source_index: str, collection_id: str, dest_index: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "source": {"index": source_index},
        "dest": {"index": dest_index or generate_fulltext_physical_index_name()},
        "script": {
            "lang": "painless",
            "source": """
ctx._source.collection_id = params.collection_id;
ctx._routing = params.collection_id;
if (ctx._source.document_id != null) {
  ctx._source.document_id = ctx._source.document_id.toString();
}
if (ctx._source.chunk_id == null) {
  ctx._source.chunk_id = ctx._id;
} else {
  ctx._source.chunk_id = ctx._source.chunk_id.toString();
}
if (ctx._source.chat_id == null && ctx._source.metadata != null && ctx._source.metadata.chat_id != null) {
  ctx._source.chat_id = ctx._source.metadata.chat_id.toString();
} else if (ctx._source.chat_id != null) {
  ctx._source.chat_id = ctx._source.chat_id.toString();
}
""",
            "params": {"collection_id": str(collection_id)},
        },
    }


def migrate_legacy_index(
    source_index: str,
    collection_id: str,
    dest_index: Optional[str] = None,
    es: Optional[Elasticsearch] = None,
) -> Dict[str, Any]:
    es = es or _get_sync_es()
    target_index = dest_index or generate_fulltext_physical_index_name()
    ensure_physical_index_exists(physical_index=target_index, es=es)
    body = build_legacy_reindex_body(source_index, collection_id, dest_index=dest_index)
    return es.reindex(body=body, wait_for_completion=True, refresh=True, conflicts="proceed")


class FulltextIndexer(BaseIndexer):
    """Fulltext index implementation"""

    def __init__(self, es_host: str = None):
        super().__init__(IndexType.FULLTEXT)
        self.es_host = es_host if es_host else settings.es_host
        config = _create_es_client_config()
        self.es = Elasticsearch(self.es_host, **config)
        self.async_es = AsyncElasticsearch(self.es_host, **config)

    def is_enabled(self, collection) -> bool:
        """Fulltext indexing follows the collection contract."""
        return parseCollectionConfig(collection.config).enable_fulltext is not False

    def _extract_chunk_data(self, part) -> Tuple[str, str, Dict[str, Any]]:
        """Extract chunk content, title and metadata from a document part"""
        if not hasattr(part, "content") or not part.content or not part.content.strip():
            return "", "", {}

        chunk_content = part.content.strip()
        chunk_metadata = part.metadata.copy() if hasattr(part, "metadata") and part.metadata else {}
        titles = chunk_metadata.get("titles", [])
        title_text = " > ".join(titles) if titles else ""

        return chunk_content, title_text, chunk_metadata

    def _process_chunks(
        self,
        document_id: int,
        doc_parts: List[Any],
        document_name: str,
        index_name: str,
        collection_id: str,
    ) -> Tuple[int, int]:
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
            self._insert_chunk(
                index_name,
                chunk_id,
                document_id,
                collection_id,
                document_name,
                chunk_content,
                title_text,
                chunk_metadata,
            )
            chunk_count += 1
            total_content_length += len(chunk_content)

        return chunk_count, total_content_length

    def _create_success_result(
        self,
        index_name: str,
        document_name: str,
        chunk_count: int,
        total_content_length: int,
        operation: str = "created",
    ) -> IndexResult:
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
            if not self.is_enabled(collection):
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "Fulltext indexing disabled", "status": "skipped"},
                )

            # Filter out non-text parts
            doc_parts = [part for part in doc_parts if hasattr(part, "content") and part.content]

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

            index_name = generate_fulltext_index_name()
            create_index(index_name)
            chunk_count, total_content_length = self._process_chunks(
                document_id,
                doc_parts,
                document.name,
                index_name,
                str(collection.id),
            )

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
            if not self.is_enabled(collection):
                return IndexResult(
                    success=True,
                    index_type=self.index_type,
                    metadata={"message": "Fulltext indexing disabled", "status": "skipped"},
                )

            document = db_ops.query_document_by_id(document_id)
            if not document:
                raise Exception(f"Document {document_id} not found")

            index_name = generate_fulltext_index_name()
            create_index(index_name)

            # Remove old chunks for this document
            try:
                self._remove_document_chunks(index_name, document_id, collection_id=str(collection.id))
                logger.debug(f"Removed old fulltext chunks for document {document_id}")
            except Exception as e:
                logger.warning(f"Failed to remove old fulltext chunks for document {document_id}: {str(e)}")

            # Filter out non-text parts
            doc_parts = [part for part in doc_parts if hasattr(part, "content") and part.content]

            # Create new chunks if there are doc_parts
            if doc_parts:
                chunk_count, total_content_length = self._process_chunks(
                    document_id,
                    doc_parts,
                    document.name,
                    index_name,
                    str(collection.id),
                )
                logger.info(f"Fulltext index updated for document {document_id} with {chunk_count} chunks")
                return self._create_success_result(
                    index_name, document.name, chunk_count, total_content_length, "updated"
                )
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
            index_name = generate_fulltext_index_name()
            deleted_count = self._remove_document_chunks(index_name, document_id, collection_id=str(collection.id))

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

    def _remove_document_chunks(self, index: str, doc_id: int, collection_id: Optional[str] = None) -> int:
        """Remove all chunks for a specific document"""
        if not self.es.indices.exists(index=index).body:
            logger.warning("index %s not exists", index)
            return 0

        try:
            filters = [{"term": {"document_id": str(doc_id)}}]
            if collection_id is not None:
                filters.append({"term": {"collection_id": str(collection_id)}})
            query = {"query": {"bool": {"filter": filters}}}
            request_kwargs = {"index": index, "body": query}
            if collection_id is not None:
                request_kwargs["routing"] = str(collection_id)
            response = self.es.delete_by_query(**request_kwargs)
            deleted_count = response.get("deleted", 0)
            logger.info(f"Deleted {deleted_count} chunks for document {doc_id} from index {index}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to remove chunks for document {doc_id} from index {index}: {str(e)}")
            return 0

    def _insert_chunk(
        self,
        index: str,
        chunk_id: str,
        doc_id: int,
        collection_id: str,
        doc_name: str,
        content: str,
        title_text: str = "",
        metadata: Dict[str, Any] = None,
    ):
        """Insert a document chunk into the fulltext index"""
        if not self.es.indices.exists(index=index).body:
            logger.warning("index %s not exists", index)
            return

        doc = {
            "collection_id": collection_id,
            "document_id": doc_id,
            "chunk_id": chunk_id,
            "chat_id": (metadata or {}).get("chat_id"),
            "name": doc_name,
            "content": content,
            "title": title_text,
            "metadata": metadata or {},
        }
        self.es.index(index=index, id=chunk_id, document=doc, routing=str(collection_id))

    async def search_document(
        self, index: str, collection_id: str, keywords: List[str], topk=3, chat_id: str = None
    ) -> List[DocumentWithScore]:
        try:
            resp = await self.async_es.indices.exists(index=index)
            if not resp.body:
                return []

            if not keywords:
                return []

            # Search in both content and title fields
            query = {
                "bool": {
                    "should": [{"match": {"content": keyword}} for keyword in keywords]
                    + [{"match": {"title": keyword}} for keyword in keywords],
                    "minimum_should_match": "80%",
                    "filter": [{"term": {"collection_id": str(collection_id)}}],
                },
            }

            # Add chat_id filter if provided
            if chat_id:
                query["bool"]["filter"].append(
                    {
                        "bool": {
                            "should": [
                                {"term": {"chat_id": str(chat_id)}},
                                {"term": {"metadata.chat_id": str(chat_id)}},
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                )
            sort = [{"_score": {"order": "desc"}}]
            resp = await self.async_es.search(
                index=index,
                query=query,
                sort=sort,
                size=topk,
                routing=str(collection_id),
            )
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
            raise FulltextSearchDegradedError(f"search failed for {index}: {str(e)}") from e


class FulltextSearchDegradedError(RuntimeError):
    """Raised when fulltext search cannot execute and the caller should degrade explicitly."""


# Global instance
fulltext_indexer = FulltextIndexer()


class KeywordExtractor:
    """Base class for keyword extraction"""

    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx

    async def extract(self, text: str) -> List[str]:
        raise NotImplementedError


class IKKeywordExtractor(KeywordExtractor):
    """Extract keywords from text using IK analyzer"""

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        config = _create_es_client_config()
        config.update(
            {
                "request_timeout": ctx.get("es_timeout", settings.es_timeout),
                "max_retries": ctx.get("es_max_retries", settings.es_max_retries),
            }
        )

        self.client = AsyncElasticsearch(ctx.get("es_host", settings.es_host), **config)
        self.index_name = ctx["index_name"]
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self) -> set:
        """Load stop words from file"""
        stop_words_path = Path(__file__).parent.parent / "misc" / "stopwords.txt"
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

            resp = await self.client.indices.analyze(index=self.index_name, body={"text": text, "analyzer": "ik_smart"})

            tokens = set()
            for item in resp.body["tokens"]:
                token = item["token"]
                if token not in self.stop_words:
                    tokens.add(token)
            return list(tokens)

        except Exception as e:
            logger.error(f"Failed to extract keywords for index {self.index_name}: {str(e)}")
            return []


class LLMKeywordExtractor(KeywordExtractor):
    """Extract keywords from text using LLM with tool calling for stable output format"""

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        self.completion_service = self._create_completion_service()

    def _create_completion_service(self) -> Optional[CompletionService]:
        """Create LLM completion service if configured"""
        try:
            # Check if LLM keyword extraction is configured. The setting stores an ApeRAG model id.
            if not settings.llm_keyword_extraction_model:
                return None

            user_id = self.ctx.get("user_id")
            if not user_id:
                logger.warning("User ID not available in context for LLM keyword extraction")
                return None
            row = db_ops.query_model_runtime(settings.llm_keyword_extraction_model, user_id)
            if not row:
                logger.warning("LLM keyword extraction model '%s' not found", settings.llm_keyword_extraction_model)
                return None
            model, account = row
            from aperag.llm.runtime.resolver import resolve_model_invocation_from_records

            invocation = resolve_model_invocation_from_records(model=model, account=account)
            provider = invocation.runner_config.get("provider")
            if not provider:
                provider = "openai" if invocation.runner_type == "openai_compatible" else invocation.provider_type

            return CompletionService(
                provider=provider,
                model=invocation.provider_model_id,
                base_url=invocation.base_url,
                api_key=invocation.api_key,
            )

        except Exception as e:
            logger.warning(f"Failed to create LLM completion service: {str(e)}")
            return None

    async def extract(self, text: str) -> List[str]:
        """Extract keywords using LLM with structured JSON output"""
        if not self.completion_service:
            raise Exception("LLM completion service not available")

        prompt = f"""Extract the most important keywords from the following text. Focus on:
1. Nouns, verbs, and adjectives that capture the main concepts
2. Remove stop words and meaningless terms
3. Keywords should be in the same language as the input text

Text: {text}

Please respond with ONLY a JSON object in the following format:
{{"keywords": ["keyword1", "keyword2", "keyword3", ...]}}

Do not include any other text or explanation, just the JSON object."""

        try:
            response = await self.completion_service.agenerate([], prompt)

            # Try to extract and parse JSON from response
            keywords = self._parse_json_response(response)
            if keywords:
                return keywords[:10]  # Limit to 10 keywords

            # Fallback to simple parsing if JSON parsing failed
            logger.warning("JSON parsing failed, falling back to simple parsing")
            return self._parse_keywords_fallback(response)

        except Exception as e:
            logger.error(f"LLM keyword extraction failed: {str(e)}")
            raise

    def _parse_json_response(self, response: str) -> List[str]:
        """Parse JSON response to extract keywords"""

        # Clean up the response
        response = response.strip()

        # Try to find JSON object in the response
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx]
            try:
                data = json.loads(json_str)
                if isinstance(data, dict) and "keywords" in data:
                    keywords = data["keywords"]
                    if isinstance(keywords, list):
                        # Filter out empty strings and ensure all items are strings
                        filtered_keywords = [str(k).strip() for k in keywords if k and str(k).strip()]
                        return filtered_keywords[:10]  # Limit to 10 keywords
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {str(e)}, response: {json_str}")
        else:
            logger.warning(f"JSON object not found in response: {response}")

        return []

    def _parse_keywords_fallback(self, response: str) -> List[str]:
        """Fallback keyword parsing method"""
        keywords = []
        for line in response.strip().split("\n"):
            keyword = line.strip()
            # Remove common prefixes and clean up
            keyword = keyword.lstrip("- *•").strip()
            if keyword and not keyword.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")):
                # Remove quotes if present
                keyword = keyword.strip("\"'")
                if keyword:
                    keywords.append(keyword)

        return keywords[:10]  # Limit to 10 keywords

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed for completion service
        pass


async def extract_keywords(text: str, ctx: Dict[str, Any]) -> List[str]:
    """
    Extract keywords from text using multiple extractors with fallback strategy.

    Priority order:
    1. LLMKeywordExtractor (if configured)
    2. IKExtractor (fallback)

    Args:
        text: Text to extract keywords from
        ctx: Context dictionary containing configuration

    Returns:
        List of extracted keywords
    """
    # Define extractors in priority order
    extractors = []

    # Add LLM extractor if configured
    if settings.llm_keyword_extraction_provider and settings.llm_keyword_extraction_model and ctx.get("user_id"):
        extractors.append(("LLM", LLMKeywordExtractor))

    # Always add IK extractor as fallback
    extractors.append(("IK", IKKeywordExtractor))

    # Try extractors in order
    for extractor_name, extractor_class in extractors:
        try:
            logger.info(f"Trying {extractor_name} keyword extractor")
            async with extractor_class(ctx) as extractor:
                keywords = await extractor.extract(text)
                if keywords:  # Only return if we got some keywords
                    logger.info(f"{extractor_name} extractor succeeded, got {len(keywords)} keywords")
                    return keywords
                else:
                    logger.warning(f"{extractor_name} extractor returned no keywords")
        except Exception as e:
            logger.warning(f"{extractor_name} extractor failed: {str(e)}")
            continue

    # If all extractors failed, return empty list
    logger.error("All keyword extractors failed")
    return []


def create_index(index: Optional[str] = None, physical_index: Optional[str] = None):
    """Ensure the shared logical fulltext index exists and is aliased."""
    return ensure_shared_index(alias=index or generate_fulltext_index_name(), physical_index=physical_index)


def delete_index(index: str):
    """Delete a legacy physical ES index.

    The shared logical index alias is intentionally protected from deletion here.
    Collection-level cleanup in the shared-index model must use delete-by-query.
    """
    if index == generate_fulltext_index_alias():
        logger.warning("Refusing to delete shared fulltext alias %s; use delete_collection_documents instead", index)
        return

    es = _get_sync_es()
    if es.indices.exists(index=index).body:
        es.indices.delete(index=index)
