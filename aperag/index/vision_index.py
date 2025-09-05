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

import base64
import json
import logging
import time
from typing import Any, List

from llama_index.core.schema import TextNode
from sqlalchemy import and_, select

from aperag.config import get_vector_db_connector
from aperag.db.models import Collection
from aperag.index.base import BaseIndexer, IndexResult, IndexType
from aperag.llm.completion.base_completion import get_collection_completion_service_sync
from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
from aperag.llm.llm_error_types import (
    CompletionError,
    InvalidConfigurationError,
    LLMError,
    is_retryable_error,
)
from aperag.schema.utils import parseCollectionConfig
from aperag.utils.utils import generate_vector_db_collection_name

logger = logging.getLogger(__name__)


class VisionIndexer(BaseIndexer):
    """Indexer for creating vision-based indexes."""

    def __init__(self):
        super().__init__(IndexType.VISION)

    def is_enabled(self, collection: Collection) -> bool:
        """Check if vision index is enabled for the collection."""
        try:
            config = parseCollectionConfig(collection.config)
            return config.enable_vision
        except Exception:
            return False

    def create_index(
        self, document_id: str, content: str, doc_parts: List[Any], collection: Collection, **kwargs
    ) -> IndexResult:
        """Create vision index for a document."""
        if not self.is_enabled(collection):
            return IndexResult(
                success=True,
                index_type=self.index_type,
                metadata={"message": "Vision index is disabled.", "status": "skipped"},
            )

        embedding_svc, vector_size = get_collection_embedding_service_sync(collection)

        try:
            completion_svc = None
            # The collection might not have an LLM configured. It will throw exceptions in this case.
            completion_svc = get_collection_completion_service_sync(collection)
        except (InvalidConfigurationError, CompletionError):
            pass

        if not embedding_svc.is_multimodal() and (completion_svc is None or not completion_svc.is_vision_model()):
            return IndexResult(
                success=True,
                index_type=self.index_type,
                metadata={
                    "message": "Neither multimodal embedding nor vision completion model is configured.",
                    "status": "skipped",
                },
            )

        # Type info are lost, can't just check `isinstance(part, AssetBinPart)`
        image_parts = [
            part for part in doc_parts if hasattr(part, "mime_type") and (part.mime_type or "").startswith("image/")
        ]
        if not image_parts:
            return IndexResult(
                success=True, index_type=self.index_type, metadata={"message": "No images found to index."}
            )

        vector_store_adaptor = get_vector_db_connector(
            collection=generate_vector_db_collection_name(collection_id=collection.id)
        )
        all_ctx_ids = []

        # Path A: Pure Vision Embedding
        if embedding_svc.is_multimodal():
            try:
                nodes: List[TextNode] = []
                image_uris = []
                for part in image_parts:
                    b64_image = base64.b64encode(part.data).decode("utf-8")
                    mime_type = part.mime_type or "image/png"
                    data_uri = f"data:{mime_type};base64,{b64_image}"
                    image_uris.append(data_uri)
                    metadata = part.metadata.copy()
                    metadata["collection_id"] = collection.id
                    metadata["document_id"] = document_id
                    metadata["source"] = metadata.get("name", "")
                    metadata["asset_id"] = part.asset_id
                    metadata["mimetype"] = mime_type
                    metadata["indexer"] = "vision"
                    metadata["index_method"] = "multimodal_embedding"
                    nodes.append(TextNode(text="", metadata=metadata))

                vectors = embedding_svc.embed_documents(image_uris)
                for i, node in enumerate(nodes):
                    node.embedding = vectors[i]

                ctx_ids = vector_store_adaptor.connector.store.add(nodes)
                all_ctx_ids.extend(ctx_ids)
                logger.info(f"Created {len(ctx_ids)} direct vision vectors for document {document_id}")
            except Exception as e:
                logger.error(f"Failed to create pure vision embedding for document {document_id}: {e}", exc_info=True)
                return IndexResult(
                    success=False,
                    index_type=self.index_type,
                    metadata={
                        "message": f"Failed to create pure vision embedding for document {document_id}: {e}",
                        "status": "failed",
                    },
                )

        # Path B: Vision-to-Text
        if completion_svc and completion_svc.is_vision_model():
            try:
                text_nodes: List[TextNode] = []
                for part in image_parts:
                    b64_image = base64.b64encode(part.data).decode("utf-8")
                    mime_type = part.mime_type or "image/png"
                    data_uri = f"data:{mime_type};base64,{b64_image}"

                    prompt = """Analyze the provided image and extract its content with high fidelity. Follow these instructions precisely and use Markdown for formatting your entire response. Do not include any introductory or conversational text.

1.  **Overall Summary:**
    *   Provide a brief, one-paragraph overview of the image's main subject, setting, and any depicted activities.

2.  **Detailed Text Extraction:**
    *   Extract all text from the image, preserving the original language. Do not translate.
    *   **Crucially, maintain the visual reading order.** For multi-column layouts, process the text column by column (e.g., left column top-to-bottom, then right column top-to-bottom).
    *   **Exclude headers and footers:** Do not extract repetitive content from the top (headers) or bottom (footers) of the page, such as page numbers, book titles, or chapter names.
    *   Replicate the original formatting using Markdown as much as possible (e.g., headings, lists, bold/italic text).
    *   For mathematical formulas or equations, represent them using LaTeX syntax (e.g., `$$...$$` for block equations, `$...$` for inline equations).
    *   For tables, reproduce them accurately using GitHub Flavored Markdown (GFM) table syntax.

3.  **Chart/Graph Analysis:**
    *   If the image contains charts, graphs, or complex tables, identify their type (e.g., bar chart, line graph, pie chart).
    *   Explain the data presented, including axes, labels, and legends.
    *   Summarize the key insights, trends, or comparisons revealed by the data.

4.  **Object and Scene Recognition:**
    *   List all significant objects, entities, and scene elements visible in the image."""

                    description = None
                    max_retries = 3
                    retry_delay = 5  # seconds
                    for attempt in range(max_retries):
                        try:
                            description = completion_svc.generate(history=[], prompt=prompt, images=[data_uri])
                            break  # Success
                        except LLMError as e:
                            if attempt < max_retries - 1 and is_retryable_error(e):
                                logger.warning(
                                    f"Retryable error generating vision-to-text for asset {part.asset_id}: {e}. "
                                    f"Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries})"
                                )
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                logger.error(
                                    f"Non-retryable error or max retries exceeded for asset {part.asset_id}: {e}",
                                    exc_info=True,
                                )
                                return IndexResult(
                                    success=False,
                                    index_type=self.index_type,
                                    metadata={
                                        "message": f"Non-retryable error or max retries exceeded for asset {part.asset_id}: {e}",
                                        "status": "failed",
                                    },
                                )
                        except Exception as e:
                            logger.error(
                                f"Unexpected error generating vision-to-text for asset {part.asset_id}: {e}",
                                exc_info=True,
                            )
                            return IndexResult(
                                success=False,
                                index_type=self.index_type,
                                metadata={
                                    "message": f"Unexpected error generating vision-to-text for asset {part.asset_id}: {e}",
                                    "status": "failed",
                                },
                            )

                    if description:
                        metadata = part.metadata.copy()
                        metadata["collection_id"] = collection.id
                        metadata["document_id"] = document_id
                        metadata["source"] = metadata.get("name", "")
                        metadata["asset_id"] = part.asset_id
                        metadata["mimetype"] = mime_type
                        metadata["indexer"] = "vision"
                        metadata["index_method"] = "vision_to_text"
                        text_nodes.append(TextNode(text=description, metadata=metadata))

                vectors = embedding_svc.embed_documents([node.get_content() for node in text_nodes])
                for i, node in enumerate(text_nodes):
                    node.embedding = vectors[i]

                ctx_ids = vector_store_adaptor.connector.store.add(text_nodes)
                all_ctx_ids.extend(ctx_ids)
                logger.info(f"Created {len(ctx_ids)} vision-to-text vectors for document {document_id}")
            except Exception as e:
                logger.error(
                    f"Failed to create vision-to-text embedding for document {document_id}: {e}", exc_info=True
                )
                return IndexResult(
                    success=False,
                    index_type=self.index_type,
                    metadata={
                        "message": f"Failed to create vision-to-text embedding for document {document_id}: {e}",
                        "status": "failed",
                    },
                )

        return IndexResult(
            success=True,
            index_type=self.index_type,
            data={"context_ids": all_ctx_ids},
            metadata={"vector_count": len(all_ctx_ids), "vector_size": vector_size},
        )

    def update_index(
        self, document_id: str, content: str, doc_parts: List[Any], collection: Collection, **kwargs
    ) -> IndexResult:
        """Update vision index for a document."""
        result = self.delete_index(document_id, collection)
        if not result.success:
            return result
        return self.create_index(document_id, content, doc_parts, collection, **kwargs)

    def delete_index(self, document_id: str, collection: Collection, **kwargs) -> IndexResult:
        """Delete vision index for a document."""

        try:
            # Get existing vector index data from DocumentIndex
            from aperag.config import get_sync_session
            from aperag.db.models import DocumentIndex, DocumentIndexType

            ctx_ids = []
            for session in get_sync_session():
                stmt = select(DocumentIndex).where(
                    and_(DocumentIndex.document_id == document_id, DocumentIndex.index_type == DocumentIndexType.VISION)
                )
                result = session.execute(stmt)
                doc_index = result.scalar_one_or_none()

                if not doc_index or not doc_index.index_data:
                    return IndexResult(
                        success=True, index_type=self.index_type, metadata={"message": "No vision index to delete"}
                    )

                index_data = json.loads(doc_index.index_data)
                ctx_ids = index_data.get("context_ids", [])

            if not ctx_ids:
                return IndexResult(
                    success=True, index_type=self.index_type, metadata={"message": "No context IDs to delete"}
                )

            # Delete vectors from vector database
            vector_db = get_vector_db_connector(
                collection=generate_vector_db_collection_name(collection_id=collection.id)
            )
            vector_db.connector.delete(ids=ctx_ids)

            logger.info(f"Deleted {len(ctx_ids)} vectors for document {document_id}")

            return IndexResult(
                success=True,
                index_type=self.index_type,
                data={"deleted_context_ids": ctx_ids},
                metadata={"deleted_vector_count": len(ctx_ids)},
            )

        except Exception as e:
            logger.error(f"Vision index deletion failed for document {document_id}: {str(e)}")
            return IndexResult(
                success=False, index_type=self.index_type, error=f"Vector index deletion failed: {str(e)}"
            )


# Global instance
vision_indexer = VisionIndexer()
