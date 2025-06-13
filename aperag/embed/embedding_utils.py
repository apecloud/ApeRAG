#!/usr/bin/env python3
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

# -*- coding: utf-8 -*-
# import faulthandler
import logging
from typing import List

from langchain_core.embeddings import Embeddings
from llama_index.core.schema import BaseNode, TextNode

from aperag.config import settings
from aperag.docparser.base import Part
from aperag.docparser.chunking import rechunk
from aperag.utils.tokenizer import get_default_tokenizer
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor

logger = logging.getLogger(__name__)

# faulthandler.enable()


def create_embeddings_and_store(
    parts: List[Part],
    vector_store_adaptor: VectorStoreConnectorAdaptor,
    embedding_model: Embeddings,
    chunk_size: int = None,
    chunk_overlap: int = None,
    tokenizer=None,
) -> List[str]:
    """
    Processes document parts, rechunks content, generates embeddings,
    and stores nodes in the vector database.

    Args:
        parts: List of document parts to process
        vector_store_adaptor: Vector store connector adaptor
        embedding_model: Embedding model to use for generating embeddings
        chunk_size: Size for chunking text (defaults to settings.chunk_size)
        chunk_overlap: Overlap size for chunking (defaults to settings.chunk_overlap_size)
        tokenizer: Tokenizer to use (defaults to default tokenizer)

    Returns:
        List[str]: A list of vector store IDs
    """
    if not parts:
        return []

    # Initialize parameters with defaults
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap_size
    tokenizer = tokenizer or get_default_tokenizer()

    nodes: List[BaseNode] = []

    # 1. Rechunk the document parts (resulting in text parts)
    # After rechunk(), parts only contains TextPart
    chunked_parts = rechunk(parts, chunk_size, chunk_overlap, tokenizer)

    # 2. Process each text chunk
    for part in chunked_parts:
        if not part.content:
            continue

        # 2.1 Prepare metadata paddings (titles, labels)
        paddings = []
        # padding titles of the hierarchy
        if "titles" in part.metadata:
            paddings.append("Breadcrumbs: " + " > ".join(part.metadata["titles"]))

        # padding user custom labels
        if "labels" in part.metadata:
            labels = []
            for item in part.metadata.get("labels", [{}]):
                if not item.get("key", None) or not item.get("value", None):
                    continue
                labels.append("%s=%s" % (item["key"], item["value"]))
            paddings.append(" ".join(labels))

        prefix = ""
        if len(paddings) > 0:
            prefix = "\n\n".join(paddings)
            logger.debug("add extra prefix for document before embedding: %s", prefix)

        # 2.2 Construct text for embedding with paddings
        if prefix:
            text = f"{prefix}\n\n{part.content}"
        else:
            text = part.content
        # 2.3 Prepare metadata for the node
        metadata = part.metadata.copy()
        metadata["source"] = metadata.get("name", "")
        # 2.4 Create TextNode
        nodes.append(TextNode(text=text, metadata=metadata))

    # 3. Generate embeddings for text chunks
    texts = [node.get_content() for node in nodes]
    vectors = embedding_model.embed_documents(texts)
    # 4. Assign embeddings to nodes
    for i in range(len(vectors)):
        nodes[i].embedding = vectors[i]

    logger.info(f"processed document with {len(parts)} parts and {len(vectors)} chunks")
    # 5. Add nodes to vector store and return results
    return vector_store_adaptor.connector.store.add(nodes)
