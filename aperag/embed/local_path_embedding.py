#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import faulthandler
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain.embeddings.base import Embeddings
from llama_index.core.schema import BaseNode, TextNode

from aperag.db.models import Document
from aperag.docparser.base import AssetBinPart, MarkdownPart, Part
from aperag.docparser.chunking import rechunk
from aperag.docparser.doc_parser import DocParser
from aperag.embed.base_embedding import DocumentBaseEmbedding
from aperag.objectstore.base import get_object_store
from aperag.readers.sensitive_filter import SensitiveFilterClassify
from aperag.utils.tokenizer import get_default_tokenizer
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor
from config import settings

logger = logging.getLogger(__name__)

faulthandler.enable()

class LocalPathEmbedding(DocumentBaseEmbedding):
    def __init__(
            self,
            *,
            filepath: str,
            file_metadata: Dict[str, Any],
            object_store_base_path: str | None = None,
            vector_store_adaptor: VectorStoreConnectorAdaptor,
            embedding_model: Embeddings = None,
            vector_size: int = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(vector_store_adaptor, embedding_model, vector_size)

        self.filepath = filepath
        self.file_metadata = file_metadata or {}
        self.object_store_base_path = object_store_base_path
        self.parser = DocParser()  # TODO: use the parser config from the collection
        self.filter = SensitiveFilterClassify(None) #todo Fixme, use a llm model
        self.chunk_size = kwargs.get('chunk_size', settings.CHUNK_SIZE)
        self.chunk_overlap = kwargs.get('chunk_overlap', settings.CHUNK_OVERLAP_SIZE)
        self.tokenizer = get_default_tokenizer()

    def parse_doc(self, ) -> list[Part]:
        filepath = Path(self.filepath)
        if not self.parser.accept(filepath.suffix):
            raise ValueError(f"unsupported file type: {filepath.suffix}")
        parts = self.parser.parse_file(filepath, self.file_metadata)
        return parts

    def load_data(self, **kwargs) -> Tuple[List[str], str, List]:
        sensitive_protect = kwargs.get('sensitive_protect', False)
        sensitive_protect_method = kwargs.get('sensitive_protect_method', Document.ProtectAction.WARNING_NOT_STORED)

        nodes: List[BaseNode] = []
        content = ""
        sensitive_info = []

        parts = self.parse_doc()
        if len(parts) == 0:
            return [], "", []
        parts = rechunk(parts, self.chunk_size, self.chunk_overlap, self.tokenizer)

        md_part = next((part for part in parts if isinstance(part, MarkdownPart)), None)
        if md_part is not None:
            content = md_part.markdown

        for part in parts:
            if not part.content:
                continue

            if md_part is None:
                content += part.content + "\n\n"

            paddings = []
            # padding titles of the hierarchy
            if "titles" in part.metadata:
                paddings.append(" ".join(part.metadata["titles"]))

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
                logger.debug("add extra prefix for document %s before embedding: %s",
                             self.filepath, prefix)

            if sensitive_protect:
                part.content, output_sensitive_info = self.filter.sensitive_filter(part.content, sensitive_protect_method)
                if output_sensitive_info != {}:
                    sensitive_info.append(output_sensitive_info)

            # embedding without the prefix #, which is usually used for padding in the LLM
            # lines = []
            # for line in text.split("\n"):
            #     lines.append(line.strip("#").strip())
            # text = "\n".join(lines)

            # embedding without the code block
            # text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

            if prefix:
                text = f"{prefix}\n\n{part.content}"
            else:
                text = part.content
            nodes.append(TextNode(text=text, metadata=part.metadata))

        if sensitive_protect and sensitive_protect_method == Document.ProtectAction.WARNING_NOT_STORED and sensitive_info != []:
            logger.info("find sensitive information: %s", self.filepath)
            return [], "", sensitive_info

        if self.object_store_base_path is not None:
            base_path = self.object_store_base_path
            obj_store = get_object_store()

            # Save markdown content
            obj_store.put(f"{base_path}/parsed.md", content.encode("utf-8"))

            # Save assets
            for part in parts:
                if not isinstance(part, AssetBinPart):
                    continue
                obj_store.put(f"{base_path}/assets/{part.asset_id}", part.data)

        texts = [node.get_content() for node in nodes]
        vectors = self.embedding.embed_documents(texts)
        for i in range(len(vectors)):
            nodes[i].embedding = vectors[i]

        logger.info(f"processed file: {self.filepath} with {len(vectors)} chunks")
        return self.connector.store.add(nodes), content, sensitive_info

    def delete(self, **kwargs) -> bool:
        return self.connector.delete(**kwargs)
