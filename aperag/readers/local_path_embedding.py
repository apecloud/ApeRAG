#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import faulthandler
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain.embeddings.base import Embeddings
from llama_index.data_structs.data_structs import Node
from llama_index.schema import NodeRelationship, RelatedNodeInfo
from llama_index.vector_stores.types import NodeWithEmbedding

from deeprag.db.models import ProtectAction
from deeprag.readers.base_embedding import DocumentBaseEmbedding
from deeprag.readers.local_path_reader import InteractiveSimpleDirectoryReader
from deeprag.readers.sensitive_filter import SensitiveFilterClassify
from deeprag.vectorstore.connector import VectorStoreConnectorAdaptor

logger = logging.getLogger(__name__)

faulthandler.enable()

class LocalPathEmbedding(DocumentBaseEmbedding):
    def __init__(
            self,
            vector_store_adaptor: VectorStoreConnectorAdaptor,
            embedding_model: Embeddings = None,
            vector_size: int = 0,
            input_file_metadata_list: Optional[List[Dict[str, Any]]] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(vector_store_adaptor, embedding_model, vector_size)
        input_files = kwargs.get("input_files", [])
        if input_files and input_file_metadata_list:
            metadata_mapping = {}
            for idx, metadata in enumerate(input_file_metadata_list):
                metadata_mapping[input_files[idx]] = metadata

            def metadata_mapping_func(path: str) -> Dict[str, Any]:
                return metadata_mapping.get(path, {})

            kwargs["file_metadata"] = metadata_mapping_func
        self.reader = InteractiveSimpleDirectoryReader(**kwargs)
        self.filter = SensitiveFilterClassify()

    def load_data(self, **kwargs) -> Tuple[List[str], str, List]:
        sensitive_protect = kwargs.get('sensitive_protect', False)
        sensitive_protect_method = kwargs.get('sensitive_protect_method', ProtectAction.WARNING_NOT_STORED)
        docs, file_name = self.reader.load_data()
        if not docs:
            return [], "", []

        nodes: List[Node] = []
        nodes_with_embedding: List[NodeWithEmbedding] = []

        texts = []
        content = ""
        embedding_docs = []
        sensitive_info = []
        
        for doc in docs:
            content += doc.text
            doc.text = doc.text.strip()

            # ignore page less than 30 characters
            text_size_threshold = 30
            if len(doc.text) < text_size_threshold:
                logger.warning("ignore page less than %d characters: %s",
                               text_size_threshold, doc.metadata.get("name", None))
                continue

            # ignore page with content ratio less than 0.5
            content_ratio = doc.metadata.get("content_ratio", 1)
            content_ratio_threshold = 0.5
            if content_ratio < content_ratio_threshold:
                logger.warning("ignore page with content ratio less than %f: %s",
                               content_ratio_threshold, doc.metadata.get("name", None))
                continue

            embedding_docs.append(doc)

            paddings = []
            # padding titles of the hierarchy
            if "titles" in doc.metadata:
                paddings.append(" ".join(doc.metadata["titles"]))

            # padding user custom labels
            if "labels" in doc.metadata:
                labels = []
                for item in doc.metadata.get("labels", [{}]):
                    if not item.get("key", None) or not item.get("value", None):
                        continue
                    labels.append("%s=%s" % (item["key"], item["value"]))
                paddings.append(" ".join(labels))

            prefix = ""
            if len(paddings) > 0:
                prefix = "\n\n".join(paddings)
                logger.info("add extra prefix for document %s before embedding: %s",
                            doc.metadata.get("name", None), prefix)

            if sensitive_protect:
                doc.text, output_sensitive_info = self.filter.sensitive_filter(doc.text, sensitive_protect_method)
                if output_sensitive_info != {}:
                    sensitive_info.append(output_sensitive_info)
            
            if prefix:
                text = f"{prefix}\n{doc.text}"
            else:
                text = doc.text
            texts.append(text)

            # embedding without the prefix #, which is usually used for padding in the LLM
            # lines = []
            # for line in text.split("\n"):
            #     lines.append(line.strip("#").strip())
            # text = "\n".join(lines)

            # embedding without the code block
            # text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
            node = Node(
                text=doc.text,
                doc_id=doc.doc_id,
            )
            node.metadata.update(doc.metadata)
            node.relationships = {
                NodeRelationship.SOURCE: RelatedNodeInfo(
                    node_id=node.node_id, metadata={"source": f"{doc.metadata['name']}"}
                )
            }
            nodes.append(node)

        if sensitive_protect and sensitive_protect_method == ProtectAction.WARNING_NOT_STORED and sensitive_info != []:
            logger.info("find sensitive infomation: %s", file_name)
            return [], "", sensitive_info

        vectors = self.embedding.embed_documents(texts)

        for i in range(len(vectors)):
            embedding_docs[i].embedding = vectors[i]
            nodes_with_embedding.append(NodeWithEmbedding(node=nodes[i], embedding=vectors[i]))

        print(f"processed file: {file_name} ")

        return self.connector.store.add(nodes_with_embedding), content, sensitive_info

    def delete(self, **kwargs) -> bool:
        return self.connector.delete(**kwargs)
