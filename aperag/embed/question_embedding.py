#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
from typing import Any, List

from langchain.embeddings.base import Embeddings
from llama_index.core.data_structs.data_structs import Node
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from llama_index.core.vector_stores.types import NodeWithEmbedding

from aperag.embed.base_embedding import DocumentBaseEmbedding
from aperag.embed.local_path_embedding import LocalPathEmbedding
from aperag.readers.question_generator import QuestionGenerator
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QuestionEmbedding(LocalPathEmbedding):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.question_generator = QuestionGenerator()
        self.questions = []
        self.max_context_window = kwargs.get("max_context_window", 3000)

    def load_data(self, **kwargs) -> list[str]:

        docs, file_name, _ = self.reader.load_data()
        if not docs:
            return [], []

        logger.info("generating questions for document: %s", file_name)
        nodes: List[NodeWithEmbedding] = []
        for doc in docs:
            doc.set_content(doc.text.strip())

            # ignore page less than 30 characters
            text_size_threshold = 30
            if len(doc.text) < text_size_threshold:
                logger.warning("ignore page less than %d characters: %s",
                               text_size_threshold, doc.metadata.get("name", None))
                continue

            # ignore page with content ratio less than 0.75
            content_ratio = doc.metadata.get("content_ratio", 1)
            content_ratio_threshold = 0.75
            if content_ratio < content_ratio_threshold:
                logger.warning("ignore page with content ratio less than %f: %s",
                               content_ratio_threshold, doc.metadata.get("name", None))
                continue

            # ignore page larger than 200 characters
            # large_doc_threshold = 200
            # if len(doc.text) > large_doc_threshold:
            #     logger.warning("ignore doc larger than %d", large_doc_threshold)
            #     continue

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
            prefix = "\n\n".join(paddings)

            # embedding without the code block
            # text = re.sub(r"```.*?```", "", doc.text, flags=re.DOTALL)
            text = f"{prefix}\n{doc.text}"
            questions = self.question_generator.gen_questions(text[:self.max_context_window])
            if len(questions) == 0:
                continue

            self.questions.extend(questions)
            logger.info("generating questions: %s", str(questions))

            for q in questions:
                node = Node(
                    text=json.dumps({"question": q, "answer": ""}),
                    doc_id=doc.doc_id,
                )
                node.metadata.update({"source": f"{doc.metadata['name']}"})
                node.relationships = {
                    NodeRelationship.SOURCE: RelatedNodeInfo(
                        node_id=node.node_id, metadata={"source": f"{doc.metadata['name']}"}
                    )
                }
                vector = self.embedding.embed_query(q)
                node.embedding = vector
                nodes.append(node)

        return self.connector.store.add(nodes), self.questions

    def delete(self, **kwargs) -> bool:
        return self.connector.delete(**kwargs)


class QuestionEmbeddingWithoutDocument(DocumentBaseEmbedding):

    def __init__(
            self,
            vector_store_adaptor: VectorStoreConnectorAdaptor,
            embedding_model: Embeddings = None,
            vector_size: int = 0,
            **kwargs: Any,
    ) -> None:
        super().__init__(vector_store_adaptor, embedding_model, vector_size)
        self.questions = []

    def load_data(self, **kwargs) -> list[str]:
        faq = kwargs.get('faq', [])

        nodes: List[Node] = []
        for qa in faq:
            question = qa["question"]
            answer = qa["answer"]
            node = Node(
                text=json.dumps({"question": question, "answer": answer}),
                doc_id='',
            )
            node.metadata.update({"source": ""})
            node.relationships = {
                NodeRelationship.SOURCE: RelatedNodeInfo(
                    node_id=node.node_id, metadata={"source": ""}
                )
            }
            vector = self.embedding.embed_query(question)
            node.embedding = vector
            nodes.append(node)
        return self.connector.store.add(nodes)

    def delete(self, **kwargs) -> bool:
        self.connector.delete(**kwargs)
