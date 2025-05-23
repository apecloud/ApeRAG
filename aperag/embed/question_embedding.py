#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
from typing import Any, List

from langchain.embeddings.base import Embeddings
from llama_index.core.schema import TextNode

from aperag.docparser.chunking import rechunk
from aperag.embed.base_embedding import DocumentBaseEmbedding
from aperag.embed.local_path_embedding import LocalPathEmbedding
from aperag.embed.question_generator import QuestionGenerator
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QuestionEmbedding(LocalPathEmbedding):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        llm_model = kwargs.get("llm_model")
        self.question_generator = QuestionGenerator(llm_model)
        self.questions = []
        self.max_context_window = kwargs.get("max_context_window", 4000)

    def load_data(self, **kwargs) -> tuple[list[str], list[str]]:
        doc_parts = self.parse_doc()
        if not doc_parts:
            return [], []

        has_room_for_prefix = False
        reserve_tokens_for_prefix = 300
        chunk_size = self.max_context_window
        if chunk_size > 5 * reserve_tokens_for_prefix:
            chunk_size -= reserve_tokens_for_prefix
            has_room_for_prefix = True
        parts = rechunk(doc_parts, chunk_size, self.chunk_overlap, self.tokenizer)

        logger.info("generating questions for document: %s", self.filepath)
        nodes: List[TextNode] = []
        for part in parts:
            if not part.content:
                continue

            text = part.content
            if has_room_for_prefix:
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
                prefix = "\n\n".join(paddings)
                text = f"{prefix}\n{text}"

            # embedding without the code block
            # text = re.sub(r"```.*?```", "", doc.text, flags=re.DOTALL)

            questions = self.question_generator.gen_questions(text)
            if len(questions) == 0:
                continue

            self.questions.extend(questions)
            logger.info("generating questions: %s", str(questions))

            for q in questions:
                node = TextNode(
                    text=json.dumps({"question": q, "answer": ""}),
                )
                node.metadata.update({"source": f"{part.metadata.get('doc_id', '')}"})
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

        nodes: List[TextNode] = []
        for qa in faq:
            question = qa["question"]
            answer = qa["answer"]
            node = TextNode(
                text=json.dumps({"question": question, "answer": answer}),
            )
            node.metadata.update({"source": ""})
            vector = self.embedding.embed_query(question)
            node.embedding = vector
            nodes.append(node)
        return self.connector.store.add(nodes)

    def delete(self, **kwargs) -> bool:
        return self.connector.delete(**kwargs)
