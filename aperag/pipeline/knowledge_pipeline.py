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
from typing import List, Optional, Tuple

from langchain_core.prompts import PromptTemplate

from aperag.context.context import ContextManager
from aperag.context.full_text import search_document
from aperag.embed.base_embedding import get_collection_embedding_service
from aperag.llm.prompts import (
    DEFAULT_CHINESE_PROMPT_TEMPLATE_V3,
    DEFAULT_KG_VECTOR_MIX_ENGLISH_PROMPT_TEMPLATE,
    DEFAULT_MODEL_MEMOTY_PROMPT_TEMPLATES,
)
from aperag.pipeline.base_pipeline import DOC_QA_REFERENCES, DOCUMENT_URLS, Message, Pipeline
from aperag.pipeline.keyword_extractor import IKExtractor
from aperag.query.query import DocumentWithScore, get_packed_answer
from aperag.rank.reranker import rerank
from aperag.schema.utils import parseCollectionConfig
from aperag.source.utils import async_run
from aperag.utils.utils import (
    generate_fulltext_index_name,
    generate_qa_vector_db_collection_name,
    generate_vector_db_collection_name,
    now_unix_milliseconds,
)
from config import settings

logger = logging.getLogger(__name__)


class KnowledgePipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.collection_id = self.collection.id
        self.collection_name = generate_vector_db_collection_name(self.collection_id)
        self.vectordb_ctx = json.loads(settings.VECTOR_DB_CONTEXT)
        self.vectordb_ctx["collection"] = self.collection_name

        config = parseCollectionConfig(self.collection.config)
        self.embedding_msp = config.embedding.model_service_provider
        self.embedding_model_name = config.embedding.model

        logging.info("KnowledgePipeline embedding model: %s, %s", self.embedding_msp, self.embedding_model_name)

        self.qa_collection_name = generate_qa_vector_db_collection_name(self.collection_id)
        self.qa_vectordb_ctx = json.loads(settings.VECTOR_DB_CONTEXT)
        self.qa_vectordb_ctx["collection"] = self.qa_collection_name

        if not self.prompt_template:
            if settings.RETRIEVE_MODE == "classic":
                self.prompt_template = DEFAULT_MODEL_MEMOTY_PROMPT_TEMPLATES.get(
                    self.model, DEFAULT_CHINESE_PROMPT_TEMPLATE_V3
                )
            else:
                self.prompt_template = DEFAULT_KG_VECTOR_MIX_ENGLISH_PROMPT_TEMPLATE
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["query", "context"])

    async def ainit(self):
        await super().ainit()
        self.embedding_model, self.vector_size = await get_collection_embedding_service(self.collection)

        self.context_manager = ContextManager(
            self.collection_name, self.embedding_model, settings.VECTOR_DB_TYPE, self.vectordb_ctx
        )

    async def new_ai_message(self, message, message_id, response, references, urls):
        return Message(
            id=message_id,
            query=message,
            response=response,
            timestamp=now_unix_milliseconds(),
            references=references,
            urls=urls,
            collection_id=self.collection_id,
            embedding_model=self.embedding_model_name,
            embedding_size=self.vector_size,
            embedding_score_threshold=self.score_threshold,
            embedding_topk=self.topk,
            llm_model=self.model,
            llm_prompt_template=self.prompt_template,
            llm_context_window=self.context_window,
        )

    async def filter_by_keywords(self, message, candidates):
        index = generate_fulltext_index_name(self.collection_id)
        async with IKExtractor({"index_name": index, "es_host": settings.ES_HOST}) as extractor:
            keywords = await extractor.extract(message)
            logger.info("[%s] extract keywords: %s", message, " | ".join(keywords))

        # find the related documents using keywords
        docs = await search_document(index, keywords, self.topk * 3)
        if not docs:
            return candidates

        doc_names = {}
        for doc in docs:
            doc_names[doc.metadata["source"]] = doc.text
            logger.info("[%s] found keyword in document %s", message, doc.metadata["source"])

        result = []
        for item in candidates:
            if item.metadata["source"] not in doc_names:
                logger.info("[%s] ignore doc %s not match keywords", message, item.metadata["source"])
                continue
            result.append(item)
        return result

    async def build_context(
        self, query_with_history: str, vector: List[float], log_prefix: str
    ) -> Tuple[str, List[DocumentWithScore]]:
        vector_context = None
        kg_context = None
        candidates = []
        if settings.RETRIEVE_MODE in ["classic", "mix"]:
            vector_context, candidates = await self._run_classic_rag(query_with_history, vector, log_prefix)
        if settings.RETRIEVE_MODE in ["local", "global", "hybrid", "graph", "mix"]:
            kg_context = await self._run_light_rag(query_with_history, log_prefix)

        if settings.RETRIEVE_MODE in ["classic"] or kg_context is None:
            return vector_context, candidates
        elif settings.RETRIEVE_MODE in ["local", "global", "hybrid", "graph"] or vector_context is None:
            return kg_context, candidates
        else:
            context = f"""
            1. From Knowledge Graph(KG):
            {kg_context}
            
            2. From Document Chunks(DC):
            {vector_context}
            """.strip()
            return context, candidates

    async def _run_classic_rag(
        self, query_with_history: str, vector: List[float], log_prefix: str
    ) -> Tuple[str, List[DocumentWithScore]]:
        """
        Executes the standard RAG pipeline: vector search, rerank, keyword filtering, context packing.
        Returns the packed context string and the list of candidate documents.
        """
        logger.info("[%s] Running standard RAG pipeline", log_prefix)
        results = await async_run(
            self.context_manager.query,
            query_with_history,
            score_threshold=self.score_threshold,
            topk=self.topk * 6,
            vector=vector,
        )
        logger.info("[%s] Found %d relevant documents in vector db", log_prefix, len(results))

        if self.bot_context != "":
            bot_context_result = DocumentWithScore(
                text=self.bot_context,  # type: ignore
                score=0,  # Use score 0 to easily identify and filter later if needed
                metadata={},  # Add empty metadata
            )
            results.append(bot_context_result)

        if len(results) > 1:
            results = await rerank(query_with_history, results)  # Use query_with_history for reranking
            logger.info("[%s] Reranked %d candidates", log_prefix, len(results))
        else:
            logger.info("[%s] No need to rerank (candidates <= 1)", log_prefix)

        candidates = results[: self.topk]

        if self.enable_keyword_recall:
            candidates = await self.filter_by_keywords(
                query_with_history.split("\n")[-1], candidates
            )  # Use original message for keywords
            logger.info("[%s] Filtered candidates by keyword, %d remaining", log_prefix, len(candidates))
        else:
            logger.info("[%s] Keyword filtering disabled", log_prefix)

        context = ""
        if len(candidates) > 0:
            # 500 is the estimated length of the prompt and memory overhead
            context_allowance = max(self.context_window - 500, 0)
            context = get_packed_answer(candidates, context_allowance)
            logger.info("[%s] Packed context generated (length: %d)", log_prefix, len(context))
        else:
            logger.info("[%s] No candidates found after filtering", log_prefix)

        return context, candidates

    async def _run_light_rag(self, query_with_history: str, log_prefix: str) -> Optional[str]:
        logger.info("[%s] Running LightRAG pipeline", log_prefix)
        from lightrag import QueryParam

        from aperag.graph import lightrag_holder
        from aperag.graph.lightrag_holder import LightRagHolder

        rag: LightRagHolder = await lightrag_holder.get_lightrag_holder(self.collection)
        param: QueryParam = QueryParam(
            mode="hybrid",
            only_need_context=True,
            top_k=self.topk,
        )
        return await rag.aquery(query=query_with_history, param=param)

    async def run(self, message, gen_references=False, message_id=""):
        log_prefix = f"{message_id}|{message}"
        logger.info("[%s] Start processing request", log_prefix)

        # --- 1. Common Setup & History Processing ---
        response = ""
        references = []
        document_url_list = []
        document_url_set = set()
        context = ""
        candidates = []  # Keep track of candidates for references/URLs

        if self.history:
            messages = await self.history.messages
        else:
            messages = []
        history_querys = [
            json.loads(msg.content)["query"] for msg in messages if msg.additional_kwargs.get("role") == "human"
        ]
        tot_history_querys = "\n".join(history_querys[-self.memory_limit_count :]) + "\n" if self.memory else ""
        query_with_history = tot_history_querys + message
        vector = self.embedding_model.embed_query(query_with_history)  # Embedding needed for standard RAG

        # --- 2. Main RAG Processing ---
        context, candidates = await self.build_context(query_with_history, vector, log_prefix)

        # --- 3. Generate LLM Answer (if needed) ---
        logger.info("[%s] Generating LLM answer.", log_prefix)
        history = []
        if self.memory and len(messages) > 0:
            history_context_allowance = max(min(self.context_window - 500 - len(context), self.memory_limit_length), 0)
            history = self.predictor.get_latest_history(
                messages=messages,
                limit_length=history_context_allowance,
                limit_count=self.memory_limit_count,
                use_ai_memory=self.use_ai_memory,
            )
            self.memory_count = len(history)
            logger.info("[%s] Prepared %d history entries for LLM.", log_prefix, len(history))

        prompt = self.prompt.format(query=message, context=context)
        logger.debug(
            "[%s] Final prompt for LLM:\n%s", log_prefix, prompt
        )  # Use debug level for potentially long prompts

        async for msg_chunk in self.predictor.agenerate_stream(history, prompt, self.memory):
            yield msg_chunk
            response += msg_chunk
        logger.info("[%s] LLM stream finished.", log_prefix)

        # Populate references and URLs from the candidates used for the context
        for result in candidates:
            # Filter out bot_context placeholder if it exists and wasn't filtered earlier
            if result.score == 0 and result.text == self.bot_context:
                continue
            references.append({"score": result.score, "text": result.text, "metadata": result.metadata})
            url = result.metadata.get("url")
            if url and url not in document_url_set:
                document_url_set.add(url)
                document_url_list.append(url)

        # --- 4. Finalization: Save Messages & Yield Metadata ---
        if self.history:
            await self.add_human_message(message, message_id)
            logger.info("[%s] Human message saved.", log_prefix)

            # Ensure AI message includes references/URLs derived from the context used
            await self.add_ai_message(message, message_id, response, references, document_url_list)
            logger.info("[%s] AI message saved.", log_prefix)

        # Yield references if requested
        if gen_references and references:
            yield DOC_QA_REFERENCES + json.dumps(references)
            logger.info("[%s] Yielded references.", log_prefix)

        # Yield document URLs if available
        if document_url_list:
            yield DOCUMENT_URLS + json.dumps(document_url_list)
            logger.info("[%s] Yielded document URLs.", log_prefix)

        logger.info("[%s] Processing finished successfully.", log_prefix)


async def create_knowledge_pipeline(**kwargs) -> KnowledgePipeline:
    pipeline = KnowledgePipeline(**kwargs)
    await pipeline.ainit()
    return pipeline
