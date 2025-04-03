import asyncio
import json
import logging
import random
import re

from langchain import PromptTemplate

from config import settings
from deeprag.context.context import ContextManager
from deeprag.context.full_text import search_document
from deeprag.llm.prompts import (
    DEFAULT_CHINESE_PROMPT_TEMPLATE_V3,
    DEFAULT_MODEL_MEMOTY_PROMPT_TEMPLATES,
)
from deeprag.pipeline.base_pipeline import KUBE_CHAT_DOC_QA_REFERENCES, KUBE_CHAT_RELATED_QUESTIONS, \
    Message, Pipeline, KUBE_CHAT_DOCUMENT_URLS
from deeprag.pipeline.keyword_extractor import IKExtractor
from deeprag.query.query import DocumentWithScore, get_packed_answer
from deeprag.readers.base_embedding import get_embedding_model, rerank
from deeprag.source.utils import async_run
from deeprag.utils.utils import (
    generate_fulltext_index_name,
    generate_qa_vector_db_collection_name,
    generate_vector_db_collection_name,
    now_unix_milliseconds,
)

logger = logging.getLogger(__name__)


class KnowledgePipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.collection_id = self.collection.id
        collection_name = generate_vector_db_collection_name(self.collection_id)
        self.vectordb_ctx = json.loads(settings.VECTOR_DB_CONTEXT)
        self.vectordb_ctx["collection"] = collection_name

        config = json.loads(self.collection.config)
        self.embedding_model_name = config.get("embedding_model", settings.EMBEDDING_MODEL)
        self.embedding_model, self.vector_size = get_embedding_model(self.embedding_model_name)

        self.context_manager = ContextManager(collection_name, self.embedding_model, settings.VECTOR_DB_TYPE,
                                              self.vectordb_ctx)

        qa_collection_name = generate_qa_vector_db_collection_name(self.collection_id)
        self.qa_vectordb_ctx = json.loads(settings.VECTOR_DB_CONTEXT)
        self.qa_vectordb_ctx["collection"] = qa_collection_name
        self.qa_context_manager = ContextManager(qa_collection_name, self.embedding_model, settings.VECTOR_DB_TYPE,
                                                 self.qa_vectordb_ctx)

        if not self.prompt_template:
            self.prompt_template = DEFAULT_MODEL_MEMOTY_PROMPT_TEMPLATES.get(self.model,
                                                                             DEFAULT_CHINESE_PROMPT_TEMPLATE_V3)
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["query", "context"])

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
            doc_names[doc["name"]] = doc["content"]
            logger.info("[%s] found keyword in document %s", message, doc["name"])

        result = []
        for item in candidates:
            if item.metadata["name"] not in doc_names:
                logger.info("[%s] ignore doc %s not match keywords", message, item.metadata["name"])
                continue
            result.append(item)
        return result

    async def run(self, message, gen_references=False, message_id=""):
        log_prefix = f"{message_id}|{message}"
        logger.info("[%s] start processing", log_prefix)

        history = []
        tot_history_querys = ''
        messages = await self.history.messages
        history_querys = [json.loads(message.content)["query"] for message in messages if message.additional_kwargs["role"] == "human"] 
        
        if self.memory:
            tot_history_querys = '\n'.join(history_querys[-self.memory_limit_count:])+'\n'
        
        references = []
        related_questions = set()
        response = ""
        document_url_list = []
        document_url_set = set()
        need_generate_answer = True
        need_related_question = True
        vector = self.embedding_model.embed_query(tot_history_querys+message)
        logger.info("[%s] embedding query end", log_prefix)
        # hyde_task = asyncio.create_task(self.generate_hyde_message(message))
                
        results = await async_run(self.qa_context_manager.query, tot_history_querys + message, score_threshold=0.5, topk=6, vector=vector)
        logger.info("[%s] find relevant qa pairs in vector db end", log_prefix)
        for result in results:
            result_text = json.loads(result.text)
            if result_text["answer"] != "" and result.score > 0.9:
                response = result_text["answer"]
            if result.score < 0.8:
                related_questions.add(result_text["question"])
                
        # if len(related_questions) >= 3:
        #     need_related_question = False
            
        if response != "":
            yield response
            
            if self.use_related_question and need_related_question:
                related_question_prompt = self.related_question_prompt.format(query=message, context=response)
                related_question_task = asyncio.create_task(self.generate_related_question(related_question_prompt))
          
        else:
            results = await async_run(self.context_manager.query, tot_history_querys + message,
                                      score_threshold=self.score_threshold, topk=self.topk * 6, vector=vector)
            logger.info("[%s] find top %d relevant context in vector db end", log_prefix, len(results))
            # hyde_message = await hyde_task
            # new_vector = self.embedding_model.embed_query(hyde_message)
            # results2 = await async_run(self.context_manager.query, message,
            #                           score_threshold=self.score_threshold, topk=self.topk * 6, vector=new_vector)
            # results_set = set([result.text for result in results])
            # results.extend(result for result in results2 if result.text not in results_set)
            
            if self.bot_context != "":
                bot_context_result = DocumentWithScore(
                    text=self.bot_context,  # type: ignore
                    score=0,
                )
                results.append(bot_context_result)
                
            if len(results) > 1:
                results = await rerank(message, results)
                logger.info("[%s] rerank candidates end", log_prefix)
            else:
                logger.info("[%s] don't need to rerank ", log_prefix)

            candidates = results[:self.topk]

            if self.enable_keyword_recall:
                candidates = await self.filter_by_keywords(message, candidates)
                logger.info("[%s] filter keyword end", log_prefix)
            else:
                logger.info("[%s] no need to filter keyword", log_prefix)

            context = ""
            if len(candidates) > 0:
                # 500 is the estimated length of the prompt
                context = get_packed_answer(candidates, max(self.context_window - 500, 0))
            else:
                if self.oops != "":
                    response = self.oops
                    yield self.oops
                    need_generate_answer = False
                if self.welcome_question:
                    related_questions.update(self.welcome_question)
                    if len(related_questions) >= 3:
                        need_related_question = False
                  
            if self.use_related_question and need_related_question:
                related_question_prompt = self.related_question_prompt.format(query=message, context=context)
                related_question_task = asyncio.create_task(self.generate_related_question(related_question_prompt))

            if need_generate_answer:
                history_querys.append(message)
                related_questions = related_questions - set(history_querys[-5:])
                if self.memory and len(messages) > 0:
                    history = self.predictor.get_latest_history(
                        messages=messages,
                        limit_length=max(min(self.context_window - 500 - len(context), self.memory_limit_length), 0),
                        limit_count=self.memory_limit_count,
                        use_ai_memory=self.use_ai_memory)
                    self.memory_count = len(history)

                prompt = self.prompt.format(query=message, context=context)
                logger.info("[%s] final prompt is\n%s", log_prefix, prompt)

                async for msg in self.predictor.agenerate_stream(history, prompt, self.memory):
                    yield msg
                    response += msg

                for result in candidates:
                    # filter bot_context
                    if result.score == 0:
                        continue
                    references.append({
                        "score": result.score,
                        "text": result.text,
                        "metadata": result.metadata
                    })
                    url = result.metadata.get("url")
                    if url and url not in document_url_set:
                        document_url_set.add(result.metadata.get("url"))
                        document_url_list.append(result.metadata.get("url"))

        await self.add_human_message(message, message_id)
        logger.info("[%s] add human message end", log_prefix)

        await self.add_ai_message(message, message_id, response, references, document_url_list)
        logger.info("[%s] add ai message end and the pipeline is succeed", log_prefix)

        if self.use_related_question:
            if need_related_question:
                related_question_generate = await related_question_task
                related_questions.update(related_question_generate)
            related_questions = list(related_questions)
            random.shuffle(related_questions)
            yield KUBE_CHAT_RELATED_QUESTIONS + str(related_questions[:3])

        if gen_references:
            yield KUBE_CHAT_DOC_QA_REFERENCES + json.dumps(references)

        if document_url_list:
            yield KUBE_CHAT_DOCUMENT_URLS + json.dumps(document_url_list)

