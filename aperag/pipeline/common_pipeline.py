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

import asyncio
import logging
import random

from langchain_core.prompts import PromptTemplate

from aperag.llm.prompts import COMMON_FILE_TEMPLATE
from aperag.pipeline.base_pipeline import RELATED_QUESTIONS, Message, Pipeline
from aperag.utils.utils import now_unix_milliseconds

logger = logging.getLogger(__name__)


class CommonPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["query"])
        self.file_prompt = PromptTemplate(template=COMMON_FILE_TEMPLATE, input_variables=["query", "context"])

    async def new_ai_message(self, message, message_id, response, references, urls):
        return Message(
            id=message_id,
            query=message,
            response=response,
            timestamp=now_unix_milliseconds(),
            urls=urls,
            references=references,
            llm_model=self.model,
            llm_prompt_template=self.prompt_template,
            llm_context_window=self.context_window,
        )

    async def run(self, message, gen_references=False, message_id="", file=None):
        log_prefix = f"{message_id}|{message}"
        logger.info("[%s] start processing", log_prefix)

        related_questions = set()
        response = ""

        need_generate_answer = True
        need_related_question = True

        if self.oops != "":
            response = self.oops
            yield self.oops
            need_generate_answer = False
        if self.welcome_question != []:
            related_questions.update(self.welcome_question)
            if len(self.welcome_question) >= 3:
                need_related_question = False


        # TODO: divide file_content into several parts and call API separately.
        context = file if file else ""
        context += self.bot_context

        if len(context) > self.context_window - 500:
            context = context[:self.context_window - 500]

        if self.use_related_question and need_related_question:
            related_question_prompt = self.related_question_prompt.format(query=message, context=context)
            related_question_task = asyncio.create_task(self.generate_related_question(related_question_prompt))

        if need_generate_answer:
            history = [{"role": "system", "content": self.prompt.format(query="")}]
            if self.memory and self.history:
                messages = await self.history.messages
                if len(messages) > 0:
                    history.extend(self.predictor.get_latest_history(
                        messages=messages,
                        limit_length=max(min(self.context_window - 500 - len(context), self.memory_limit_length), 0),
                        limit_count=self.memory_limit_count,
                        use_ai_memory=self.use_ai_memory))
                    self.memory_count = len(history)

            if context:
                prompt = self.file_prompt.format(query=message, context=context)
            else:
                prompt = self.prompt.format(query=message)
            logger.info("[%s] final prompt is\n%s", log_prefix, prompt)

            async for msg in self.predictor.agenerate_stream(history, message, self.memory):
                yield msg
                response += msg

            if self.history:
                await self.add_human_message(message, message_id)
                logger.info("[%s] add human message end", log_prefix)

                await self.add_ai_message(message, message_id, response, references=[], urls=[])
                logger.info("[%s] add ai message end and the pipeline is succeed", log_prefix)

            if self.use_related_question:
                if need_related_question:
                    related_question_generate = await related_question_task
                    related_questions.update(related_question_generate)
                related_questions = list(related_questions)
                random.shuffle(related_questions)
                yield RELATED_QUESTIONS + str(related_questions[:3])

async def create_common_pipeline(**kwargs) -> CommonPipeline:
    pipeline = CommonPipeline(**kwargs)
    await pipeline.ainit()
    return pipeline