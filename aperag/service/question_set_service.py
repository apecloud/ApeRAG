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

from asgiref.sync import sync_to_async
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import Question, QuestionSet, User
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.llm.completion.base_completion import get_completion_service
from aperag.schema import view_models
from aperag.service.document_service import document_service
from aperag.utils import llm_response
from aperag.utils.tokenizer import get_default_tokenizer

logger = logging.getLogger(__name__)


class QuestionSetService:
    def __init__(self, session: AsyncSession = None):
        if session is None:
            self.db_ops = async_db_ops
        else:
            self.db_ops = AsyncDatabaseOps(session)

    async def create_question_set(self, request: view_models.QuestionSetCreate, user_id: str) -> QuestionSet:
        """Creates a new question set."""
        db_question_set = QuestionSet(
            user_id=user_id,
            collection_id=request.collection_id,
            name=request.name,
            description=request.description,
        )

        questions_to_create = []
        if request.questions:
            questions_to_create = [
                Question(
                    question_type=q.question_type,
                    question_text=q.question_text,
                    ground_truth=q.ground_truth,
                )
                for q in request.questions
            ]

        return await self.db_ops.create_question_set(db_question_set, questions_to_create)

    async def get_question_set(self, qs_id: str, user_id: str) -> QuestionSet | None:
        """Gets a question set by its ID."""
        return await self.db_ops.get_question_set_by_id(qs_id, user_id)

    async def list_question_sets(
        self, user_id: str, collection_id: str | None, page: int, page_size: int
    ) -> tuple[list[QuestionSet], int]:
        """Lists all question sets for a user."""
        return await self.db_ops.list_question_sets_by_user(
            user_id=user_id, collection_id=collection_id, page=page, page_size=page_size
        )

    async def update_question_set(
        self, qs_id: str, request: view_models.QuestionSetUpdate, user_id: str
    ) -> QuestionSet | None:
        """Updates a question set."""
        return await self.db_ops.update_question_set(qs_id, user_id, request.name, request.description)

    async def delete_question_set(self, qs_id: str, user_id: str) -> bool:
        """Deletes a question set."""
        return await self.db_ops.delete_question_set_by_id(qs_id, user_id)

    async def add_questions(self, qs_id: str, request: view_models.QuestionsAdd) -> list[Question]:
        """Adds multiple questions to a question set."""
        questions_to_create = [
            Question(
                question_set_id=qs_id,
                question_text=q.question_text,
                ground_truth=q.ground_truth,
                question_type=q.question_type,
            )
            for q in request.questions
        ]
        return await self.db_ops.create_questions_in_bulk(questions_to_create)

    async def update_question(self, q_id: str, request: view_models.QuestionUpdate) -> Question | None:
        """Updates a question."""
        return await self.db_ops.update_question(
            q_id, request.question_text, request.ground_truth, request.question_type
        )

    async def delete_question(self, q_id: str) -> bool:
        """Deletes a question."""
        return await self.db_ops.delete_question_by_id(q_id)

    async def list_questions_by_set_id(self, qs_id: str, page: int, page_size: int) -> tuple[list[Question], int]:
        """Lists all questions for a question set."""
        return await self.db_ops.list_questions_by_set_id(qs_id, page, page_size)

    async def list_all_questions(self, qs_id: str) -> list[Question]:
        """Lists all questions for a question set."""
        return await self.db_ops.list_all_questions_by_set_id(qs_id)

    async def generate_questions(
        self, request: view_models.QuestionSetGenerate, user: User
    ) -> list[view_models.Question]:
        provider_name = request.llm_config.model_service_provider
        model_name = request.llm_config.model_name
        custom_llm_provider = request.llm_config.custom_llm_provider

        # 1. Get model details
        model_info = await async_db_ops.query_llm_provider_model(provider_name, "completion", model_name)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{request.llm_config}' not found.")

        context_window = model_info.context_window or 32 * 1024  # Default to 32k if not specified

        # 2. Fetch documents from the collection
        try:
            docs = await document_service.list_documents(user.id, request.collection_id)
        except Exception as e:
            logger.error(f"Failed to list documents for collection {request.collection_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve documents from the knowledge base.")

        # 3. Construct DOCUMENT_CONTENT, respecting context window
        document_content = ""

        # Leave 30% buffer for prompt and estimation error.
        max_tokens = int(context_window * 4 * 0.7)

        tokenizer = get_default_tokenizer()

        doc_tokens = 0
        for doc in docs.items:
            rest_tokens = max_tokens - doc_tokens
            try:
                preview = await document_service.get_document_preview(user.id, request.collection_id, doc.id)
                if preview.markdown_content and len(preview.markdown_content.strip()) > 0:
                    content = f"---------- **Doc Name: {preview.doc_filename}** ----------\n\n{preview.markdown_content}\n\n---------- Doc End ----------\n\n"
                    tokens = len(tokenizer(content))
                    if tokens >= rest_tokens:
                        # The number of tokens occupied by Chinese characters and English characters is not the same.
                        # Here, it is simply assumed that one char is one token.
                        content = content[:rest_tokens]
                        document_content += content
                        break

                    document_content += content
                    doc_tokens += tokens
            except Exception as e:
                logger.warning(f"Failed to get preview for document {doc.id}: {e}")
                continue

        if not document_content.strip():
            raise HTTPException(status_code=400, detail="The selected knowledge base is empty or could not be read.")

        # 4. Construct the final prompt
        prompt = (request.prompt or "").replace("{DOCUMENT_CONTENT}", document_content)
        prompt = prompt.replace("{NUMBER_OF_QUESTIONS}", str(request.question_count))

        if len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail="The prompt is empty.")

        # 5. Call the LLM
        try:
            completion_service = await sync_to_async(get_completion_service)(
                model_name, provider_name, custom_llm_provider, user.id, 0.1
            )
            response_text = await completion_service.agenerate(history=[], prompt=prompt)
        except Exception as e:
            logger.error(f"Failed to generate questions with model {request.llm_config}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate questions: {e}")

        # 6. Parse the response
        try:
            questions_data = llm_response.parse_json(response_text)
            questions = [view_models.Question(**q) for q in questions_data]
            return questions
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}\nResponse was:\n{response_text}")
            raise HTTPException(
                status_code=500, detail="Failed to parse the generated questions from the model's response."
            )


question_set_service = QuestionSetService()
