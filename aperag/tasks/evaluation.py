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

from aperag.agent.response_types import AgentErrorResponse
from aperag.chat.history.message import StoredChatMessagePart
from aperag.config import get_async_session
from aperag.db.models import Evaluation, EvaluationStatus, Question
from aperag.exceptions import CollectionNotFoundException
from aperag.llm.completion.base_completion import get_completion_service
from aperag.service.agent_chat_service import AgentChatService
from aperag.service.collection_service import collection_service

logger = logging.getLogger(__name__)


class EvaluationTask:
    async def run_evaluation(self, evaluation_id: str):
        """
        Main logic for running an evaluation task.
        """
        logger.info(f"Running evaluation for ID: {evaluation_id}")
        async for session in get_async_session():
            # 1. Fetching the evaluation and its questions from the DB.
            evaluation = await session.get(Evaluation, evaluation_id)
            if not evaluation:
                logger.error(f"Evaluation {evaluation_id} not found.")
                return

            # 2. Updating the evaluation status to RUNNING.
            evaluation.status = EvaluationStatus.RUNNING
            session.add(evaluation)
            await session.commit()

            from sqlalchemy import select

            stmt = select(Question).where(Question.question_set_id == evaluation.question_set_id)
            result = await session.execute(stmt)
            questions = result.scalars().all()

            if not questions:
                logger.warning(
                    f"No questions found for question set {evaluation.question_set_id}, finishing evaluation."
                )
                evaluation.status = EvaluationStatus.COMPLETED
                await session.commit()
                return

            evaluation.total_questions = len(questions)
            await session.commit()

            agent_service = AgentChatService(db_session=session)
            total_score = 0

            collections = []
            try:
                collection = await collection_service.get_collection(evaluation.user_id, evaluation.collection_id)
                collections.append(collection)
            except CollectionNotFoundException:
                # TODO: let the evaluation fail
                pass

            for i, question in enumerate(questions):
                # 4. Calling the agent to get an answer using the internal evaluation method.
                result = await agent_service.chat_for_evaluation(
                    query=question.question_text,
                    user_id=evaluation.user_id,
                    model_name=evaluation.agent_llm_config.get("model_name"),
                    model_service_provider=evaluation.agent_llm_config.get("model_service_provider"),
                    custom_llm_provider=evaluation.agent_llm_config.get("custom_llm_provider"),
                    collections=collections,
                )

                rag_answer = ""
                rag_answer_details = {}
                if isinstance(result, StoredChatMessagePart):
                    rag_answer = result.content
                    rag_answer_details = result.model_dump()
                elif isinstance(result, AgentErrorResponse):
                    logger.error(f"Agent failed to produce an answer for question {question.id}, {result}")
                    rag_answer = json.dumps(result)
                    rag_answer_details = result

                # 5. Calling the judge LLM to get a score and reasoning.
                judge_prompt = f"""你是一个客观、严谨的 RAG 系统回答质量评估专家。请根据以下信息，对 RAG 系统的回答进行评分。

**评分标准 (5分制):**
- 5分 (完美回答): 事实100%准确，完全基于来源，全面回答了问题，无任何冗余，语言流畅。
- 4分 (高质量回答): 绝大部分信息准确，可能有极微小瑕疵，基本完整，可读性好。
- 3分 (中等质量回答): 包含部分正确信息，但也有明显错误或遗漏，需要用户自行辨别。
- 2分 (低质量回答): 包含大量错误信息，或未能解决问题，可能会误导用户。
- 1分 (错误或无法回答): 完全错误，产生幻觉，或拒绝回答。

**待评估信息:**
1.  **原始问题:**
    ```
    {question.question_text}
    ```
2.  **标准答案 (Ground Truth):**
    ```
    {question.ground_truth}
    ```
3.  **RAG 系统回答:**
    ```
    {rag_answer}
    ```

**你的任务:**
请以 JSON 格式输出你的评判结果，包含两个字段：`score` (1-5的整数) 和 `reasoning` (解释你打分原因的字符串)。
"""
                try:
                    llm_service = get_completion_service(
                        model_name=evaluation.judge_llm_config.get("model_name"),
                        model_service_provider=evaluation.judge_llm_config.get("model_service_provider"),
                        custom_llm_provider=evaluation.judge_llm_config.get("custom_llm_provider"),
                        user_id=evaluation.user_id,
                    )
                    judge_response_str = await llm_service.agenerate(prompt=judge_prompt)
                    judge_response = json.loads(judge_response_str)
                    llm_judge_score = judge_response.get("score", 0)
                    llm_judge_reasoning = judge_response.get("reasoning", "Failed to parse reasoning.")
                except Exception as e:
                    logger.error(f"Failed to judge answer for question {question.id}: {e}")
                    llm_judge_score = 0
                    llm_judge_reasoning = f"Error during judgment: {e}"

                total_score += llm_judge_score

                # 6. Storing the result in the evaluation_results table.
                from aperag.db.models import EvaluationResult

                eval_result = EvaluationResult(
                    evaluation_id=evaluation.id,
                    question_id=question.id,
                    question_text=question.question_text,
                    ground_truth=question.ground_truth,
                    rag_answer=rag_answer,
                    rag_answer_details=rag_answer_details,
                    llm_judge_score=llm_judge_score,
                    llm_judge_reasoning=llm_judge_reasoning,
                )
                session.add(eval_result)

                # 7. Updating the progress.
                evaluation.completed_questions = i + 1
                await session.commit()

            # 8. Once complete, updating the final status and average score.
            if evaluation.total_questions > 0:
                evaluation.average_score = total_score / evaluation.total_questions
            evaluation.status = EvaluationStatus.COMPLETED
            await session.commit()
            logger.info(f"Evaluation {evaluation_id} completed with average score: {evaluation.average_score}")


evaluation_task = EvaluationTask()
