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
import os
from datetime import timedelta
from typing import Optional

import httpx
import redis.asyncio as async_redis
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from aperag.config import get_async_session
from aperag.db.models import (
    Evaluation,
    EvaluationItem,
    EvaluationItemStatus,
    EvaluationStatus,
)
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.db.redis_manager import RedisConnectionManager
from aperag.exceptions import CollectionNotFoundException
from aperag.llm.completion.base_completion import get_completion_service
from aperag.schema import view_models
from aperag.service.collection_service import collection_service
from aperag.utils import llm_response
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)

# Concurrency limits and timeouts from environment variables
MAX_CONCURRENT_EVALUATIONS = int(os.getenv("MAX_CONCURRENT_EVALUATIONS", 5))
MAX_CONCURRENT_PROCESSING_TASKS_PER_EVALUATION = int(os.getenv("MAX_CONCURRENT_PROCESSING_TASKS_PER_EVALUATION", 1))
EVALUATION_ITEM_PROCESSING_TASK_TIMEOUT_MINUTES = int(os.getenv("EVALUATION_ITEM_PROCESSING_TASK_TIMEOUT_MINUTES", 15))
APERAG_API_BASE_URL = os.getenv("APERAG_API_BASE_URL", "http://localhost:8000")


class EvaluationExecutor:
    """Evaluation workflow orchestrator"""

    def __init__(self, engine: Optional[AsyncEngine]):
        self.engine = engine

    async def schedule_evaluations(self):
        """
        Logic: Periodically scans for PENDING evaluations and schedules them to run,
        respecting the concurrency limit. It also acts as a coordinator to recover
        from crashed or stuck evaluations.
        """
        from config.celery_tasks import initialize_evaluation_task

        logger.info("Scanning for pending and running evaluations...")

        async for session in get_async_session(self.engine):
            # 1. Schedule new evaluations
            running_count_stmt = select(func.count(Evaluation.id)).where(Evaluation.status == EvaluationStatus.RUNNING)
            running_count = (await session.execute(running_count_stmt)).scalar_one()

            if running_count < MAX_CONCURRENT_EVALUATIONS:
                slots_available = MAX_CONCURRENT_EVALUATIONS - running_count
                pending_stmt = (
                    select(Evaluation)
                    .where(Evaluation.status == EvaluationStatus.PENDING)
                    .order_by(Evaluation.gmt_created)
                    .limit(slots_available)
                )
                pending_evaluations = (await session.execute(pending_stmt)).scalars().all()
                for evaluation in pending_evaluations:
                    logger.info(f"Triggering initialize_evaluation_task for evaluation {evaluation.id}")
                    initialize_evaluation_task.delay(evaluation.id)

            # 2. Coordinator logic for running evaluations
            running_evals_stmt = select(Evaluation).where(Evaluation.status == EvaluationStatus.RUNNING)
            running_evaluations = (await session.execute(running_evals_stmt)).scalars().all()

            for evaluation in running_evaluations:
                await self._coordinate_evaluation(session, evaluation)

    async def _coordinate_evaluation(self, session: AsyncSession, evaluation: Evaluation):
        """Coordinator logic for a single running evaluation."""
        from config.celery_tasks import process_evaluation_batch_task

        # Check for stuck items
        stuck_threshold = utc_now() - timedelta(minutes=EVALUATION_ITEM_PROCESSING_TASK_TIMEOUT_MINUTES)
        stuck_items_stmt = (
            update(EvaluationItem)
            .where(EvaluationItem.evaluation_id == evaluation.id)
            .where(EvaluationItem.status == EvaluationItemStatus.RUNNING)
            .where(EvaluationItem.gmt_updated < stuck_threshold)
            .values(status=EvaluationItemStatus.PENDING)
        )
        result = await session.execute(stuck_items_stmt)
        if result.rowcount > 0:
            logger.warning(f"Reset {result.rowcount} stuck items for evaluation {evaluation.id}")
            await session.commit()

        # Trigger a new batch processing task to drive the evaluation in case the process restarted.
        process_evaluation_batch_task.delay(evaluation.id)

    def _get_evaluation_processing_redis_lock(
        self, evaluation_id: str, expire_time: int, redis_client: async_redis.Redis
    ):
        from aperag.concurrent_control.redis_lock import RedisLock

        lock_name = f"evaluation_processing:{evaluation_id}"

        # Note: don't use aperag.concurrent_control.get_or_create_lock(), because it uses
        #       threading.Lock() internally, which should be avoided in an async context.
        lock = RedisLock(lock_name, expire_time=expire_time, redis_client=redis_client)
        return lock

    async def initialize_evaluation(self, evaluation_id: str):
        """
        Logic: Initializes an evaluation. It checks prerequisites,
        creates all EvaluationItem records, and transitions the Evaluation
        status to RUNNING.
        """
        # This import is deferred to avoid circular dependency issues with Celery tasks.
        from config.celery_tasks import process_evaluation_batch_task

        logger.info(f"Initializing evaluation {evaluation_id}")
        async for session in get_async_session(self.engine):
            try:
                evaluation = await session.get(Evaluation, evaluation_id)
                if not evaluation:
                    logger.error(f"Evaluation {evaluation_id} not found.")
                    return

                # Transition status to RUNNING
                evaluation.status = EvaluationStatus.RUNNING
                session.add(evaluation)

                await session.commit()
                logger.info(f"Evaluation {evaluation.id} successfully initialized and set to RUNNING.")

                # Trigger process_evaluation_task to start processing
                process_evaluation_batch_task.delay(evaluation.id)

            except Exception as e:
                logger.exception(
                    f"An unexpected error occurred during evaluation initialization for {evaluation_id}: {e}"
                )
                async for error_session in get_async_session(self.engine):
                    evaluation = await error_session.get(Evaluation, evaluation_id)
                    if evaluation:
                        evaluation.status = EvaluationStatus.FAILED
                        evaluation.error_message = f"Initialization failed: {str(e)}"
                        await error_session.commit()

    async def process_evaluation_batch(self, evaluation_id: str):
        """
        Logic: Acts as the scheduler and finalizer for an evaluation.
        It's a short-lived, lock-protected task that checks the state and
        dispatches new item tasks if there are available concurrency slots.
        """
        from config.celery_tasks import process_evaluation_item_task

        # Using a dedicated redis client because we are running inside asyncio.run(), which
        # creates an event loop each time.
        async for redis_client in RedisConnectionManager.new_async_client():
            lock = self._get_evaluation_processing_redis_lock(evaluation_id, expire_time=60, redis_client=redis_client)
            try:
                if not await lock.acquire(timeout=5):
                    logger.info(
                        f"Could not acquire batch lock for evaluation {evaluation_id}. Another task may be running."
                    )
                    return

                async for session in get_async_session(self.engine):
                    evaluation = await session.get(Evaluation, evaluation_id)
                    if not evaluation or evaluation.gmt_deleted or evaluation.status != EvaluationStatus.RUNNING:
                        logger.info(f"Evaluation {evaluation_id} is not in a runnable state. Halting.")
                        return

                    running_items_stmt = select(func.count(EvaluationItem.id)).where(
                        EvaluationItem.evaluation_id == evaluation_id,
                        EvaluationItem.status == EvaluationItemStatus.RUNNING,
                    )
                    running_count = (await session.execute(running_items_stmt)).scalar_one()

                    slots_available = MAX_CONCURRENT_PROCESSING_TASKS_PER_EVALUATION - running_count
                    if slots_available <= 0:
                        logger.debug(f"No available slots for evaluation {evaluation_id}. Concurrency full.")
                        return

                    pending_items_stmt = (
                        select(EvaluationItem)
                        .where(EvaluationItem.evaluation_id == evaluation_id)
                        .where(EvaluationItem.status == EvaluationItemStatus.PENDING)
                        .order_by(EvaluationItem.gmt_created)
                        .limit(slots_available)
                    )
                    items_to_process = (await session.execute(pending_items_stmt)).scalars().all()

                    if not items_to_process:
                        if running_count == 0:
                            logger.info(f"All items processed for evaluation {evaluation.id}. Finalizing.")
                            await self._finalize_evaluation(session, evaluation)
                        else:
                            logger.debug(
                                f"No pending items for evaluation {evaluation.id}, but {running_count} are still running."
                            )
                        return

                    for item in items_to_process:
                        logger.debug(f"Dispatching task for evaluation item {item.id}")
                        process_evaluation_item_task.delay(evaluation_id, item.id)

            except Exception as e:
                logger.exception(f"An unexpected error occurred in batch processor for evaluation {evaluation_id}: {e}")
            finally:
                if lock.is_locked():
                    await lock.release()

    async def process_evaluation_item(self, evaluation_id: str, item_id: str):
        """
        Logic: Processes a single evaluation item. This is the actual worker task.
        It uses optimistic locking to claim the item.
        """
        from config.celery_tasks import process_evaluation_batch_task

        async for session in get_async_session(self.engine):
            try:
                update_stmt = (
                    update(EvaluationItem)
                    .where(EvaluationItem.id == item_id)
                    .where(EvaluationItem.status == EvaluationItemStatus.PENDING)
                    .values(status=EvaluationItemStatus.RUNNING)
                )
                result = await session.execute(update_stmt)
                await session.commit()

                if not result.rowcount:
                    logger.info(f"Skipping item {item_id} as it's not in PENDING state (likely already processed).")
                    return

                item_to_process = await session.get(EvaluationItem, item_id)
                evaluation = await session.get(Evaluation, evaluation_id)

                if not evaluation:
                    logger.error(f"Evaluation {evaluation_id} not found for item {item_id}.")
                    await session.commit()
                    return

                await self._process_single_item(session, evaluation, item_to_process)

                process_evaluation_batch_task.delay(evaluation_id)

            except Exception as e:
                logger.exception(f"An unexpected error occurred while processing item {item_id}: {e}")
                await session.rollback()
                # Use a new session to safely update the item's status to FAILED.
                async for error_session in get_async_session(self.engine):
                    await error_session.execute(
                        update(EvaluationItem)
                        .where(EvaluationItem.id == item_id)
                        .where(EvaluationItem.status == EvaluationItemStatus.RUNNING)
                        .values(
                            status=EvaluationItemStatus.FAILED,
                            llm_judge_score=0,
                            llm_judge_reasoning=f"Error during processing: {e}",
                        )
                    )
                    await error_session.commit()

                process_evaluation_batch_task.delay(evaluation_id)

    async def _finalize_evaluation(self, session: AsyncSession, evaluation: Evaluation):
        """
        Checks if all items are done, calculates the final score, and updates the
        evaluation status to COMPLETED or FAILED using an optimistic lock.
        """
        logger.info(f"Attempting to finalize evaluation {evaluation.id}.")

        # 1. Verify that all items are in a terminal state (COMPLETED or FAILED)
        pending_or_running_stmt = select(func.count(EvaluationItem.id)).where(
            EvaluationItem.evaluation_id == evaluation.id,
            EvaluationItem.status.in_([EvaluationItemStatus.PENDING, EvaluationItemStatus.RUNNING]),
        )
        pending_or_running_count = (await session.execute(pending_or_running_stmt)).scalar_one()

        if pending_or_running_count > 0:
            logger.warning(
                f"Finalization of evaluation {evaluation.id} aborted: "
                f"{pending_or_running_count} items are still PENDING or RUNNING."
            )
            return

        # 2. Calculate final scores and check for failed items
        score_stmt = select(func.sum(EvaluationItem.llm_judge_score)).where(
            EvaluationItem.evaluation_id == evaluation.id
        )
        total_score = (await session.execute(score_stmt)).scalar_one_or_none() or 0

        completed_items_stmt = select(func.count(EvaluationItem.id)).where(
            EvaluationItem.evaluation_id == evaluation.id,
            EvaluationItem.status.in_([EvaluationItemStatus.COMPLETED, EvaluationItemStatus.FAILED]),
        )
        completed_count = (await session.execute(completed_items_stmt)).scalar_one()

        failed_items_stmt = select(func.count(EvaluationItem.id)).where(
            EvaluationItem.evaluation_id == evaluation.id,
            EvaluationItem.status == EvaluationItemStatus.FAILED,
        )
        failed_count = (await session.execute(failed_items_stmt)).scalar_one()

        average_score = 0
        if evaluation.total_questions > 0:
            average_score = total_score / evaluation.total_questions

        # 3. Determine final status and update evaluation using optimistic locking
        final_status = EvaluationStatus.COMPLETED
        error_message = None
        if failed_count > 0:
            final_status = EvaluationStatus.FAILED
            error_message = f"Evaluation failed because {failed_count} of {completed_count} items failed."

        update_values = {
            "status": final_status,
            "average_score": average_score,
            "completed_questions": completed_count,
            "gmt_updated": utc_now(),
        }
        if error_message:
            update_values["error_message"] = error_message

        update_stmt = (
            update(Evaluation)
            .where(Evaluation.id == evaluation.id, Evaluation.status == EvaluationStatus.RUNNING)
            .values(**update_values)
        )
        result = await session.execute(update_stmt)
        await session.commit()

        if result.rowcount > 0:
            logger.info(
                f"Evaluation {evaluation.id} successfully finalized and marked as {final_status}. "
                f"Average score: {average_score}, Completed items: {completed_count}/{evaluation.total_questions}"
            )
            if error_message:
                logger.error(f"Evaluation {evaluation.id} failed with message: {error_message}")
        else:
            logger.warning(
                f"Could not finalize evaluation {evaluation.id}. "
                "It was not in RUNNING state or was modified by another process."
            )

    async def _call_agent_chat_api(
        self, session: AsyncSession, user_id: str, evaluation: Evaluation, question_text: str
    ) -> dict:
        """Calls the internal agent chat API via HTTP."""

        db_ops = AsyncDatabaseOps(session)
        aperag_api_keys = await db_ops.query_api_keys(user_id, is_system=True)
        for item in aperag_api_keys:
            aperag_api_key = item.key
        if not aperag_api_key:
            # Auto-create a new system aperag API key for the user if none exists
            logger.info(f"No aperag API key found for user {user_id}, creating a new system key")
            try:
                api_key_result = await db_ops.create_api_key(user=user_id, description="aperag", is_system=True)
                aperag_api_key = api_key_result.key
                logger.info(f"Successfully created new system aperag API key for user {user_id}")
            except Exception as e:
                error_msg = f"Failed to create aperag API key for user {user_id}: {str(e)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

        url = f"{APERAG_API_BASE_URL}/api/v1/evaluations/chat_with_agent"
        headers = {"Authorization": f"Bearer {aperag_api_key}", "Content-Type": "application/json"}
        payload = view_models.EvaluationChatWithAgentRequest(
            collection_id=evaluation.collection_id,
            agent_llm_config=view_models.LLMConfig(**evaluation.agent_llm_config),
            question_text=question_text,
        )

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, headers=headers, json=payload.model_dump(), timeout=300)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error calling agent chat API: {e.response.status_code} - {e.response.text}")
                raise Exception(f"Agent chat API returned status {e.response.status_code}") from e
            except httpx.RequestError as e:
                logger.error(f"Request error calling agent chat API: {e}")
                raise Exception("Failed to connect to agent chat API") from e

    async def _process_single_item(
        self, session: AsyncSession, evaluation: Evaluation, item_to_process: EvaluationItem
    ):
        """Process one evaluation item: call agent via API, call judge, and update DB."""
        try:
            agent_result_json = await self._call_agent_chat_api(
                session, evaluation.user_id, evaluation, item_to_process.question_text
            )

            # Check if the response is an AgentErrorResponse
            if agent_result_json.get("type") == "error":
                logger.error(f"Agent failed for question {item_to_process.question_id}: {agent_result_json}")
                item_to_process.rag_answer = json.dumps(agent_result_json)
                item_to_process.rag_answer_details = agent_result_json
            elif agent_result_json.get("messages") is not None:
                # Process successful response
                resp = view_models.ChatSuccessResponse(**agent_result_json)
                full_answer = ""
                for msg in resp.messages:
                    if msg.type == "message":
                        full_answer += msg.data + "\n\n" if msg.data else ""
                item_to_process.rag_answer = full_answer
                item_to_process.rag_answer_details = agent_result_json
            else:
                raise RuntimeError(f"unhandled response of agent chat API: {agent_result_json}")

            await self._judge_result(session, evaluation, item_to_process)
            item_to_process.status = EvaluationItemStatus.COMPLETED

        except Exception as e:
            logger.error(
                f"Failed to process item {item_to_process.id} for evaluation {evaluation.id}: {e}", exc_info=True
            )
            item_to_process.status = EvaluationItemStatus.FAILED
            item_to_process.llm_judge_score = 0
            item_to_process.llm_judge_reasoning = f"Error during processing: {e}"

        evaluation.completed_questions = (evaluation.completed_questions or 0) + 1  # TODO: try to remove this field
        await session.commit()
        logger.info(
            f"Successfully processed item {item_to_process.id}. Progress: {evaluation.completed_questions}/{evaluation.total_questions}"
        )

    async def _judge_result(self, session: AsyncSession, evaluation: Evaluation, item_to_process: EvaluationItem):
        """Call the judge LLM to score the RAG answer."""
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
    {item_to_process.question_text}
    ```
2.  **标准答案 (Ground Truth):**
    ```
    {item_to_process.ground_truth}
    ```
3.  **RAG 系统回答:**
    ```
    {item_to_process.rag_answer}
    ```

**你的任务:**
请以 JSON 格式输出你的评判结果，包含两个字段：`score` (1-5的整数) 和 `reasoning` (解释你打分原因的字符串，使用跟问题或标准答案相同的语言进行解释，即如果问题是英文的，你就用英文来解释)。
"""
        llm_service = get_completion_service(
            model_name=evaluation.judge_llm_config.get("model_name"),
            model_service_provider=evaluation.judge_llm_config.get("model_service_provider"),
            custom_llm_provider=evaluation.judge_llm_config.get("custom_llm_provider"),
            user_id=evaluation.user_id,
        )
        judge_response_str = await llm_service.agenerate(history=[], prompt=judge_prompt)
        try:
            judge_response = llm_response.parse_json(judge_response_str)
            item_to_process.llm_judge_score = judge_response.get("score", 0)
            item_to_process.llm_judge_reasoning = judge_response.get("reasoning", "No reason.")
        except Exception:
            item_to_process.llm_judge_score = 0
            item_to_process.llm_judge_reasoning = "Failed to parse JSON. LLM response: " + judge_response_str


class EvaluationService:
    def __init__(self, session: AsyncSession = None):
        if session is None:
            self.db_ops = async_db_ops
        else:
            self.db_ops = AsyncDatabaseOps(session)

    def _convert_db_evaluation_to_view_model(self, db_eval: Evaluation) -> view_models.Evaluation:
        """Converts an Evaluation DB model to a Pydantic view model."""
        if db_eval is None:
            return None

        # Handle LLM config conversion from JSON/dict to Pydantic model
        agent_llm_config = view_models.LLMConfig(**db_eval.agent_llm_config) if db_eval.agent_llm_config else None
        judge_llm_config = view_models.LLMConfig(**db_eval.judge_llm_config) if db_eval.judge_llm_config else None

        return view_models.Evaluation(
            id=db_eval.id,
            user_id=db_eval.user_id,
            name=db_eval.name,
            collection_id=db_eval.collection_id,
            question_set_id=db_eval.question_set_id,
            agent_llm_config=agent_llm_config,
            judge_llm_config=judge_llm_config,
            status=db_eval.status,
            total_questions=db_eval.total_questions,
            completed_questions=db_eval.completed_questions,
            average_score=db_eval.average_score,
            gmt_created=db_eval.gmt_created,
            gmt_updated=db_eval.gmt_updated,
        )

    async def create_evaluation(self, request: view_models.EvaluationCreate, user_id: str) -> Evaluation:
        """Creates a new evaluation task."""

        # Basic configuration checks
        question_set = await self.db_ops.get_question_set_by_id(request.question_set_id, user_id)
        if not question_set:
            raise ValueError("QuestionSet not found.")

        questions = await self.db_ops.list_all_questions_by_set_id(request.question_set_id)
        if not questions:
            raise ValueError("QuestionSet contains no questions.")

        try:
            await collection_service.get_collection(user_id, request.collection_id)
        except CollectionNotFoundException:
            raise ValueError("Collection not found.")

        def check_llm_config(cfg: view_models.LLMConfig):
            if not cfg.model_name or not cfg.model_service_provider or not cfg.custom_llm_provider:
                return False
            return True

        if not check_llm_config(request.agent_llm_config) or not check_llm_config(request.judge_llm_config):
            raise ValueError("LLM configuration is missing.")

        # Create evaluation
        db_evaluation = Evaluation(
            user_id=user_id,
            name=request.name,
            collection_id=request.collection_id,
            question_set_id=request.question_set_id,
            agent_llm_config=request.agent_llm_config.model_dump(),
            judge_llm_config=request.judge_llm_config.model_dump(),
            total_questions=len(questions),
            status=EvaluationStatus.PENDING,
        )
        evaluation = await self.db_ops.create_evaluation(db_evaluation, questions)
        if evaluation:
            # Trigger the scheduler to pick it up
            from config.celery_tasks import reconcile_evaluations_task

            reconcile_evaluations_task.delay()

        return evaluation

    async def get_evaluation(self, eval_id: str, user_id: str) -> view_models.EvaluationDetail | None:
        """Gets an evaluation by its ID and enriches it with related data."""
        from aperag.service.collection_service import collection_service
        from aperag.service.question_set_service import question_set_service

        db_eval = await self.db_ops.get_evaluation_by_id(eval_id, user_id)
        if not db_eval:
            return None

        # Fetch related object names
        collection_name = "Unknown"
        try:
            collection = await collection_service.get_collection(user_id, db_eval.collection_id)
            if collection:
                collection_name = collection.title
        except Exception:
            logger.warning(f"Could not fetch collection {db_eval.collection_id} for evaluation {eval_id}")

        question_set_name = "Unknown"
        try:
            qs = await question_set_service.get_question_set(db_eval.question_set_id, user_id)
            if qs:
                question_set_name = qs.name
        except Exception:
            logger.warning(f"Could not fetch question set {db_eval.question_set_id} for evaluation {eval_id}")

        # Convert to Pydantic model and add extra fields
        eval_detail = view_models.EvaluationDetail(
            id=db_eval.id,
            name=db_eval.name,
            status=db_eval.status,
            average_score=db_eval.average_score,
            gmt_created=db_eval.gmt_created,
            gmt_updated=db_eval.gmt_updated,
            collection_name=collection_name,
            question_set_name=question_set_name,
            config=view_models.Config1(
                collection_id=db_eval.collection_id,
                question_set_id=db_eval.question_set_id,
                agent_llm_config=db_eval.agent_llm_config,
                judge_llm_config=db_eval.judge_llm_config,
            ),
            items=[],  # Results will be loaded separately in the view
        )
        return eval_detail

    async def get_evaluation_items(self, eval_id: str) -> list[EvaluationItem]:
        """Gets all evaluation items for a given evaluation."""
        return await self.db_ops.get_evaluation_items_by_eval_id(eval_id)

    async def list_evaluations(
        self, user_id: str, page: int, page_size: int
    ) -> tuple[list[view_models.Evaluation], int]:
        """Lists all evaluations for a user."""
        db_items, total = await self.db_ops.list_evaluations_by_user(user_id, page, page_size)
        items = [self._convert_db_evaluation_to_view_model(item) for item in db_items]
        return items, total

    async def delete_evaluation(self, eval_id: str, user_id: str) -> bool:
        """Deletes an evaluation."""
        # TODO: Add logic to stop the running task if it's in progress
        return await self.db_ops.delete_evaluation_by_id(eval_id, user_id)

    async def pause_evaluation(self, eval_id: str, user_id: str) -> Evaluation | None:
        """Pauses a running evaluation."""
        return await self.db_ops.update_evaluation_status(
            eval_id,
            user_id,
            EvaluationStatus.PAUSED,
            [EvaluationStatus.RUNNING, EvaluationStatus.PENDING],
        )

    async def resume_evaluation(self, eval_id: str, user_id: str) -> Evaluation | None:
        """Resumes a paused evaluation by setting it back to pending."""
        evaluation = await self.db_ops.update_evaluation_status(
            eval_id, user_id, EvaluationStatus.PENDING, [EvaluationStatus.PAUSED]
        )
        if evaluation:
            # Trigger the scheduler to pick it up
            from config.celery_tasks import reconcile_evaluations_task

            reconcile_evaluations_task.delay()
        return evaluation

    async def retry_evaluation(self, eval_id: str, user_id: str, scope: str) -> Evaluation | None:
        """Retries items in an evaluation based on the scope."""
        evaluation = await self.db_ops.retry_evaluation(eval_id, user_id, scope)
        if evaluation:
            # Trigger the scheduler to pick it up
            from config.celery_tasks import reconcile_evaluations_task

            reconcile_evaluations_task.delay()
        return evaluation


# Global service instances
evaluation_service = EvaluationService()
