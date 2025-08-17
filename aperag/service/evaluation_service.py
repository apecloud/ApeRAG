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

from celery import group
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from aperag.agent.response_types import AgentErrorResponse
from aperag.chat.history.message import StoredChatMessagePart
from aperag.config import get_async_session
from aperag.db.models import (
    Evaluation,
    EvaluationItem,
    EvaluationItemStatus,
    EvaluationStatus,
    Question,
    QuestionSet,
)
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import CollectionNotFoundException
from aperag.llm.completion.base_completion import get_completion_service
from aperag.schema import view_models
from aperag.service.agent_chat_service import AgentChatService
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)

# Concurrency limits and timeouts from environment variables
MAX_CONCURRENT_EVALUATIONS = int(os.getenv("MAX_CONCURRENT_EVALUATIONS", 5))
MAX_CONCURRENT_PROCESSING_TASKS_PER_EVALUATION = int(os.getenv("MAX_CONCURRENT_PROCESSING_TASKS_PER_EVALUATION", 5))
EVALUATION_ITEM_PROCESSING_TASK_TIMEOUT_MINUTES = int(os.getenv("EVALUATION_ITEM_PROCESSING_TASK_TIMEOUT_MINUTES", 15))


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
        from config.celery_tasks import process_evaluation_task

        # A. Check for stuck items
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

        # B. Check for orphaned 'total commander' task
        lock = self._get_evaluation_processing_redis_lock(evaluation.id, expire_time=30)
        if await lock.acquire(timeout=3):
            logger.warning(
                f"Acquired lock for running evaluation {evaluation.id}. "
                "The previous processing task may have crashed. Restarting."
            )
            await lock.release()
            process_evaluation_task.delay(evaluation.id)
            return

        # C. Check for premature completion
        pending_count_stmt = select(func.count(EvaluationItem.id)).where(
            EvaluationItem.evaluation_id == evaluation.id,
            EvaluationItem.status.in_([EvaluationItemStatus.PENDING, EvaluationItemStatus.RUNNING]),
        )
        pending_count = (await session.execute(pending_count_stmt)).scalar_one()
        if pending_count == 0:
            logger.warning(f"Evaluation {evaluation.id} is running but has no pending items. Triggering finalization.")
            await self._finalize_evaluation(session, evaluation)

    def _get_evaluation_processing_redis_lock(self, evaluation_id: str, expire_time: int):
        from aperag.concurrent_control.redis_lock import RedisLock

        lock_name = f"evaluation_processing:{evaluation_id}"

        # Note: don't use aperag.concurrent_control.get_or_create_lock(), because it uses
        #       threading.Lock() internally, which should be avoided in an async context.
        lock = RedisLock(lock_name, expire_time=expire_time)
        return lock

    async def initialize_evaluation(self, evaluation_id: str):
        """
        Logic: Initializes an evaluation. It checks prerequisites,
        creates all EvaluationItem records, and transitions the Evaluation
        status to RUNNING.
        """
        # This import is deferred to avoid circular dependency issues with Celery tasks.
        from aperag.service.collection_service import collection_service
        from config.celery_tasks import process_evaluation_task

        logger.info(f"Initializing evaluation {evaluation_id}")
        async for session in get_async_session(self.engine):
            try:
                evaluation = await session.get(Evaluation, evaluation_id)
                if not evaluation:
                    logger.error(f"Evaluation {evaluation_id} not found.")
                    return

                # 1. Basic configuration checks
                question_set = await session.get(QuestionSet, evaluation.question_set_id)
                if not question_set:
                    raise ValueError("QuestionSet not found.")

                stmt_questions = select(Question).where(Question.question_set_id == evaluation.question_set_id)
                questions = (await session.execute(stmt_questions)).scalars().all()
                if not questions:
                    raise ValueError("QuestionSet contains no questions.")

                try:
                    await collection_service.get_collection(evaluation.user_id, evaluation.collection_id)
                except CollectionNotFoundException:
                    raise ValueError("Collection not found.")

                if not evaluation.agent_llm_config or not evaluation.judge_llm_config:
                    raise ValueError("LLM configuration is missing.")

                # 2. Transition status to RUNNING and create result entries in a transaction
                async with session.begin_nested():
                    evaluation.status = EvaluationStatus.RUNNING
                    evaluation.total_questions = len(questions)
                    session.add(evaluation)

                    for question in questions:
                        eval_item = EvaluationItem(
                            evaluation_id=evaluation.id,
                            question_id=question.id,
                            question_text=question.question_text,
                            ground_truth=question.ground_truth,
                            status=EvaluationItemStatus.PENDING,
                        )
                        session.add(eval_item)

                await session.commit()
                logger.info(f"Evaluation {evaluation.id} successfully initialized and set to RUNNING.")

                # 3. Trigger process_evaluation_task to start processing
                process_evaluation_task.delay(evaluation.id)

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

    async def process_evaluation(self, evaluation_id: str):
        """
        Logic: Acts as the 'total commander' for an evaluation. It acquires a long-running,
        renewable lock and processes all items in batches, while checking for pause/delete signals.
        """
        from aperag.concurrent_control.redis_lock import redis_lock_with_renewal
        from config.celery_tasks import process_evaluation_item_task

        processing_lock = lock = self._get_evaluation_processing_redis_lock(evaluation_id, expire_time=120)

        try:
            async with redis_lock_with_renewal(processing_lock, renewal_interval=20) as lock:
                logger.info(f"Acquired commander lock for evaluation {evaluation_id}. Starting processing.")
                async for session in get_async_session(self.engine):
                    while True:
                        if not lock.is_locked():
                            logger.warning(f"Commander lock for evaluation {evaluation_id} was lost. Aborting.")
                            return

                        evaluation = await session.get(Evaluation, evaluation_id)
                        if not evaluation or evaluation.gmt_deleted:
                            logger.warning(f"Evaluation {evaluation.id} has been deleted. Halting.")
                            return
                        if evaluation.status == EvaluationStatus.PAUSED:
                            logger.info(f"Evaluation {evaluation.id} is PAUSED. Halting.")
                            return
                        if evaluation.status != EvaluationStatus.RUNNING:
                            logger.warning(
                                f"Evaluation {evaluation.id} is not RUNNING (current: {evaluation.status}). Halting."
                            )
                            return

                        pending_items_stmt = (
                            select(EvaluationItem)
                            .where(EvaluationItem.evaluation_id == evaluation_id)
                            .where(EvaluationItem.status == EvaluationItemStatus.PENDING)
                            .order_by(EvaluationItem.gmt_created)
                            .limit(MAX_CONCURRENT_PROCESSING_TASKS_PER_EVALUATION)
                        )
                        items_to_process = (await session.execute(pending_items_stmt)).scalars().all()

                        if not items_to_process:
                            logger.info(f"No more pending items for evaluation {evaluation.id}.")
                            break

                        # TODO: Optimization: start the next task as soon as one completes

                        # Create and execute a group of tasks
                        task_group = group(process_evaluation_item_task.s(item.id) for item in items_to_process)
                        result = task_group.apply_async()
                        result.get()  # Wait for the batch to complete

                    # Finalize after the loop finishes
                    await self._finalize_evaluation(session, evaluation)

        except (RuntimeError, TimeoutError):
            logger.info(
                f"Could not acquire commander lock for evaluation {evaluation_id}. Another task may be running."
            )
        except Exception as e:
            logger.exception(f"An unexpected error occurred in commander for evaluation {evaluation_id}: {e}")

    async def process_evaluation_item(self, item_id: str):
        """
        Logic: Processes a single evaluation item. This is the actual worker task.
        It uses optimistic locking to claim the item.
        """
        async for session in get_async_session(self.engine):
            try:
                # 1. Optimistically try to claim the item by updating its status from PENDING to RUNNING.
                update_stmt = (
                    update(EvaluationItem)
                    .where(EvaluationItem.id == item_id, EvaluationItem.status == EvaluationItemStatus.PENDING)
                    .values(status=EvaluationItemStatus.RUNNING)
                )
                result = await session.execute(update_stmt)

                # If the update affected 0 rows, it means the item was not PENDING
                # (e.g., already picked up by another worker). So we can safely skip it.
                if result.rowcount == 0:
                    logger.info(f"Skipping item {item_id} as it's not in PENDING state (likely already processed).")
                    await session.commit()  # Commit to end the transaction even if we do nothing.
                    return

                # 2. Fetch the item we just claimed.
                item_to_process = await session.get(EvaluationItem, item_id)
                if not item_to_process:
                    # This should ideally not happen if the update succeeded.
                    logger.error(f"EvaluationItem {item_id} not found after successful status update.")
                    await session.commit()
                    return

                evaluation = await session.get(Evaluation, item_to_process.evaluation_id)
                if not evaluation:
                    logger.error(f"Evaluation {item_to_process.evaluation_id} not found for item {item_id}.")
                    await session.commit()
                    return

                # 3. Process the item.
                await self._process_single_item(session, evaluation, item_to_process)

            except Exception as e:
                logger.exception(f"An unexpected error occurred while processing item {item_id}: {e}")
                await session.rollback()
                # Use a new session to safely update the item's status to FAILED.
                async for error_session in get_async_session(self.engine):
                    await error_session.execute(
                        update(EvaluationItem)
                        .where(EvaluationItem.id == item_id, EvaluationItem.status == EvaluationItemStatus.RUNNING)
                        .values(
                            status=EvaluationItemStatus.FAILED,
                            llm_judge_score=0,
                            llm_judge_reasoning=f"Error during processing: {e}",
                        )
                    )
                    await error_session.commit()

    async def _finalize_evaluation(self, session: AsyncSession, evaluation: Evaluation):
        """
        Checks if all items are done, calculates the final score, and updates the
        evaluation status to COMPLETED using an optimistic lock.
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

        # 2. Calculate final scores
        score_stmt = select(func.sum(EvaluationItem.llm_judge_score)).where(
            EvaluationItem.evaluation_id == evaluation.id
        )
        total_score = (await session.execute(score_stmt)).scalar_one_or_none() or 0

        completed_items_stmt = select(func.count(EvaluationItem.id)).where(
            EvaluationItem.evaluation_id == evaluation.id,
            EvaluationItem.status.in_([EvaluationItemStatus.COMPLETED, EvaluationItemStatus.FAILED]),
        )
        completed_count = (await session.execute(completed_items_stmt)).scalar_one()

        average_score = 0
        if evaluation.total_questions > 0:
            average_score = total_score / evaluation.total_questions

        # 3. Update evaluation status using optimistic locking
        update_stmt = (
            update(Evaluation)
            .where(Evaluation.id == evaluation.id, Evaluation.status == EvaluationStatus.RUNNING)
            .values(
                status=EvaluationStatus.COMPLETED,
                average_score=average_score,
                completed_questions=completed_count,
                gmt_updated=utc_now(),
            )
        )
        result = await session.execute(update_stmt)
        await session.commit()

        if result.rowcount > 0:
            logger.info(
                f"Evaluation {evaluation.id} successfully finalized and marked as COMPLETED. "
                f"Average score: {average_score}, Completed items: {completed_count}/{evaluation.total_questions}"
            )
        else:
            logger.warning(
                f"Could not finalize evaluation {evaluation.id}. "
                "It was not in RUNNING state or was modified by another process."
            )

    async def _process_single_item(
        self, session: AsyncSession, evaluation: Evaluation, item_to_process: EvaluationItem
    ):
        """Process one evaluation item: call agent, call judge, and update DB."""
        from aperag.service.collection_service import collection_service

        try:
            agent_service = AgentChatService(db_session=session)
            collections = []
            try:
                collection = await collection_service.get_collection(evaluation.user_id, evaluation.collection_id)
                collections.append(collection)
            except CollectionNotFoundException:
                raise Exception(f"Collection {evaluation.collection_id} not found during processing.")

            agent_result = await agent_service.chat_for_evaluation(
                query=item_to_process.question_text,
                user_id=evaluation.user_id,
                model_name=evaluation.agent_llm_config.get("model_name"),
                model_service_provider=evaluation.agent_llm_config.get("model_service_provider"),
                custom_llm_provider=evaluation.agent_llm_config.get("custom_llm_provider"),
                collections=collections,
            )

            if isinstance(agent_result, StoredChatMessagePart):
                item_to_process.rag_answer = agent_result.content
                item_to_process.rag_answer_details = agent_result.model_dump()
            elif isinstance(agent_result, AgentErrorResponse):
                logger.error(f"Agent failed for question {item_to_process.question_id}: {agent_result}")
                item_to_process.rag_answer = json.dumps(agent_result)
                item_to_process.rag_answer_details = agent_result

            await self._judge_result(session, evaluation, item_to_process)
            item_to_process.status = EvaluationItemStatus.COMPLETED

        except Exception as e:
            logger.error(f"Failed to process item {item_to_process.id} for evaluation {evaluation.id}: {e}")
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
请以 JSON 格式输出你的评判结果，包含两个字段：`score` (1-5的整数) 和 `reasoning` (解释你打分原因的字符串)。
"""
        llm_service = get_completion_service(
            model_name=evaluation.judge_llm_config.get("model_name"),
            model_service_provider=evaluation.judge_llm_config.get("model_service_provider"),
            custom_llm_provider=evaluation.judge_llm_config.get("custom_llm_provider"),
            user_id=evaluation.user_id,
        )
        judge_response_str = await llm_service.agenerate(prompt=judge_prompt)
        try:
            judge_response = json.loads(judge_response_str)
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
        # TODO: Add logic to trigger async task runner
        db_evaluation = Evaluation(
            user_id=user_id,
            name=request.name,
            collection_id=request.collection_id,
            question_set_id=request.question_set_id,
            agent_llm_config=request.agent_llm_config.model_dump(),
            judge_llm_config=request.judge_llm_config.model_dump(),
        )
        return await self.db_ops.create_evaluation(db_evaluation)

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
            results=[],  # Results will be loaded separately in the view
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


# Global service instances
evaluation_service = EvaluationService()
