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

from fastapi import APIRouter, Depends, HTTPException, Query

from aperag.db.models import User
from aperag.schema import view_models
from aperag.service.evaluation_service import EVALUATION_DISABLED_MESSAGE, EvaluationDisabledError, evaluation_service
from aperag.service.question_set_service import question_set_service
from aperag.views.auth import required_user


async def evaluation_feature_disabled():
    raise HTTPException(status_code=410, detail=EVALUATION_DISABLED_MESSAGE)


router = APIRouter(tags=["evaluation"], dependencies=[Depends(evaluation_feature_disabled)])

MAX_QUESTIONS_PER_SET = 1000


# region Question Set Management
@router.get("/question-sets", response_model=view_models.QuestionSetList)
async def list_question_sets(
    collection_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    user: User = Depends(required_user),
):
    items, total = await question_set_service.list_question_sets(
        user_id=user.id, collection_id=collection_id, page=page, page_size=page_size
    )
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/question-sets", response_model=view_models.QuestionSet)
async def create_question_set(
    request: view_models.QuestionSetCreate,
    user: User = Depends(required_user),
):
    if len(request.questions) > MAX_QUESTIONS_PER_SET:
        raise HTTPException(
            status_code=400, detail=f"A question set can have a maximum of {MAX_QUESTIONS_PER_SET} questions."
        )
    return await question_set_service.create_question_set(request, user.id)


@router.post("/question-sets/generate", response_model=view_models.QuestionSetDetail)
async def generate_question_set(
    request: view_models.QuestionSetGenerate,
    user: User = Depends(required_user),
):
    questions = await question_set_service.generate_questions(request, user)

    return view_models.QuestionSetDetail(
        name=f"Generated Questions for {request.collection_id}",
        collection_id=request.collection_id,
        questions=questions,
    )


@router.get("/question-sets/{qs_id}", response_model=view_models.QuestionSetDetail)
async def get_question_set(
    qs_id: str,
    user: User = Depends(required_user),
):
    qs = await question_set_service.get_question_set(qs_id, user.id)
    if not qs:
        raise HTTPException(status_code=404, detail="Question set not found")

    # Load questions for the detail view
    questions = await question_set_service.list_all_questions(qs_id)
    return view_models.QuestionSetDetail(
        id=qs.id,
        user_id=qs.user_id,
        collection_id=qs.collection_id,
        name=qs.name,
        description=qs.description,
        gmt_created=qs.gmt_created,
        gmt_updated=qs.gmt_updated,
        questions=[
            view_models.Question(
                id=q.id,
                question_set_id=q.question_set_id,
                question_type=q.question_type,
                question_text=q.question_text,
                ground_truth=q.ground_truth,
                gmt_created=q.gmt_created,
                gmt_updated=q.gmt_updated,
            )
            for q in questions
        ],
    )


@router.put("/question-sets/{qs_id}", response_model=view_models.QuestionSet)
async def update_question_set(
    qs_id: str,
    request: view_models.QuestionSetUpdate,
    user: User = Depends(required_user),
):
    qs = await question_set_service.update_question_set(qs_id, request, user.id)
    if not qs:
        raise HTTPException(status_code=404, detail="Question set not found")
    return qs


@router.delete("/question-sets/{qs_id}", status_code=204)
async def delete_question_set(
    qs_id: str,
    user: User = Depends(required_user),
):
    if not await question_set_service.delete_question_set(qs_id, user.id):
        raise HTTPException(status_code=404, detail="Question set not found")


@router.post("/question-sets/{qs_id}/questions", response_model=list[view_models.Question])
async def add_questions(
    qs_id: str,
    request: view_models.QuestionsAdd,
    user: User = Depends(required_user),
):
    qs = await question_set_service.get_question_set(qs_id, user.id)
    if not qs:
        raise HTTPException(status_code=404, detail="Question set not found")

    # Get current question count
    _, total_questions = await question_set_service.list_questions_by_set_id(qs_id, page=1, page_size=1)
    if total_questions + len(request.questions) > MAX_QUESTIONS_PER_SET:
        raise HTTPException(
            status_code=400,
            detail=f"Adding these questions would exceed the maximum of {MAX_QUESTIONS_PER_SET} questions per set.",
        )

    return await question_set_service.add_questions(qs_id, request)


@router.put("/question-sets/{qs_id}/questions/{q_id}", response_model=view_models.Question)
async def update_question(
    qs_id: str,
    q_id: str,
    request: view_models.QuestionUpdate,
    user: User = Depends(required_user),
):
    qs = await question_set_service.get_question_set(qs_id, user.id)
    if not qs:
        raise HTTPException(status_code=404, detail="Question set not found")

    q = await question_set_service.update_question(q_id, request)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    return q


@router.delete("/question-sets/{qs_id}/questions/{q_id}", status_code=204)
async def delete_question(
    qs_id: str,
    q_id: str,
    user: User = Depends(required_user),
):
    qs = await question_set_service.get_question_set(qs_id, user.id)
    if not qs:
        raise HTTPException(status_code=404, detail="Question set not found")

    await question_set_service.delete_question(q_id)


# endregion


# region Evaluation Management
@router.get("/evaluations", response_model=view_models.EvaluationList)
async def list_evaluations(
    collection_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    user: User = Depends(required_user),
):
    items, total = await evaluation_service.list_evaluations(
        user_id=user.id, collection_id=collection_id, page=page, page_size=page_size
    )
    return view_models.EvaluationList(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/evaluations", response_model=view_models.Evaluation)
async def create_evaluation(
    request: view_models.EvaluationCreate,
    user: User = Depends(required_user),
):
    try:
        return await evaluation_service.create_evaluation(request, user.id)
    except EvaluationDisabledError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc


@router.get("/evaluations/{eval_id}", response_model=view_models.EvaluationDetail)
async def get_evaluation(
    eval_id: str,
    user: User = Depends(required_user),
):
    evaluation_detail = await evaluation_service.get_evaluation(eval_id, user.id)
    if not evaluation_detail:
        raise HTTPException(status_code=404, detail="Evaluation not found")

    # Load evaluation items for the detail view
    items = await evaluation_service.get_evaluation_items(eval_id)
    evaluation_detail.items = [
        view_models.EvaluationItem(
            id=item.id,
            evaluation_id=item.evaluation_id,
            question_id=item.question_id,
            status=item.status,
            question_text=item.question_text,
            ground_truth=item.ground_truth,
            rag_answer=item.rag_answer,
            rag_answer_details=item.rag_answer_details,
            llm_judge_score=item.llm_judge_score,
            llm_judge_reasoning=item.llm_judge_reasoning,
            gmt_created=item.gmt_created,
            gmt_updated=item.gmt_updated,
        )
        for item in items
    ]

    return evaluation_detail


@router.delete("/evaluations/{eval_id}", status_code=204)
async def delete_evaluation(
    eval_id: str,
    user: User = Depends(required_user),
):
    if not await evaluation_service.delete_evaluation(eval_id, user.id):
        raise HTTPException(status_code=404, detail="Evaluation not found")


@router.post("/evaluations/{eval_id}/pause", response_model=view_models.Evaluation)
async def pause_evaluation(
    eval_id: str,
    user: User = Depends(required_user),
):
    try:
        evaluation = await evaluation_service.pause_evaluation(eval_id, user.id)
    except EvaluationDisabledError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation


@router.post("/evaluations/{eval_id}/resume", response_model=view_models.Evaluation)
async def resume_evaluation(
    eval_id: str,
    user: User = Depends(required_user),
):
    try:
        evaluation = await evaluation_service.resume_evaluation(eval_id, user.id)
    except EvaluationDisabledError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation


@router.post("/evaluations/{eval_id}/retry", response_model=view_models.Evaluation)
async def retry_evaluation(
    eval_id: str,
    scope: str = Query("failed", enum=["failed", "all"]),
    user: User = Depends(required_user),
):
    try:
        evaluation = await evaluation_service.retry_evaluation(eval_id, user.id, scope)
    except EvaluationDisabledError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation


@router.post(
    "/evaluations/chat_with_agent",
    response_model=view_models.EvaluationChatWithAgentResponse,
)
async def chat_with_agent_for_evaluation(
    request: view_models.EvaluationChatWithAgentRequest,
    user: User = Depends(required_user),
):
    raise HTTPException(status_code=410, detail=EVALUATION_DISABLED_MESSAGE)


# endregion
