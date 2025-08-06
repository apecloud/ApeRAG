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

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from aperag.schema import view_models
from aperag.service.evaluation_service import evaluation_service
from aperag.service.question_set_service import question_set_service
from aperag.views.auth import current_user

router = APIRouter(tags=["evaluation"])


# region Question Set Management
@router.get("/question-sets", response_model=view_models.QuestionSetList)
async def list_question_sets(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    user: view_models.User = Depends(current_user),
):
    items, total = await question_set_service.list_question_sets(user.id, page, page_size)
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/question-sets", response_model=view_models.QuestionSet)
async def create_question_set(
    request: view_models.QuestionSetCreate,
    user: view_models.User = Depends(current_user),
):
    return await question_set_service.create_question_set(request, user.id)


@router.post("/question-sets/upload", response_model=view_models.QuestionSet)
async def upload_question_set(
    file: UploadFile,
    user: view_models.User = Depends(current_user),
):
    # TODO: Implement file parsing and question creation logic
    raise HTTPException(status_code=501, detail="Not Implemented")


@router.post("/question-sets/generate", response_model=view_models.QuestionSet)
async def generate_question_set(
    request: view_models.QuestionSetGenerate,
    user: view_models.User = Depends(current_user),
):
    # TODO: Implement question generation logic via async task
    raise HTTPException(status_code=501, detail="Not Implemented")


@router.get("/question-sets/{qs_id}", response_model=view_models.QuestionSetDetail)
async def get_question_set(
    qs_id: str,
    user: view_models.User = Depends(current_user),
):
    qs = await question_set_service.get_question_set(qs_id, user.id)
    if not qs:
        raise HTTPException(status_code=404, detail="Question set not found")
    # TODO: Load questions for the detail view
    return qs


@router.put("/question-sets/{qs_id}", response_model=view_models.QuestionSet)
async def update_question_set(
    qs_id: str,
    request: view_models.QuestionSetUpdate,
    user: view_models.User = Depends(current_user),
):
    qs = await question_set_service.update_question_set(qs_id, request, user.id)
    if not qs:
        raise HTTPException(status_code=404, detail="Question set not found")
    return qs


@router.delete("/question-sets/{qs_id}", status_code=204)
async def delete_question_set(
    qs_id: str,
    user: view_models.User = Depends(current_user),
):
    # TODO: check if the question set belongs to the user
    if not await question_set_service.delete_question_set(qs_id, user.id):
        raise HTTPException(status_code=404, detail="Question set not found")


@router.post("/question-sets/{qs_id}/questions", response_model=view_models.Question)
async def add_question(
    qs_id: str,
    request: view_models.Question,
    user: view_models.User = Depends(current_user),
):
    return await question_set_service.add_question(qs_id, request, user.id)


@router.put("/question-sets/{qs_id}/questions/{q_id}", response_model=view_models.Question)
async def update_question(
    qs_id: str,
    q_id: str,
    request: view_models.QuestionUpdate,
    user: view_models.User = Depends(current_user),
):
    # TODO: check if the question set belongs to the user
    q = await question_set_service.update_question(q_id, request)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    return q


@router.delete("/question-sets/{qs_id}/questions/{q_id}", status_code=204)
async def delete_question(
    qs_id: str,
    q_id: str,
    user: view_models.User = Depends(current_user),
):
    # TODO: check if the question set belongs to the user
    await question_set_service.delete_question(q_id)


# endregion


# region Evaluation Management
@router.get("/evaluations", response_model=view_models.EvaluationList)
async def list_evaluations(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    user: view_models.User = Depends(current_user),
):
    items, total = await evaluation_service.list_evaluations(user.id, page, page_size)
    return view_models.EvaluationList(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/evaluations", response_model=view_models.Evaluation)
async def create_evaluation(
    request: view_models.EvaluationCreate,
    user: view_models.User = Depends(current_user),
):
    # The request model is generated from OpenAPI spec, so it should match the new structure.
    # No changes needed here as long as the view_models are up to date.
    return await evaluation_service.create_evaluation(request, user.id)


@router.get("/evaluations/{eval_id}", response_model=view_models.EvaluationDetail)
async def get_evaluation(
    eval_id: str,
    user: view_models.User = Depends(current_user),
):
    evaluation = await evaluation_service.get_evaluation(eval_id, user.id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    # TODO: Load evaluation results for the detail view
    return evaluation


@router.delete("/evaluations/{eval_id}", status_code=204)
async def delete_evaluation(
    eval_id: str,
    user: view_models.User = Depends(current_user),
):
    if not await evaluation_service.delete_evaluation(eval_id, user.id):
        raise HTTPException(status_code=404, detail="Evaluation not found")


# endregion
