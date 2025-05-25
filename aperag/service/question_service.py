from http import HTTPStatus
from asgiref.sync import sync_to_async
from django.utils import timezone
from aperag.db import models as db_models
from aperag.schema.view_models import Question, QuestionList
from aperag.views.utils import fail, success
from aperag.db.ops import PagedQuery, query_collection, query_question, query_questions, query_document
from aperag.tasks.index import generate_questions, update_collection_status, update_index_for_question
from celery import chain, group
from aperag.schema import view_models

async def create_questions(user: str, collection_id: str) -> view_models.Question:
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    if collection.status == db_models.Collection.Status.QUESTION_PENDING:
        return fail(HTTPStatus.BAD_REQUEST, "Collection is generating questions")
    collection.status = db_models.Collection.Status.QUESTION_PENDING
    await collection.asave()
    documents = await sync_to_async(collection.document_set.exclude)(status=db_models.Document.Status.DELETED)
    generate_tasks = []
    async for document in documents:
        generate_tasks.append(generate_questions.si(document.id))
    generate_group = group(*generate_tasks)
    callback_chain = chain(generate_group, update_collection_status.s(collection.id))
    callback_chain.delay()
    return success({})

async def update_question(user: str, collection_id: str, question_in: view_models.QuestionUpdate) -> view_models.Question:
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    if not question_in.id:
        question_instance = db_models.Question(
            user=collection.user,
            collection_id=collection.id,
            status=db_models.Question.Status.PENDING,
        )
        await question_instance.asave()
    else:
        question_instance = await query_question(user, question_in.id)
        if question_instance is None:
            return fail(HTTPStatus.NOT_FOUND, "Question not found")
    question_instance.question = question_in.question
    question_instance.answer = question_in.answer if question_in.answer else ""
    question_instance.status = db_models.Question.Status.PENDING
    await sync_to_async(question_instance.documents.clear)()
    if question_in.relate_documents:
        for document_id in question_in.relate_documents:
            document = await query_document(user, collection_id, document_id)
            if document is None or document.status == db_models.Document.Status.DELETED:
                return fail(HTTPStatus.NOT_FOUND, "Document not found")
            await sync_to_async(question_instance.documents.add)(document)
    else:
        question_in.relate_documents = []
    await question_instance.asave()
    update_index_for_question.delay(question_instance.id)
    return success(Question(
        id=question_instance.id,
        question=question_instance.question,
        answer=question_instance.answer,
        relate_documents=question_in.relate_documents,
    ))

async def delete_question(user: str, collection_id: str, question_id: str) -> view_models.Question:
    question = await query_question(user, question_id)
    if question is None:
        return fail(HTTPStatus.NOT_FOUND, "Question not found")
    question.status = db_models.Question.Status.DELETED
    question.gmt_deleted = timezone.now()
    await question.asave()
    update_index_for_question.delay(question.id)
    docs = await sync_to_async(question.documents.exclude)(status=db_models.Document.Status.DELETED)
    doc_ids = []
    async for doc in docs:
        doc_ids.append(doc.id)
    return success(Question(
        id=question.id,
        question=question.question,
        answer=question.answer,
        relate_documents=doc_ids,
    ))

async def list_questions(user: str, collection_id: str, pq: PagedQuery) -> view_models.QuestionList:
    pr = await query_questions(user, collection_id, pq)
    response = []
    async for question in pr.data:
        response.append(Question(
            id=question.id,
            question=question.question,
            answer=question.answer,
            relate_documents=question.documents,
        ))
    return success(QuestionList(items=response), pr=pr)

async def get_question(user: str, collection_id: str, question_id: str) -> view_models.Question:
    question = await query_question(user, question_id)
    docs = await sync_to_async(question.documents.exclude)(status=db_models.Document.Status.DELETED)
    doc_ids = []
    async for doc in docs:
        doc_ids.append(doc.id)
    return success(Question(
        id=question.id,
        question=question.question,
        answer=question.answer,
        relate_documents=doc_ids,
    )) 