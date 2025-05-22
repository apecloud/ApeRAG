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
import tarfile
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import IO

import py7zr
import rarfile
from asgiref.sync import async_to_sync
from celery import Task
from django.core.files.base import ContentFile
from django.utils import timezone

from aperag.context.full_text import insert_document, remove_document
from aperag.db.models import Collection, Document, MessageFeedback, Question
from aperag.embed.base_embedding import get_collection_embedding_service
from aperag.embed.local_path_embedding import LocalPathEmbedding
from aperag.embed.qa_embedding import QAEmbedding
from aperag.embed.question_embedding import QuestionEmbedding, QuestionEmbeddingWithoutDocument
from aperag.graph import lightrag_holder
from aperag.graph.lightrag_holder import LightRagHolder
from aperag.objectstore.base import get_object_store
from aperag.readers.base_readers import DEFAULT_FILE_READER_CLS, SUPPORTED_COMPRESSED_EXTENSIONS
from aperag.schema.utils import parseCollectionConfig
from aperag.source.base import get_source
from aperag.source.feishu.client import FeishuNoPermission, FeishuPermissionDenied
from aperag.utils.tokenizer import get_default_tokenizer
from aperag.utils.utils import (
    generate_fulltext_index_name,
    generate_qa_vector_db_collection_name,
    generate_vector_db_collection_name,
)
from config import settings
from config.celery import app
from config.vector_db import get_vector_db_connector

logger = logging.getLogger(__name__)


class CustomLoadDocumentTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        document_id = args[0]
        document = Document.objects.get(id=document_id)
        if document.status != Document.Status.WARNING:
            document.status = Document.Status.COMPLETE
        document.save()
        logger.info(f"index for document {document.name} success")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        document_id = args[0]
        document = Document.objects.get(id=document_id)
        document.status = Document.Status.FAILED
        document.save()
        logger.error(f"index for document {document.name} error:{exc}")


class CustomDeleteDocumentTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        document_id = args[0]
        document = Document.objects.get(id=document_id)
        logger.info(f"remove qdrant points for document {document.name} success")
        document.status = Document.Status.DELETED
        document.gmt_deleted = timezone.now()
        document.name = document.name + "-" + str(uuid.uuid4())
        document.save()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        document_id = args[0]
        document = Document.objects.get(id=document_id)
        document.status = Document.Status.FAILED
        document.save()
        logger.error(f"remove_index(): index delete from vector db failed:{exc}")

    # def after_return(self, status, retval, task_id, args, kwargs, einfo):
    #     print(retval)
    #     return super(CustomLoadDocumentTask, self).after_return(status, retval, task_id, args, kwargs, einfo)


class SensitiveInformationFound(Exception):
    """
    raised when Sensitive information found in document
    """


def uncompress(fileobj: IO[bytes], suffix: str, dest: str):
    if suffix == '.zip':
        with zipfile.ZipFile(fileobj, 'r') as zf:
            for name in zf.namelist():
                try:
                    name_utf8 = name.encode('cp437').decode('utf-8')
                except Exception:
                    name_utf8 = name
                zf.extract(name, dest)
                if name_utf8 != name:
                    os.rename(os.path.join(dest, name), os.path.join(dest, name_utf8))
    elif suffix in ['.rar', '.r00']:
        with rarfile.RarFile(fileobj, 'r') as rf:
            rf.extractall(dest)
    elif suffix == '.7z':
        with py7zr.SevenZipFile(fileobj, 'r') as z7:
            z7.extractall(dest)
    elif suffix in ['.tar', '.gz', '.xz', '.bz2', '.tar.gz', '.tar.xz', '.tar.bz2', '.tar.7z']:
        with tarfile.open(fileobj=fileobj, mode='r:*') as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unsupported file format {suffix}")


def uncompress_file(document: Document):
    MAX_EXTRACTED_SIZE = 5000 * 1024 * 1024  # 5 GB
    obj_store = get_object_store()

    with tempfile.TemporaryDirectory(prefix=f"aperag_unzip_{document.id}_") as temp_dir_path_str:
        tmp_dir = Path(temp_dir_path_str)
        obj = obj_store.get(document.object_path)
        if obj is None:
            raise Exception(f"object '{document.object_path}' is not found")
        suffix = Path(document.object_path).suffix
        with obj:
            uncompress(obj, suffix, tmp_dir)

        extracted_files = []
        total_size = 0
        for root, dirs, file_names in os.walk(tmp_dir):
            for name in file_names:
                path = Path(os.path.join(root, name))
                if path.suffix in SUPPORTED_COMPRESSED_EXTENSIONS:
                    continue
                if path.suffix not in DEFAULT_FILE_READER_CLS.keys():
                    continue
                extracted_files.append(path)
                total_size += path.stat().st_size

                if total_size > MAX_EXTRACTED_SIZE:
                    raise Exception("Extracted size exceeded limit")

        for extracted_file_path in extracted_files:
            with extracted_file_path.open(mode="rb") as extracted_file:  # open in binary
                document_instance = Document(
                    user=document.user,
                    name=document.name + "/" + extracted_file_path.name,
                    status=Document.Status.PENDING,
                    size=extracted_file_path.stat().st_size,
                    collection_id=document.collection_id,
                )
                # Upload to object store
                upload_path = f"{document_instance.object_store_base_path()}/original{suffix}"
                obj_store.put(upload_path, extracted_file)

                document_instance.object_path = upload_path
                document_instance.metadata = json.dumps({
                    "object_path": upload_path,
                    "uncompressed": "true"
                })
                document_instance.save()
                add_index_for_local_document.delay(document_instance.id)

    return


@app.task(base=CustomLoadDocumentTask, bind=True, ignore_result=True)
def add_index_for_local_document(self, document_id):
    try:
        add_index_for_document(document_id)
    except Exception as e:
        for info in e.args:
            if isinstance(info, str) and "sensitive information" in info:
                raise e
        raise self.retry(exc=e, countdown=5, max_retries=1)

@app.task(base=CustomLoadDocumentTask, bind=True, track_started=True)
def add_index_for_document(self, document_id):
    """
        Celery task to do an embedding for a given Document and save the results in vector database.
        Args:
            document_id: the document in Django Module
    """
    document = Document.objects.get(id=document_id)
    document.status = Document.Status.RUNNING
    document.save()

    source = None
    local_doc = None
    metadata = json.loads(document.metadata)
    metadata["doc_id"] = document_id
    collection = async_to_sync(document.get_collection)()
    try:
        if document.object_path and Path(document.object_path).suffix in SUPPORTED_COMPRESSED_EXTENSIONS:
            config = parseCollectionConfig(collection.config)
            if config.source != "system":
                return
            uncompress_file(document)
            return
        else:
            source = get_source(parseCollectionConfig(collection.config))
            local_doc = source.prepare_document(name=document.name, metadata=metadata)
            if document.size == 0:
                document.size = os.path.getsize(local_doc.path)

            embedding_model, vector_size = async_to_sync(get_collection_embedding_service)(collection)
            loader = LocalPathEmbedding(filepath=local_doc.path,
                                        file_metadata=local_doc.metadata,
                                        embedding_model=embedding_model,
                                        vector_size=vector_size,
                                        vector_store_adaptor=get_vector_db_connector(
                                            collection=generate_vector_db_collection_name(
                                                collection_id=collection.id)),
                                        chunk_size=settings.CHUNK_SIZE,
                                        chunk_overlap=settings.CHUNK_OVERLAP_SIZE,
                                        tokenizer=get_default_tokenizer())

            config = parseCollectionConfig(collection.config)
            sensitive_protect = config.sensitive_protect or False
            sensitive_protect_method = config.sensitive_protect_method or Document.ProtectAction.WARNING_NOT_STORED
            ctx_ids, content, sensitive_info = loader.load_data(sensitive_protect=sensitive_protect,
                                                                sensitive_protect_method=sensitive_protect_method)
            document.sensitive_info = sensitive_info
            if sensitive_protect and sensitive_info:
                if sensitive_protect_method == Document.ProtectAction.WARNING_NOT_STORED:
                    raise SensitiveInformationFound()
                else:
                    document.status = Document.Status.WARNING
            logger.info(f"add ctx qdrant points: {ctx_ids} for document {local_doc.path}")

            # only index the document that have points in the vector database
            if ctx_ids:
                index = generate_fulltext_index_name(collection.id)
                insert_document(index, document.id, local_doc.name, content)

            relate_ids = {
                "ctx": ctx_ids,
            }
            document.relate_ids = json.dumps(relate_ids)

            enable_knowledge_graph = config.enable_knowledge_graph or False
            if enable_knowledge_graph:
                add_lightrag_index(content, document, local_doc)

    except FeishuNoPermission:
        raise Exception("no permission to access document %s" % document.name)
    except FeishuPermissionDenied:
        raise Exception("permission denied to access document %s" % document.name)
    except SensitiveInformationFound:
        raise Exception("sensitive information found in document %s" % document.name)
    except Exception as e:
        raise e
    finally:
        document.save()
        if local_doc and source:
            source.cleanup_document(local_doc.path)


@app.task(base=CustomDeleteDocumentTask, bind=True, track_started=True)
def remove_index(self, document_id):
    """
    remove the doc embedding index from vector store db
    :param self:
    :param document_id:
    """
    document = Document.objects.get(id=document_id)
    try:
        collection = async_to_sync(document.get_collection)()
        index = generate_fulltext_index_name(collection.id)
        remove_document(index, document.id)

        if document.relate_ids == "":
            return

        relate_ids = json.loads(document.relate_ids)
        vector_db = get_vector_db_connector(
            collection=generate_vector_db_collection_name(collection_id=collection.id)
        )
        ctx_relate_ids = relate_ids.get("ctx", [])
        vector_db.connector.delete(ids=ctx_relate_ids)
        logger.info(f"remove ctx qdrant points: {ctx_relate_ids} for document {document.name}")

        rag: LightRagHolder = async_to_sync(lightrag_holder.get_lightrag_holder)(collection=collection)
        async_to_sync(rag.adelete_by_doc_id)(document_id)

    except Exception as e:
        raise e


@app.task(base=CustomLoadDocumentTask, bind=True, track_started=True)
def update_index_for_local_document(self, document_id):
    try:
        update_index_for_document(document_id)
    except Exception as e:
        for info in e.args:
            if isinstance(info, str) and "sensitive information" in info:
                raise e
        raise self.retry(exc=e, countdown=5, max_retries=1)


@app.task(base=CustomLoadDocumentTask, bind=True, track_started=True)
def update_index_for_document(self, document_id):
    document = Document.objects.get(id=document_id)
    document.status = Document.Status.RUNNING
    document.save()

    try:
        relate_ids = json.loads(document.relate_ids) if document.relate_ids.strip() else {}
        collection = async_to_sync(document.get_collection)()
        source = get_source(parseCollectionConfig(collection.config))
        metadata = json.loads(document.metadata)
        metadata["doc_id"] = document_id
        local_doc = source.prepare_document(name=document.name, metadata=metadata)

        embedding_model, vector_size = async_to_sync(get_collection_embedding_service)(collection)
        loader = LocalPathEmbedding(filepath=local_doc.path,
                                    file_metadata=local_doc.metadata,
                                    embedding_model=embedding_model,
                                    vector_size=vector_size,
                                    vector_store_adaptor=get_vector_db_connector(
                                        collection=generate_vector_db_collection_name(
                                            collection_id=collection.id)),
                                    chunk_size=settings.CHUNK_SIZE,
                                    chunk_overlap=settings.CHUNK_OVERLAP_SIZE,
                                    tokenizer=get_default_tokenizer())
        loader.connector.delete(ids=relate_ids.get("ctx", []))

        config = parseCollectionConfig(collection.config)
        sensitive_protect = config.sensitive_protect or False
        sensitive_protect_method = config.sensitive_protect_method or Document.ProtectAction.WARNING_NOT_STORED
        ctx_ids, content, sensitive_info = loader.load_data(sensitive_protect=sensitive_protect,
                                                            sensitive_protect_method=sensitive_protect_method)
        document.sensitive_info = sensitive_info
        if sensitive_protect and sensitive_info != []:
            if sensitive_protect_method == Document.ProtectAction.WARNING_NOT_STORED:
                raise SensitiveInformationFound()
            else:
                document.status = Document.Status.WARNING
        logger.info(f"add ctx qdrant points: {ctx_ids} for document {local_doc.path}")

        # only index the document that have points in the vector database
        if ctx_ids:
            index = generate_fulltext_index_name(collection.id)
            insert_document(index, document.id, local_doc.name, content)

        relate_ids = {
            "ctx": ctx_ids,
        }
        document.relate_ids = json.dumps(relate_ids)
        logger.info(f"update qdrant points: {document.relate_ids} for document {local_doc.path}")

        enable_knowledge_graph = config.enable_knowledge_graph or False
        if enable_knowledge_graph:
            add_lightrag_index(content, document, local_doc)

    except FeishuNoPermission:
        raise Exception("no permission to access document %s" % document.name)
    except FeishuPermissionDenied:
        raise Exception("permission denied to access document %s" % document.name)
    except SensitiveInformationFound:
        raise Exception("sensitive information found in document %s" % document.name)
    except Exception as e:
        logger.error(e)
        raise Exception("an error occur %s" % e)
    finally:
        document.save()

    source.cleanup_document(local_doc.path)


def add_lightrag_index(content, document, local_doc):
    logger.info(f"Begin indexing document for LightRAG (ID: {document.id})")
    collection = async_to_sync(document.get_collection)()
    rag: LightRagHolder = async_to_sync(lightrag_holder.get_lightrag_holder)(collection=collection)
    rag.insert(content, ids=document.id, file_paths=local_doc.path)
    lightrag_docs = async_to_sync(rag.get_processed_docs)()
    if not lightrag_docs or str(document.id) not in lightrag_docs:
        error_msg = f"Error indexing document for LightRAG (ID: {document.id}). No processed document found."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    logger.info(f"Successfully indexed document for LightRAG (ID: {document.id})")


@app.task
def message_feedback(**kwargs):
    feedback_id = kwargs["feedback_id"]
    feedback = MessageFeedback.objects.get(id=feedback_id)
    feedback.status = MessageFeedback.Status.RUNNING
    feedback.save()

    qa_collection_name = generate_qa_vector_db_collection_name(collection=feedback.collection.id)
    vector_store_adaptor = get_vector_db_connector(collection=qa_collection_name)
    embedding_model, _ = async_to_sync(get_collection_embedding_service)(feedback.collection)
    ids = [i for i in str(feedback.relate_ids or "").split(',') if i]
    if ids:
        vector_store_adaptor.connector.delete(ids=ids)
    ids = QAEmbedding(feedback.question, feedback.revised_answer, vector_store_adaptor, embedding_model).load_data()
    if ids:
        feedback.relate_ids = ",".join(ids)
        feedback.save()

    feedback.status = MessageFeedback.Status.COMPLETE
    feedback.save()

@app.task
def generate_questions(document_id):
    try:
        document = Document.objects.get(id=document_id)
        collection = async_to_sync(document.get_collection)()
        embedding_model, _ = async_to_sync(get_collection_embedding_service)(collection)

        source = get_source(parseCollectionConfig(collection.config))
        metadata = json.loads(document.metadata)
        metadata["doc_id"] = document_id
        local_doc = source.prepare_document(name=document.name, metadata=metadata)
        q_loaders = QuestionEmbedding(filepath=local_doc.path,
                                file_metadata=local_doc.metadata,
                                embedding_model=embedding_model,
                                llm_model=None, #todo Fixme
                                vector_store_adaptor=get_vector_db_connector(
                                    collection=generate_qa_vector_db_collection_name(
                                        collection=collection.id)))
        ids, questions = q_loaders.load_data()
        for relate_id, question in zip(ids, questions):
            question_instance = Question(
                user=document.user,
                question=question,
                answer='',
                status=Question.Status.ACTIVE,
                collection_id=collection.id,
                relate_id=relate_id
            )
            question_instance.save()
            question_instance.documents.add(document)
    except Exception as e:
        logger.error(e)
        raise Exception("an error occur %s" % e)

@app.task
def update_index_for_question(question_id):
    try:
        question = Question.objects.get(id=question_id)
        embedding_model, _ = async_to_sync(get_collection_embedding_service)(question.collection)

        q_loaders = QuestionEmbeddingWithoutDocument(embedding_model=embedding_model,
                                vector_store_adaptor=get_vector_db_connector(
                                    collection=generate_qa_vector_db_collection_name(
                                        collection=question.collection.id)))
        if question.relate_id is not None:
            q_loaders.delete(ids=[question.relate_id])
        if question.status != Question.Status.DELETED:
            ids = q_loaders.load_data(faq=[{"question": question.question, "answer": question.answer}])
            question.relate_id = ids[0]
            question.status = Question.Status.ACTIVE
            question.save()
    except Exception as e:
        logger.error(e)
        raise Exception("an error occur %s" % e)


@app.task
def update_collection_status(status, collection_id):
    Collection.objects.filter(
        id=collection_id,
        status=Collection.Status.QUESTION_PENDING
    ).update(status=Collection.Status.ACTIVE)