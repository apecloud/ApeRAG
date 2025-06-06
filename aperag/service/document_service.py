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
from http import HTTPStatus
from typing import List

from asgiref.sync import sync_to_async
from django.db import IntegrityError
from django.utils import timezone
from ninja.files import UploadedFile

from aperag.apps import QuotaType
from aperag.db import models as db_models
from aperag.db.ops import (
    PagedQuery,
    query_collection,
    query_document,
    query_documents,
    query_documents_count,
    query_user_quota,
)
from aperag.docparser.doc_parser import DocParser
from aperag.objectstore.base import get_object_store
from aperag.schema import view_models
from aperag.schema.view_models import Document, DocumentList
from aperag.tasks.crawl_web import crawl_domain
from aperag.tasks.index import add_index_for_local_document, remove_index, update_index_for_document
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS
from aperag.views.utils import fail, success, validate_url
from config import settings

logger = logging.getLogger(__name__)


def build_document_response(document: db_models.Document) -> view_models.Document:
    """Build Document response object for API return."""
    return Document(
        id=document.id,
        name=document.name,
        status=document.status,
        vector_index_status=document.vector_index_status,
        fulltext_index_status=document.fulltext_index_status,
        graph_index_status=document.graph_index_status,
        size=document.size,
        created=document.gmt_created,
        updated=document.gmt_updated,
    )


async def create_documents(user: str, collection_id: str, files: List[UploadedFile]) -> view_models.DocumentList:
    # 1. Validate the number of files
    if len(files) > 50:
        return fail(HTTPStatus.BAD_REQUEST, "documents are too many,add document failed")
    # 2. Query the collection and check its status
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    if collection.status != db_models.Collection.Status.ACTIVE:
        return fail(HTTPStatus.BAD_REQUEST, "Collection is not active")
    # 3. Check document count against the limit
    if settings.MAX_DOCUMENT_COUNT:
        document_limit = await query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
        if document_limit is None:
            document_limit = settings.MAX_DOCUMENT_COUNT
        if await query_documents_count(user, collection_id) >= document_limit:
            return fail(HTTPStatus.FORBIDDEN, f"document number has reached the limit of {document_limit}")
    # 4. Get supported file extensions
    supported_file_extensions = DocParser().supported_extensions()
    supported_file_extensions += SUPPORTED_COMPRESSED_EXTENSIONS
    response = []
    # 5. Process each uploaded file
    for item in files:
        # 5.1 Validate file type and size
        file_suffix = os.path.splitext(item.name)[1].lower()
        if file_suffix not in supported_file_extensions:
            return fail(HTTPStatus.BAD_REQUEST, f"unsupported file type {file_suffix}")
        if item.size > settings.MAX_DOCUMENT_SIZE:
            return fail(HTTPStatus.BAD_REQUEST, "file size is too large")
        try:
            # 5.2 Create Document instance in database
            document_instance = db_models.Document(
                user=user,
                name=item.name,
                status=db_models.Document.Status.PENDING,
                size=item.size,
                collection_id=collection.id,
            )
            await document_instance.asave()
            # 5.3 Upload file to object storage
            obj_store = get_object_store()
            upload_path = f"{document_instance.object_store_base_path()}/original{file_suffix}"
            await sync_to_async(obj_store.put)(upload_path, item)
            # 5.4 Update document metadata and save
            document_instance.object_path = upload_path
            document_instance.metadata = json.dumps(
                {
                    "object_path": upload_path,
                }
            )
            await document_instance.asave()
            # 5.5 Add document to response and trigger indexing task
            response.append(build_document_response(document_instance))
            add_index_for_local_document.delay(document_instance.id)
        except IntegrityError:
            # 5.6 Handle duplicate document name
            return fail(HTTPStatus.BAD_REQUEST, f"document {item.name} already exists")
        except Exception:
            # 5.7 Handle other exceptions during processing
            logger.exception("add document failed")
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, "add document failed")
    # 6. Return success response with created documents
    return success(DocumentList(items=response))


async def create_url_document(user: str, collection_id: str, urls: List[str]) -> view_models.DocumentList:
    response = {"failed_urls": []}
    collection = await query_collection(user, collection_id)
    if collection is None:
        return fail(HTTPStatus.NOT_FOUND, "Collection not found")
    if settings.MAX_DOCUMENT_COUNT:
        document_limit = await query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
        if document_limit is None:
            document_limit = settings.MAX_DOCUMENT_COUNT
        if await query_documents_count(user, collection_id) >= document_limit:
            return fail(HTTPStatus.FORBIDDEN, f"document number has reached the limit of {document_limit}")
    failed_urls = []
    for url in urls:
        if not validate_url(url):
            failed_urls.append(url)
            continue
        if ".html" not in url:
            document_name = url + ".html"
        else:
            document_name = url
        document_instance = db_models.Document(
            user=user,
            name=document_name,
            status=db_models.Document.Status.PENDING,
            collection_id=collection.id,
            size=0,
        )
        await document_instance.asave()
        string_data = json.dumps(url)
        document_instance.metadata = json.dumps(
            {
                "url": string_data,
            }
        )
        await document_instance.asave()
        add_index_for_local_document.delay(document_instance.id)
        crawl_domain.delay(url, url, collection_id, user, max_pages=2)
    if len(failed_urls) != 0:
        response["message"] = "Some URLs failed validation,eg. https://example.com/path?query=123#fragment"
        response["failed_urls"] = failed_urls
    return success(DocumentList(items=response))


async def list_documents(user: str, collection_id: str, pq: PagedQuery) -> view_models.DocumentList:
    pr = await query_documents([user, settings.ADMIN_USER], collection_id, pq)
    response = []
    async for document in pr.data:
        response.append(build_document_response(document))
    return success(DocumentList(items=response), pr=pr)


async def get_document(user: str, collection_id: str, document_id: str) -> view_models.Document:
    document = await query_document(user, collection_id, document_id)
    if document is None:
        return fail(HTTPStatus.NOT_FOUND, "Document not found")
    return success(build_document_response(document))


async def update_document(
    user: str, collection_id: str, document_id: str, document_in: view_models.DocumentUpdate
) -> view_models.Document:
    instance = await query_document(user, collection_id, document_id)
    if instance is None:
        return fail(HTTPStatus.NOT_FOUND, "Document not found")
    if instance.status == db_models.Document.Status.DELETING:
        return fail(HTTPStatus.BAD_REQUEST, "Document is deleting")
    if document_in.config:
        try:
            config = json.loads(document_in.config)
            metadata = json.loads(instance.metadata)
            metadata["labels"] = config["labels"]
            instance.metadata = json.dumps(metadata)
        except Exception:
            return fail(HTTPStatus.BAD_REQUEST, "invalid document config")
    await instance.asave()
    update_index_for_document.delay(instance.id)
    return success(build_document_response(instance))


async def delete_document(user: str, collection_id: str, document_id: str) -> view_models.Document:
    document = await query_document(user, collection_id, document_id)
    if document is None:
        logger.info(f"document {document_id} not found, maybe has already been deleted")
        return success({})
    if document.status == db_models.Document.Status.DELETING:
        logger.info(f"document {document_id} is deleting, ignore delete")
        return success({})
    obj_store = get_object_store()
    await sync_to_async(obj_store.delete_objects_by_prefix)(f"{document.object_store_base_path()}/")
    document.status = db_models.Document.Status.DELETING
    document.gmt_deleted = timezone.now()
    await document.asave()
    remove_index.delay(document.id)
    return success(build_document_response(document))


async def delete_documents(user: str, collection_id: str, document_ids: List[str]):
    documents = await query_documents([user], collection_id, None)
    ok = []
    failed = []
    async for document in documents.data:
        if document.id not in document_ids:
            continue
        try:
            document.status = db_models.Document.Status.DELETING
            document.gmt_deleted = timezone.now()
            await document.asave()
            remove_index.delay(document.id)
            ok.append(document.id)
        except Exception as e:
            logger.exception(e)
            failed.append(document.id)
    return success({"success": ok, "failed": failed})
