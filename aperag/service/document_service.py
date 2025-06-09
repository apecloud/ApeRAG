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
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.config import settings
from aperag.db import models as db_models
from aperag.db.ops import (
    AsyncDatabaseOps,
    async_db_ops,
)
from aperag.docparser.doc_parser import DocParser
from aperag.objectstore.base import get_object_store
from aperag.schema import view_models
from aperag.schema.view_models import Document, DocumentList
from aperag.tasks.crawl_web import crawl_domain
from aperag.tasks.index import add_index_for_local_document, remove_index, update_index_for_document
from aperag.utils.constant import QuotaType
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS
from aperag.views.utils import fail, success, validate_url

logger = logging.getLogger(__name__)


class DocumentService:
    """Document service that handles business logic for documents"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    def build_document_response(self, document: db_models.Document) -> view_models.Document:
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

    async def create_documents(
        self, user: str, collection_id: str, files: List[UploadFile]
    ) -> view_models.DocumentList:
        if len(files) > 50:
            return fail(HTTPStatus.BAD_REQUEST, "documents are too many,add document failed")

        collection = await self.db_ops.query_collection(user, collection_id)
        if collection is None:
            return fail(HTTPStatus.NOT_FOUND, "Collection not found")
        if collection.status != db_models.CollectionStatus.ACTIVE:
            return fail(HTTPStatus.BAD_REQUEST, "Collection is not active")

        if settings.max_document_count:
            document_limit = await self.db_ops.query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
            if document_limit is None:
                document_limit = settings.max_document_count
            if await self.db_ops.query_documents_count(user, collection_id) >= document_limit:
                return fail(HTTPStatus.FORBIDDEN, f"document number has reached the limit of {document_limit}")

        supported_file_extensions = DocParser().supported_extensions()
        supported_file_extensions += SUPPORTED_COMPRESSED_EXTENSIONS

        async def _create_documents_operation(session):
            db_ops_session = AsyncDatabaseOps(session)
            response = []

            for item in files:
                file_suffix = os.path.splitext(item.filename)[1].lower()
                if file_suffix not in supported_file_extensions:
                    raise ValueError(f"unsupported file type {file_suffix}")
                if item.size > settings.max_document_size:
                    raise ValueError("file size is too large")

                # Use DatabaseOps to create document
                document_instance = await db_ops_session.create_document(
                    user=user, collection_id=collection.id, name=item.filename, size=item.size
                )

                obj_store = get_object_store()
                upload_path = f"{document_instance.object_store_base_path()}/original{file_suffix}"

                # Read file content from UploadFile
                file_content = await item.read()
                # Reset file pointer for potential future use
                await item.seek(0)

                # Use sync_to_async to call the synchronous put method with file content
                await sync_to_async(obj_store.put)(upload_path, file_content)

                # Update document with object path
                metadata = json.dumps({"object_path": upload_path})
                updated_doc = await db_ops_session.update_document_by_id(
                    user, collection_id, document_instance.id, metadata
                )
                updated_doc.object_path = upload_path
                session.add(updated_doc)
                await session.flush()
                await session.refresh(updated_doc)

                response.append(self.build_document_response(updated_doc))
                add_index_for_local_document.delay(updated_doc.id)

            return response

        try:
            result = await self.db_ops.execute_with_transaction(_create_documents_operation)
            return success(DocumentList(items=result))
        except ValueError as e:
            return fail(HTTPStatus.BAD_REQUEST, str(e))
        except Exception:
            logger.exception("add document failed")
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, "add document failed")

    async def create_url_document(self, user: str, collection_id: str, urls: List[str]) -> view_models.DocumentList:
        collection = await self.db_ops.query_collection(user, collection_id)
        if collection is None:
            return fail(HTTPStatus.NOT_FOUND, "Collection not found")

        if settings.max_document_count:
            document_limit = await self.db_ops.query_user_quota(user, QuotaType.MAX_DOCUMENT_COUNT)
            if document_limit is None:
                document_limit = settings.max_document_count
            if await self.db_ops.query_documents_count(user, collection_id) >= document_limit:
                return fail(HTTPStatus.FORBIDDEN, f"document number has reached the limit of {document_limit}")

        async def _create_url_documents_operation(session):
            db_ops_session = AsyncDatabaseOps(session)
            failed_urls = []

            for url in urls:
                if not validate_url(url):
                    failed_urls.append(url)
                    continue

                document_name = url + ".html" if ".html" not in url else url

                document_instance = await db_ops_session.create_document(
                    user=user, collection_id=collection.id, name=document_name, size=0
                )

                string_data = json.dumps(url)
                metadata = json.dumps({"url": string_data})
                await db_ops_session.update_document_by_id(user, collection_id, document_instance.id, metadata)

                add_index_for_local_document.delay(document_instance.id)
                crawl_domain.delay(url, url, collection_id, user, max_pages=2)

            response = {"failed_urls": failed_urls}
            if len(failed_urls) != 0:
                response["message"] = "Some URLs failed validation,eg. https://example.com/path?query=123#fragment"

            return response

        try:
            result = await self.db_ops.execute_with_transaction(_create_url_documents_operation)
            return success(DocumentList(items=result))
        except Exception:
            logger.exception("create url document failed")
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, "create url document failed")

    async def list_documents(self, user: str, collection_id: str) -> view_models.DocumentList:
        documents = await self.db_ops.query_documents([user], collection_id)
        response = []
        for document in documents:
            response.append(self.build_document_response(document))
        return success(DocumentList(items=response))

    async def get_document(self, user: str, collection_id: str, document_id: str) -> view_models.Document:
        document = await self.db_ops.query_document(user, collection_id, document_id)
        if document is None:
            return fail(HTTPStatus.NOT_FOUND, "Document not found")
        return success(self.build_document_response(document))

    async def update_document(
        self, user: str, collection_id: str, document_id: str, document_in: view_models.DocumentUpdate
    ) -> view_models.Document:
        instance = await self.db_ops.query_document(user, collection_id, document_id)
        if instance is None:
            return fail(HTTPStatus.NOT_FOUND, "Document not found")
        if instance.status == db_models.DocumentStatus.DELETING:
            return fail(HTTPStatus.BAD_REQUEST, "Document is deleting")

        async def _update_document_operation(session):
            if document_in.config:
                try:
                    config = json.loads(document_in.config)
                    metadata = json.loads(instance.metadata)
                    metadata["labels"] = config["labels"]
                    updated_metadata = json.dumps(metadata)

                    db_ops_session = AsyncDatabaseOps(session)
                    updated_doc = await db_ops_session.update_document_by_id(
                        user, collection_id, document_id, updated_metadata
                    )

                    if not updated_doc:
                        raise ValueError("Document not found")

                    update_index_for_document.delay(updated_doc.id)
                    return self.build_document_response(updated_doc)
                except json.JSONDecodeError:
                    raise ValueError("invalid document config")
            else:
                return self.build_document_response(instance)

        try:
            result = await self.db_ops.execute_with_transaction(_update_document_operation)
            return success(result)
        except ValueError as e:
            status_code = HTTPStatus.NOT_FOUND if "not found" in str(e) else HTTPStatus.BAD_REQUEST
            return fail(status_code, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to update document: {str(e)}")

    async def delete_document(self, user: str, collection_id: str, document_id: str) -> view_models.Document:
        document = await self.db_ops.query_document(user, collection_id, document_id)
        if document is None:
            logger.info(f"document {document_id} not found, maybe has already been deleted")
            return success({})
        if document.status == db_models.DocumentStatus.DELETING:
            logger.info(f"document {document_id} is deleting, ignore delete")
            return success({})

        async def _delete_document_operation(session):
            obj_store = get_object_store()
            await sync_to_async(obj_store.delete_objects_by_prefix)(f"{document.object_store_base_path()}/")

            db_ops_session = AsyncDatabaseOps(session)
            deleted_doc = await db_ops_session.delete_document_by_id(user, collection_id, document_id)

            if deleted_doc:
                remove_index.delay(document_id)
                return self.build_document_response(deleted_doc)
            else:
                raise ValueError("Document not found")

        try:
            result = await self.db_ops.execute_with_transaction(_delete_document_operation)
            return success(result)
        except ValueError as e:
            return fail(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to delete document: {str(e)}")

    async def delete_documents(self, user: str, collection_id: str, document_ids: List[str]):
        async def _delete_documents_operation(session):
            db_ops_session = AsyncDatabaseOps(session)
            success_ids, failed_ids = await db_ops_session.delete_documents_by_ids(user, collection_id, document_ids)

            for doc_id in success_ids:
                remove_index.delay(doc_id)

            return {"success": success_ids, "failed": failed_ids}

        try:
            result = await self.db_ops.execute_with_transaction(_delete_documents_operation)
            return success(result)
        except Exception as e:
            logger.exception("Failed to delete documents")
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to delete documents: {str(e)}")


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
document_service = DocumentService()


# Keep existing functions for backward compatibility
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
