import logging

from aperag.db.models import DocumentIndexType
from aperag.tasks.models import IndexTaskResult, LocalDocumentInfo, ParsedDocumentData
from aperag.tasks.utils import parse_document_content

logger = logging.getLogger(__name__)


class DocumentIndexTask:
    """
    Document index task orchestrator with structured data handling
    """

    # ========== Fine-grained Task Methods ==========

    def parse_document(self, document_id: str) -> ParsedDocumentData:
        """
        Parse document content and return structured data

        Args:
            document_id: Document ID to parse

        Returns:
            ParsedDocumentData containing all parsed information
        """
        logger.info(f"Parsing document {document_id}")

        from aperag.tasks.utils import get_document_and_collection

        document, collection = get_document_and_collection(document_id)
        content, doc_parts, local_doc = parse_document_content(document, collection)

        # Create structured local document info
        local_doc_info = LocalDocumentInfo(path=local_doc.path, is_temp=getattr(local_doc, "is_temp", False))

        # Return structured parsed data
        return ParsedDocumentData(
            document_id=document_id,
            collection_id=collection.id,
            content=content,
            doc_parts=doc_parts,
            file_path=local_doc.path,
            local_doc_info=local_doc_info,
        )

    def create_index(self, document_id: str, index_type: str, parsed_data: ParsedDocumentData) -> IndexTaskResult:
        """
        Create a single index for a document using parsed data

        Args:
            document_id: Document ID
            index_type: Type of index to create
            parsed_data: Parsed document data

        Returns:
            IndexTaskResult containing operation result
        """
        logger.info(f"Creating {index_type} index for document {document_id}")

        # Get collection
        from aperag.tasks.utils import get_document_and_collection

        _, collection = get_document_and_collection(document_id)

        try:
            if index_type == DocumentIndexType.VECTOR.value:
                from aperag.index.vector_index import vector_indexer

                result = vector_indexer.create_index(
                    document_id=document_id,
                    content=parsed_data.content,
                    doc_parts=parsed_data.doc_parts,
                    collection=collection,
                    file_path=parsed_data.file_path,
                )
                if not result.success:
                    raise Exception(result.error)
                result_data = result.data or {"success": True}

            elif index_type == DocumentIndexType.FULLTEXT.value:
                from aperag.index.fulltext_index import fulltext_indexer

                result = fulltext_indexer.create_index(
                    document_id=document_id,
                    content=parsed_data.content,
                    doc_parts=parsed_data.doc_parts,
                    collection=collection,
                    file_path=parsed_data.file_path,
                )
                if not result.success:
                    raise Exception(result.error)
                result_data = result.data or {"success": True}

            elif index_type == DocumentIndexType.GRAPH.value:
                from aperag.index.graph_index import graph_indexer

                if not graph_indexer.is_enabled(collection):
                    logger.info(f"Graph indexing disabled for document {document_id}")
                    result_data = {"success": True, "message": "Graph indexing disabled"}
                else:
                    from aperag.graph.lightrag_manager import process_document_for_celery

                    result = process_document_for_celery(
                        collection=collection,
                        content=parsed_data.content,
                        doc_id=document_id,
                        file_path=parsed_data.file_path,
                    )
                    if result.get("status") != "success":
                        error_msg = result.get("message", "Unknown error")
                        raise Exception(f"Graph indexing failed: {error_msg}")
                    result_data = result
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            return IndexTaskResult.success_result(
                index_type=index_type,
                document_id=document_id,
                data=result_data,
                message=f"Successfully created {index_type} index",
            )

        except Exception as e:
            error_msg = f"Failed to create {index_type} index: {str(e)}"
            logger.error(f"Document {document_id}: {error_msg}")
            return IndexTaskResult.failed_result(index_type=index_type, document_id=document_id, error=error_msg)

    def delete_index(self, document_id: str, index_type: str) -> IndexTaskResult:
        """
        Delete a single index for a document

        Args:
            document_id: Document ID
            index_type: Type of index to delete

        Returns:
            IndexTaskResult containing operation result
        """
        logger.info(f"Deleting {index_type} index for document {document_id}")

        from aperag.tasks.utils import get_document_and_collection

        _, collection = get_document_and_collection(document_id)

        try:
            if index_type == DocumentIndexType.VECTOR.value:
                from aperag.index.vector_index import vector_indexer

                result = vector_indexer.delete_index(document_id, collection)
                if not result.success:
                    raise Exception(result.error)

            elif index_type == DocumentIndexType.FULLTEXT.value:
                from aperag.index.fulltext_index import fulltext_indexer

                result = fulltext_indexer.delete_index(document_id, collection)
                if not result.success:
                    raise Exception(result.error)

            elif index_type == DocumentIndexType.GRAPH.value:
                from aperag.index.graph_index import graph_indexer

                if graph_indexer.is_enabled(collection):
                    from aperag.graph.lightrag_manager import delete_document_for_celery

                    result = delete_document_for_celery(collection=collection, doc_id=document_id)
                    if result.get("status") != "success":
                        error_msg = result.get("message", "Unknown error")
                        raise Exception(f"Graph index deletion failed: {error_msg}")
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            return IndexTaskResult.success_result(
                index_type=index_type, document_id=document_id, message=f"Successfully deleted {index_type} index"
            )

        except Exception as e:
            error_msg = f"Failed to delete {index_type} index: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return IndexTaskResult.failed_result(index_type=index_type, document_id=document_id, error=error_msg)

    def update_index(self, document_id: str, index_type: str, parsed_data: ParsedDocumentData) -> IndexTaskResult:
        """
        Update a single index for a document using parsed data

        Args:
            document_id: Document ID
            index_type: Type of index to update
            parsed_data: Parsed document data

        Returns:
            IndexTaskResult containing operation result
        """
        logger.info(f"Updating {index_type} index for document {document_id}")

        # Get collection
        from aperag.tasks.utils import get_document_and_collection

        _, collection = get_document_and_collection(document_id)

        try:
            if index_type == DocumentIndexType.VECTOR.value:
                from aperag.index.vector_index import vector_indexer

                result = vector_indexer.update_index(
                    document_id=document_id,
                    content=parsed_data.content,
                    doc_parts=parsed_data.doc_parts,
                    collection=collection,
                    file_path=parsed_data.file_path,
                )
                if not result.success:
                    raise Exception(result.error)
                result_data = result.data or {"success": True}

            elif index_type == DocumentIndexType.FULLTEXT.value:
                from aperag.index.fulltext_index import fulltext_indexer

                result = fulltext_indexer.update_index(
                    document_id=document_id,
                    content=parsed_data.content,
                    doc_parts=parsed_data.doc_parts,
                    collection=collection,
                    file_path=parsed_data.file_path,
                )
                if not result.success:
                    raise Exception(result.error)
                result_data = result.data or {"success": True}

            elif index_type == DocumentIndexType.GRAPH.value:
                from aperag.index.graph_index import graph_indexer

                if not graph_indexer.is_enabled(collection):
                    logger.info(f"Graph indexing disabled for document {document_id}")
                    result_data = {"success": True, "message": "Graph indexing disabled"}
                else:
                    from aperag.graph.lightrag_manager import process_document_for_celery

                    result = process_document_for_celery(
                        collection=collection,
                        content=parsed_data.content,
                        doc_id=document_id,
                        file_path=parsed_data.file_path,
                    )
                    if result.get("status") != "success":
                        error_msg = result.get("message", "Unknown error")
                        raise Exception(f"Graph indexing failed: {error_msg}")
                    result_data = result
            else:
                raise ValueError(f"Unknown index type: {index_type}")

            return IndexTaskResult.success_result(
                index_type=index_type,
                document_id=document_id,
                data=result_data,
                message=f"Successfully updated {index_type} index",
            )

        except Exception as e:
            error_msg = f"Failed to update {index_type} index: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return IndexTaskResult.failed_result(index_type=index_type, document_id=document_id, error=error_msg)


document_index_task = DocumentIndexTask()
