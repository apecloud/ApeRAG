"""
Simplified index operations - core utilities for document indexing
Provides document parsing utilities and batch processing capabilities
"""

import json
import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


def get_document_and_collection(document_id: str):
    """Get document and collection objects"""
    from aperag.db.ops import db_ops
    
    document = db_ops.query_document_by_id(int(document_id))
    if not document:
        raise ValueError(f"Document {document_id} not found")
    
    collection = db_ops.query_collection_by_id(document.collection_id)
    if not collection:
        raise ValueError(f"Collection {document.collection_id} not found")
    
    return document, collection


def parse_document_content(document, collection) -> Tuple[str, List[Any], Any]:
    """Parse document content for indexing (shared across all index types)"""
    from aperag.core.processors.document_parser import document_parser
    from aperag.schema.utils import parseCollectionConfig
    from aperag.source.base import get_source
    
    # Get document source and prepare local file
    source = get_source(parseCollectionConfig(collection.config))
    metadata = json.loads(document.doc_metadata or "{}")
    metadata["doc_id"] = document.id
    local_doc = source.prepare_document(name=document.name, metadata=metadata)
    
    try:
        # Parse document to get content and parts
        parsing_result = document_parser.process_document_parsing(
            local_doc.path,
            local_doc.metadata,
            document.object_store_base_path()
        )
        
        return parsing_result.content, parsing_result.doc_parts, local_doc
    except Exception as e:
        # Cleanup on error
        source.cleanup_document(local_doc.path)
        raise e


def cleanup_local_document(local_doc, collection):
    """Cleanup local document after processing"""
    from aperag.schema.utils import parseCollectionConfig
    from aperag.source.base import get_source
    source = get_source(parseCollectionConfig(collection.config))
    source.cleanup_document(local_doc.path)


# The individual index operations have been removed to reduce code duplication.
# Tasks now call indexers directly, using the shared utilities above for document parsing. 