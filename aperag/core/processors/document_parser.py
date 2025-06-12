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
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from aperag.db.models import Document, DocumentStatus
from aperag.db.ops import db_ops
from aperag.docparser.doc_parser import DocParser
from aperag.objectstore.base import get_object_store
from aperag.tasks.async_interface import TaskResult
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS, uncompress

logger = logging.getLogger(__name__)


class DocumentParsingResult:
    """Result of document parsing operation"""
    
    def __init__(self, doc_parts: List[Any], content: str, metadata: Optional[Dict[str, Any]] = None):
        self.doc_parts = doc_parts
        self.content = content
        self.metadata = metadata or {}


class DocumentParser:
    """Document parsing and processing logic"""
    
    # Configuration constants
    MAX_EXTRACTED_SIZE = 5000 * 1024 * 1024  # 5 GB
    
    def __init__(self):
        self.logger = logger
    
    def parse_document(self, filepath: str, file_metadata: Dict[str, Any]) -> List[Any]:
        """
        Parse document into parts using DocParser.
        
        Args:
            filepath: Path to the document file
            file_metadata: Metadata associated with the document
            
        Returns:
            List of document parts (MarkdownPart, AssetBinPart, etc.)
            
        Raises:
            ValueError: If the file type is unsupported
        """
        parser = DocParser()  # TODO: use the parser config from the collection
        filepath_obj = Path(filepath)
        
        if not parser.accept(filepath_obj.suffix):
            raise ValueError(f"unsupported file type: {filepath_obj.suffix}")
        
        parts = parser.parse_file(filepath_obj, file_metadata)
        self.logger.info(f"Parsed document {filepath} into {len(parts)} parts")
        return parts
    
    def save_processed_content_and_assets(self, doc_parts: List[Any], object_store_base_path: Optional[str]) -> str:
        """
        Save processed content and assets to object storage.
        
        Args:
            doc_parts: List of document parts from DocParser
            object_store_base_path: Base path for object storage, if None, skip saving
            
        Returns:
            Full markdown content of the document
            
        Raises:
            Exception: If object storage operations fail
        """
        from aperag.docparser.base import AssetBinPart, MarkdownPart
        
        content = ""
        
        # Extract full markdown content if available
        md_part = next((part for part in doc_parts if isinstance(part, MarkdownPart)), None)
        if md_part is not None:
            content = md_part.markdown
        
        # Save to object storage if base path is provided
        if object_store_base_path is not None:
            base_path = object_store_base_path
            obj_store = get_object_store()
            
            # Save markdown content
            md_upload_path = f"{base_path}/parsed.md"
            md_data = content.encode("utf-8")
            obj_store.put(md_upload_path, md_data)
            self.logger.info(f"uploaded markdown content to {md_upload_path}, size: {len(md_data)}")
            
            # Save assets
            asset_count = 0
            for part in doc_parts:
                if not isinstance(part, AssetBinPart):
                    continue
                asset_upload_path = f"{base_path}/assets/{part.asset_id}"
                obj_store.put(asset_upload_path, part.data)
                asset_count += 1
                self.logger.info(f"uploaded asset to {asset_upload_path}, size: {len(part.data)}")
            
            self.logger.info(f"Saved {asset_count} assets to object storage")
        
        return content
    
    def extract_content_from_parts(self, doc_parts: List[Any]) -> str:
        """
        Extract content from document parts when no MarkdownPart is available.
        
        Args:
            doc_parts: List of document parts
            
        Returns:
            Concatenated content from all text parts
        """
        from aperag.docparser.base import MarkdownPart
        
        # Check if MarkdownPart exists
        md_part = next((part for part in doc_parts if isinstance(part, MarkdownPart)), None)
        if md_part is not None:
            return md_part.markdown
        
        # If no MarkdownPart, concatenate content from other parts
        content_parts = []
        for part in doc_parts:
            if hasattr(part, "content") and part.content:
                content_parts.append(part.content)
        
        return "\n\n".join(content_parts)
    
    def process_document_parsing(self, filepath: str, file_metadata: Dict[str, Any], 
                               object_store_base_path: Optional[str] = None) -> DocumentParsingResult:
        """
        Complete document parsing workflow
        
        Args:
            filepath: Path to the document file
            file_metadata: Metadata associated with the document
            object_store_base_path: Base path for object storage
            
        Returns:
            DocumentParsingResult containing parsed parts and content
        """
        try:
            # Parse document into parts
            doc_parts = self.parse_document(filepath, file_metadata)
            
            # Save processed content and assets to object storage
            content = self.save_processed_content_and_assets(doc_parts, object_store_base_path)
            
            return DocumentParsingResult(
                doc_parts=doc_parts,
                content=content,
                metadata={"parts_count": len(doc_parts)}
            )
            
        except Exception as e:
            raise Exception(f"Document parsing failed for {filepath}: {str(e)}")
    
    def handle_compressed_file(self, document: Document, supported_file_extensions: List[str], 
                             task_scheduler=None) -> TaskResult:
        """
        Handle compressed file extraction and create document specs for extracted files
        
        Args:
            document: Document object representing the compressed file
            supported_file_extensions: List of supported file extensions
            task_scheduler: Legacy parameter, ignored in new system
            
        Returns:
            TaskResult indicating success or failure
        """
        try:
            obj_store = get_object_store()
            supported_file_extensions = supported_file_extensions or []
            
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
                        if path.suffix.lower() in SUPPORTED_COMPRESSED_EXTENSIONS:
                            continue
                        if path.suffix.lower() not in supported_file_extensions:
                            continue
                        extracted_files.append(path)
                        total_size += path.stat().st_size
                        
                        if total_size > self.MAX_EXTRACTED_SIZE:
                            raise Exception("Extracted size exceeded limit")
                
                # Create document instances for extracted files
                created_documents = []
                
                for extracted_file_path in extracted_files:
                    with extracted_file_path.open(mode="rb") as extracted_file:  # open in binary
                        document_instance = Document(
                            user=document.user,
                            name=document.name + "/" + extracted_file_path.name,
                            status=DocumentStatus.PENDING,
                            size=extracted_file_path.stat().st_size,
                            collection_id=document.collection_id,
                        )
                        
                        # Upload to object store
                        upload_path = f"{document_instance.object_store_base_path()}/original{suffix}"
                        obj_store.put(upload_path, extracted_file)
                        
                        document_instance.object_path = upload_path
                        document_instance.doc_metadata = json.dumps({
                            "object_path": upload_path, 
                            "uncompressed": "true"
                        })
                        
                        # Save document using db_ops
                        def _operation(session):
                            session.add(document_instance)
                            session.flush()
                            session.refresh(document_instance)  # Refresh to get the generated ID
                            return document_instance
                        
                        db_ops._execute_transaction(_operation)
                        created_documents.append(document_instance.id)
                        
                        # Create index specs using new declarative system
                        try:
                            import asyncio

                            from aperag.db.ops import get_session
                            from aperag.index.manager import document_index_manager
                            
                            async def create_specs():
                                async with get_session() as session:
                                    await document_index_manager.create_document_indexes(
                                        session, str(document_instance.id), document.user
                                    )
                                    await session.commit()
                            
                            asyncio.run(create_specs())
                            self.logger.info(f"Created index specs for extracted document {document_instance.id}")
                        except Exception as e:
                            self.logger.warning(f"Failed to create index specs for document {document_instance.id}: {str(e)}")
                            # Don't fail the whole operation for this
            
            return TaskResult(
                success=True,
                data={"extracted_documents": created_documents},
                metadata={"total_extracted": len(created_documents), "total_size": total_size}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle compressed file {document.object_path}: {str(e)}")
            return TaskResult(
                success=False,
                error=f"Compressed file handling failed: {str(e)}"
            )


# Global parser instance
document_parser = DocumentParser() 