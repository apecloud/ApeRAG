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

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from aperag.docparser.doc_parser import DocParser
from aperag.objectstore.base import get_object_store

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

    def parse_document(
        self, filepath: str, file_metadata: Dict[str, Any], parser_config: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Parse document into parts using DocParser.

        Args:
            filepath: Path to the document file
            file_metadata: Metadata associated with the document
            parser_config: Configuration for the parser

        Returns:
            List of document parts (MarkdownPart, AssetBinPart, etc.)

        Raises:
            ValueError: If the file type is unsupported
        """
        parser = DocParser(parser_config=parser_config)
        filepath_obj = Path(filepath)

        if not parser.accept(filepath_obj.suffix):
            raise ValueError(f"unsupported file type: {filepath_obj.suffix}")

        parts = parser.parse_file(filepath_obj, file_metadata)
        logger.info(f"Parsed document {filepath} into {len(parts)} parts")
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
        from aperag.docparser.base import AssetBinPart, MarkdownPart, PdfPart

        content = ""

        # Extract full markdown content if available
        md_part = next((part for part in doc_parts if isinstance(part, MarkdownPart)), None)
        if md_part is not None:
            content = md_part.markdown

        pdf_part = next((part for part in doc_parts if isinstance(part, PdfPart)), None)
        if pdf_part is not None:
            doc_parts.remove(pdf_part)

        # Save to object storage if base path is provided
        if object_store_base_path is not None:
            base_path = object_store_base_path
            obj_store = get_object_store()

            # Save markdown content
            md_upload_path = f"{base_path}/parsed.md"
            md_data = content.encode("utf-8")
            obj_store.put(md_upload_path, md_data)
            logger.info(f"uploaded markdown content to {md_upload_path}, size: {len(md_data)}")

            if pdf_part is not None:
                converted_pdf_upload_path = f"{base_path}/converted.pdf"
                obj_store.put(converted_pdf_upload_path, pdf_part.data)
                logger.info(f"uploaded converted pdf to {md_upload_path}, size: {len(pdf_part.data)}")

            # Save assets
            asset_count = 0
            to_be_removed = []
            for part in doc_parts:
                if not isinstance(part, AssetBinPart):
                    continue
                to_be_removed.append(part)

                asset_upload_path = f"{base_path}/assets/{part.asset_id}"
                obj_store.put(asset_upload_path, part.data)
                asset_count += 1
                logger.info(f"uploaded asset to {asset_upload_path}, size: {len(part.data)}")

            for part in to_be_removed:
                doc_parts.remove(part)

            logger.info(f"Saved {asset_count} assets to object storage")

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

    def process_document_parsing(
        self,
        filepath: str,
        file_metadata: Dict[str, Any],
        object_store_base_path: Optional[str] = None,
        parser_config: Optional[Dict[str, Any]] = None,
    ) -> DocumentParsingResult:
        """
        Complete document parsing workflow

        Args:
            filepath: Path to the document file
            file_metadata: Metadata associated with the document
            object_store_base_path: Base path for object storage
            parser_config: Configuration for the parser

        Returns:
            DocumentParsingResult containing parsed parts and content
        """
        try:
            # Parse document into parts
            doc_parts = self.parse_document(filepath, file_metadata, parser_config)

            # Save processed content and assets to object storage
            content = self.save_processed_content_and_assets(doc_parts, object_store_base_path)

            return DocumentParsingResult(doc_parts=doc_parts, content=content, metadata={"parts_count": len(doc_parts)})

        except Exception as e:
            raise Exception(f"Document parsing failed for {filepath}: {str(e)}")


# Global parser instance
document_parser = DocumentParser()
