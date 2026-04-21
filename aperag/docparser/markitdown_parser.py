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

import tempfile
import zipfile
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from aperag.docparser.base import BaseParser, FallbackError, ParserError, Part
from aperag.docparser.parse_md import parse_md
from aperag.docparser.utils import convert_office_doc, get_soffice_cmd

SUPPORTED_EXTENSIONS = [
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".csv",
    ".tsv",
    ".xml",
    ".rst",
    ".eml",
    ".ipynb",
    ".pdf",
    ".docx",
    ".doc",  # convert to .docx first
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",  # convert to .pptx first
    ".epub",
]


class MarkItDownParser(BaseParser):
    name = "markitdown"

    def supported_extensions(self) -> list[str]:
        return SUPPORTED_EXTENSIONS

    def parse_file(self, path: Path, metadata: dict[str, Any] = {}, **kwargs) -> list[Part]:
        extension = path.suffix.lower()
        target_format = None
        if extension == ".doc":
            target_format = ".docx"
        elif extension == ".ppt":
            target_format = ".pptx"
        if target_format:
            if get_soffice_cmd() is None:
                raise FallbackError(
                    "Legacy Office conversion requires soffice",
                    parser_name=self.name,
                    code="missing_dependency",
                    detail="Install LibreOffice/OpenOffice or upload .docx/.pptx instead.",
                )
            with tempfile.TemporaryDirectory() as temp_dir:
                converted = convert_office_doc(path, Path(temp_dir), target_format)
                return self._parse_file(converted, metadata, **kwargs)
        return self._parse_file(path, metadata, **kwargs)

    def _parse_file(self, path: Path, metadata: dict[str, Any] = {}, **kwargs) -> list[Part]:
        try:
            self._validate_binary_container(path)
            mid = MarkItDown()
            result = mid.convert_local(path, keep_data_uris=True)
            return parse_md(result.markdown, metadata)
        except ParserError:
            raise
        except Exception as e:
            raise ParserError(
                f"MarkItDown failed to parse {path.suffix.lower() or 'unknown'} file",
                parser_name=self.name,
                code="parse_failed",
                detail=str(e),
            ) from e

    def _validate_binary_container(self, path: Path) -> None:
        extension = path.suffix.lower()
        if extension == ".pdf":
            self._validate_pdf_header(path)
            return

        required_members = {
            ".docx": ("[Content_Types].xml", "word/document.xml"),
            ".xlsx": ("[Content_Types].xml", "xl/workbook.xml"),
            ".pptx": ("[Content_Types].xml", "ppt/presentation.xml"),
        }.get(extension)
        if required_members is None:
            return

        try:
            with zipfile.ZipFile(path) as archive:
                members = set(archive.namelist())
        except zipfile.BadZipFile as exc:
            raise ParserError(
                f"Corrupted {extension} document",
                parser_name=self.name,
                code="corrupted_document",
                detail=f"{extension} is not a valid OOXML package: {exc}",
            ) from exc

        missing_members = [member for member in required_members if member not in members]
        if missing_members:
            raise ParserError(
                f"Corrupted {extension} document",
                parser_name=self.name,
                code="corrupted_document",
                detail=f"{extension} is missing required OOXML entries: {', '.join(missing_members)}",
            )

    def _validate_pdf_header(self, path: Path) -> None:
        header = path.read_bytes()[:5]
        if header != b"%PDF-":
            raise ParserError(
                "Corrupted .pdf document",
                parser_name=self.name,
                code="corrupted_document",
                detail="Missing %PDF file header.",
            )
