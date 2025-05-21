import tempfile
from pathlib import Path
from typing import Any

from markitdown import MarkItDown

from aperag.docparser.base import BaseParser, FallbackError, Part
from aperag.docparser.parse_md import parse_md
from aperag.docparser.utils import convert_office_doc, get_soffice_cmd

SUPPORTED_EXTENSIONS = [
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".html",
    ".htm",
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
                raise FallbackError("soffice command not found")
            with tempfile.TemporaryDirectory() as temp_dir:
                converted = convert_office_doc(path, Path(temp_dir), target_format)
                return self._parse_file(converted, metadata, **kwargs)
        return self._parse_file(path, metadata, **kwargs)

    def _parse_file(self, path: Path, metadata: dict[str, Any] = {}, **kwargs) -> list[Part]:
        mid = MarkItDown()
        result = mid.convert_local(path, keep_data_uris=True)
        return parse_md(result.markdown, metadata)
