import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.data.read_api import read_local_office
from magic_pdf.libs import config_reader
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

from aperag.readers.markdown_reader import MarkdownReader

logger = logging.getLogger(__name__)


class MinerUReader(BaseReader):

    def _set_config_path(self):
        path = Path(os.environ.get("MINERU_CONFIG_JSON", "./magic-pdf.json"))
        config_reader.CONFIG_FILE_NAME = str(path.absolute())

    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        """Parse file."""

        supported_suffixes: Set[str] = {".pdf", ".doc", ".docx", ".ppt", ".pptx"}
        if file.suffix.lower() not in supported_suffixes:
            raise ValueError(f"Unsupported file type: {file.suffix}. Supported types are: {', '.join(supported_suffixes)}")

        self._set_config_path()

        temp_dir = os.environ.get("MINERU_TEMP_FILE_DIR", None)
        temp_dir_obj: tempfile.TemporaryDirectory | None = None
        if not temp_dir:
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_dir = temp_dir_obj.name

        try:
            local_md_dir = os.path.join(temp_dir, "output")
            local_image_dir = os.path.join(temp_dir, "output/images")
            image_dir = str(os.path.basename(local_image_dir))
            input_file_name = file.stem

            os.makedirs(local_image_dir, exist_ok=True)

            image_writer = FileBasedDataWriter(local_image_dir)
            md_writer = FileBasedDataWriter(local_md_dir)

            parse_method = SupportedPdfParseMethod.OCR
            ds: PymuDocDataset = None
            if file.suffix == ".pdf":
                reader1 = FileBasedDataReader("")
                pdf_bytes = reader1.read(str(file))
                ds = PymuDocDataset(pdf_bytes)
                parse_method = ds.classify()
            else:
                # Note: this requires the "soffice" command to convert office docs into PDF.
                # The "soffice" command is part of LibreOffice, can be installed via:
                #   apt-get install libreoffice
                #   brew install libreoffice
                ds = read_local_office(str(file))[0]

            if parse_method == SupportedPdfParseMethod.OCR:
                ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer).dump_md(
                    md_writer, f"{input_file_name}.md", image_dir
                )
            else:
                ds.apply(doc_analyze, ocr=False).pipe_txt_mode(image_writer).dump_md(
                    md_writer, f"{input_file_name}.md", image_dir
                )

            md_file = os.path.join(local_md_dir, f"{input_file_name}.md")
            return MarkdownReader().load_data(Path(md_file), metadata=metadata)
        except:
            logger.exception("MinerUReader failed")
            raise
        finally:
            if temp_dir_obj is not None:
                temp_dir_obj.cleanup()
