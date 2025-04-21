import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Generic, List, Optional, Type, TypeVar, get_args

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


OFFICE_DOC_SUFFIXES = set((".doc", ".docx", ".ppt", ".pptx"))
SUPPORTED_SUFFIXES = set((".pdf",)).union(OFFICE_DOC_SUFFIXES)


FallbackReader = TypeVar("FallbackReader")


class MinerUReader(BaseReader, Generic[FallbackReader]):

    _use_fallback_reader: bool = False

    def _set_config_path(self) -> bool:
        path = Path(os.environ.get("MINERU_CONFIG_JSON", "./magic-pdf.json"))
        if not path.exists():
            logger.warning(f"MinerUReader config {path} is not found, fallback to use {self._fallback_reader_cls().__name__}.")
            return False

        # Adjust the model dir when running in docker.
        new_cache_dir = os.environ.get("MINERU_ADJUST_MODEL_CACHE_DIR", None)
        if new_cache_dir is not None:
            cfg = {}
            with open(str(path), "r") as f:
                cfg = json.loads(f.read())
            self._adjust_cache_path(cfg, "models-dir", new_cache_dir)
            self._adjust_cache_path(cfg, "layoutreader-model-dir", new_cache_dir)
            content = json.dumps(cfg)

            path = Path(tempfile.gettempdir()) / "magic-pdf-mod.json"

            should_write_file = True
            if path.exists():
                curr_content = ""
                with path.open("r") as f:
                    curr_content = f.read()
                should_write_file = (content != curr_content)
            if should_write_file:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as f:
                    f.write(content)

        config_reader.CONFIG_FILE_NAME = str(path.absolute())

        return True

    def _adjust_cache_path(self, cfg: dict, key: str, new_cache_dir: str):
        cache_dir_name = ".cache"
        orig = cfg.get(key, "")
        pos = orig.find(cache_dir_name)
        if pos == -1:
            return
        cfg[key] = str(Path(new_cache_dir + "/" + orig[pos+len(cache_dir_name):]))

    def _fallback_reader_cls(self) -> Type[FallbackReader]:
        return get_args(self.__orig_class__)[0]

    def _check_soffice(self) -> bool:
        if shutil.which("soffice") is None:
            logger.warning(
                f"soffice command was not found, fallback to use {self._fallback_reader_cls().__name__}."
            )
            return False
        return True

    def use_fallback_reader(self, v: bool):
        self._use_fallback_reader = v

    def _should_use_fallback_reader(self, file_suffix: str) -> bool:
        if self._use_fallback_reader:
            return True
        if not self._set_config_path():
            return True
        if file_suffix in OFFICE_DOC_SUFFIXES:
            return not self._check_soffice()
        return False

    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        """Parse file."""

        file_suffix = file.suffix.lower()
        if file_suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {file_suffix}. Supported types are: {', '.join(SUPPORTED_SUFFIXES)}")

        if self._should_use_fallback_reader(file_suffix):
            reader_cls = self._fallback_reader_cls()
            return reader_cls().load_data(file, metadata)

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
