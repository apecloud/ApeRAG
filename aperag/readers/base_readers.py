from typing import Dict, Type

from llama_index.core.readers.base import BaseReader
from llama_index.readers.file.docs.base import DocxReader, PDFReader
from llama_index.readers.file.ipynb import IPYNBReader
from llama_index.readers.file.mbox import MboxReader
from llama_index.readers.file.tabular import PandasCSVReader

from aperag.readers.compose_audio_reader import ComposeAudioReader
from aperag.readers.compose_image_reader import ComposeImageReader
from aperag.readers.compressed_file_reader import CompressedFileReader
from aperag.readers.doc_reader import MyDocReader
from aperag.readers.epub_reader import EpubReader
from aperag.readers.excel_reader import ExcelReader
from aperag.readers.html_reader import HtmlReader
from aperag.readers.markdown_reader import MarkdownReader
from aperag.readers.mineru_reader import MinerUReader
from aperag.readers.ppt_reader import PptReader
from aperag.readers.pptx_reader import PptxReader

DEFAULT_FILE_READER_CLS: Dict[str, Type[BaseReader]] = {
    ".pdf": MinerUReader[PDFReader],
    ".docx": MinerUReader[DocxReader],
    ".doc": MinerUReader[MyDocReader],
    ".pptx": MinerUReader[PptxReader],
    ".ppt": MinerUReader[PptReader],
    ".html": HtmlReader,
    ".xlxs": ExcelReader,
    ".jpg": ComposeImageReader,
    ".png": ComposeImageReader,
    ".jpeg": ComposeImageReader,
    ".mp3": ComposeAudioReader,
    ".mp4": ComposeAudioReader,
    ".mpeg": ComposeAudioReader,
    ".mpga": ComposeAudioReader,
    ".m4a": ComposeAudioReader,
    ".wav": ComposeAudioReader,
    ".webm": ComposeAudioReader,
    ".csv": PandasCSVReader,
    ".epub": EpubReader,
    ".md": MarkdownReader,
    ".mbox": MboxReader,
    ".ipynb": IPYNBReader,
    ".txt": MarkdownReader,
    ".zip": CompressedFileReader,
    ".rar": CompressedFileReader,
    ".7z": CompressedFileReader,
    ".tar": CompressedFileReader,
    ".gz": CompressedFileReader,
    ".xz": CompressedFileReader,
    ".bz2": CompressedFileReader,
    ".tar.gz": CompressedFileReader,
    ".tar.xz": CompressedFileReader,
    ".tar.bz2": CompressedFileReader,
    ".tar.7z": CompressedFileReader
}


FULLTEXT_SUFFIX = {
    ".md": True,
    ".txt": True,
}
