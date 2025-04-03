from typing import Dict, Type

from llama_index.readers.base import BaseReader
from llama_index.readers.file.docs_reader import DocxReader, PDFReader
from llama_index.readers.file.ipynb_reader import IPYNBReader
from llama_index.readers.file.mbox_reader import MboxReader
from llama_index.readers.file.tabular_reader import PandasCSVReader

from deeprag.readers.compose_audio_reader import ComposeAudioReader
from deeprag.readers.compose_image_reader import ComposeImageReader
from deeprag.readers.compressed_file_reader import CompressedFileReader
from deeprag.readers.doc_reader import MyDocReader
from deeprag.readers.epub_reader import EpubReader
from deeprag.readers.excel_reader import ExcelReader
from deeprag.readers.html_reader import HtmlReader
from deeprag.readers.markdown_reader import MarkdownReader
from deeprag.readers.ppt_reader import PptReader
from deeprag.readers.pptx_reader import PptxReader

DEFAULT_FILE_READER_CLS: Dict[str, Type[BaseReader]] = {
    ".pdf": PDFReader,
    ".docx": DocxReader,
    # ".docx": MyDocxReader,
    ".doc": MyDocReader,
    ".pptx": PptxReader,
    ".ppt": PptReader,
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


SUPPORTED_COMPRESSED_EXTENSIONS = ['.zip', '.rar', '.r00', '.7z', '.tar', '.gz', '.xz', '.bz2', '.tar.gz', '.tar.xz', '.tar.bz2', '.tar.7z']

