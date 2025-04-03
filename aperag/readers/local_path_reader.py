"""Simple reader that reads files of different formats from a directory."""
import logging
import os
from typing import Callable, Dict, List, Optional

from llama_index.readers.base import BaseReader
from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index.readers.schema.base import Document

from deeprag.readers.base_readers import DEFAULT_FILE_READER_CLS

logger = logging.getLogger(__name__)


class InteractiveSimpleDirectoryReader(SimpleDirectoryReader):
    """Simple directory reader.

    Can read files into separate documents, or concatenates
    files into one document text.

    Args:
        input_dir (str): Path to the directory.
        input_files (List): List of file paths to read
            (Optional; overrides input_dir, exclude)
        exclude (List): glob of python file paths to exclude (Optional)
        exclude_hidden (bool): Whether to exclude hidden files (dotfiles).
        errors (str): how encoding and decoding errors are to be handled,
              see https://docs.python.org/3/library/functions.html#open
        recursive (bool): Whether to recursively search in subdirectories.
            False by default.
        required_exts (Optional[List[str]]): List of required extensions.
            Default is None.
        file_extractor (Optional[Dict[str, BaseReader]]): A mapping of file
            extension to a BaseReader class that specifies how to convert that file
            to text. If not specified, use default from DEFAULT_FILE_READER_CLS.
        num_files_limit (Optional[int]): Maximum number of files to read.
            Default is None.
        file_metadata (Optional[Callable[str, Dict]]): A function that takes
            in a filename and returns a Dict of metadata for the Document.
            Default is None.
    """

    def __init__(
            self,
            input_dir: Optional[str] = None,
            input_files: Optional[List] = None,
            exclude: Optional[List] = None,
            exclude_hidden: bool = True,
            errors: str = "ignore",
            recursive: bool = False,
            required_exts: Optional[List[str]] = None,
            file_extractor: Optional[Dict[str, BaseReader]] = None,
            num_files_limit: Optional[int] = None,
            file_metadata: Optional[Callable[[str], Dict]] = None,
    ) -> None:
        """Initialize with parameters."""
        super().__init__(
            input_dir,
            input_files,
            exclude,
            exclude_hidden,
            errors,
            recursive,
            required_exts,
            file_extractor,
            num_files_limit,
            file_metadata,
        )
        self.file_metadata = file_metadata
        self.process_files = self.input_files or self.phase_dir()

    def load_data(self) -> {List[Document], str}:
        """Load data from the input directory.

        Args:
            concatenate (bool): whether to concatenate all text docs into a single doc.
                If set to True, file metadata is ignored. False by default.
                This setting does not apply to image docs (always one doc per image).

        Returns:
            List[Document]: A list of documents.

        """
        documents = []
        if not self.process_files:
            return documents, ""

        input_file = self.process_files.pop(0)
        metadata: Optional[dict] = None
        if self.file_metadata is not None:
            metadata = self.file_metadata(str(input_file))
        else:
            metadata = {"name": str(input_file)}

        if input_file.suffix.lower() in list(DEFAULT_FILE_READER_CLS.keys()):
            # use file readers
            if input_file.suffix not in self.file_extractor:
                # instantiate file reader if not already
                reader_cls = DEFAULT_FILE_READER_CLS[input_file.suffix]
                self.file_extractor[input_file.suffix] = reader_cls()
            reader = self.file_extractor[input_file.suffix]
            # todo: support more kind of reader
            docs = reader.load_data(input_file)  # metadata for llama_index 0.6.35
            for doc in docs:
                doc.metadata.update(metadata)
            documents.extend(docs)
        else:
            logger.warning(f"Unsupported file extension: {input_file.suffix}")

        return documents, input_file

    def phase_dir(self) -> List:
        files = []
        if not self.input_dir:
            return files
        try:
            files = os.listdir(self.input_dir)
            return files
        except Exception as e:
            logger.error(f"phase dir failed: {e}")
