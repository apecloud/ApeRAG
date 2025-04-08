import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.readers.file.docs import DocxReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class MyDocReader(BaseReader):
    """Doc Parser."""
    """
    Convert the 'doc' type to 'docx' using 'win32com,' and then parse it using the 'docx' type
    """

    def __init__(self):
        self.reader = DocxReader()

    def convert_to_docx(self, input_filename: str,
                        output_directory: str,
                        target_format: str = "docx",
                        target_filter: Optional[str] = "MS Word 2007 XML", ):

        """Converts a .doc file to a .docx file using the libreoffice CLI.

        Parameters
        ----------
        input_filename: str
            The name of the .doc file to convert to .docx
        output_directory: str
            The output directory for the convert .docx file
        target_format: str
            The desired output format
        target_filter: str
            The output filter name to use when converting. See references below
            for details.

        References
        ----------
        https://stackoverflow.com/questions/52277264/convert-doc-to-docx-using-soffice-not-working
        https://git.libreoffice.org/core/+/refs/heads/master/filter/source/config/fragments/filters

        """
        if target_filter is not None:
            target_format = f"{target_format}:{target_filter}"
        # NOTE(robinson) - In the future can also include win32com client as a fallback for windows
        # users who do not have LibreOffice installed
        # ref: https://stackoverflow.com/questions/38468442/
        #       multiple-doc-to-docx-file-conversion-using-python
        command = [
            "soffice",
            "--headless",
            "--convert-to",
            target_format,
            "--outdir",
            output_directory,
            input_filename,
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            output, error = process.communicate()
            logger.debug(output)
            logger.debug(error)
        except FileNotFoundError:
            raise FileNotFoundError(
                """soffice command was not found. Please install libreoffice
    on your system and try again.

    - Install instructions: https://www.libreoffice.org/get-help/install-howto/
    - Mac: https://formulae.brew.sh/cask/libreoffice
    - Debian: https://wiki.debian.org/LibreOffice""",
            )

    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        tempdir = tempfile.gettempdir()
        base_filename, _ = os.path.splitext(file)
        self.convert_to_docx(file, tempdir, "docx")
        return self.reader.load_data(Path(os.path.join(tempdir, f"{base_filename}.docx")), extra_info=metadata)
