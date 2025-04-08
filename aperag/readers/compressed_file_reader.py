import logging
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)
class CompressedFileReader(BaseReader):

    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        return [
            Document(
                text="",
                metadata=metadata or {},
            )
        ]



