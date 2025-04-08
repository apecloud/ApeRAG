from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader


class TxtReader(BaseReader):
    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        """Parse file."""
        try:
            with open(file, encoding="utf-8") as f:
                content = f.read()
                return [Document(text=content, metadata=metadata or {})]
        except Exception as e:
            print(f"txt reader error:{e}")
