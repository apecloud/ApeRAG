from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader


class HtmlReader(BaseReader):
    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        """Parse file."""
        try:
            from unstructured.partition.html import partition_html
            elements = partition_html(filename=str(file))
            content = ''.join(item.text for item in elements)
            return [Document(text=content, metadata=metadata or {})]
        except Exception as e:
            print(f"html reader error:{e}")
