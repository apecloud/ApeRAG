from pathlib import Path
from typing import Dict, List, Optional

from llama_index import Document
from llama_index.readers.base import BaseReader


class ExcelReader(BaseReader):
    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        """Parse file."""
        try:
            from unstructured.partition.xlsx import partition_xlsx
            elements = partition_xlsx(filename=str(file))
            content = ''.join(item.text for item in elements)
            return [Document(text=content, metadata=metadata or {})]
        except Exception as e:
            print(f"excel reader error:{e}")
