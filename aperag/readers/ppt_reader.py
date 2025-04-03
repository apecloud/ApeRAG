from pathlib import Path
from typing import Dict, List, Optional

from llama_index import Document
from llama_index.readers.base import BaseReader


class PptReader(BaseReader):
    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        """Parse file."""
        try:

            import os
            import subprocess

            ppt_path = str(file)
            pdf_path = file.with_suffix('.pdf')

            try:
                subprocess.run(
                    ['soffice', '--headless', '--convert-to', 'pdf', '--outdir', os.path.dirname(pdf_path), ppt_path],
                    check=True)
                from unstructured.partition.pdf import partition_pdf
                elements = partition_pdf(filename=str(file))
                content = ''.join(item.text for item in elements)
                return [Document(text=content, metadata=metadata or {})]
            finally:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    print("PDF which converted in PPT has been deleted")


        except Exception as e:
            print(f"ppt reader error:{e}")
