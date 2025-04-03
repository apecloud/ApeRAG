from typing import Any, Dict, Iterator

from aperag.source.base import LocalDocument, RemoteDocument, Source
from aperag.source.utils import gen_temporary_file


class UploadSource(Source):

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)

    def scan_documents(self) -> Iterator[RemoteDocument]:
        return iter([])

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        path = metadata.get("path", "")
        if not path:
            raise Exception("empty upload path")
        with open(path, "rb") as f:
            content = f.read()
        temp_file = gen_temporary_file(name)
        temp_file.write(content)
        temp_file.close()
        metadata["name"] = name
        return LocalDocument(name=name, path=temp_file.name, metadata=metadata)

    def sync_enabled(self):
        return False
