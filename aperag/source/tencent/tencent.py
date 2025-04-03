from typing import Any, Dict, Iterator

from deeprag.source.base import LocalDocument, RemoteDocument, Source
from deeprag.source.tencent.client import TencentClient


class TencentSource(Source):
    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        self.client = TencentClient(ctx)

    def scan_documents(self) -> Iterator[RemoteDocument]:
        return self.client.scan_documents()

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        return self.client.prepare_document(name, metadata, source="download")

    def sync_enabled(self):
        return True
