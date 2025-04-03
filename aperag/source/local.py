import logging
import os
from datetime import datetime
from typing import Any, Dict, Iterator

from aperag.source.base import CustomSourceInitializationError, LocalDocument, RemoteDocument, Source

logger = logging.getLogger(__name__)


class LocalSource(Source):

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        self.path = ctx["path"]
        if not os.path.isdir(self.path):
            raise CustomSourceInitializationError("input is not a dir")

    def scan_documents(self) -> Iterator[RemoteDocument]:
        if not os.path.isdir(self.path):
            logger.error(f"{self.path} is not a dir")
            return iter([])

        documents = []
        logger.debug(f"phrase dir is {self.path}")
        for root, dirs, files in os.walk(self.path):
            for file in files:
                file_path = os.path.join(root, file)
                # maybe add a field to record the local file ref rather than upload local file
                try:
                    file_stat = os.stat(file_path)
                    modified_time = datetime.utcfromtimestamp(file_stat.st_mtime)
                    doc = RemoteDocument(
                        name=file_path,
                        size=file_stat.st_size,
                        modified_time=modified_time
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"scanning local source {file_path} error {e}")
                    raise e
        return documents

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        metadata["name"] = name
        return LocalDocument(name=name, path=name, metadata=metadata)

    def cleanup_document(self, filepath: str):
        pass

    def sync_enabled(self):
        return True
