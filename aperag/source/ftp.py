import logging
import os
from datetime import datetime
from ftplib import FTP
from typing import Any, Dict, Iterator

from aperag.source.base import CustomSourceInitializationError, LocalDocument, RemoteDocument, Source
from aperag.source.utils import gen_temporary_file

logger = logging.getLogger(__name__)


class FTPSource(Source):

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        self.path = ctx["path"]
        self.host = ctx["host"]
        self.port = ctx["port"]
        self.user = ctx["username"]
        self.password = ctx["password"]
        try:
            self.ftp = FTP()
            self.ftp.connect(str(self.host), self.port, timeout=3)
            self.ftp.login(self.user, self.password)
        except Exception:
            raise CustomSourceInitializationError("Error connecting to FTP server")

    def isDir(self, path):
        try:
            self.ftp.cwd(path)
            return True
        except Exception:
            return False

    def _deal_the_path(self, ftp, path="/"):
        if not self.isDir(path):
            size = ftp.size(path)
            mtime = ftp.sendcmd('MDTM ' + path)[4:]
            modified_time = datetime.strptime(f"{mtime}", "%Y%m%d%H%M%S")
            doc = RemoteDocument(
                name=path,
                size=size,
                modified_time=modified_time,
            )
            yield doc

        queue = [path]  # dir queue
        while len(queue) > 0:
            curPath = queue[0]
            queue = queue[1:]
            ftp.cwd(curPath)  # Switch to the specified path
            files = ftp.nlst()  # Get the list of files in the current path
            for file in files:
                file_path = os.path.join(curPath, file)
                if self.isDir(file_path):
                    queue.append(file_path)
                    self.ftp.cwd(curPath)
                else:
                    size = ftp.size(file_path)
                    mtime = ftp.sendcmd('MDTM ' + file_path)[4:]
                    doc = RemoteDocument(
                        name=file_path,
                        size=size,
                        metadata={
                            "modified_time": datetime.strptime(f"{mtime}", "%Y%m%d%H%M%S"),
                        }
                    )
                    yield doc

    def scan_documents(self) -> Iterator[RemoteDocument]:
        try:
            return self._deal_the_path(self.ftp, self.path)
        except Exception as e:
            raise e

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        temp_file = gen_temporary_file(name)
        self.ftp.retrbinary("RETR " + name, temp_file.write)
        temp_file.close()
        metadata["name"] = name
        return LocalDocument(name=name, path=temp_file.name, metadata=metadata)

    def close(self):
        self.ftp.quit()

    def sync_enabled(self):
        return True
