import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

from pydantic import BaseModel


class RemoteDocument(BaseModel):
    """
    RemoteDocument is a document residing in a remote location.

    name: str - name of the document, maybe a s3 object key, a ftp file path, a local file path, etc.
    size: int - size of the document in bytes
    metadata: Dict[str, Any] - metadata of the document
    """
    name: str
    size: Optional[int]
    metadata: Dict[str, Any] = {}


class LocalDocument(BaseModel):
    """
    LocalDocument is a document that is downloaded from the RemoteDocument.

    name: str - name of the document, maybe a s3 object key, a ftp file path, a local file path, etc.
    path: str - path of the document on the local file system
    size: int - size of the document in bytes
    metadata: Dict[str, Any] - metadata of the document
    """
    name: str
    path: str
    size: Optional[int]
    metadata: Dict[str, Any] = {}


class CustomSourceInitializationError(Exception):
    pass


class Source(ABC):
    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx

    @abstractmethod
    def scan_documents(self) -> Iterator[RemoteDocument]:
        raise NotImplementedError

    @abstractmethod
    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        raise NotImplementedError

    def cleanup_document(self, filepath: str):
        os.remove(filepath)

    def close(self):
        pass

    @abstractmethod
    def sync_enabled(self):
        raise NotImplementedError


def get_source(ctx: Dict[str, Any]):
    source = None
    match ctx["source"]:
        case "system":
            from deeprag.source.upload import UploadSource
            source = UploadSource(ctx)
        case "local":
            from deeprag.source.local import LocalSource
            source = LocalSource(ctx)
        case "s3":
            from deeprag.source.s3 import S3Source
            source = S3Source(ctx)
        case "oss":
            from deeprag.source.oss import OSSSource
            source = OSSSource(ctx)
        case "feishu":
            from deeprag.source.feishu.feishu import FeishuSource
            source = FeishuSource(ctx)
        case "ftp":
            from deeprag.source.ftp import FTPSource
            source = FTPSource(ctx)
        case "email":
            from deeprag.source.Email import EmailSource
            source = EmailSource(ctx)
        case "url":
            from deeprag.source.url import URLSource
            source = URLSource(ctx)
        case "tencent":
            from deeprag.source.tencent.tencent import TencentSource
            source = TencentSource(ctx)
        case "github":
            from deeprag.source.github import GitHubSource
            source = GitHubSource(ctx)
    return source
