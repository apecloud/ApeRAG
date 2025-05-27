# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

from pydantic import BaseModel

from aperag.schema.view_models import CollectionConfig


class RemoteDocument(BaseModel):
    """
    RemoteDocument is a document residing in a remote location.

    name: str - name of the document, maybe a s3 object key, a ftp file path, a local file path, etc.
    size: int - size of the document in bytes
    metadata: Dict[str, Any] - metadata of the document
    """

    name: str
    size: Optional[int] = None
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
    size: Optional[int] = None
    metadata: Dict[str, Any] = {}


class CustomSourceInitializationError(Exception):
    pass


class Source(ABC):
    def __init__(self, ctx: CollectionConfig):
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


def get_source(collectionConfig: CollectionConfig) -> Source:
    source = None
    match collectionConfig.source:
        case "system":
            from aperag.source.upload import UploadSource

            source = UploadSource(collectionConfig)
        case "local":
            from aperag.source.local import LocalSource

            source = LocalSource(collectionConfig)
        case "s3":
            from aperag.source.s3 import S3Source

            source = S3Source(collectionConfig)
        case "oss":
            from aperag.source.oss import OSSSource

            source = OSSSource(collectionConfig)
        case "feishu":
            from aperag.source.feishu.feishu import FeishuSource

            source = FeishuSource(collectionConfig)
        case "ftp":
            from aperag.source.ftp import FTPSource

            source = FTPSource(collectionConfig)
        case "email":
            from aperag.source.Email import EmailSource

            source = EmailSource(collectionConfig)
        case "url":
            from aperag.source.url import URLSource

            source = URLSource(collectionConfig)
        case "tencent":
            from aperag.source.tencent.tencent import TencentSource

            source = TencentSource(collectionConfig)
        case "git":
            from aperag.source.github import GitHubSource

            source = GitHubSource(collectionConfig)
    return source
