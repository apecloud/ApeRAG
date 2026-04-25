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

from aperag.schema.common import CollectionConfig


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
    if collectionConfig.source != "system":
        raise CustomSourceInitializationError(f"unsupported collection source: {collectionConfig.source}")

    from aperag.platform.source.upload import UploadSource

    return UploadSource(collectionConfig)
