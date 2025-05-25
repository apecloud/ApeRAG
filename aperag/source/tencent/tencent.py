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

from typing import Any, Dict, Iterator

from aperag.source.base import LocalDocument, RemoteDocument, Source
from aperag.source.tencent.client import TencentClient
from aperag.schema.view_models import CollectionConfig


class TencentSource(Source):
    def __init__(self, ctx: CollectionConfig):
        super().__init__(ctx)
        self.client = TencentClient(ctx)

    def scan_documents(self) -> Iterator[RemoteDocument]:
        return self.client.scan_documents()

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        return self.client.prepare_document(name, metadata, source="download")

    def sync_enabled(self):
        return True
