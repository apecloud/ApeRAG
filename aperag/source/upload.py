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

import shutil
from typing import Any, Dict, Iterator

from aperag.objectstore.base import get_object_store
from aperag.source.base import LocalDocument, RemoteDocument, Source
from aperag.source.utils import gen_temporary_file
from aperag.schema.view_models import CollectionConfig


class UploadSource(Source):

    def __init__(self, ctx: CollectionConfig):
        super().__init__(ctx)

    def scan_documents(self) -> Iterator[RemoteDocument]:
        return iter([])

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        obj_path = metadata.get("object_path", "")
        if not obj_path:
            raise Exception("empty object path")
        obj_store = get_object_store()
        obj = obj_store.get(obj_path)
        if obj is None:
            raise Exception(f"object '{obj_path}' is not found")
        with gen_temporary_file(name) as temp_file, obj:
            shutil.copyfileobj(obj, temp_file)
            filepath = temp_file.name
        metadata["name"] = name
        return LocalDocument(name=name, path=filepath, metadata=metadata)

    def sync_enabled(self):
        return False
