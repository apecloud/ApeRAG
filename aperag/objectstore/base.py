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


from abc import ABC, abstractmethod
from typing import IO

from aperag.config import settings


class ObjectStore(ABC):
    @abstractmethod
    def put(self, path: str, data: bytes | IO[bytes]): ...

    @abstractmethod
    def get(self, path: str) -> IO[bytes] | None: ...

    @abstractmethod
    def obj_exists(self, path: str) -> bool: ...

    @abstractmethod
    def delete(self, path: str): ...

    @abstractmethod
    def delete_objects_by_prefix(self, path_prefix: str): ...


def get_object_store() -> ObjectStore:
    match settings.object_store_type:
        case "local":
            from aperag.objectstore.local import Local, LocalConfig

            # Convert pydantic model to dict for unpacking
            local_config_dict = (
                settings.object_store_local_config.model_dump() if settings.object_store_local_config else {}
            )
            return Local(LocalConfig(**local_config_dict))
        case "s3":
            from aperag.objectstore.s3 import S3, S3Config

            # Convert pydantic model to dict for unpacking
            s3_config_dict = settings.object_store_s3_config.model_dump() if settings.object_store_s3_config else {}
            return S3(S3Config(**s3_config_dict))
