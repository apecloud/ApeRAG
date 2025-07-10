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
from typing import IO, AsyncIterator, Tuple

from aperag.config import settings


class ObjectStore(ABC):
    """Abstract base class for synchronous object storage operations."""

    @abstractmethod
    def put(self, path: str, data: bytes | IO[bytes]):
        """
        Uploads an object to the specified path.

        Args:
            path: The destination path for the object.
            data: The object data, can be bytes or a file-like object.
        """
        ...

    @abstractmethod
    def get(self, path: str) -> IO[bytes] | None:
        """
        Retrieves an object from the specified path.

        Args:
            path: The path of the object to retrieve.

        Returns:
            A file-like object (stream) of the object's content, or None if not found.
        """
        ...

    @abstractmethod
    def get_obj_size(self, path: str) -> int | None:
        """
        Gets the size of an object in bytes.

        Args:
            path: The path of the object.

        Returns:
            The size of the object in bytes, or None if the object does not exist.
        """
        ...

    @abstractmethod
    def stream_range(self, path: str, start: int, end: int | None = None) -> Tuple[IO[bytes], int] | None:
        """
        Streams a specific byte range of an object.

        Args:
            path: The path of the object.
            start: The starting byte position (inclusive).
            end: The ending byte position (inclusive). If None, reads to the end of the file.

        Returns:
            A tuple containing a file-like object for the specified range and the actual
            number of bytes in the stream, or None if the object does not exist.
            The returned stream should be closed by the caller, preferably using a 'with' statement.
        """
        ...

    @abstractmethod
    def obj_exists(self, path: str) -> bool:
        """
        Checks if an object exists at the specified path.

        Args:
            path: The path of the object.

        Returns:
            True if the object exists, False otherwise.
        """
        ...

    @abstractmethod
    def delete(self, path: str):
        """
        Deletes an object at the specified path.

        If the object does not exist, this method should not raise an error.

        Args:
            path: The path of the object to delete.
        """
        ...

    @abstractmethod
    def delete_objects_by_prefix(self, path_prefix: str):
        """
        Deletes all objects whose paths start with the given prefix.

        Args:
            path_prefix: The prefix to match for deletion.
        """
        ...


class AsyncObjectStore(ABC):
    """Abstract base class for asynchronous object storage operations."""

    @abstractmethod
    async def put(self, path: str, data: bytes | IO[bytes]):
        """
        Asynchronously uploads an object to the specified path.

        Args:
            path: The destination path for the object.
            data: The object data, can be bytes or a file-like object.
        """
        ...

    @abstractmethod
    async def get(self, path: str) -> Tuple[AsyncIterator[bytes], int] | None:
        """
        Asynchronously retrieves an object from the specified path as a stream of bytes.

        Args:
            path: The path of the object to retrieve.

        Returns:
            A tuple containing an async iterator yielding chunks of the object's content
            and the total object size in bytes, or None if not found.
            The underlying resources are automatically managed.
        """
        ...

    @abstractmethod
    async def get_obj_size(self, path: str) -> int | None:
        """
        Asynchronously gets the size of an object in bytes.

        Args:
            path: The path of the object.

        Returns:
            The size of the object in bytes, or None if the object does not exist.
        """
        ...

    @abstractmethod
    async def stream_range(
        self, path: str, start: int, end: int | None = None
    ) -> Tuple[AsyncIterator[bytes], int] | None:
        """
        Asynchronously streams a specific byte range of an object.

        Args:
            path: The path of the object.
            start: The starting byte position (inclusive).
            end: The ending byte position (inclusive). If None, reads to the end of the file.

        Returns:
            A tuple containing an async iterator yielding chunks of the specified range
            and the total length of the content in the stream, or None if the object
            does not exist. The underlying resources are automatically managed.
        """
        ...

    @abstractmethod
    async def obj_exists(self, path: str) -> bool:
        """
        Asynchronously checks if an object exists at the specified path.

        Args:
            path: The path of the object.

        Returns:
            True if the object exists, False otherwise.
        """
        ...

    @abstractmethod
    async def delete(self, path: str):
        """
        Asynchronously deletes an object at the specified path.

        If the object does not exist, this method should not raise an error.

        Args:
            path: The path of the object to delete.
        """
        ...

    @abstractmethod
    async def delete_objects_by_prefix(self, path_prefix: str):
        """
        Asynchronously deletes all objects whose paths start with the given prefix.

        Args:
            path_prefix: The prefix to match for deletion.
        """
        ...


def get_object_store() -> ObjectStore:
    """
    Factory function to get a synchronous ObjectStore instance based on settings.
    """
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


def get_async_object_store() -> AsyncObjectStore:
    """
    Factory function to get an asynchronous AsyncObjectStore instance based on settings.
    """
    match settings.object_store_type:
        case "local":
            from aperag.objectstore.local import AsyncLocal, LocalConfig

            local_config_dict = (
                settings.object_store_local_config.model_dump() if settings.object_store_local_config else {}
            )
            return AsyncLocal(LocalConfig(**local_config_dict))
        case "s3":
            from aperag.objectstore.s3 import AsyncS3, S3Config

            s3_config_dict = settings.object_store_s3_config.model_dump() if settings.object_store_s3_config else {}
            return AsyncS3(S3Config(**s3_config_dict))
