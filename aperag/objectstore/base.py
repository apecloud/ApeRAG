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
from typing import IO, AsyncIterator, Optional, Tuple

from aperag.config import settings

# Process-level singletons. Created on first call to
# get_object_store() / get_async_object_store() and reused for the
# life of the process. This avoids creating a new boto3 client (or
# local path resolver) on every call — document_service alone calls
# the factory 6 times per request.
_SYNC_STORE: Optional["ObjectStore"] = None
_ASYNC_STORE: Optional["AsyncObjectStore"] = None


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

    @abstractmethod
    def list_objects_by_prefix(self, path_prefix: str) -> list[str]:
        """
        Lists all object paths that start with the given prefix.

        Args:
            path_prefix: The prefix to match.

        Returns:
            A list of object paths (relative to the store root) that match the prefix.
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

    @abstractmethod
    async def list_objects_by_prefix(self, path_prefix: str) -> list[str]:
        """
        Asynchronously lists all object paths that start with the given prefix.

        Args:
            path_prefix: The prefix to match.

        Returns:
            A list of object paths (relative to the store root) that match the prefix.
        """
        ...


def get_object_store() -> ObjectStore:
    """Return a process-level singleton synchronous ObjectStore.

    The instance is created on first call and reused for the life of
    the process. This avoids the overhead of creating a new boto3
    client / local resolver on every call.
    """
    global _SYNC_STORE
    if _SYNC_STORE is not None:
        return _SYNC_STORE

    match settings.object_store_type:
        case "local":
            from aperag.objectstore.local import Local, LocalConfig

            local_config_dict = (
                settings.object_store_local_config.model_dump() if settings.object_store_local_config else {}
            )
            _SYNC_STORE = Local(LocalConfig(**local_config_dict))
        case "s3":
            from aperag.objectstore.s3 import S3

            if settings.object_store_s3_config is None:
                raise RuntimeError("S3 object store configured but OBJECT_STORE_S3_* env vars are missing")
            _SYNC_STORE = S3(settings.object_store_s3_config)
        case _:
            raise ValueError(f"Unsupported object_store_type: {settings.object_store_type!r}")

    return _SYNC_STORE


def get_async_object_store() -> AsyncObjectStore:
    """Return a process-level singleton asynchronous AsyncObjectStore."""
    global _ASYNC_STORE
    if _ASYNC_STORE is not None:
        return _ASYNC_STORE

    match settings.object_store_type:
        case "local":
            from aperag.objectstore.local import AsyncLocal, LocalConfig

            local_config_dict = (
                settings.object_store_local_config.model_dump() if settings.object_store_local_config else {}
            )
            _ASYNC_STORE = AsyncLocal(LocalConfig(**local_config_dict))
        case "s3":
            from aperag.objectstore.s3 import AsyncS3

            if settings.object_store_s3_config is None:
                raise RuntimeError("S3 object store configured but OBJECT_STORE_S3_* env vars are missing")
            _ASYNC_STORE = AsyncS3(settings.object_store_s3_config)
        case _:
            raise ValueError(f"Unsupported object_store_type: {settings.object_store_type!r}")

    return _ASYNC_STORE
