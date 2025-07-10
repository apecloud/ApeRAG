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

import io
import logging
import os
import shutil
from pathlib import Path
from typing import IO, AsyncIterator, Tuple

from asgiref.sync import sync_to_async
from pydantic import BaseModel

from aperag.objectstore.base import AsyncObjectStore, ObjectStore

logger = logging.getLogger(__name__)


class LocalConfig(BaseModel):
    root_dir: str


class RangedFileStream(IO[bytes]):
    """
    A file-like object that reads a specific range of bytes from an underlying file handle.
    This class is a context manager and can be used in a 'with' statement.
    """

    def __init__(self, file_handle: IO[bytes], start: int, end: int):
        self._handle = file_handle
        self._start = start
        self._end = end
        self._pos = start
        self._handle.seek(start)

    def read(self, size: int = -1) -> bytes:
        """
        Reads bytes from the stream, respecting the defined range.
        """
        bytes_left = self._end - self._pos + 1
        if bytes_left <= 0:
            return b""

        read_size = bytes_left if size < 0 else min(size, bytes_left)
        data = self._handle.read(read_size)
        self._pos += len(data)
        return data

    def seek(self, offset: int, whence: int = 0) -> int:
        """
        Seeks to a position within the allowed range.
        """
        if whence == 0:  # Absolute
            self._pos = self._start + offset
        elif whence == 1:  # Relative to current
            self._pos += offset
        elif whence == 2:  # Relative to end
            self._pos = self._end + 1 + offset
        else:
            raise ValueError(f"Invalid whence value: {whence}")

        self._pos = max(self._start, min(self._pos, self._end + 1))
        self._handle.seek(self._pos)
        return self._pos - self._start

    def tell(self) -> int:
        """
        Returns the current position relative to the start of the range.
        """
        return self._pos - self._start

    def close(self) -> None:
        """
        Closes the underlying file handle.
        """
        self._handle.close()

    def __enter__(self) -> "RangedFileStream":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Add other IO methods if needed, delegating to self._handle
    def isatty(self) -> bool:
        return self._handle.isatty()

    def readable(self) -> bool:
        return self._handle.readable()

    def writable(self) -> bool:
        return False  # This stream is read-only

    def seekable(self) -> bool:
        return self._handle.seekable()


class Local(ObjectStore):
    def __init__(self, cfg: LocalConfig):
        self.cfg = cfg

        # Resolve root_dir to an absolute, canonical path (handles '..', symlinks)
        self._base_storage_path = Path(self.cfg.root_dir).resolve()

        try:
            self._base_storage_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create base storage directory {self._base_storage_path}: {e}")
            raise

    def _resolve_object_path(self, path: str) -> Path:
        """
        Resolves the relative object path to an absolute path on the filesystem,
        ensuring it's within the configured storage directory.
        """
        # Normalize path to remove leading slashes and break into components.
        # This ensures 'path' is treated as relative and cleans it.
        path_components = Path(path.lstrip("/")).parts
        if not path_components:  # Handle empty path string or path being just "/"
            raise ValueError("Object path cannot be empty or just root.")

        # Explicitly disallow '..' components in the input path for security.
        if ".." in path_components:
            raise ValueError("Invalid path: '..' components are not allowed in object paths.")

        # Construct the prospective full path.
        # Since self._base_storage_path is absolute, this will also be absolute.
        prospective_full_path = self._base_storage_path.joinpath(*path_components)

        # Security check: Verify that the constructed path is genuinely within the base storage directory.
        # We normalize prospective_full_path using os.path.abspath for a robust comparison.
        # self._base_storage_path is already absolute and resolved.
        normalized_prospective_path = Path(os.path.abspath(prospective_full_path))

        if (
            self._base_storage_path != normalized_prospective_path
            and self._base_storage_path not in normalized_prospective_path.parents
        ):
            logger.error(
                f"Path traversal attempt or invalid path: input '{path}' resolved to "
                f"'{normalized_prospective_path}', which is not under base_path '{self._base_storage_path}'"
            )
            raise ValueError("Invalid path: access attempt outside designated storage area.")

        return prospective_full_path

    def put(self, path: str, data: bytes | IO[bytes]):
        full_path = self._resolve_object_path(path)
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with full_path.open("wb") as f:
                if isinstance(data, bytes):
                    f.write(data)
                else:  # IO[bytes]
                    shutil.copyfileobj(data, f)
        except OSError as e:
            logger.error(f"Failed to write object to {full_path}: {e}")
            # Re-raise to signal failure, potentially wrapping or adding context
            raise IOError(f"Failed to write object to {full_path}") from e

    def get(self, path: str) -> IO[bytes] | None:
        try:
            full_path = self._resolve_object_path(path)
            if full_path.is_file():
                return full_path.open("rb")
            return None
        except ValueError:  # From _resolve_object_path for invalid paths
            logger.warning(f"Invalid path provided for get: {path}")
            return None
        except OSError as e:
            # Log access errors (e.g., permissions) but treat as "not found" for the caller.
            logger.warning(
                f"Failed to access object at {path} (resolved to {full_path if 'full_path' in locals() else 'unknown'}) for get: {e}"
            )
            return None

    def get_obj_size(self, path: str) -> int | None:
        try:
            full_path = self._resolve_object_path(path)
            if full_path.is_file():
                return full_path.stat().st_size
            return None
        except ValueError:  # From _resolve_object_path for invalid paths
            logger.warning(f"Invalid path provided for get: {path}")
            return None
        except OSError as e:
            # Log access errors (e.g., permissions) but treat as "not found" for the caller.
            logger.warning(
                f"Failed to access object at {path} (resolved to {full_path if 'full_path' in locals() else 'unknown'}) for get: {e}"
            )
            return None

    def stream_range(self, path: str, start: int, end: int | None = None) -> Tuple[IO[bytes], int] | None:
        try:
            full_path = self._resolve_object_path(path)
        except ValueError:  # Path validation error
            return None

        try:
            if not full_path.is_file():
                return None

            file_size = full_path.stat().st_size
        except OSError as e:
            logger.warning(f"Failed to access object at {path} for streaming: {e}")
            return None

        if start < 0 or start >= file_size:
            raise ValueError("Start position is out of file bounds.")

        # If end is None or beyond the file, read to the end.
        actual_end = file_size - 1 if end is None or end >= file_size else end
        content_length = actual_end - start + 1

        if content_length <= 0:
            return io.BytesIO(b""), 0

        try:
            file_handle = full_path.open("rb")
            ranged_stream = RangedFileStream(file_handle, start, actual_end)
            return ranged_stream, content_length
        except OSError as e:
            logger.warning(f"Failed to open object at {path} for streaming: {e}")
            return None

    def obj_exists(self, path: str) -> bool:
        try:
            full_path = self._resolve_object_path(path)
            return full_path.is_file()
        except ValueError:  # From _resolve_object_path
            return False  # Invalid path means object doesn't exist there
        except OSError:  # Catch potential permission errors etc. as object not accessible/existing
            return False

    def _cleanup_empty_dirs(self, dir_path: Path):
        """
        Recursively deletes empty parent directories of a given path until a non-empty
        directory or the base storage path is reached.
        """
        # Loop until we reach a non-empty directory, the base path, or outside of it
        while dir_path.is_dir() and dir_path != self._base_storage_path and self._base_storage_path in dir_path.parents:
            try:
                # Check if directory is empty. An empty directory has no items.
                if not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    logger.debug(f"Removed empty directory: {dir_path}")
                    # Move up to the parent directory for the next iteration
                    dir_path = dir_path.parent
                else:
                    # Directory is not empty, stop cleanup
                    break
            except OSError as e:
                # Log error and stop, as we can't proceed.
                # This could happen due to permissions or if the dir was deleted by another process.
                logger.warning(f"Could not remove directory {dir_path} during cleanup: {e}")
                break

    def delete(self, path: str):
        try:
            full_path = self._resolve_object_path(path)
            if not full_path.is_file():
                # If it's not a file (e.g., doesn't exist or is a dir), do nothing.
                # unlink(missing_ok=True) handles non-existence, but this is an extra guard.
                return

            # missing_ok=True requires Python 3.8+
            full_path.unlink(missing_ok=True)
            # After deleting the file, try to clean up empty parent directories.
            self._cleanup_empty_dirs(full_path.parent)
        except ValueError:  # From _resolve_object_path
            logger.warning(f"Invalid path provided for delete: {path}")
            # Path is invalid, so object effectively doesn't exist at that path to delete. Do nothing.
            return
        except OSError as e:
            # This might happen if it's a directory or due to permissions.
            # If missing_ok=True is used, FileNotFoundError is handled.
            # Other OSErrors (like IsADirectoryError, PermissionError) should be logged if the path still exists.
            if "full_path" in locals() and full_path.exists():  # Check if 'full_path' was defined and still exists
                logger.error(f"Failed to delete object at {path} (resolved to {full_path}): {e}")
                raise IOError(f"Failed to delete object at {path}") from e
            # If it doesn't exist, then missing_ok=True behavior is achieved, or it was an invalid path.

    def delete_objects_by_prefix(self, path_prefix: str):
        # Normalize the prefix to be relative and use forward slashes for consistent matching
        normalized_prefix = path_prefix.lstrip("/").replace("\\", "/")

        # An empty prefix (after normalization) would mean deleting everything under _base_storage_path.
        # This is a destructive operation, so we require a non-empty prefix.
        if not normalized_prefix:
            logger.warning(
                "Attempted to delete objects with an empty or root prefix. "
                "This operation is skipped for safety. Provide a specific prefix."
            )
            return

        files_deleted_count = 0
        # Keep track of parent directories of deleted files to check for emptiness later.
        parent_dirs_to_check = set()
        try:
            # The path_prefix is relative to the conceptual root of the object store.
            # We iterate files under _base_storage_path and check their relative path.
            # Use a list to realize the generator from rglob, avoiding issues with modifying the file system while iterating
            paths_to_check = list(self._base_storage_path.rglob("*"))
            for item_path in paths_to_check:
                if item_path.is_file():
                    try:
                        # Get path relative to the effective object store root (_base_storage_path)
                        relative_to_base_str = str(item_path.relative_to(self._base_storage_path)).replace("\\", "/")
                        if relative_to_base_str.startswith(normalized_prefix):
                            # Add parent directory to the set for later cleanup check.
                            parent_dirs_to_check.add(item_path.parent)
                            item_path.unlink()
                            files_deleted_count += 1
                    except ValueError:
                        # Should not happen if rglob is from _base_storage_path and item_path is under it.
                        logger.debug(f"Item {item_path} not relative to {self._base_storage_path}, skipping.")
                    except OSError as e:
                        logger.error(f"Failed to delete file {item_path} during prefix deletion: {e}")

            # After deleting all matching files, clean up empty directories.
            # We process them in reverse order of path length to handle nested empty dirs correctly.
            sorted_dirs = sorted(list(parent_dirs_to_check), key=lambda p: len(str(p)), reverse=True)
            for dir_path in sorted_dirs:
                self._cleanup_empty_dirs(dir_path)

            if files_deleted_count > 0:
                logger.info(f"Deleted {files_deleted_count} objects with prefix '{path_prefix}'.")
            else:
                logger.info(f"No objects found with prefix '{path_prefix}' to delete.")

        except Exception as e:
            logger.error(f"Error during deletion of objects with prefix '{path_prefix}': {e}")
            raise IOError(f"Error during deletion of objects with prefix '{path_prefix}'") from e


class AsyncLocal(AsyncObjectStore):
    """Asynchronous wrapper for the Local object store."""

    def __init__(self, cfg: LocalConfig):
        self._sync_store = Local(cfg)
        self.chunk_size = 32 * 1024

    async def put(self, path: str, data: bytes | IO[bytes]):
        return await sync_to_async(self._sync_store.put)(path=path, data=data)

    async def get(self, path: str) -> Tuple[AsyncIterator[bytes], int] | None:
        # First, get the size and a sync stream handle without blocking
        size = await sync_to_async(self._sync_store.get_obj_size)(path=path)
        if size is None:
            return None

        stream_handle = await sync_to_async(self._sync_store.get)(path=path)
        if stream_handle is None:
            # This case should be rare if size is not None, but handle it for safety
            return None

        # Now, return an async generator that wraps the sync stream
        async def generator():
            try:
                while True:
                    chunk = await sync_to_async(stream_handle.read)(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
            finally:
                await sync_to_async(stream_handle.close)()

        return generator(), size

    async def get_obj_size(self, path: str) -> int | None:
        return await sync_to_async(self._sync_store.get_obj_size)(path=path)

    async def stream_range(
        self, path: str, start: int, end: int | None = None
    ) -> Tuple[AsyncIterator[bytes], int] | None:
        # Get the ranged stream and content length synchronously first
        range_info = await sync_to_async(self._sync_store.stream_range)(path=path, start=start, end=end)
        if range_info is None:
            return None

        stream_handle, content_length = range_info

        # Define an async generator to wrap the synchronous ranged stream
        async def generator():
            try:
                while True:
                    chunk = await sync_to_async(stream_handle.read)(self.chunk_size)
                    if not chunk:
                        break
                    yield chunk
            finally:
                await sync_to_async(stream_handle.close)()

        return generator(), content_length

    async def obj_exists(self, path: str) -> bool:
        return await sync_to_async(self._sync_store.obj_exists)(path=path)

    async def delete(self, path: str):
        return await sync_to_async(self._sync_store.delete)(path=path)

    async def delete_objects_by_prefix(self, path_prefix: str):
        return await sync_to_async(self._sync_store.delete_objects_by_prefix)(path_prefix=path_prefix)
