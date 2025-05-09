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

import logging
import os
import shutil
from pathlib import Path
from typing import IO

from pydantic import BaseModel

from aperag.objectstore.base import ObjectStore

logger = logging.getLogger(__name__)


class LocalConfig(BaseModel):
    root_dir: str


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

    def obj_exists(self, path: str) -> bool:
        try:
            full_path = self._resolve_object_path(path)
            return full_path.is_file()
        except ValueError:  # From _resolve_object_path
            return False  # Invalid path means object doesn't exist there
        except OSError:  # Catch potential permission errors etc. as object not accessible/existing
            return False

    def delete(self, path: str):
        try:
            full_path = self._resolve_object_path(path)
            # missing_ok=True requires Python 3.8+
            full_path.unlink(missing_ok=True)
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
        try:
            # The path_prefix is relative to the conceptual root of the object store.
            # We iterate files under _base_storage_path and check their relative path.
            for item_path in self._base_storage_path.rglob("*"):
                if item_path.is_file():
                    try:
                        # Get path relative to the effective object store root (_base_storage_path)
                        relative_to_base_str = str(item_path.relative_to(self._base_storage_path)).replace("\\", "/")
                        if relative_to_base_str.startswith(normalized_prefix):
                            item_path.unlink()
                            files_deleted_count += 1
                    except ValueError:
                        # Should not happen if rglob is from _base_storage_path and item_path is under it.
                        logger.debug(f"Item {item_path} not relative to {self._base_storage_path}, skipping.")
                    except OSError as e:
                        logger.error(f"Failed to delete file {item_path} during prefix deletion: {e}")

            if files_deleted_count > 0:
                logger.info(f"Deleted {files_deleted_count} objects with prefix '{path_prefix}'.")
            else:
                logger.info(f"No objects found with prefix '{path_prefix}' to delete.")

        except Exception as e:
            logger.error(f"Error during deletion of objects with prefix '{path_prefix}': {e}")
            raise IOError(f"Error during deletion of objects with prefix '{path_prefix}'") from e
