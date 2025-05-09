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
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from aperag.objectstore.local import Local, LocalConfig


@pytest.fixture
def local_config():
    """Provides a LocalConfig with a temporary root directory."""
    # tempfile.TemporaryDirectory() creates a unique directory that will be cleaned up.
    with tempfile.TemporaryDirectory() as tmpdir:
        yield LocalConfig(root_dir=tmpdir)
    # tmpdir is automatically removed here


@pytest.fixture
def local_service(local_config: LocalConfig) -> Local:
    """Provides a Local service instance."""
    # The Local class __init__ will create the root_dir if it doesn't exist.
    # In this case, local_config.root_dir already exists due to TemporaryDirectory.
    return Local(cfg=local_config)


def test_local_init(local_config: LocalConfig):
    service = Local(local_config)
    assert service.cfg == local_config
    assert service._base_storage_path == Path(local_config.root_dir).resolve()
    assert service._base_storage_path.exists()
    assert service._base_storage_path.is_dir()


def test_put_and_get_bytes(local_service: Local):
    file_path = "test_file_bytes.txt"
    file_content = b"Hello, Local FS from bytes!"

    local_service.put(file_path, file_content)

    retrieved_content_stream = local_service.get(file_path)
    assert retrieved_content_stream is not None
    retrieved_content = retrieved_content_stream.read()
    retrieved_content_stream.close()

    assert retrieved_content == file_content

    # Verify actual file on disk
    expected_disk_path = local_service._base_storage_path / file_path
    assert expected_disk_path.exists()
    assert expected_disk_path.read_bytes() == file_content


def test_put_and_get_io(local_service: Local):
    file_path = "test_file_io.txt"
    file_content_str = "Hello, Local FS from IO!"
    file_content_bytes = file_content_str.encode("utf-8")
    file_io = io.BytesIO(file_content_bytes)

    local_service.put(file_path, file_io)
    file_io.close()  # Good practice

    retrieved_content_stream = local_service.get(file_path)
    assert retrieved_content_stream is not None
    retrieved_content = retrieved_content_stream.read()
    retrieved_content_stream.close()

    assert retrieved_content == file_content_bytes

    expected_disk_path = local_service._base_storage_path / file_path
    assert expected_disk_path.exists()
    assert expected_disk_path.read_bytes() == file_content_bytes


def test_put_creates_subdirectories(local_service: Local):
    file_path = "sub/dir/test_file_subdir.txt"
    file_content = b"Content in subdirectory"

    local_service.put(file_path, file_content)

    expected_disk_path = local_service._base_storage_path / file_path
    assert expected_disk_path.exists()
    assert expected_disk_path.read_bytes() == file_content
    assert (local_service._base_storage_path / "sub").is_dir()
    assert (local_service._base_storage_path / "sub" / "dir").is_dir()

    retrieved_content_stream = local_service.get(file_path)
    assert retrieved_content_stream is not None
    retrieved_content = retrieved_content_stream.read()
    retrieved_content_stream.close()
    assert retrieved_content == file_content


def test_obj_exists(local_service: Local):
    existing_file = "i_exist_local.txt"
    non_existing_file = "i_dont_exist_local.txt"
    content = b"some data"

    local_service.put(existing_file, content)

    assert local_service.obj_exists(existing_file)
    assert not local_service.obj_exists(non_existing_file)
    assert not local_service.obj_exists("sub/dir/" + non_existing_file)


def test_delete_object(local_service: Local):
    file_to_delete = "to_be_deleted_local.txt"
    content = b"delete me"

    local_service.put(file_to_delete, content)
    assert local_service.obj_exists(file_to_delete)
    expected_disk_path = local_service._base_storage_path / file_to_delete
    assert expected_disk_path.exists()

    local_service.delete(file_to_delete)
    assert not local_service.obj_exists(file_to_delete)
    assert not expected_disk_path.exists()


def test_delete_non_existent_object(local_service: Local):
    non_existent_file = f"this_file_really_does_not_exist_for_deletion_{uuid.uuid4().hex}.txt"
    assert not local_service.obj_exists(non_existent_file)
    expected_disk_path = local_service._base_storage_path / non_existent_file
    assert not expected_disk_path.exists()

    try:
        local_service.delete(non_existent_file)
    except Exception as e:
        pytest.fail(f"local_service.delete() raised an exception for a non-existent object: {e}")

    assert not local_service.obj_exists(non_existent_file)
    assert not expected_disk_path.exists()


def test_delete_object_when_root_dir_removed_after_init(local_config: LocalConfig):
    # Initialize service (root_dir is created by LocalConfig or Local's __init__)
    service = Local(local_config)
    object_path = "some_object_in_soon_to_be_gone_bucket.txt"

    # Put an object to ensure it would be there
    service.put(object_path, b"test data")
    assert service.obj_exists(object_path)

    # Simulate external removal of the entire root directory
    shutil.rmtree(service._base_storage_path)
    assert not service._base_storage_path.exists()

    try:
        # Attempt to delete the object. Since the base path is gone,
        # _resolve_object_path will still return a Path, but unlink(missing_ok=True)
        # on a non-existent path (because its base is gone) should not error.
        service.delete(object_path)
    except Exception as e:
        pytest.fail(
            f"Local.delete() raised an unexpected exception when root_dir was removed: {type(e).__name__} - {e}"
        )

    # The object (and its containing directory) should definitely not exist.
    # obj_exists might also return False gracefully.
    assert not service.obj_exists(object_path)


def test_get_non_existent_object(local_service: Local):
    assert local_service.get("this_file_does_not_exist_local.txt") is None


def test_get_object_when_root_dir_removed_after_init(local_config: LocalConfig):
    service = Local(local_config)
    object_path = "test_get_gone_root.txt"
    service.put(object_path, b"data")
    assert service.obj_exists(object_path)

    shutil.rmtree(service._base_storage_path)
    assert not service._base_storage_path.exists()

    result = service.get(object_path)
    assert result is None, "get() should return None when the root_dir (bucket) does not exist."


def test_delete_objects_by_prefix_simple(local_service: Local):
    prefix = "logs/today/"
    files_to_delete = [
        f"{prefix}log1.txt",
        f"{prefix}log2.txt",
        f"{prefix}sub_folder/log3.txt",
    ]
    other_file = "other_data/data.txt"
    all_files = files_to_delete + [other_file]

    for f_path in all_files:
        local_service.put(f_path, f"content of {f_path}".encode())
        assert local_service.obj_exists(f_path)
        assert (local_service._base_storage_path / f_path).exists()

    local_service.delete_objects_by_prefix(prefix)

    for f_path in files_to_delete:
        assert not local_service.obj_exists(f_path), f"File {f_path} should have been deleted"
        assert not (local_service._base_storage_path / f_path).exists()

    assert local_service.obj_exists(other_file), "Other file should still exist"
    assert (local_service._base_storage_path / other_file).exists()

    # Check that parent directories of deleted files might still exist if they contain other files
    # or if they were part of other_file's path.
    # The current delete_objects_by_prefix does not remove empty directories.
    assert (local_service._base_storage_path / "logs").is_dir()  # Parent of "logs/today/"
    assert (local_service._base_storage_path / "logs" / "today").is_dir()  # Parent of deleted files
    assert (local_service._base_storage_path / "logs" / "today" / "sub_folder").is_dir()


def test_delete_objects_by_prefix_no_objects(local_service: Local):
    prefix = "empty_prefix_local/"
    # Ensure no objects exist with this prefix (and the prefix itself doesn't exist as a dir)

    try:
        local_service.delete_objects_by_prefix(prefix)
    except Exception as e:
        pytest.fail(f"delete_objects_by_prefix raised an error for a non-existent prefix: {e}")

    # Verify no new directories or files were created
    assert not (local_service._base_storage_path / prefix).exists()


def test_delete_objects_by_prefix_deletes_many_files(local_service: Local):
    prefix = "many_files_local/"
    num_files = 50

    for i in range(num_files):
        f_path = f"{prefix}file_{i}.txt"
        local_service.put(f_path, f"content_{i}".encode())

    assert local_service.obj_exists(f"{prefix}file_{num_files // 2}.txt")

    local_service.delete_objects_by_prefix(prefix)

    for i in range(num_files):
        assert not local_service.obj_exists(f"{prefix}file_{i}.txt")

    # The directory "many_files_local" should still exist but be empty of files.
    # Subdirectories are not created in this test case.
    assert (local_service._base_storage_path / prefix).is_dir()
    assert not list((local_service._base_storage_path / prefix).glob("*"))  # Check if it's empty


def test_resolve_object_path_security(local_service: Local):
    # Valid paths
    assert local_service._resolve_object_path("file.txt") == local_service._base_storage_path / "file.txt"
    assert local_service._resolve_object_path("a/b/c.txt") == local_service._base_storage_path / "a" / "b" / "c.txt"
    assert (
        local_service._resolve_object_path("/leading/slash.txt")
        == local_service._base_storage_path / "leading" / "slash.txt"
    )

    # Invalid paths - path traversal
    with pytest.raises(ValueError, match="Invalid path: '..' components are not allowed"):
        local_service._resolve_object_path("../../../etc/passwd")

    with pytest.raises(ValueError, match="Invalid path: '..' components are not allowed"):
        local_service._resolve_object_path("some/../../path.txt")

    # Test empty or root path
    with pytest.raises(ValueError, match="Object path cannot be empty or just root"):
        local_service._resolve_object_path("")
    with pytest.raises(ValueError, match="Object path cannot be empty or just root"):
        local_service._resolve_object_path("/")


def test_init_with_non_creatable_root_dir():
    # Choose a path that likely cannot be created, e.g., inside a non-existent user's home
    # or a protected system directory. This is OS-dependent.
    # A simpler way for testing is to point to a file as root_dir.
    with tempfile.NamedTemporaryFile() as tmp_file:
        non_dir_path = tmp_file.name
        config = LocalConfig(root_dir=non_dir_path)
        with pytest.raises(OSError):  # Expecting "Not a directory" or similar
            Local(cfg=config)

    # Test with a path where parent doesn't exist and we don't have perms (harder to set up robustly x-platform)
    # For now, the file-as-root test is a good proxy for "cannot create directory".
    # If root_dir was "/proc/some_new_dir" on Linux, mkdir would fail.
    if os.name != "nt":  # Avoid windows specific paths like /dev/null
        protected_path = "/dev/null/cannot_create_this"  # /dev/null is a file
        config_protected = LocalConfig(root_dir=protected_path)
        with pytest.raises(OSError):
            Local(cfg=config_protected)
