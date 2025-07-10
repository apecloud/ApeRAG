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
import shutil
import tempfile
import uuid

import pytest

from aperag.objectstore.local import AsyncLocal, LocalConfig


@pytest.fixture
def local_config():
    """Provides a LocalConfig with a temporary root directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield LocalConfig(root_dir=tmpdir)


@pytest.fixture
def async_local_service(local_config: LocalConfig) -> AsyncLocal:
    """Provides an AsyncLocal service instance."""
    return AsyncLocal(cfg=local_config)


@pytest.mark.asyncio
async def test_put_and_get_bytes(async_local_service: AsyncLocal):
    file_path = "test_async_bytes.txt"
    file_content = b"Hello, Async Local FS!"

    await async_local_service.put(file_path, file_content)

    get_info = await async_local_service.get(file_path)
    assert get_info is not None
    iterator, size = get_info
    assert size == len(file_content)
    retrieved_content = b"".join([chunk async for chunk in iterator])
    assert retrieved_content == file_content


@pytest.mark.asyncio
async def test_put_and_get_io(async_local_service: AsyncLocal):
    file_path = "test_async_io.txt"
    file_content_bytes = b"Hello from async IO!"
    file_io = io.BytesIO(file_content_bytes)

    await async_local_service.put(file_path, file_io)
    file_io.close()

    get_info = await async_local_service.get(file_path)
    assert get_info is not None
    iterator, size = get_info
    assert size == len(file_content_bytes)
    retrieved_content = b"".join([chunk async for chunk in iterator])
    assert retrieved_content == file_content_bytes


@pytest.mark.asyncio
async def test_put_creates_subdirectories(async_local_service: AsyncLocal):
    file_path = "sub/dir/test_async_subdir.txt"
    file_content = b"Content in async subdirectory"

    await async_local_service.put(file_path, file_content)

    assert await async_local_service.obj_exists(file_path)
    get_info = await async_local_service.get(file_path)
    assert get_info is not None
    iterator, size = get_info
    assert size == len(file_content)
    retrieved_content = b"".join([chunk async for chunk in iterator])
    assert retrieved_content == file_content


@pytest.mark.asyncio
async def test_obj_exists(async_local_service: AsyncLocal):
    existing_file = "i_exist_async.txt"
    non_existing_file = "i_dont_exist_async.txt"
    await async_local_service.put(existing_file, b"data")

    assert await async_local_service.obj_exists(existing_file)
    assert not await async_local_service.obj_exists(non_existing_file)


@pytest.mark.asyncio
async def test_delete(async_local_service: AsyncLocal):
    file_path = "to_delete_async.txt"
    await async_local_service.put(file_path, b"delete me")
    assert await async_local_service.obj_exists(file_path)

    await async_local_service.delete(file_path)
    assert not await async_local_service.obj_exists(file_path)


@pytest.mark.asyncio
async def test_delete_non_existent_object(async_local_service: AsyncLocal):
    non_existent_file = f"async_non_existent_{uuid.uuid4().hex}.txt"
    try:
        await async_local_service.delete(non_existent_file)
    except Exception as e:
        pytest.fail(f"delete() raised an exception for a non-existent object: {e}")


@pytest.mark.asyncio
async def test_get_non_existent_object(async_local_service: AsyncLocal):
    assert await async_local_service.get("non_existent_async.txt") is None


@pytest.mark.asyncio
async def test_get_obj_size(async_local_service: AsyncLocal):
    file_path = "test_async_size.txt"
    file_content = b"size is 12"
    await async_local_service.put(file_path, file_content)

    assert await async_local_service.get_obj_size(file_path) == len(file_content)
    assert await async_local_service.get_obj_size("non_existent.txt") is None


@pytest.mark.asyncio
async def test_stream_range_full_file(async_local_service: AsyncLocal):
    file_path = "test_async_stream_full.txt"
    file_content = b"This is the full content."
    await async_local_service.put(file_path, file_content)

    range_info = await async_local_service.stream_range(file_path, 0)
    assert range_info is not None
    iterator, length = range_info
    assert length == len(file_content)
    content = b"".join([chunk async for chunk in iterator])
    assert content == file_content


@pytest.mark.asyncio
async def test_stream_range_partial_middle(async_local_service: AsyncLocal):
    file_path = "test_async_stream_partial_middle.txt"
    file_content = b"0123456789abcdef"
    await async_local_service.put(file_path, file_content)

    range_info = await async_local_service.stream_range(file_path, 5, 10)
    assert range_info is not None
    iterator, length = range_info
    assert length == 6
    content = b"".join([chunk async for chunk in iterator])
    assert content == b"56789a"


@pytest.mark.asyncio
async def test_stream_range_partial_start(async_local_service: AsyncLocal):
    file_path = "test_async_stream_partial_start.txt"
    file_content = b"0123456789"
    await async_local_service.put(file_path, file_content)

    # Stream from byte 4 to the end
    range_info = await async_local_service.stream_range(file_path, 4)
    assert range_info is not None
    iterator, length = range_info
    assert length == 6
    content = b"".join([chunk async for chunk in iterator])
    assert content == b"456789"


@pytest.mark.asyncio
async def test_stream_range_exceeds_bounds(async_local_service: AsyncLocal):
    file_path = "test_async_stream_exceeds.txt"
    file_content = b"short file"
    await async_local_service.put(file_path, file_content)

    range_info = await async_local_service.stream_range(file_path, 2, 1000)
    assert range_info is not None
    iterator, length = range_info
    assert length == len(file_content) - 2
    content = b"".join([chunk async for chunk in iterator])
    assert content == b"ort file"


@pytest.mark.asyncio
async def test_stream_range_invalid_start(async_local_service: AsyncLocal):
    file_path = "test_async_stream_invalid_start.txt"
    file_content = b"content"
    await async_local_service.put(file_path, file_content)

    with pytest.raises(ValueError):
        await async_local_service.stream_range(file_path, 100)


@pytest.mark.asyncio
async def test_stream_range_non_existent_file(async_local_service: AsyncLocal):
    assert await async_local_service.stream_range("non_existent.txt", 0) is None


@pytest.mark.asyncio
async def test_delete_by_prefix(async_local_service: AsyncLocal):
    prefix = "logs_async/"
    files_to_delete = [f"{prefix}log1.txt", f"{prefix}log2.txt"]
    other_file = "other_data_async/data.txt"

    for f in files_to_delete:
        await async_local_service.put(f, b"log data")
    await async_local_service.put(other_file, b"other data")

    await async_local_service.delete_objects_by_prefix(prefix)

    for f in files_to_delete:
        assert not await async_local_service.obj_exists(f)
    assert await async_local_service.obj_exists(other_file)


@pytest.mark.asyncio
async def test_delete_objects_by_prefix_deletes_many_files(async_local_service: AsyncLocal):
    prefix = "many_files_async/"
    num_files = 50

    for i in range(num_files):
        f_path = f"{prefix}file_{i}.txt"
        await async_local_service.put(f_path, f"content_{i}".encode())

    assert await async_local_service.obj_exists(f"{prefix}file_{num_files // 2}.txt")

    await async_local_service.delete_objects_by_prefix(prefix)

    for i in range(num_files):
        assert not await async_local_service.obj_exists(f"{prefix}file_{i}.txt")

    # With the new cleanup logic, the directory should be removed after becoming empty.
    assert not (async_local_service._sync_store._base_storage_path / prefix).exists()


@pytest.mark.asyncio
async def test_delete_objects_by_prefix_no_objects(async_local_service: AsyncLocal):
    prefix = "empty_prefix_async/"
    try:
        await async_local_service.delete_objects_by_prefix(prefix)
    except Exception as e:
        pytest.fail(f"delete_objects_by_prefix raised an error for a non-existent prefix: {e}")


@pytest.mark.asyncio
async def test_get_object_when_root_dir_removed_after_init(local_config: LocalConfig):
    service = AsyncLocal(local_config)
    object_path = "test_get_gone_root_async.txt"
    await service.put(object_path, b"data")
    assert await service.obj_exists(object_path)

    # Simulate external removal of the entire root directory
    shutil.rmtree(service._sync_store._base_storage_path)
    assert not service._sync_store._base_storage_path.exists()

    result = await service.get(object_path)
    assert result is None, "get() should return None when the root_dir does not exist."


@pytest.mark.asyncio
async def test_delete_object_when_root_dir_removed_after_init(local_config: LocalConfig):
    service = AsyncLocal(local_config)
    object_path = "some_object_in_soon_to_be_gone_bucket_async.txt"

    await service.put(object_path, b"test data")
    assert await service.obj_exists(object_path)

    # Simulate external removal of the entire root directory
    shutil.rmtree(service._sync_store._base_storage_path)
    assert not service._sync_store._base_storage_path.exists()

    try:
        await service.delete(object_path)
    except Exception as e:
        pytest.fail(
            f"AsyncLocal.delete() raised an unexpected exception when root_dir was removed: {type(e).__name__} - {e}"
        )

    assert not await service.obj_exists(object_path)


@pytest.mark.asyncio
async def test_delete_removes_empty_parent_directories(async_local_service: AsyncLocal):
    """Tests that deleting a file also removes all its parent directories if they become empty."""
    file_path = "a/b/c/d_async.txt"
    await async_local_service.put(file_path, b"some data")

    # Verify that the nested directories were created
    base_path = async_local_service._sync_store._base_storage_path
    dir_a = base_path / "a"
    dir_b = dir_a / "b"
    dir_c = dir_b / "c"
    assert dir_a.is_dir()
    assert dir_b.is_dir()
    assert dir_c.is_dir()

    # Delete the only file in the nested structure
    await async_local_service.delete(file_path)

    # Assert that the file is gone
    assert not await async_local_service.obj_exists(file_path)
    # Assert that all parent directories have been cleaned up because they became empty
    assert not dir_c.exists()
    assert not dir_b.exists()
    assert not dir_a.exists()


@pytest.mark.asyncio
async def test_delete_does_not_remove_non_empty_parent_directory(async_local_service: AsyncLocal):
    """Tests that deleting a file does not remove its parent if it contains other files."""
    file_path1 = "a/b/1_async.txt"
    file_path2 = "a/b/2_async.txt"
    await async_local_service.put(file_path1, b"data 1")
    await async_local_service.put(file_path2, b"data 2")

    base_path = async_local_service._sync_store._base_storage_path
    dir_path = base_path / "a" / "b"
    assert dir_path.is_dir()

    # Delete one of the files
    await async_local_service.delete(file_path1)

    # Assert that the first file is gone, but the second remains
    assert not await async_local_service.obj_exists(file_path1)
    assert await async_local_service.obj_exists(file_path2)
    # Assert that the parent directory still exists because it's not empty
    assert dir_path.is_dir()


@pytest.mark.asyncio
async def test_delete_by_prefix_removes_empty_directories(async_local_service: AsyncLocal):
    """Tests that deleting by prefix also cleans up directories that become empty."""
    prefix_to_delete = "level1_async/level2/"
    files_to_delete = [
        f"{prefix_to_delete}file1.txt",
        f"{prefix_to_delete}sub/file2.txt",
    ]
    other_file_in_parent = "level1_async/other.txt"

    for f_path in files_to_delete:
        await async_local_service.put(f_path, b"data")
    await async_local_service.put(other_file_in_parent, b"other data")

    base_path = async_local_service._sync_store._base_storage_path
    level1_dir = base_path / "level1_async"
    level2_dir = level1_dir / "level2"
    sub_dir = level2_dir / "sub"

    assert level1_dir.is_dir()
    assert level2_dir.is_dir()
    assert sub_dir.is_dir()

    # Perform the prefix deletion
    await async_local_service.delete_objects_by_prefix(prefix_to_delete)

    # Assert that all files under the prefix are gone
    for f_path in files_to_delete:
        assert not await async_local_service.obj_exists(f_path)

    # Assert that the directories that became empty are also gone
    assert not sub_dir.exists()
    assert not level2_dir.exists()
    # Assert that the parent directory `level1_async` still exists because it contains `other.txt`
    assert level1_dir.is_dir()
    assert await async_local_service.obj_exists(other_file_in_parent)
