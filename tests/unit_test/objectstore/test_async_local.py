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
