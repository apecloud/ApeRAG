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

import aioboto3
import pytest
import pytest_asyncio

from aperag.objectstore.s3 import AsyncS3, S3Config

# Note: moto and aioboto3 are not compatible, so the pytest-aioboto3 library needs
# to be installed for this test file to run correctly.

TEST_BUCKET_NAME = "test-async-aperag-bucket"
TEST_REGION = "us-east-1"


@pytest.fixture
def s3_config() -> S3Config:
    """Provides a standard S3 configuration for tests."""
    # Set endpoint to empty string so that the moto patch from pytest-aioboto3
    # can inject the correct mock endpoint URL.
    return S3Config(
        endpoint="",
        access_key="testing",
        secret_key="testing",
        bucket=TEST_BUCKET_NAME,
        region=TEST_REGION,
        use_path_style=True,
    )


@pytest_asyncio.fixture
async def async_s3_service(moto_patch_session, s3_config: S3Config):
    """
    Provides a mocked AsyncS3 service instance using pytest-aioboto3.
    This fixture sets up a mocked session, creates a bucket, and injects
    the session into the service instance to ensure mocks are applied correctly.
    """
    # moto_patch_session activates the mock. Now we create a session
    # that will be patched to go to the mock server.
    session = aioboto3.Session()
    async with session.client("s3", region_name=s3_config.region) as s3_client:
        await s3_client.create_bucket(Bucket=s3_config.bucket)

    # Inject the patched session into our service class
    service = AsyncS3(cfg=s3_config, session=session)
    yield service


@pytest.mark.asyncio
async def test_put_and_get_bytes(async_s3_service: AsyncS3):
    file_path = "async_test.txt"
    content = b"Hello from async S3!"
    await async_s3_service.put(file_path, content)

    get_info = await async_s3_service.get(file_path)
    assert get_info is not None
    iterator, size = get_info
    assert size == len(content)
    read_content = b"".join([chunk async for chunk in iterator])
    assert read_content == content


@pytest.mark.asyncio
async def test_put_and_get_io(async_s3_service: AsyncS3):
    file_path = "async_test_io.txt"
    content = b"Hello from async S3 IO!"
    await async_s3_service.put(file_path, io.BytesIO(content))

    get_info = await async_s3_service.get(file_path)
    assert get_info is not None
    iterator, size = get_info
    assert size == len(content)
    read_content = b"".join([chunk async for chunk in iterator])
    assert read_content == content


@pytest.mark.asyncio
async def test_obj_exists(async_s3_service: AsyncS3):
    await async_s3_service.put("exists.txt", b"data")
    assert await async_s3_service.obj_exists("exists.txt")
    assert not await async_s3_service.obj_exists("not-exists.txt")


@pytest.mark.asyncio
async def test_delete(async_s3_service: AsyncS3):
    file_path = "to_delete_async.txt"
    await async_s3_service.put(file_path, b"delete me")
    assert await async_s3_service.obj_exists(file_path)

    await async_s3_service.delete(file_path)
    assert not await async_s3_service.obj_exists(file_path)


@pytest.mark.asyncio
async def test_delete_non_existent(async_s3_service: AsyncS3):
    try:
        await async_s3_service.delete("non-existent-for-delete.txt")
    except Exception as e:
        pytest.fail(f"Deleting non-existent object raised an error: {e}")


@pytest.mark.asyncio
async def test_get_non_existent(async_s3_service: AsyncS3):
    assert await async_s3_service.get("non-existent-for-get.txt") is None


@pytest.mark.asyncio
async def test_get_obj_size(async_s3_service: AsyncS3):
    file_path = "size_test_async.txt"
    content = b"12345"
    await async_s3_service.put(file_path, content)

    assert await async_s3_service.get_obj_size(file_path) == 5
    assert await async_s3_service.get_obj_size("non-existent.txt") is None


@pytest.mark.asyncio
async def test_stream_range_full(async_s3_service: AsyncS3):
    file_path = "range_test_full_async.txt"
    content = b"0123456789"
    await async_s3_service.put(file_path, content)

    range_info = await async_s3_service.stream_range(file_path, 0)
    assert range_info is not None
    iterator, length = range_info
    assert length == len(content)
    read_content = b"".join([chunk async for chunk in iterator])
    assert read_content == content


@pytest.mark.asyncio
async def test_stream_range_partial(async_s3_service: AsyncS3):
    file_path = "range_test_partial_async.txt"
    content = b"0123456789"
    await async_s3_service.put(file_path, content)

    range_info = await async_s3_service.stream_range(file_path, 3, 8)
    assert range_info is not None
    iterator, length = range_info
    assert length == 6
    read_content = b"".join([chunk async for chunk in iterator])
    assert read_content == b"345678"


@pytest.mark.asyncio
async def test_stream_range_invalid(async_s3_service: AsyncS3):
    file_path = "range_test_invalid_async.txt"
    content = b"0123456789"
    await async_s3_service.put(file_path, content)

    assert await async_s3_service.stream_range(file_path, 20) is None


@pytest.mark.asyncio
async def test_delete_by_prefix(async_s3_service: AsyncS3):
    prefix = "async_logs/"
    files_to_delete = [f"{prefix}log1.txt", f"{prefix}log2.txt"]
    other_file = "other_async_data/data.txt"

    for f in files_to_delete:
        await async_s3_service.put(f, b"log data")
    await async_s3_service.put(other_file, b"other data")

    await async_s3_service.delete_objects_by_prefix(prefix)

    for f in files_to_delete:
        assert not await async_s3_service.obj_exists(f)
    assert await async_s3_service.obj_exists(other_file)


@pytest.mark.asyncio
async def test_delete_by_prefix_pagination(async_s3_service: AsyncS3):
    prefix = "many_files_async/"
    num_files = 15
    for i in range(num_files):
        await async_s3_service.put(f"{prefix}file_{i}.txt", f"content_{i}".encode())

    await async_s3_service.delete_objects_by_prefix(prefix)

    assert not await async_s3_service.obj_exists(f"{prefix}file_0.txt")
    assert not await async_s3_service.obj_exists(f"{prefix}file_{num_files - 1}.txt")
