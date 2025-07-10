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

import pytest
from moto import mock_aws

from aperag.objectstore.s3 import AsyncS3, S3Config

TEST_BUCKET_NAME = "test-async-aperag-bucket"
TEST_REGION = "us-east-1"


@pytest.fixture
def s3_config() -> S3Config:
    return S3Config(
        endpoint="http://localhost:9000",
        access_key="testing",
        secret_key="testing",
        bucket=TEST_BUCKET_NAME,
        region=TEST_REGION,
        use_path_style=True,
    )


@pytest.fixture
async def async_s3_service(s3_config: S3Config) -> AsyncS3:
    """Provides an AsyncS3 service instance mocked with moto."""
    with mock_aws():
        service = AsyncS3(s3_config)
        # Manually create the bucket for the test
        await service._ensure_conn()
        async with service.session.client("s3", region_name=s3_config.region) as client:
            await client.create_bucket(Bucket=s3_config.bucket)
        yield service


@pytest.mark.asyncio
async def test_put_and_get(async_s3_service: AsyncS3):
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
async def test_get_obj_size(async_s3_service: AsyncS3):
    file_path = "size_test_async.txt"
    content = b"12345"
    await async_s3_service.put(file_path, content)

    assert await async_s3_service.get_obj_size(file_path) == 5
    assert await async_s3_service.get_obj_size("non-existent.txt") is None


@pytest.mark.asyncio
async def test_stream_range(async_s3_service: AsyncS3):
    file_path = "range_test_async.txt"
    content = b"0123456789"
    await async_s3_service.put(file_path, content)

    range_info = await async_s3_service.stream_range(file_path, 3, 8)
    assert range_info is not None
    iterator, length = range_info
    assert length == 6
    read_content = b"".join([chunk async for chunk in iterator])
    assert read_content == b"345678"


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
