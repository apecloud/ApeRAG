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
import uuid

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
    """
    Provides S3 configuration, switching between MinIO and moto based on env var.
    """
    if os.environ.get("TEST_TARGET_S3") == "minio":
        # Config for real MinIO integration tests
        return S3Config(
            endpoint=os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
            access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            bucket=os.environ.get("MINIO_TEST_BUCKET", "pytest-async-minio-test-bucket"),
            region=os.environ.get("MINIO_REGION", "us-east-1"),
            use_path_style=True,
        )
    else:
        # Config for moto mock tests
        return S3Config(
            endpoint="",  # Must be empty for pytest-aioboto3 to patch
            access_key="testing",
            secret_key="testing",
            bucket=TEST_BUCKET_NAME,
            region=TEST_REGION,
            use_path_style=True,
        )


@pytest_asyncio.fixture
async def async_s3_service_real(s3_config: S3Config):
    """
    Provides a real AsyncS3 service instance.
    """
    is_minio_target = os.environ.get("TEST_TARGET_S3") == "minio"
    if not is_minio_target:
        yield None
        return

    session = aioboto3.Session()
    service = AsyncS3(cfg=s3_config, session=session)

    # Setup: Ensure bucket exists
    async with service.session.client("s3", **service._get_client_kwargs()) as client:
        try:
            await client.head_bucket(Bucket=s3_config.bucket)
        except client.exceptions.ClientError:
            await client.create_bucket(Bucket=s3_config.bucket)

    yield service

    # Teardown: clean up all objects in the bucket if it was a MinIO test
    try:
        await service.delete_objects_by_prefix("")
    except Exception as e:
        print(f"Warning: Failed to clean up bucket {s3_config.bucket} after test: {e}")


@pytest_asyncio.fixture
async def async_s3_service_moto(s3_config: S3Config, moto_patch_session):
    """
    Provides a mocked AsyncS3 service instance with pytest-aioboto3.
    """
    is_minio_target = os.environ.get("TEST_TARGET_S3") == "minio"
    if is_minio_target:
        yield None
        return

    session = aioboto3.Session()
    service = AsyncS3(cfg=s3_config, session=session)

    # Setup: Ensure bucket exists
    async with service.session.client("s3", **service._get_client_kwargs()) as client:
        try:
            await client.head_bucket(Bucket=s3_config.bucket)
        except client.exceptions.ClientError:
            await client.create_bucket(Bucket=s3_config.bucket)

    yield service


@pytest_asyncio.fixture
async def async_s3_service(async_s3_service_real, async_s3_service_moto):
    """
    Provides a mocked or real AsyncS3 service instance.
    - If TEST_TARGET_S3=minio, it's an integration test against MinIO.
    - Otherwise, it's a mocked test using pytest-aioboto3.
    """
    is_minio_target = os.environ.get("TEST_TARGET_S3") == "minio"
    if is_minio_target:
        yield async_s3_service_real
        return

    yield async_s3_service_moto


@pytest_asyncio.fixture
async def async_s3_service_with_prefix(s3_config: S3Config, async_s3_service: AsyncS3):
    """
    Provides an AsyncS3 service instance with a prefix path.
    This fixture depends on the already-configured async_s3_service to determine
    the session and other parameters, but overrides the prefix.
    """
    # Create a new config instance for the prefix test
    config_with_prefix = s3_config.model_copy(deep=True)
    config_with_prefix.prefix_path = "my/test/prefix/"

    # We can reuse the session from the main service fixture
    service = AsyncS3(cfg=config_with_prefix, session=async_s3_service.session)
    yield service
    # Cleanup is handled by the main async_s3_service fixture


@pytest.mark.asyncio
async def test_put_and_get_bytes(async_s3_service: AsyncS3):
    file_path = f"async_test_bytes_{uuid.uuid4()}.txt"
    content = b"Hello from async S3!"
    await async_s3_service.put(file_path, content)

    get_info = await async_s3_service.get(file_path)
    assert get_info is not None
    iterator, size = get_info
    assert size == len(content)
    read_content = b"".join([chunk async for chunk in iterator])
    assert read_content == content
    await async_s3_service.delete(file_path)


@pytest.mark.asyncio
async def test_put_and_get_io(async_s3_service: AsyncS3):
    file_path = f"async_test_io_{uuid.uuid4()}.txt"
    content = b"Hello from async S3 IO!"
    await async_s3_service.put(file_path, io.BytesIO(content))

    get_info = await async_s3_service.get(file_path)
    assert get_info is not None
    iterator, size = get_info
    assert size == len(content)
    read_content = b"".join([chunk async for chunk in iterator])
    assert read_content == content
    await async_s3_service.delete(file_path)


@pytest.mark.asyncio
async def test_put_and_get_with_prefix_path_config(async_s3_service_with_prefix: AsyncS3):
    service = async_s3_service_with_prefix
    base_path = f"data/file_{uuid.uuid4()}.txt"
    content = b"prefixed content"

    await service.put(base_path, content)

    # Verify that the object is created under the correct S3 key.
    expected_s3_key = service._final_path(base_path)
    # We need a client to check the raw object
    raw_client_kwargs = service._get_client_kwargs()
    async with service.session.client("s3", **raw_client_kwargs) as client:
        response = await client.get_object(Bucket=service.cfg.bucket, Key=expected_s3_key)
        body = response["Body"]
        assert await body.read() == content
        body.close()

    # Retrieve using service.get.
    get_info = await service.get(base_path)
    assert get_info is not None
    iterator, size = get_info
    assert size == len(content)
    retrieved_content = b"".join([chunk async for chunk in iterator])
    assert retrieved_content == content
    await service.delete(base_path)


@pytest.mark.asyncio
async def test_obj_exists(async_s3_service: AsyncS3):
    file_path = f"exists_{uuid.uuid4()}.txt"
    await async_s3_service.put(file_path, b"data")
    assert await async_s3_service.obj_exists(file_path)
    assert not await async_s3_service.obj_exists(f"not-exists-{uuid.uuid4()}.txt")
    await async_s3_service.delete(file_path)


@pytest.mark.asyncio
async def test_obj_exists_with_prefix(async_s3_service_with_prefix: AsyncS3):
    service = async_s3_service_with_prefix
    base_path = f"check_existence_{uuid.uuid4()}.log"
    content = b"log data"

    await service.put(base_path, content)

    assert await service.obj_exists(base_path)
    assert not await service.obj_exists(f"other_dir/non_existent_{uuid.uuid4()}.log")
    await service.delete(base_path)


@pytest.mark.asyncio
async def test_delete(async_s3_service: AsyncS3):
    file_path = f"to_delete_async_{uuid.uuid4()}.txt"
    await async_s3_service.put(file_path, b"delete me")
    assert await async_s3_service.obj_exists(file_path)

    await async_s3_service.delete(file_path)
    assert not await async_s3_service.obj_exists(file_path)


@pytest.mark.asyncio
async def test_delete_object_with_prefix(async_s3_service_with_prefix: AsyncS3):
    service = async_s3_service_with_prefix
    base_path = f"removable_{uuid.uuid4()}.dat"
    content = b"temporary data"

    await service.put(base_path, content)
    assert await service.obj_exists(base_path)

    await service.delete(base_path)
    assert not await service.obj_exists(base_path)


@pytest.mark.asyncio
async def test_delete_non_existent(async_s3_service: AsyncS3):
    try:
        await async_s3_service.delete("non-existent-for-delete.txt")
    except Exception as e:
        pytest.fail(f"Deleting non-existent object raised an error: {e}")


@pytest.mark.asyncio
async def test_get_non_existent(async_s3_service: AsyncS3):
    assert await async_s3_service.get(f"non-existent-for-get-{uuid.uuid4()}.txt") is None


@pytest.mark.asyncio
async def test_get_obj_size(async_s3_service: AsyncS3):
    file_path = f"size_test_async_{uuid.uuid4()}.txt"
    content = b"12345"
    await async_s3_service.put(file_path, content)

    assert await async_s3_service.get_obj_size(file_path) == 5
    assert await async_s3_service.get_obj_size(f"non-existent-{uuid.uuid4()}.txt") is None
    await async_s3_service.delete(file_path)


@pytest.mark.asyncio
async def test_stream_range_full(async_s3_service: AsyncS3):
    file_path = f"range_test_full_async_{uuid.uuid4()}.txt"
    content = b"0123456789"
    await async_s3_service.put(file_path, content)

    range_info = await async_s3_service.stream_range(file_path, 0)
    assert range_info is not None
    iterator, length = range_info
    assert length == len(content)
    read_content = b"".join([chunk async for chunk in iterator])
    assert read_content == content
    await async_s3_service.delete(file_path)


@pytest.mark.asyncio
async def test_stream_range_partial(async_s3_service: AsyncS3):
    file_path = f"range_test_partial_async_{uuid.uuid4()}.txt"
    content = b"0123456789"
    await async_s3_service.put(file_path, content)

    range_info = await async_s3_service.stream_range(file_path, 3, 8)
    assert range_info is not None
    iterator, length = range_info
    assert length == 6
    read_content = b"".join([chunk async for chunk in iterator])
    assert read_content == b"345678"
    await async_s3_service.delete(file_path)


@pytest.mark.asyncio
async def test_stream_range_exceeds_bounds(async_s3_service: AsyncS3):
    file_path = f"range_test_exceeds_async_{uuid.uuid4()}.txt"
    content = b"short s3 file"
    await async_s3_service.put(file_path, content)

    # End is beyond the file length, should stream to the actual end
    range_info = await async_s3_service.stream_range(file_path, 2, 1000)
    assert range_info is not None
    iterator, length = range_info
    assert length == len(content) - 2
    read_content = b"".join([chunk async for chunk in iterator])
    assert read_content == b"ort s3 file"
    await async_s3_service.delete(file_path)


@pytest.mark.asyncio
async def test_stream_range_invalid(async_s3_service: AsyncS3):
    file_path = f"range_test_invalid_async_{uuid.uuid4()}.txt"
    content = b"0123456789"
    await async_s3_service.put(file_path, content)

    # An invalid range (starting beyond the end of the file) should return None
    # because the S3 backend will return an InvalidRange error.
    assert await async_s3_service.stream_range(file_path, 20) is None
    await async_s3_service.delete(file_path)


@pytest.mark.asyncio
async def test_stream_range_non_existent_file(async_s3_service: AsyncS3):
    assert await async_s3_service.stream_range(f"non_existent_{uuid.uuid4()}.txt", 0) is None


@pytest.mark.asyncio
async def test_delete_by_prefix(async_s3_service: AsyncS3):
    prefix = f"async_logs_{uuid.uuid4()}/"
    files_to_delete = [f"{prefix}log1.txt", f"{prefix}log2.txt"]
    other_file = f"other_async_data_{uuid.uuid4()}/data.txt"

    for f in files_to_delete:
        await async_s3_service.put(f, b"log data")
    await async_s3_service.put(other_file, b"other data")

    await async_s3_service.delete_objects_by_prefix(prefix)

    for f in files_to_delete:
        assert not await async_s3_service.obj_exists(f)
    assert await async_s3_service.obj_exists(other_file)
    await async_s3_service.delete(other_file)


@pytest.mark.asyncio
async def test_delete_objects_by_prefix_with_config_prefix(async_s3_service_with_prefix: AsyncS3):
    service = async_s3_service_with_prefix
    app_specific_prefix = f"app_logs/specific_run_{uuid.uuid4()}/"

    files_to_delete_relative = [
        f"{app_specific_prefix}run1.log",
        f"{app_specific_prefix}run2.log",
    ]
    other_file_relative = f"app_logs/other_run_{uuid.uuid4()}/run.log"

    for rel_path in files_to_delete_relative:
        await service.put(rel_path, f"content of {rel_path}".encode())
    await service.put(other_file_relative, b"other app log content")

    for rel_path in files_to_delete_relative:
        assert await service.obj_exists(rel_path)
    assert await service.obj_exists(other_file_relative)

    await service.delete_objects_by_prefix(app_specific_prefix)

    for rel_path in files_to_delete_relative:
        assert not await service.obj_exists(rel_path)
    assert await service.obj_exists(other_file_relative)
    await service.delete(other_file_relative)


@pytest.mark.asyncio
async def test_delete_objects_by_prefix_no_objects(async_s3_service: AsyncS3):
    prefix = f"empty_prefix_async_{uuid.uuid4()}/"
    try:
        await async_s3_service.delete_objects_by_prefix(prefix)
    except Exception as e:
        pytest.fail(f"delete_objects_by_prefix raised an error for a non-existent prefix: {e}")


@pytest.mark.asyncio
async def test_delete_by_prefix_pagination(async_s3_service: AsyncS3):
    prefix = f"many_files_async_{uuid.uuid4()}/"
    # We test with a number that would require at least one paginated call in a real scenario
    # but keep it reasonable for tests.
    num_files = 15
    for i in range(num_files):
        await async_s3_service.put(f"{prefix}file_{i}.txt", f"content_{i}".encode())

    await async_s3_service.delete_objects_by_prefix(prefix)

    assert not await async_s3_service.obj_exists(f"{prefix}file_0.txt")
    assert not await async_s3_service.obj_exists(f"{prefix}file_{num_files - 1}.txt")


@pytest.mark.asyncio
async def test_get_object_from_non_existent_bucket(async_s3_service: AsyncS3):
    non_existent_bucket_name = f"non-existent-bucket-{uuid.uuid4().hex}"

    # Create a new config and service for this test
    temp_config = async_s3_service.cfg.model_copy(deep=True)
    temp_config.bucket = non_existent_bucket_name
    service_for_non_existent_bucket = AsyncS3(cfg=temp_config, session=async_s3_service.session)

    result = await service_for_non_existent_bucket.get("some_random_object.txt")
    assert result is None, "get() should return None when the bucket does not exist."


@pytest.mark.asyncio
async def test_delete_object_from_non_existent_bucket(async_s3_service: AsyncS3):
    non_existent_bucket_name = f"non-existent-bucket-{uuid.uuid4().hex}"

    temp_config = async_s3_service.cfg.model_copy(deep=True)
    temp_config.bucket = non_existent_bucket_name
    service_for_non_existent_bucket = AsyncS3(cfg=temp_config, session=async_s3_service.session)

    try:
        await service_for_non_existent_bucket.delete("some_object.txt")
    except Exception as e:
        pytest.fail(f"AsyncS3.delete() raised an unexpected exception for a non-existent bucket: {e}")
