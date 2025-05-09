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
from typing import Generator

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from aperag.objectstore.s3 import S3, S3Config

TEST_BUCKET_NAME = "test-aperag-bucket"
TEST_REGION = "us-east-1"


@pytest.fixture
def minio_s3_config() -> S3Config:
    return S3Config(
        endpoint=os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
        access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        bucket=os.environ.get("MINIO_TEST_BUCKET", "pytest-minio-test-bucket"),
        region=os.environ.get("MINIO_REGION", "us-east-1"),
        prefix_path=None,
        use_path_style=True,
    )


@pytest.fixture
def s3_config() -> S3Config:
    return S3Config(
        endpoint="http://localhost:9000",  # Moto intercepts requests to localhost:9000 or s3.amazonaws.com by default.
        access_key="testing",
        secret_key="testing",
        bucket=TEST_BUCKET_NAME,
        region=TEST_REGION,
        prefix_path=None,
        use_path_style=True,  # Moto requires path-style addressing in some cases.
    )


@pytest.fixture
def s3_target_config(minio_s3_config: S3Config, s3_config: S3Config) -> S3Config:
    if os.environ.get("TEST_TARGET_S3") == "minio":
        return minio_s3_config
    return s3_config


@pytest.fixture(scope="function")
def s3_client_real_or_moto(s3_target_config: S3Config) -> Generator[boto3.client, None, None]:
    """
    Provides a boto3 S3 client.
    If TEST_TARGET_S3=minio, connects to MinIO.
    Otherwise, uses moto.
    Handles bucket creation for the test session and basic cleanup for MinIO.
    """
    is_minio_target = os.environ.get("TEST_TARGET_S3") == "minio"

    # boto3 client parameters
    client_params = {
        "endpoint_url": s3_target_config.endpoint,
        "aws_access_key_id": s3_target_config.access_key,
        "aws_secret_access_key": s3_target_config.secret_key,
        "region_name": s3_target_config.region,
    }
    if s3_target_config.use_path_style:
        from botocore.client import Config as BotoConfig

        client_params["config"] = BotoConfig(s3={"addressing_style": "path"})

    if is_minio_target:
        client = boto3.client("s3", **client_params)
        try:
            client.head_bucket(Bucket=s3_target_config.bucket)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404" or "NotFound" in str(e) or "NoSuchBucket" in str(e):
                print(f"Creating bucket {s3_target_config.bucket} on MinIO.")
                client.create_bucket(Bucket=s3_target_config.bucket)
            else:
                raise
        yield client
        # Basic cleanup for MinIO: empty the bucket
        clean_bucket(client, s3_target_config.bucket, delete_bucket=False)
    else:  # Use moto
        with mock_aws():  # moto's mock_aws context
            # For moto, the client_params for endpoint might be ignored by moto if it's localhost:9000
            # but it's good to be consistent. Moto primarily uses region_name.
            moto_client_params = {
                "region_name": s3_target_config.region,
                "aws_access_key_id": s3_target_config.access_key,  # Moto uses these
                "aws_secret_access_key": s3_target_config.secret_key,
            }
            client = boto3.client("s3", **moto_client_params)
            try:  # Ensure bucket exists in moto
                client.head_bucket(Bucket=s3_target_config.bucket)
            except ClientError:
                client.create_bucket(Bucket=s3_target_config.bucket)
            yield client


def clean_bucket(client: boto3.client, bucket_name: str, delete_bucket: bool = False):
    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket_name):
            if "Contents" in page:
                delete_keys = {"Objects": [{"Key": obj["Key"]} for obj in page["Contents"]]}
                if delete_keys["Objects"]:
                    client.delete_objects(Bucket=bucket_name, Delete=delete_keys)
        if delete_bucket:
            client.delete_bucket(Bucket=bucket_name)
    except ClientError as e:
        print(f"Warning: Error during clean_bucket: {e}")
        pass  # Ignore if bucket already gone or other issues during cleanup


@pytest.fixture
def s3_service(s3_target_config: S3Config, s3_client_real_or_moto: boto3.client) -> S3:
    """
    Provides an S3 service instance configured for the target (MinIO or moto).
    The s3_client_real_or_moto fixture ensures the bucket exists.
    """
    service = S3(s3_target_config)
    service.conn = s3_client_real_or_moto
    return service


@pytest.fixture
def s3_service_with_prefix(s3_target_config: S3Config, s3_client_real_or_moto: boto3.client) -> S3:
    # Create a new config instance for the prefix test
    config_with_prefix = S3Config(
        endpoint=s3_target_config.endpoint,
        access_key=s3_target_config.access_key,
        secret_key=s3_target_config.secret_key,
        bucket=s3_target_config.bucket,  # Use the same bucket
        region=s3_target_config.region,
        prefix_path="my/test/prefix/",
        use_path_style=s3_target_config.use_path_style,
    )
    service = S3(config_with_prefix)
    service.conn = s3_client_real_or_moto
    return service


def test_s3_init(s3_target_config: S3Config):
    service = S3(s3_target_config)
    assert service.cfg == s3_target_config
    assert service.conn is None


def test_ensure_conn(s3_service: S3):
    s3_service.conn = None  # Reset
    s3_service._ensure_conn()
    assert s3_service.conn is not None, "Connection should be established."
    # Subsequent calls should return immediately without creating a new connection.
    conn_before = s3_service.conn
    s3_service._ensure_conn()
    assert s3_service.conn == conn_before, "Subsequent calls should return the existing connection."


def test_bucket_exists(s3_service: S3, s3_target_config: S3Config):
    # Test an existing bucket.
    assert s3_service.bucket_exists(s3_target_config.bucket)

    # Test a non-existent bucket.
    assert not s3_service.bucket_exists("non-existent-bucket-" + uuid.uuid4().hex)


def test_put_and_get_bytes(s3_service: S3):
    file_path = "test_file_bytes.txt"
    file_content = b"Hello, S3 from bytes!"

    s3_service.put(file_path, file_content)

    retrieved_content_stream = s3_service.get(file_path)
    retrieved_content = retrieved_content_stream.read()
    retrieved_content_stream.close()

    assert retrieved_content == file_content


def test_put_and_get_io(s3_service: S3):
    file_path = "test_file_io.txt"
    file_content_str = "Hello, S3 from IO!"
    file_content_bytes = file_content_str.encode("utf-8")
    file_io = io.BytesIO(file_content_bytes)

    s3_service.put(file_path, file_io)

    # Reopen BytesIO for reading (if the original stream was consumed).
    # Or get directly from S3.
    retrieved_content_stream = s3_service.get(file_path)
    retrieved_content = retrieved_content_stream.read()
    retrieved_content_stream.close()

    assert retrieved_content == file_content_bytes


def test_put_and_get_with_prefix_path_config(s3_service_with_prefix: S3, s3_client_real_or_moto: boto3.client):
    service = s3_service_with_prefix
    base_path = "data/file.txt"
    content = b"prefixed content"

    service.put(base_path, content)

    # Verify that the object is created under the correct S3 key.
    expected_s3_key = f"{service.cfg.prefix_path.rstrip('/')}/{base_path.lstrip('/')}"
    response = s3_client_real_or_moto.get_object(Bucket=service.cfg.bucket, Key=expected_s3_key)
    assert response["Body"].read() == content

    # Retrieve using service.get.
    retrieved_stream = service.get(base_path)
    assert retrieved_stream.read() == content
    retrieved_stream.close()


def test_put_should_create_bucket_if_it_does_not_exist(s3_service: S3, s3_client_real_or_moto: boto3.client):
    original_bucket_name = s3_service.cfg.bucket
    non_existent_bucket_name = "test-bucket-non-existent-abcde"

    # Ensure it does not exist.
    if s3_service.bucket_exists(non_existent_bucket_name):
        clean_bucket(s3_client_real_or_moto, non_existent_bucket_name, delete_bucket=True)

    s3_service.cfg.bucket = non_existent_bucket_name

    base_path = "data/file.txt"
    content = b"prefixed content"

    s3_service.put(base_path, content)
    assert s3_service.bucket_exists(non_existent_bucket_name)

    # Clean up
    clean_bucket(s3_client_real_or_moto, non_existent_bucket_name, delete_bucket=True)

    s3_service.cfg.bucket = original_bucket_name


def test_final_path_logic(s3_target_config: S3Config):
    # 1. No prefix_path.
    s3_target_config.prefix_path = None
    service_no_prefix = S3(s3_target_config)
    assert service_no_prefix._final_path("some/path.txt") == "some/path.txt"
    assert service_no_prefix._final_path("/leading/slash.txt") == "/leading/slash.txt"  # lstrip in put.

    # 2. With prefix_path.
    s3_target_config.prefix_path = "myprefix"
    service_with_prefix = S3(s3_target_config)
    assert service_with_prefix._final_path("file.txt") == "myprefix/file.txt"
    assert service_with_prefix._final_path("sub/file.txt") == "myprefix/sub/file.txt"

    # 3. prefix_path ending with /.
    s3_target_config.prefix_path = "myprefix/"
    service_with_trailing_slash_prefix = S3(s3_target_config)
    assert service_with_trailing_slash_prefix._final_path("file.txt") == "myprefix/file.txt"
    assert service_with_trailing_slash_prefix._final_path("sub/file.txt") == "myprefix/sub/file.txt"

    # 4. path starting with /.
    s3_target_config.prefix_path = "myprefix"
    service_with_leading_slash_path = S3(s3_target_config)
    assert service_with_leading_slash_path._final_path("/file.txt") == "myprefix/file.txt"
    assert service_with_leading_slash_path._final_path("/sub/file.txt") == "myprefix/sub/file.txt"

    # 5. prefix_path ending with /, path starting with /.
    s3_target_config.prefix_path = "myprefix/"
    service_both_slashes = S3(s3_target_config)
    assert service_both_slashes._final_path("/file.txt") == "myprefix/file.txt"
    assert service_both_slashes._final_path("/sub/file.txt") == "myprefix/sub/file.txt"

    # 6. Empty prefix_path.
    s3_target_config.prefix_path = ""
    service_empty_prefix = S3(s3_target_config)
    assert service_empty_prefix._final_path("file.txt") == "file.txt"
    assert service_empty_prefix._final_path("/file.txt") == "/file.txt"


def test_obj_exists(s3_service: S3):
    existing_file = "i_exist.txt"
    non_existing_file = "i_dont_exist.txt"
    content = b"some data"

    s3_service.put(existing_file, content)

    assert s3_service.obj_exists(existing_file)
    assert not s3_service.obj_exists(non_existing_file)


def test_obj_exists_with_prefix(s3_service_with_prefix: S3):
    service = s3_service_with_prefix
    base_path = "check_existence.log"
    content = b"log data"

    service.put(base_path, content)

    assert service.obj_exists(base_path)
    assert not service.obj_exists("other_dir/non_existent.log")


def test_delete_object(s3_service: S3):
    file_to_delete = "to_be_deleted.txt"
    content = b"delete me"

    s3_service.put(file_to_delete, content)
    assert s3_service.obj_exists(file_to_delete)

    s3_service.delete(file_to_delete)
    assert not s3_service.obj_exists(file_to_delete)


def test_delete_object_with_prefix(s3_service_with_prefix: S3):
    service = s3_service_with_prefix
    base_path = "removable.dat"
    content = b"temporary data"

    service.put(base_path, content)
    assert service.obj_exists(base_path)

    service.delete(base_path)
    assert not service.obj_exists(base_path)


def test_delete_objects_by_prefix_simple(s3_service: S3):
    prefix = "logs/today/"
    files_to_delete = [
        f"{prefix}log1.txt",
        f"{prefix}log2.txt",
        f"{prefix}sub_folder/log3.txt",
    ]
    other_file = "other_data/data.txt"

    for f_path in files_to_delete:
        s3_service.put(f_path, f"content of {f_path}".encode())
    s3_service.put(other_file, b"other content")

    for f_path in files_to_delete:
        assert s3_service.obj_exists(f_path)
    assert s3_service.obj_exists(other_file)

    s3_service.delete_objects_by_prefix(prefix)

    for f_path in files_to_delete:
        assert not s3_service.obj_exists(f_path), f"File {f_path} should have been deleted"
    assert s3_service.obj_exists(other_file), "Other file should still exist"


def test_delete_objects_by_prefix_with_config_prefix(s3_service_with_prefix: S3):
    service = s3_service_with_prefix  # cfg.prefix_path = "my/test/prefix/"

    # This path_prefix is relative to the S3 object store root,
    # but _final_path will prepend the service's prefix_path.
    # So, objects will be under "my/test/prefix/app_logs/specific_run/"
    app_specific_prefix = "app_logs/specific_run/"

    files_to_delete_relative = [
        f"{app_specific_prefix}run1.log",
        f"{app_specific_prefix}run2.log",
    ]
    other_file_relative = "app_logs/other_run/run.log"  # Will be under "my/test/prefix/app_logs/other_run/"

    # Put objects using the service (which applies its own prefix).
    for rel_path in files_to_delete_relative:
        service.put(rel_path, f"content of {rel_path}".encode())
    service.put(other_file_relative, b"other app log content")

    # Check existence using the service
    for rel_path in files_to_delete_relative:
        assert service.obj_exists(rel_path)
    assert service.obj_exists(other_file_relative)

    # Delete using the app_specific_prefix. The service will prepend its own prefix when constructing the final path.
    service.delete_objects_by_prefix(app_specific_prefix)

    for rel_path in files_to_delete_relative:
        assert not service.obj_exists(rel_path), f"File {rel_path} should have been deleted"
    assert service.obj_exists(other_file_relative), "Other app log file should still exist"


def test_delete_objects_by_prefix_pagination(s3_service: S3, s3_client_real_or_moto: boto3.client):
    prefix = "many_files/"
    num_files = 1005  # More than 1000 to test pagination in list_objects_v2
    # Create more than 1000 files to test pagination.
    for i in range(num_files):
        s3_service.put(f"{prefix}file_{i}.txt", f"content_{i}".encode())

    # Verify one file exists
    assert s3_service.obj_exists(f"{prefix}file_500.txt")

    s3_service.delete_objects_by_prefix(prefix)

    # Verify all files under the prefix are deleted
    # Check a few samples; checking all 1005 would be slow.
    assert not s3_service.obj_exists(f"{prefix}file_0.txt")
    assert not s3_service.obj_exists(f"{prefix}file_500.txt")
    assert not s3_service.obj_exists(f"{prefix}file_{num_files - 1}.txt")

    # Double-check with a direct s3_client_real_or_moto call.
    response = s3_client_real_or_moto.list_objects_v2(
        Bucket=s3_service.cfg.bucket, Prefix=prefix
    )  # List objects under the prefix.
    assert "Contents" not in response or not response["Contents"]


def test_delete_objects_by_prefix_no_objects(s3_service: S3):
    prefix = "empty_prefix/"
    # Ensure no objects exist with this prefix
    s3_service.delete_objects_by_prefix(prefix)  # Should not raise an error


def test_get_non_existent_object(s3_service: S3):
    assert s3_service.get("this_file_does_not_exist.txt") is None


def test_get_object_from_non_existent_bucket(s3_service: S3):
    original_bucket_name = s3_service.cfg.bucket
    non_existent_bucket_name = "non-existent-bucket-" + uuid.uuid4().hex

    s3_service.cfg.bucket = non_existent_bucket_name

    result = s3_service.get("some_random_object.txt")
    assert result is None, "get() should return None when the bucket does not exist."

    s3_service.cfg.bucket = original_bucket_name


def test_delete_non_existent_object(s3_service: S3):
    non_existent_file = f"this_file_really_does_not_exist_for_deletion_{uuid.uuid4().hex}.txt"
    # Ensure it doesn't exist
    assert not s3_service.obj_exists(non_existent_file)

    try:
        s3_service.delete(non_existent_file)
    except Exception as e:
        pytest.fail(f"s3_service.delete() raised an exception for a non-existent object: {e}")

    # Optionally, re-check it still doesn't exist (though delete shouldn't create it)
    assert not s3_service.obj_exists(non_existent_file)


def test_delete_object_from_non_existent_bucket(s3_target_config: S3Config, s3_client_real_or_moto: boto3.client):
    # Use a bucket name that is highly unlikely to exist
    non_existent_bucket_name = f"non-existent-bucket-{uuid.uuid4().hex}"

    # Create a new config object for the non-existent bucket
    temp_config = s3_target_config.model_copy(deep=True)
    temp_config.bucket = non_existent_bucket_name

    # Create a new S3 service instance with this temporary config
    # This service instance will use the provided s3_client_real_or_moto if its conn is set,
    # but it will operate on the non_existent_bucket_name.
    s3_service_for_non_existent_bucket = S3(temp_config)
    s3_service_for_non_existent_bucket.conn = s3_client_real_or_moto # Use the same client connection

    try:
        s3_service_for_non_existent_bucket.delete("some_object_in_non_existent_bucket.txt")
    except Exception as e:
        pytest.fail(
            f"S3.delete() raised an unexpected exception for a non-existent bucket: {type(e).__name__} - {e}"
        )
