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
import sys
from io import BytesIO
from typing import IO, AsyncIterator, Tuple

import aioboto3
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel

from aperag.objectstore.base import AsyncObjectStore, ObjectStore

logger = logging.getLogger(__name__)


class S3Config(BaseModel):
    """Backward-compatible runtime config for the S3 object store.

    ``aperag.config.S3Config`` is still used to load environment-backed
    settings, but the object-store module needs a plain model that callers and
    tests can instantiate directly with explicit keyword arguments.
    """

    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    region: str | None = None
    prefix_path: str | None = None
    use_path_style: bool = False


class S3(ObjectStore):
    """Sync S3-compatible object store (MinIO, AWS S3, Alibaba OSS, etc.)."""

    def __init__(self, cfg) -> None:
        self.conn = None
        self.cfg = cfg
        self._checked_bucket = None

    def _ensure_conn(self):
        if self.conn is not None:
            return

        try:
            s3_params = {
                "endpoint_url": self.cfg.endpoint,
                "region_name": self.cfg.region,
                "aws_access_key_id": self.cfg.access_key,
                "aws_secret_access_key": self.cfg.secret_key,
            }
            config: Config | None = None
            if self.cfg.use_path_style:
                config = Config(s3={"addressing_style": "path"})
            self.conn = boto3.client("s3", config=config, **s3_params)
        except Exception:
            logger.exception(
                "Failed to connect to S3 at region=%s endpoint=%s",
                self.cfg.region,
                self.cfg.endpoint,
            )

    def _ensure_bucket(self):
        self._ensure_conn()
        if self._checked_bucket == self.cfg.bucket:
            return
        if self.bucket_exists(self.cfg.bucket):
            self._checked_bucket = self.cfg.bucket
            return
        self.conn.create_bucket(Bucket=self.cfg.bucket)

    def _final_path(self, path: str) -> str:
        if self.cfg.prefix_path:
            return f"{self.cfg.prefix_path.rstrip('/')}/{path.lstrip('/')}"
        return path

    def bucket_exists(self, bucket: str) -> bool:
        self._ensure_conn()
        try:
            self.conn.head_bucket(Bucket=bucket)
            exists = True
        except self.conn.exceptions.NoSuchBucket:
            exists = False
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                exists = False
            else:
                raise
        return exists

    def put(self, path: str, data: bytes | IO[bytes]):
        self._ensure_bucket()
        path = self._final_path(path)
        if isinstance(data, bytes):
            data = BytesIO(data)
        return self.conn.upload_fileobj(data, self.cfg.bucket, path)

    def get(self, path: str) -> IO[bytes] | None:
        self._ensure_conn()
        path = self._final_path(path)
        try:
            r = self.conn.get_object(Bucket=self.cfg.bucket, Key=path)
            return r["Body"]
        except (self.conn.exceptions.NoSuchKey, self.conn.exceptions.NoSuchBucket):
            return None

    def get_obj_size(self, path: str) -> int | None:
        self._ensure_conn()
        path = self._final_path(path)
        try:
            response = self.conn.head_object(Bucket=self.cfg.bucket, Key=path)
            return response.get("ContentLength")
        except (self.conn.exceptions.NoSuchKey, self.conn.exceptions.NoSuchBucket, ClientError):
            return None

    def stream_range(self, path: str, start: int, end: int | None = None) -> Tuple[IO[bytes], int] | None:
        self._ensure_conn()
        path = self._final_path(path)

        # Get total file size to validate range
        total_size = self.get_obj_size(path)
        if total_size is None:
            return None  # Object doesn't exist

        if start < 0 or start >= total_size:
            raise ValueError("Start position is out of file bounds.")

        # Format the range header
        range_str = f"bytes={start}-"
        if end is not None:
            # Ensure end is within bounds
            actual_end = min(end, total_size - 1)
            range_str += str(actual_end)
        else:
            actual_end = total_size - 1

        content_length = actual_end - start + 1
        if content_length <= 0:
            return BytesIO(b""), 0

        try:
            response = self.conn.get_object(Bucket=self.cfg.bucket, Key=path, Range=range_str)
            return response["Body"], content_length
        except ClientError as e:
            # Catching ClientError here because an invalid range can also cause it.
            # We log the warning and return None, letting the caller handle it.
            logger.warning(f"Failed to stream range for S3 object at {path}: {e}")
            return None

    def obj_exists(self, path: str) -> bool:
        self._ensure_conn()
        path = self._final_path(path)
        try:
            if self.conn.head_object(Bucket=self.cfg.bucket, Key=path):
                return True
        except self.conn.exceptions.NoSuchKey:
            return False
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            else:
                raise

    def delete(self, path: str):
        self._ensure_conn()
        path = self._final_path(path)
        try:
            self.conn.delete_object(Bucket=self.cfg.bucket, Key=path)
        except (self.conn.exceptions.NoSuchKey, self.conn.exceptions.NoSuchBucket):
            # Ignore
            return

    def delete_objects_by_prefix(self, path_prefix: str):
        self._ensure_conn()
        path_prefix = self._final_path(path_prefix)

        all_objects_to_delete = []
        continuation_token = None

        while True:
            list_kwargs = {"Bucket": self.cfg.bucket, "Prefix": path_prefix}
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = self.conn.list_objects_v2(**list_kwargs)

            if "Contents" in response:
                for obj in response["Contents"]:
                    all_objects_to_delete.append({"Key": obj["Key"]})

            if not response.get("IsTruncated"):  # If not truncated, we're done listing.
                break

            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:  # Safety break if IsTruncated is True but no token
                logger.warning(
                    f"ListObjectsV2 response was truncated but no NextContinuationToken was provided for prefix {path_prefix} in bucket {self.cfg.bucket}. Stopping."
                )
                break

        if not all_objects_to_delete:
            return

        for i in range(0, len(all_objects_to_delete), 1000):
            delete_batch = all_objects_to_delete[i : i + 1000]
            self.conn.delete_objects(Bucket=self.cfg.bucket, Delete={"Objects": delete_batch, "Quiet": True})

    def list_objects_by_prefix(self, path_prefix: str) -> list[str]:
        self._ensure_conn()
        full_prefix = self._final_path(path_prefix)
        prefix_strip_len = len(self._final_path("")) if self.cfg.prefix_path else 0
        result = []
        continuation_token = None

        while True:
            list_kwargs = {"Bucket": self.cfg.bucket, "Prefix": full_prefix}
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token

            response = self.conn.list_objects_v2(**list_kwargs)
            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    if prefix_strip_len:
                        key = key[prefix_strip_len:]
                    result.append(key)

            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:
                break

        return result


class AsyncS3(AsyncObjectStore):
    """Async S3-compatible object store.

    Uses a **lazy-initialized, long-lived** aioboto3 client for
    non-streaming operations (``put``, ``delete``, ``obj_exists``,
    ``get_obj_size``, prefix ops). Streaming operations (``get``,
    ``stream_range``) open a dedicated client context so the response
    body can outlive the method call.

    Before this change, every single operation opened and closed a new
    S3 client — measurable overhead on hot paths like batch document
    parsing that calls ``put`` dozens of times.
    """

    def __init__(self, cfg) -> None:
        self._session = aioboto3.Session()
        self.cfg = cfg
        self._checked_bucket = None
        self._client = None
        self._client_ctx = None

    def _get_client_kwargs(self):
        params = {
            "region_name": self.cfg.region,
            "aws_access_key_id": self.cfg.access_key,
            "aws_secret_access_key": self.cfg.secret_key,
        }
        if self.cfg.endpoint:
            params["endpoint_url"] = self.cfg.endpoint
        if self.cfg.use_path_style:
            params["config"] = Config(s3={"addressing_style": "path"})
        return params

    async def _ensure_client(self):
        """Lazy-init a long-lived S3 client for non-streaming ops."""
        if self._client is not None:
            return
        self._client_ctx = self._session.client("s3", **self._get_client_kwargs())
        self._client = await self._client_ctx.__aenter__()

    async def _ensure_bucket(self):
        if self._checked_bucket == self.cfg.bucket:
            return
        await self._ensure_client()
        try:
            await self._client.head_bucket(Bucket=self.cfg.bucket)
            self._checked_bucket = self.cfg.bucket
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                await self._client.create_bucket(Bucket=self.cfg.bucket)
                self._checked_bucket = self.cfg.bucket
            else:
                raise

    def _final_path(self, path: str) -> str:
        if self.cfg.prefix_path:
            return f"{self.cfg.prefix_path.rstrip('/')}/{path.lstrip('/')}"
        return path

    async def bucket_exists(self, bucket: str) -> bool:
        await self._ensure_client()
        try:
            await self._client.head_bucket(Bucket=bucket)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                return False
            raise

    async def put(self, path: str, data: bytes | IO[bytes]):
        await self._ensure_bucket()
        path = self._final_path(path)
        if isinstance(data, bytes):
            data = BytesIO(data)
        await self._client.upload_fileobj(data, self.cfg.bucket, path)

    async def get(self, path: str) -> Tuple[AsyncIterator[bytes], int] | None:
        # Streaming: need a dedicated client context so the response
        # body can outlive this method call.
        client_context = self._session.client("s3", **self._get_client_kwargs())
        client = await client_context.__aenter__()
        try:
            response = await client.get_object(Bucket=self.cfg.bucket, Key=self._final_path(path))
            stream = response["Body"]
            size = response["ContentLength"]
        except ClientError as e:
            await client_context.__aexit__(*sys.exc_info())
            if e.response["Error"]["Code"] in ("NoSuchKey", "NoSuchBucket"):
                return None
            raise
        except Exception:
            await client_context.__aexit__(*sys.exc_info())
            raise

        async def generator():
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                stream.close()
                await client_context.__aexit__(None, None, None)

        return generator(), size

    async def get_obj_size(self, path: str) -> int | None:
        await self._ensure_client()
        try:
            response = await self._client.head_object(Bucket=self.cfg.bucket, Key=self._final_path(path))
            return response.get("ContentLength")
        except ClientError:
            return None

    async def stream_range(
        self, path: str, start: int, end: int | None = None
    ) -> Tuple[AsyncIterator[bytes], int] | None:
        if start < 0 or (end is not None and end < start):
            raise ValueError("Invalid range: start/end positions are illogical.")

        range_str = f"bytes={start}-"
        if end is not None:
            range_str += str(end)

        # Streaming: dedicated client context.
        client_context = self._session.client("s3", **self._get_client_kwargs())
        client = await client_context.__aenter__()
        try:
            response = await client.get_object(Bucket=self.cfg.bucket, Key=self._final_path(path), Range=range_str)
            stream = response["Body"]
            content_length = response["ContentLength"]
        except ClientError as e:
            await client_context.__aexit__(*sys.exc_info())
            if e.response["Error"]["Code"] in ("InvalidRange", "NoSuchKey", "NoSuchBucket"):
                logger.warning("Failed to stream range for S3 object at %s with range '%s': %s", path, range_str, e)
                return None
            raise
        except Exception:
            await client_context.__aexit__(*sys.exc_info())
            raise

        async def generator():
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                stream.close()
                await client_context.__aexit__(None, None, None)

        return generator(), content_length

    async def obj_exists(self, path: str) -> bool:
        await self._ensure_client()
        try:
            await self._client.head_object(Bucket=self.cfg.bucket, Key=self._final_path(path))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    async def delete(self, path: str):
        await self._ensure_client()
        try:
            await self._client.delete_object(Bucket=self.cfg.bucket, Key=self._final_path(path))
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("NoSuchKey", "NoSuchBucket"):
                raise

    async def delete_objects_by_prefix(self, path_prefix: str):
        await self._ensure_client()
        path_prefix = self._final_path(path_prefix)

        all_objects_to_delete = []
        paginator = self._client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=self.cfg.bucket, Prefix=path_prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    all_objects_to_delete.append({"Key": obj["Key"]})

        if not all_objects_to_delete:
            return

        for i in range(0, len(all_objects_to_delete), 1000):
            delete_batch = all_objects_to_delete[i : i + 1000]
            await self._client.delete_objects(Bucket=self.cfg.bucket, Delete={"Objects": delete_batch, "Quiet": True})

    async def list_objects_by_prefix(self, path_prefix: str) -> list[str]:
        await self._ensure_client()
        full_prefix = self._final_path(path_prefix)
        prefix_strip_len = len(self._final_path("")) if self.cfg.prefix_path else 0
        result = []

        paginator = self._client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=self.cfg.bucket, Prefix=full_prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if prefix_strip_len:
                        key = key[prefix_strip_len:]
                    result.append(key)

        return result
