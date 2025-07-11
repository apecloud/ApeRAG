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
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    region: str | None = None
    prefix_path: str | None = None
    use_path_style: bool = False


class S3(ObjectStore):
    def __init__(self, cfg: S3Config):
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
            logger.exception(f"Fail to connect at region {self.region} or endpoint {self.endpoint_url}")

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


class AsyncS3(AsyncObjectStore):
    def __init__(self, cfg: S3Config, session: aioboto3.Session | None = None):
        self.session = session
        self.cfg = cfg
        self._checked_bucket = None

    async def _ensure_conn(self):
        if self.session is not None:
            return

        try:
            self.session = aioboto3.Session()
        except Exception:
            logger.exception("Failed to create aioboto3 session")
            raise

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

    async def _ensure_bucket(self, client):
        if self._checked_bucket == self.cfg.bucket:
            return
        if await self.bucket_exists(self.cfg.bucket):
            self._checked_bucket = self.cfg.bucket
            return
        await client.create_bucket(Bucket=self.cfg.bucket)

    def _final_path(self, path: str) -> str:
        if self.cfg.prefix_path:
            return f"{self.cfg.prefix_path.rstrip('/')}/{path.lstrip('/')}"
        return path

    async def bucket_exists(self, bucket: str) -> bool:
        await self._ensure_conn()
        async with self.session.client("s3", **self._get_client_kwargs()) as client:
            try:
                await client.head_bucket(Bucket=bucket)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                    return False
                raise

    async def put(self, path: str, data: bytes | IO[bytes]):
        await self._ensure_conn()
        path = self._final_path(path)
        if isinstance(data, bytes):
            data = BytesIO(data)

        async with self.session.client("s3", **self._get_client_kwargs()) as client:
            await self._ensure_bucket(client)
            await client.upload_fileobj(data, self.cfg.bucket, path)

    async def get(self, path: str) -> Tuple[AsyncIterator[bytes], int] | None:
        await self._ensure_conn()
        path = self._final_path(path)

        client_context = self.session.client("s3", **self._get_client_kwargs())
        client = await client_context.__aenter__()
        try:
            response = await client.get_object(Bucket=self.cfg.bucket, Key=path)
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
        await self._ensure_conn()
        path = self._final_path(path)
        try:
            async with self.session.client("s3", **self._get_client_kwargs()) as client:
                response = await client.head_object(Bucket=self.cfg.bucket, Key=path)
                return response.get("ContentLength")
        except ClientError:
            return None

    async def stream_range(
        self, path: str, start: int, end: int | None = None
    ) -> Tuple[AsyncIterator[bytes], int] | None:
        await self._ensure_conn()
        path = self._final_path(path)

        if start < 0 or (end is not None and end < start):
            raise ValueError("Invalid range: start/end positions are illogical.")

        range_str = f"bytes={start}-"
        if end is not None:
            range_str += str(end)

        client_context = self.session.client("s3", **self._get_client_kwargs())
        client = await client_context.__aenter__()
        try:
            response = await client.get_object(Bucket=self.cfg.bucket, Key=path, Range=range_str)
            stream = response["Body"]
            content_length = response["ContentLength"]
        except ClientError as e:
            await client_context.__aexit__(*sys.exc_info())
            if e.response["Error"]["Code"] in ("InvalidRange", "NoSuchKey", "NoSuchBucket"):
                logger.warning(f"Failed to stream range for S3 object at {path} with range '{range_str}': {e}")
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
        await self._ensure_conn()
        path = self._final_path(path)
        try:
            async with self.session.client("s3", **self._get_client_kwargs()) as client:
                await client.head_object(Bucket=self.cfg.bucket, Key=path)
                return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    async def delete(self, path: str):
        await self._ensure_conn()
        path = self._final_path(path)
        try:
            async with self.session.client("s3", **self._get_client_kwargs()) as client:
                await client.delete_object(Bucket=self.cfg.bucket, Key=path)
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("NoSuchKey", "NoSuchBucket"):
                raise

    async def delete_objects_by_prefix(self, path_prefix: str):
        await self._ensure_conn()
        path_prefix = self._final_path(path_prefix)

        async with self.session.client("s3", **self._get_client_kwargs()) as client:
            all_objects_to_delete = []
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self.cfg.bucket, Prefix=path_prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        all_objects_to_delete.append({"Key": obj["Key"]})

            if not all_objects_to_delete:
                return

            for i in range(0, len(all_objects_to_delete), 1000):
                delete_batch = all_objects_to_delete[i : i + 1000]
                await client.delete_objects(Bucket=self.cfg.bucket, Delete={"Objects": delete_batch, "Quiet": True})
