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
from io import BytesIO
from typing import IO

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel

from aperag.objectstore.base import ObjectStore

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
            logging.exception(f"Fail to connect at region {self.region} or endpoint {self.endpoint_url}")

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
