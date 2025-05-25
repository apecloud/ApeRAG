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
from datetime import datetime
from typing import Any, Dict, Iterator

import oss2

from aperag.source.base import CustomSourceInitializationError, LocalDocument, RemoteDocument, Source
from aperag.source.utils import find_duplicate_paths, gen_temporary_file
from aperag.schema.view_models import CollectionConfig

logger = logging.getLogger(__name__)


class OSSSource(Source):

    def __init__(self, ctx: CollectionConfig):
        super().__init__(ctx)
        self.access_key_id = ctx.access_key_id
        self.access_key_secret = ctx.secret_access_key
        self.bucket_objs = []
        self.bucket_name = ctx.bucket
        self.endpoint = ctx.region
        self.dir = ctx.dir
        self.buckets = self._connect_buckets()

    def _connect_buckets(self):
        if self.bucket_name != '':
            new_bucket_obj = {}
            new_bucket_obj['bucket'] = self.bucket_name
            new_bucket_obj['dir'] = self.dir
            self.bucket_objs.append(new_bucket_obj)
        bucket_dirs = []
        for bucket_obj in self.bucket_objs:
            bucket_dirs.append('/' + bucket_obj['dir'])
        duplicates = find_duplicate_paths(bucket_dirs)
        if len(duplicates) != 0:
            raise CustomSourceInitializationError(f"There is duplicate dir in bucket dirs eg.({duplicates[0][0]},{duplicates[0][1]})")
        buckets = {}
        for bucket_obj in self.bucket_objs:
            try:
                bucket_name = bucket_obj['bucket']
                auth = oss2.Auth(self.access_key_id, self.access_key_secret)
                bucket = oss2.Bucket(auth, endpoint=self.endpoint, bucket_name=bucket_name, connect_timeout=3)
                bucket.get_bucket_info()
                buckets[bucket_name] = bucket
            except oss2.exceptions.ClientError:
                raise CustomSourceInitializationError("Error connecting to OSS server. Invalid parameter")
            except oss2.exceptions.AccessDenied:
                raise CustomSourceInitializationError("Error connecting to OSS server. Access denied")
            except oss2.exceptions.NoSuchBucket:
                raise CustomSourceInitializationError("Error connecting to OSS server. Bucket does not exist")
            except oss2.exceptions.RequestError:
                raise CustomSourceInitializationError("Error connecting to OSS server. Request error")
            except oss2.exceptions.ServerError:
                raise CustomSourceInitializationError("Error connecting to OSS server. Server error")
        return buckets

    def scan_documents(self) -> Iterator[RemoteDocument]:
        for bucket_obj in self.bucket_objs:
            bucket_name = bucket_obj['bucket']
            file_path = bucket_obj['dir']
            for obj in oss2.ObjectIterator(self.buckets[bucket_name], prefix=file_path):  # get file in given directory
                try:
                    doc = RemoteDocument(
                        name=obj.key,
                        size=obj.size,
                        metadata={
                            "modified_time": datetime.utcfromtimestamp(int(obj.last_modified)),
                            "bucket_name": bucket_name,
                        }
                    )
                    yield doc
                except Exception as e:
                    logger.error(f"scanning_oss_add_index() {obj.key} error {e}")
                    raise e

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        bucket_name = metadata.get("bucket_name", self.bucket_name)
        content = self.buckets[bucket_name].get_object(name).read()
        temp_file = gen_temporary_file(name)
        temp_file.write(content)
        temp_file.close()
        metadata["name"] = name
        return LocalDocument(name=name, path=temp_file.name, metadata=metadata)

    def sync_enabled(self):
        return True
