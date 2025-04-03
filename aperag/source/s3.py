import logging
from datetime import datetime
from typing import Any, Dict, Iterator

import boto3
import botocore

from deeprag.source.base import CustomSourceInitializationError, LocalDocument, RemoteDocument, Source
from deeprag.source.utils import find_duplicate_paths, gen_temporary_file

logger = logging.getLogger(__name__)


class S3Source(Source):

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__(ctx)
        self.access_key_id = ctx["access_key_id"]
        self.access_key_secret = ctx["secret_access_key"]
        self.bucket_objs = ctx.get("buckets", [])
        self.bucket_name = ctx.get("bucket", "")
        self.region = ctx.get("region")
        self.dir = ctx.get("dir", "")
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
            bucket_name = bucket_obj['bucket']
            try:
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.access_key_secret,
                    region_name=self.region,
                    config=botocore.config.Config(connect_timeout=3)
                )
                # check if bucket exists, and you have permission to access it
                s3_client.head_bucket(Bucket=bucket_name)
                buckets[bucket_name] = s3_client.Bucket(bucket_name)
            except botocore.exceptions.ClientError:
                raise CustomSourceInitializationError("Error connecting to S3 server. Invalid parameter")
            except botocore.exceptions.NoCredentialsError:
                raise CustomSourceInitializationError("Error connecting to S3 server. No valid AWS credentials provided")
            except botocore.exceptions.EndpointConnectionError:
                raise CustomSourceInitializationError("Error connecting to S3 server. Unable to reach the endpoint")
            except botocore.exceptions.WaiterError:
                raise CustomSourceInitializationError("Error connecting to S3 server. Connection timed out")
        return buckets

    def scan_documents(self) -> Iterator[RemoteDocument]:
        for bucket_obj in self.bucket_objs:
            bucket_name = bucket_obj['bucket']
            file_path = bucket_obj['dir']
            for obj in self.buckets[bucket_name].objects.filter(Prefix=file_path):
                try:
                    doc = RemoteDocument(
                        name=obj.key,
                        size=obj.size,
                        metadata={
                            "modified_time": datetime.utcfromtimestamp(int(obj.last_modified.timestamp())),
                            "bucket_name": bucket_name,
                        }
                    )
                    yield doc
                except Exception as e:
                    logger.error(f"scanning_s3_add_index() {obj.key} error {e}")
                    raise e

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        bucket_name = metadata.get("bucket_name", self.bucket_name)
        obj = self.buckets[bucket_name].Object(name)
        content = obj.get()["Body"].read()
        temp_file = gen_temporary_file(name)
        temp_file.write(content)
        temp_file.close()
        metadata["name"] = name
        return LocalDocument(name=name, path=temp_file.name, metadata=metadata)

    def sync_enabled(self):
        return True
