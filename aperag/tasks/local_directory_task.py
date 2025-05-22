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
import os
import time
from typing import Tuple

from aperag.db.models import Collection, Document
from aperag.db.ops import query_collection, query_documents
from aperag.docparser.doc_parser import DocParser, get_default_config
from aperag.schema.utils import parseCollectionConfig
from aperag.tasks.index import add_index_for_document, remove_index, update_index_for_document
from aperag.utils.uncompress import SUPPORTED_COMPRESSED_EXTENSIONS
from config.celery import app

logger = logging.getLogger(__name__)


class filestat:
    def __init__(self, latest_time="", file_size=0):
        self.latest_time = latest_time
        self.file_size = file_size


@app.task
def update_local_directory_index(user, collection_id):
    logger.debug(f"update_index_cron_job() : update collection{collection_id} start ")

    collection: Collection = query_collection(user=user, collection_id=collection_id)
    # scan the directory
    collectionConfig = parseCollectionConfig(collection.config)
    # docments_in_direct = dict[str:filestat]
    # documents_in_db = dict[str:int]
    supported_file_extensions = DocParser().supported_extensions()  # TODO: apply collection config
    supported_file_extensions += SUPPORTED_COMPRESSED_EXTENSIONS
    _, docments_in_direct = scan_local_direct(collectionConfig.path, supported_file_extensions)  # full_filename to file info
    # scan the db
    documents = query_documents([collection.user], collection.id)
    documents_in_db = {}
    for i, doc in enumerate(documents):
        documents_in_db[doc.name] = i

    for filename in docments_in_direct.keys():
        if filename not in documents_in_db:
            # for added
            file_stat = os.stat(filename)
            document_instance = Document(
                user=collection.user,
                name=filename,
                status=Document.Status.PENDING,
                size=file_stat.st_size,
                collection_id=collection.id,
                metadata=time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(file_stat.st_mtime)
                ),
            )
            document_instance.save()
            add_index_for_document.delay(document_instance.id)
            # read from local direct
        else:
            # for update
            document_in_db = documents[documents_in_db[filename]]
            if update_strategies(
                docments_in_direct[filename],
                filestat(
                    latest_time=document_in_db.metadata, file_size=document_in_db.size
                ),
            ):
                update_index_for_document.delay(document_in_db.id)
    # for delete
    for filename in documents_in_db.keys():
        if filename not in docments_in_direct.keys():
            remove_index.delay(documents[documents_in_db[filename]].id)

    logger.debug(f"update_index_cron_job() : update collection{collection_id} end")


def update_strategies(stat_in_direct: filestat, stat_in_db: filestat) -> bool:
    """
    update_strategies: if the file diff meet the update requirement return true
    :param stat_in_direct:
    :param stat_in_db:
    :return:
    """
    if stat_in_direct.latest_time == stat_in_db.latest_time:
        logger.debug("update_strategies : no need update")
        return False
    if (
        abs(stat_in_direct.file_size - stat_in_db.file_size)
        < stat_in_db.file_size * 0.2
    ):  # only size, need more check
        logger.debug("update_strategies : no need update")
        return False
    logger.debug("update_strategies : need update")
    return True


def scan_local_direct(directory, supported_file_extensions: list[str]) -> Tuple[bool, dict]:
    """
    return a dict, key is full-path name of a file, value is the file stat
    :param directory: scan directory
    :return:
    """
    result = {}
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                if (
                    os.path.splitext(file_path)[1].lower()
                    in supported_file_extensions
                ):
                    file_stat = os.stat(file_path)
                    temp = filestat(
                        latest_time=time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(file_stat.st_mtime)
                        ),
                        file_size=file_stat.st_size,
                    )
                    result[file_path] = temp
        return True, result
    except Exception as e:
        logger.error(f"update_local_directory_index - scan_local_direct(): error{e}")
        return False, None
