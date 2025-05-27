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

import tempfile
from typing import Any, Dict, Iterator

from aperag.schema.view_models import CollectionConfig
from aperag.source.base import LocalDocument, RemoteDocument, Source
from aperag.utils.spider.base_spider import WebCannotBeCrawledException, url_selector


def download_web_text_to_temp_file(url, name):
    html_content, prefix = url_selector(url, name)
    if len(html_content) == 0:
        raise WebCannotBeCrawledException("can't crawl the web")
    temp_file = tempfile.NamedTemporaryFile(
        prefix=prefix,
        delete=False,
        suffix=".html",
    )
    temp_file.write(html_content.encode("utf-8"))
    temp_file.close()
    return temp_file


class URLSource(Source):
    def __init__(self, ctx: CollectionConfig):
        super().__init__(ctx)

    def sync_enabled(self):
        return False

    def scan_documents(self) -> Iterator[RemoteDocument]:
        return iter([])

    def prepare_document(self, name: str, metadata: Dict[str, Any]) -> LocalDocument:
        url = metadata["url"]
        result_url = url.replace('"', "")
        temp_file_path = download_web_text_to_temp_file(result_url, name).name
        metadata["name"] = name
        return LocalDocument(name=name, path=temp_file_path, metadata=metadata)
