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

import json
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from aperag.chat.utils import get_sync_redis_client
from aperag.db.models import Collection, Document
from aperag.tasks.index import add_index_for_local_document
from config.celery import app


@app.task(bind=True, max_retries=3)
def crawl_domain(self, root_url, url, collection_id, user, max_pages):
    redis_key = f"{url}"
    # redis_conn = redis.from_url(settings.MEMORY_REDIS_URL)
    redis_conn = get_sync_redis_client()

    collection = Collection.objects.get(id=collection_id)
    redis_set_key = f"crawled_urls:{collection_id}:{root_url}"
    if (
        redis_conn.sismember(redis_set_key, redis_key)
        or redis_conn.scard(redis_set_key) >= max_pages
        or collection.status == Collection.Status.DELETED
    ):
        return

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        redis_conn.sadd(redis_set_key, redis_key)
        root_parts = urlparse(root_url)

        soup = BeautifulSoup(response.content, "html.parser")
        if root_url != url:
            if ".html" not in url:
                document_name = url + ".html"
            else:
                document_name = url
            document_instance = Document(
                user=user,
                name=document_name,
                status=Document.Status.PENDING,
                collection_id=collection.id,
                size=0,
            )
            document_instance.save()
            string_data = json.dumps(url)
            document_instance.metadata = json.dumps({
                "url": string_data,
            })
            document_instance.save()
            add_index_for_local_document.delay(document_instance.id)
        for link in soup.find_all("a", href=True):
            sub_url = urljoin(url, link["href"]).split("#")[0]
            sub_parts = urlparse(sub_url)
            sub_redis_key = f"{sub_url}"
            if root_parts.scheme != sub_parts.scheme or root_parts.netloc != sub_parts.netloc:
                continue
            if (
                sub_url == url
                or redis_conn.sismember(redis_set_key, sub_redis_key)
                or not sub_parts.path.startswith(root_parts.path)
            ):
                continue
            if re.match(r"https?://[\w.]+", sub_url):
                crawl_domain.delay(root_url, sub_url, collection_id, user, max_pages)

    except Exception as e:
        print(f"Error crawling {url}: {e}")
        self.retry(exc=e)
    # finally:
    #     # use synchronized redis_client here, need to be released manually
    #     redis_conn.connection_pool.disconnect()


@app.task
def cleanup(collectionid, redis_conn):
    for key in redis_conn.scan_iter(f"{collectionid}:*"):
        redis_conn.srem("crawled_urls", key)
