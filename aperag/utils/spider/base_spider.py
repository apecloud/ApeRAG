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

import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from aperag.utils.spider.stackoverflow import get_stackoverflow
from aperag.utils.spider.zhihu import get_zhihu


class WebCannotBeCrawledException(BaseException):
    pass


def get_default(url: str, max_retries: int = 3, retry_delay: int = 3):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            return soup
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Failed to fetch the page.")
                return None


def url_selector(url, name):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    prefix = name.strip("/").replace("/", "--")
    if "zhihu" in domain:
        return get_zhihu(url), prefix
    if "stackoverflow" in domain:
        return get_stackoverflow(url), prefix
    else:
        return get_default(url=url, max_retries=3, retry_delay=5), prefix
