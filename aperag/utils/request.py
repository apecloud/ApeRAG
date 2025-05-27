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

from aperag.utils.constant import KEY_USER_ID


def get_user(request):
    return request.META.get(KEY_USER_ID, "")


def get_urls(request):
    body_str = request.body.decode("utf-8")

    data = json.loads(body_str)

    urls = [item["url"] for item in data["urls"]]

    return urls
