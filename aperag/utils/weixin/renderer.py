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
from typing import Any, Mapping, Type

from django.http import HttpRequest
from ninja.responses import NinjaJSONEncoder


class MyJSONRenderer:
    media_type = "application/json"
    charset: str = "utf-8"
    encoder_class: Type[json.JSONEncoder] = NinjaJSONEncoder

    def render(request: HttpRequest, data: Any, *, response_status: int) -> Any:
        encoder_class: Type[json.JSONEncoder] = NinjaJSONEncoder
        json_dumps_params: Mapping[str, Any] = {}

        # prevent Chinese characters from being converted to unicode
        return json.dumps(data, ensure_ascii=False, cls=encoder_class, **json_dumps_params)
