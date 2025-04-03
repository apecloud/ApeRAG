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