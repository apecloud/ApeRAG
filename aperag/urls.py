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

from django.urls import path
from ninja import NinjaAPI
from ninja.errors import AuthenticationError, ValidationError

from aperag.utils.weixin.renderer import MyJSONRenderer
from aperag.views import main
from aperag.views.api_key import router as api_key_router
from aperag.views.auth import router as auth_router
from aperag.views.chat_completion import router as chat_completion_router
from aperag.views.config import router as config_router
from aperag.views.dingtalk import router as dingtalk_router
from aperag.views.feishu import router as feishu_router
from aperag.views.flow import router as flow_router
from aperag.views.main import router as main_router
from aperag.views.tencent import router as tencent_router
from aperag.views.utils import auth_errors, auth_middleware, validation_errors
from aperag.views.weixin import router as weixin_router

api = NinjaAPI(renderer=MyJSONRenderer)
api.add_exception_handler(ValidationError, validation_errors)
api.add_exception_handler(AuthenticationError, auth_errors)
api.add_router("/", main_router, auth=auth_middleware)
api.add_router("/", api_key_router, auth=auth_middleware)
api.add_router("/", auth_router)
api.add_router("/", flow_router)
api.add_router("/config", config_router)
api.add_router("/feishu", feishu_router)
api.add_router("/weixin", weixin_router)
api.add_router("/tencent", tencent_router)
api.add_router("/dingtalk", dingtalk_router)
api.add_exception_handler(ValidationError, validation_errors)
api.add_exception_handler(AuthenticationError, auth_errors)

chat_completion_api = NinjaAPI(renderer=MyJSONRenderer, docs_url=None, urls_namespace="chat_completion")
chat_completion_api.add_router("/", chat_completion_router, auth=auth_middleware)

urlpatterns = [
    path("v1/", chat_completion_api.urls),
    path("api/v1/", api.urls),
    path("aperag/dashboard/", main.dashboard, name="dashboard"),
]
