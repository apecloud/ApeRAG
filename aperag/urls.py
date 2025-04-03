from django.urls import path
from ninja import NinjaAPI
from ninja.errors import AuthenticationError, ValidationError

from deeprag.auth.validator import AdminAuth, GlobalHTTPAuth
from deeprag.utils.weixin.renderer import MyJSONRenderer
from deeprag.views import main
from deeprag.views.admin import router as admin_router
from deeprag.views.config import router as config_router
from deeprag.views.dingtalk import router as dingtalk_router
from deeprag.views.feishu import router as feishu_router
from deeprag.views.main import router as main_router
from deeprag.views.tencent import router as tencent_router
from deeprag.views.utils import auth_errors, validation_errors
from deeprag.views.web import router as web_router
from deeprag.views.weixin import router as weixin_router

api = NinjaAPI(renderer=MyJSONRenderer)
api.add_exception_handler(ValidationError, validation_errors)
api.add_exception_handler(AuthenticationError, auth_errors)

api.add_router("/", main_router, auth=GlobalHTTPAuth())
api.add_router("/", web_router)
api.add_router("/config", config_router)
api.add_router("/feishu", feishu_router)
api.add_router("/weixin", weixin_router)
api.add_router("/tencent", tencent_router)
api.add_router("/dingtalk", dingtalk_router)

admin_api = NinjaAPI(renderer=MyJSONRenderer, docs_url=None, urls_namespace="admin")
api.add_exception_handler(ValidationError, validation_errors)
api.add_exception_handler(AuthenticationError, auth_errors)
admin_api.add_router("/admin", admin_router, auth=AdminAuth())

urlpatterns = [
    path("", admin_api.urls),
    path("v1/", api.urls),
    path('deeprag/dashboard/', main.dashboard, name='dashboard'),
]
