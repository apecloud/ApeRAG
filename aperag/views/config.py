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

from ninja import Router

from aperag.db.ops import query_first_user_exists
from aperag.schema.view_models import Auth, Auth0, Authing, Config, Logto
from aperag.views.utils import success
from config import settings

router = Router()


@router.get("")
async def config_view(request) -> Config:
    auth = Auth(
        type=settings.AUTH_TYPE,
    )
    match settings.AUTH_TYPE:
        case "auth0":
            auth.auth0 = Auth0(
                auth_domain=settings.AUTH0_DOMAIN,
                auth_app_id=settings.AUTH0_CLIENT_ID,
            )
        case "authing":
            auth.authing = Authing(
                auth_domain=settings.AUTHING_DOMAIN,
                auth_app_id=settings.AUTHING_APP_ID,
            )
        case "logto":
            auth.logto = Logto(
                auth_domain="http://" + settings.LOGTO_DOMAIN,
                auth_app_id=settings.LOGTO_APP_ID,
            )
        case "cookie":
            pass
        case _:
            raise ValueError(f"Unsupported auth type: {settings.AUTH_TYPE}")

    result = Config(
        auth=auth,
        admin_user_exists=await query_first_user_exists(),
    )
    return success(result)
