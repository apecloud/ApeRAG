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

from fastapi import APIRouter

from aperag.config import AsyncSessionDep, settings
from aperag.db.ops import query_first_user_exists
from aperag.schema.view_models import Auth, Auth0, Authing, Config, Logto
from aperag.views.utils import success

router = APIRouter()


@router.get("")
async def config_view(session: AsyncSessionDep) -> Config:
    auth = Auth(
        type=settings.auth_type,
    )
    match settings.auth_type:
        case "auth0":
            auth.auth0 = Auth0(
                auth_domain=settings.auth0_domain,
                auth_app_id=settings.auth0_client_id,
            )
        case "authing":
            auth.authing = Authing(
                auth_domain=settings.authing_domain,
                auth_app_id=settings.authing_app_id,
            )
        case "logto":
            auth.logto = Logto(
                auth_domain="http://" + settings.logto_domain,
                auth_app_id=settings.logto_app_id,
            )
        case "cookie":
            pass
        case _:
            raise ValueError(f"Unsupported auth type: {settings.auth_type}")

    result = Config(
        auth=auth,
        admin_user_exists=await query_first_user_exists(session),
    )
    return success(result)
