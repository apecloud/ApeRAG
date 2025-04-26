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

from ninja import Router

from aperag.db.ops import query_config
from aperag.views.utils import success
from config import settings

router = Router()


@router.get("")
def config(request):
    auth = {
        "type": settings.AUTH_TYPE,
    }

    match settings.AUTH_TYPE:
        case "auth0":
            auth["auth_domain"] = settings.AUTH0_DOMAIN
            auth["auth_app_id"] = settings.AUTH0_CLIENT_ID
        case "authing":
            auth["auth_domain"] = settings.AUTHING_DOMAIN
            auth["auth_app_id"] = settings.AUTHING_APP_ID
        case "logto":
            auth["auth_domain"] = "http://" + settings.LOGTO_DOMAIN
            auth["auth_app_id"] = settings.LOGTO_APP_ID

    values = query_config(key="public_ips")
    public_ips = json.loads(values)

    return success({"auth": auth, "public_ips": public_ips})
