import json

from ninja import NinjaAPI, Router

from config import settings
from deeprag.db.ops import query_config
from deeprag.views.utils import success

api = NinjaAPI(version="1.0.0", urls_namespace="config")

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
