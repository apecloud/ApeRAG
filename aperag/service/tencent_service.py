import time

from aperag.chat.utils import get_async_redis_client


async def save_tencent_code_and_uri(state, code, redirect_uri):
    redis_client = get_async_redis_client()
    await redis_client.set("tencent_code_" + state, code)
    await redis_client.set("tencent_redirect_uri_" + state, redirect_uri)
    await redis_client.expireat("tencent_code_" + state, int(time.time()) + 300)
