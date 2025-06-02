from fastapi import FastAPI
from sqlmodel import SQLModel

# from aperag.views.tencent import router as tencent_router
# from aperag.views.weixin import router as wexin_router
from aperag.config import engine
from aperag.views.api_key import router as api_key_router

# from aperag.views.auth import router as auth_router
from aperag.views.chat_completion import router as chat_completion_router
from aperag.views.config import router as config_router

# from aperag.views.dingtalk import router as dingtalk_router
# from aperag.views.feishu import router as feishu_router
from aperag.views.flow import router as flow_router
from aperag.views.main import router as main_router

app = FastAPI()


@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)


app.include_router(main_router, prefix="/api/v1")
app.include_router(api_key_router, prefix="/api/v1")
# app.include_router(auth_router, prefix="/api/v1")
app.include_router(flow_router, prefix="/api/v1")
app.include_router(chat_completion_router, prefix="/v1")
app.include_router(config_router, prefix="/api/v1/config")
# app.include_router(dingtalk_router, prefix="/api/v1/dingtalk")
# app.include_router(feishu_router, prefix="/api/v1/feishu")
# app.include_router(tencent_router, prefix="/api/v1/tencent")
# app.include_router(wexin_router, prefix="/api/v1/weixin")
