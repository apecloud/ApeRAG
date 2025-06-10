from fastapi import FastAPI

from aperag.views.api_key import router as api_key_router
from aperag.views.auth import router as auth_router
from aperag.views.chat_completion import router as chat_completion_router
from aperag.views.config import router as config_router
from aperag.views.flow import router as flow_router
from aperag.views.main import router as main_router

app = FastAPI()


app.include_router(auth_router, prefix="/api/v1")
app.include_router(main_router, prefix="/api/v1")
app.include_router(api_key_router, prefix="/api/v1")
app.include_router(flow_router, prefix="/api/v1")
app.include_router(chat_completion_router, prefix="/v1")
app.include_router(config_router, prefix="/api/v1/config")
