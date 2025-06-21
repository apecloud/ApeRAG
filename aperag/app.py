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

from fastapi import FastAPI

from aperag.exception_handlers import register_exception_handlers
from aperag.llm.litellm_track import register_opik_llm_track
from aperag.views.api_key import router as api_key_router
from aperag.views.audit import router as audit_router
from aperag.views.auth import router as auth_router
from aperag.views.chat_completion import router as chat_completion_router
from aperag.views.config import router as config_router
from aperag.views.flow import router as flow_router
from aperag.views.llm import router as llm_router
from aperag.views.main import router as main_router

app = FastAPI(
    title="ApeRAG API",
    description="Knowledge management and retrieval system",
    version="1.0.0",
)

# Register global exception handlers
register_exception_handlers(app)

register_opik_llm_track()

app.include_router(auth_router, prefix="/api/v1")
app.include_router(main_router, prefix="/api/v1")
app.include_router(api_key_router, prefix="/api/v1")
app.include_router(audit_router, prefix="/api/v1")  # Add audit router
app.include_router(flow_router, prefix="/api/v1")
app.include_router(llm_router, prefix="/api/v1")
app.include_router(chat_completion_router, prefix="/v1")
app.include_router(config_router, prefix="/api/v1/config")
