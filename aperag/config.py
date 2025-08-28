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
import os
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Dict, Generator, Optional

from dotenv import load_dotenv
from fastapi import Depends
from pydantic import Field
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from aperag.vectorstore.connector import VectorStoreConnectorAdaptor

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(os.path.join(BASE_DIR, ".env"))


class S3Config(BaseSettings):
    endpoint: str = Field("http://127.0.0.1:9000", alias="OBJECT_STORE_S3_ENDPOINT")
    access_key: str = Field("minioadmin", alias="OBJECT_STORE_S3_ACCESS_KEY")
    secret_key: str = Field("minioadmin", alias="OBJECT_STORE_S3_SECRET_KEY")
    bucket: str = Field("aperag", alias="OBJECT_STORE_S3_BUCKET")
    region: Optional[str] = Field(None, alias="OBJECT_STORE_S3_REGION")
    prefix_path: Optional[str] = Field(None, alias="OBJECT_STORE_S3_PREFIX_PATH")
    use_path_style: bool = Field(True, alias="OBJECT_STORE_S3_USE_PATH_STYLE")


class LocalObjectStoreConfig(BaseSettings):
    root_dir: str = Field(".objects", alias="OBJECT_STORE_LOCAL_ROOT_DIR")


class Config(BaseSettings):
    # Debug mode
    debug: bool = Field(False, alias="DEBUG")

    # Postgres atomic fields
    postgres_host: str = Field("127.0.0.1", alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    postgres_db: str = Field("postgres", alias="POSTGRES_DB")
    postgres_user: str = Field("postgres", alias="POSTGRES_USER")
    postgres_password: str = Field("postgres", alias="POSTGRES_PASSWORD")

    # Redis atomic fields
    redis_host: str = Field("127.0.0.1", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_user: str = Field("default", alias="REDIS_USER")
    redis_password: str = Field("password", alias="REDIS_PASSWORD")

    # Elasticsearch atomic fields
    es_host_name: str = Field("127.0.0.1", alias="ES_HOST_NAME")
    es_port: int = Field(9200, alias="ES_PORT")
    es_user: str = Field("", alias="ES_USER")
    es_password: str = Field("", alias="ES_PASSWORD")
    es_protocol: str = Field("http", alias="ES_PROTOCOL")

    # Database
    database_url: Optional[str] = Field(None, alias="DATABASE_URL")

    # Database connection pool settings
    db_pool_size: int = Field(20, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(40, alias="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(60, alias="DB_POOL_TIMEOUT")
    db_pool_recycle: int = Field(3600, alias="DB_POOL_RECYCLE")
    db_pool_pre_ping: bool = Field(True, alias="DB_POOL_PRE_PING")

    # Auth
    auth_type: str = Field("none", alias="AUTH_TYPE")
    jwt_secret: str = Field("SECRET", alias="JWT_SECRET")
    oauth_redirect_url: str = Field("http://localhost:3000/auth/callback", alias="OAUTH_REDIRECT_URL")
    google_oauth_client_id: str = Field("", alias="GOOGLE_OAUTH_CLIENT_ID")
    google_oauth_client_secret: str = Field("", alias="GOOGLE_OAUTH_CLIENT_SECRET")
    github_oauth_client_id: str = Field("", alias="GITHUB_OAUTH_CLIENT_ID")
    github_oauth_client_secret: str = Field("", alias="GITHUB_OAUTH_CLIENT_SECRET")
    auth0_domain: str = Field("aperag-dev.auting.cn", alias="AUTH0_DOMAIN")
    auth0_client_id: str = Field("", alias="AUTH0_CLIENT_ID")
    authing_domain: str = Field("aperag.authing.cn", alias="AUTHING_DOMAIN")
    authing_app_id: str = Field("", alias="AUTHING_APP_ID")
    logto_domain: str = Field("aperag.authing.cn", alias="LOGTO_DOMAIN")
    logto_app_id: str = Field("", alias="LOGTO_APP_ID")

    # Celery
    celery_broker_url: Optional[str] = Field(None, alias="CELERY_BROKER_URL")
    celery_result_backend: Optional[str] = None  # Will be set in __post_init__
    celery_beat_scheduler: str = "django_celery_beat.schedulers:DatabaseScheduler"
    celery_worker_send_task_events: bool = True
    celery_task_send_sent_event: bool = True
    celery_task_track_started: bool = True

    local_queue_name: str = Field("", alias="LOCAL_QUEUE_NAME")

    # Model configs
    model_configs: Dict[str, Any] = {}

    # Embedding
    embedding_max_chunks_in_batch: int = Field(10, alias="EMBEDDING_MAX_CHUNKS_IN_BATCH")

    # Memory backend
    memory_redis_url: Optional[str] = Field(None, alias="MEMORY_REDIS_URL")

    # Vector DB
    vector_db_type: str = Field("qdrant", alias="VECTOR_DB_TYPE")
    vector_db_context: str = Field(
        '{"url":"http://localhost", "port":6333, "distance":"Cosine"}', alias="VECTOR_DB_CONTEXT"
    )

    # Object store
    object_store_type: str = Field("local", alias="OBJECT_STORE_TYPE")
    object_store_local_config: Optional[LocalObjectStoreConfig] = None
    object_store_s3_config: Optional[S3Config] = None

    # Limits
    max_bot_count: int = Field(10, alias="MAX_BOT_COUNT")
    max_collection_count: int = Field(50, alias="MAX_COLLECTION_COUNT")
    max_document_count: int = Field(1000, alias="MAX_DOCUMENT_COUNT")
    max_document_size: int = Field(100 * 1024 * 1024, alias="MAX_DOCUMENT_SIZE")
    max_conversation_count: int = Field(100, alias="MAX_CONVERSATION_COUNT")

    # Chunking
    chunk_size: int = Field(400, alias="CHUNK_SIZE")
    chunk_overlap_size: int = Field(20, alias="CHUNK_OVERLAP_SIZE")

    # Fulltext search
    es_host: Optional[str] = Field(None, alias="ES_HOST")
    es_timeout: int = Field(30, alias="ES_TIMEOUT")  # ES request timeout in seconds
    es_max_retries: int = Field(3, alias="ES_MAX_RETRIES")  # Max retries for ES requests

    # LLM keyword extraction
    llm_keyword_extraction_provider: str = Field("", alias="LLM_KEYWORD_EXTRACTION_PROVIDER")
    llm_keyword_extraction_model: str = Field("", alias="LLM_KEYWORD_EXTRACTION_MODEL")

    # Qianfan
    qianfan_api_key: str = Field("", alias="QIANFAN_API_KEY")
    qianfan_secret_key: str = Field("", alias="QIANFAN_SECRET_KEY")

    # OCR/ASR
    whisper_host: str = Field("", alias="WHISPER_HOST")
    paddleocr_host: str = Field("", alias="PADDLEOCR_HOST")
    docray_host: str = Field("", alias="DOCRAY_HOST")

    # Register mode
    register_mode: str = Field("unlimited", alias="REGISTER_MODE")

    # Cache
    cache_enabled: bool = Field(True, alias="CACHE_ENABLED")
    cache_ttl: int = Field(86400, alias="CACHE_TTL")

    # Opik
    opik_api_key: str = Field("", alias="OPIK_API_KEY")
    opik_workspace: str = Field("", alias="OPIK_WORKSPACE")

    # OpenTelemetry/Jaeger Tracing
    otel_enabled: bool = Field(True, alias="OTEL_ENABLED")
    otel_service_name: str = Field("aperag", alias="OTEL_SERVICE_NAME")
    otel_service_version: str = Field("1.0.0", alias="OTEL_SERVICE_VERSION")
    jaeger_enabled: bool = Field(False, alias="JAEGER_ENABLED")
    jaeger_endpoint: Optional[str] = Field(None, alias="JAEGER_ENDPOINT")
    otel_console_enabled: bool = Field(False, alias="OTEL_CONSOLE_ENABLED")
    otel_fastapi_enabled: bool = Field(True, alias="OTEL_FASTAPI_ENABLED")
    otel_sqlalchemy_enabled: bool = Field(True, alias="OTEL_SQLALCHEMY_ENABLED")
    otel_mcp_enabled: bool = Field(True, alias="OTEL_MCP_ENABLED")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load model configs from file
        import json
        import os

        json_path = os.path.join(BASE_DIR, "model_configs.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                self.model_configs = json.load(f)

        # DATABASE_URL
        if not self.database_url:
            self.database_url = (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        # CELERY_BROKER_URL
        if not self.celery_broker_url:
            self.celery_broker_url = (
                f"redis://{self.redis_user}:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
            )

        # CELERY_RESULT_BACKEND
        if not self.celery_result_backend:
            self.celery_result_backend = self.celery_broker_url

        # MEMORY_REDIS_URL
        if not self.memory_redis_url:
            self.memory_redis_url = (
                f"redis://{self.redis_user}:{self.redis_password}@{self.redis_host}:{self.redis_port}/1"
            )
        # ES_HOST
        if not self.es_host:
            if self.es_user and self.es_password:
                self.es_host = (
                    f"{self.es_protocol}://{self.es_user}:{self.es_password}@{self.es_host_name}:{self.es_port}"
                )
            else:
                self.es_host = f"{self.es_protocol}://{self.es_host_name}:{self.es_port}"
        # Object store config
        if self.object_store_type == "local":
            self.object_store_local_config = LocalObjectStoreConfig()
        elif self.object_store_type == "s3":
            self.object_store_s3_config = S3Config()
        else:
            raise ValueError(
                f"Unsupported OBJECT_STORE_TYPE: {self.object_store_type}. Supported types are: local, s3."
            )


def get_sync_database_url(url: str):
    """Convert async database URL to sync version for celery"""
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://")
    if url.startswith("postgres+asyncpg://"):
        return url.replace("postgres+asyncpg://", "postgres://")
    if url.startswith("sqlite+aiosqlite://"):
        return url.replace("sqlite+aiosqlite://", "sqlite://")
    return url


def get_async_database_url(url: str):
    """Convert sync database URL to async version"""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://")
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://")
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://")
    return url


def new_async_engine():
    return create_async_engine(
        get_async_database_url(settings.database_url),
        echo=settings.debug,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=settings.db_pool_pre_ping,
    )


def new_sync_engine():
    return create_engine(
        get_sync_database_url(settings.database_url),
        echo=settings.debug,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=settings.db_pool_pre_ping,
    )


settings = Config()

# Database connection pool settings from configuration
async_engine = new_async_engine()
sync_engine = new_sync_engine()


async def get_async_session(engine=None) -> AsyncGenerator[AsyncSession, None]:
    if engine is None:
        engine = async_engine
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


def get_sync_session(engine=None) -> Generator[Session, None, None]:
    if engine is None:
        engine = sync_engine
    sync_session = sessionmaker(engine)
    with sync_session() as session:
        yield session


def with_sync_session(func):
    """Decorator to inject sync session into sync functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        for session in get_sync_session():
            return func(session, *args, **kwargs)

    return wrapper


def with_async_session(func):
    """Decorator to inject async session into async functions"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        async for session in get_async_session():
            return await func(session, *args, **kwargs)

    return wrapper


AsyncSessionDep = Annotated[AsyncSession, Depends(get_async_session)]
SyncSessionDep = Annotated[Session, Depends(get_sync_session)]


def get_vector_db_connector(collection: str) -> VectorStoreConnectorAdaptor:
    # todo: specify the collection for different user
    # one person one collection
    ctx = json.loads(settings.vector_db_context)
    ctx["collection"] = collection
    return VectorStoreConnectorAdaptor(settings.vector_db_type, ctx=ctx)
