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

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(os.path.join(BASE_DIR, ".env"))


class S3Config(BaseSettings):
    endpoint: str = Field("http://127.0.0.1:9000", env="OBJECT_STORE_S3_ENDPOINT")
    access_key: str = Field("minioadmin", env="OBJECT_STORE_S3_ACCESS_KEY")
    secret_key: str = Field("minioadmin", env="OBJECT_STORE_S3_SECRET_KEY")
    bucket: str = Field("aperag", env="OBJECT_STORE_S3_BUCKET")
    region: Optional[str] = Field(None, env="OBJECT_STORE_S3_REGION")
    prefix_path: Optional[str] = Field(None, env="OBJECT_STORE_S3_PREFIX_PATH")
    use_path_style: bool = Field(True, env="OBJECT_STORE_S3_USE_PATH_STYLE")


class LocalObjectStoreConfig(BaseSettings):
    root_dir: str = Field(".objects", env="OBJECT_STORE_LOCAL_ROOT_DIR")


class Config(BaseSettings):
    # Debug mode
    debug: bool = Field(False, env="DEBUG")

    # Database
    database_url: str = Field(f"sqlite:///{BASE_DIR}/db.sqlite3", env="DATABASE_URL")

    # Media root
    media_root: str = Field(".", env="MEDIA_ROOT")

    # Session
    session_cookie_name: str = Field("sessionid", env="SESSION_COOKIE_NAME")

    # Auth
    auth_type: str = Field("none", env="AUTH_TYPE")
    auth0_domain: str = Field("aperag-dev.auting.cn", env="AUTH0_DOMAIN")
    auth0_client_id: str = Field("", env="AUTH0_CLIENT_ID")
    authing_domain: str = Field("aperag.authing.cn", env="AUTHING_DOMAIN")
    authing_app_id: str = Field("", env="AUTHING_APP_ID")
    logto_domain: str = Field("aperag.authing.cn", env="LOGTO_DOMAIN")
    logto_app_id: str = Field("", env="LOGTO_APP_ID")

    # Celery
    celery_broker_url: str = Field("redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: Optional[str] = None  # Will be set in __post_init__
    celery_beat_scheduler: str = "django_celery_beat.schedulers:DatabaseScheduler"
    celery_worker_send_task_events: bool = True
    celery_task_send_sent_event: bool = True
    celery_task_track_started: bool = True

    local_queue_name: str = Field("", env="LOCAL_QUEUE_NAME")

    # Model configs
    model_configs: Dict[str, Any] = {}

    # Embedding
    embedding_max_chunks_in_batch: int = Field(16, env="EMBEDDING_MAX_CHUNKS_IN_BATCH")

    # Rerank
    rerank_backend: str = Field("local", env="RERANK_BACKEND")
    rerank_service_url: str = Field("http://localhost:9997", env="RERANK_SERVICE_URL")
    rerank_service_model: str = Field("", env="RERANK_SERVICE_MODEL")
    rerank_service_token_api_key: str = Field("", env="RERANK_SERVICE_TOKEN_API_KEY")

    # Memory backend
    memory_redis_url: str = Field("redis://127.0.0.1:6379/1", env="MEMORY_REDIS_URL")

    # Vector DB
    vector_db_type: str = Field("qdrant", env="VECTOR_DB_TYPE")
    vector_db_context: str = Field(
        '{"url":"http://localhost", "port":6333, "distance":"Cosine"}', env="VECTOR_DB_CONTEXT"
    )

    # Object store
    object_store_type: str = Field("local", env="OBJECT_STORE_TYPE")
    object_store_local_config: Optional[LocalObjectStoreConfig] = None
    object_store_s3_config: Optional[S3Config] = None

    # Prometheus, static root
    static_root: Path = BASE_DIR / "static"

    # Feishu
    feishu_app_id: str = Field("", env="FEISHU_APP_ID")
    feishu_app_secret: str = Field("", env="FEISHU_APP_SECRET")
    feishu_encrypt_key: str = Field("", env="FEISHU_ENCRYPT_KEY")

    # Limits
    max_bot_count: int = Field(10, env="MAX_BOT_COUNT")
    max_collection_count: int = Field(50, env="MAX_COLLECTION_COUNT")
    max_document_count: int = Field(1000, env="MAX_DOCUMENT_COUNT")
    max_document_size: int = Field(100 * 1024 * 1024, env="MAX_DOCUMENT_SIZE")
    max_conversation_count: int = Field(100, env="MAX_CONVERSATION_COUNT")

    # Chunking
    chunk_size: int = Field(400, env="CHUNK_SIZE")
    chunk_overlap_size: int = Field(20, env="CHUNK_OVERLAP_SIZE")

    # Redis
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: str = Field("6379", env="REDIS_PORT")
    redis_username: str = Field("", env="REDIS_USERNAME")
    redis_password: str = Field("", env="REDIS_PASSWORD")

    # Fulltext search
    enable_fulltext_search: bool = Field(True, env="ENABLE_FULLTEXT_SEARCH")
    es_host: str = Field("http://localhost:9200", env="ES_HOST")

    # Qianfan
    qianfan_api_key: str = Field("", env="QIANFAN_API_KEY")
    qianfan_secret_key: str = Field("", env="QIANFAN_SECRET_KEY")

    # OCR/ASR
    whisper_host: str = Field("", env="WHISPER_HOST")
    paddleocr_host: str = Field("", env="PADDLEOCR_HOST")
    docray_host: str = Field("", env="DOCRAY_HOST")

    # Register mode
    register_mode: str = Field("unlimited", env="REGISTER_MODE")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set celery_result_backend if not set
        if not self.celery_result_backend:
            self.celery_result_backend = self.celery_broker_url

        # Load model configs from file
        import json
        import os

        json_path = os.path.join(BASE_DIR, "model_configs.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                self.model_configs = json.load(f)

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
    elif url.startswith("postgres+asyncpg://"):
        return url.replace("postgres+asyncpg://", "postgres://")
    else:
        return url


def get_async_database_url(url: str):
    """Convert sync database URL to async version for celery"""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://")
    elif url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://")
    else:
        return url


settings = Config()

async_engine = create_async_engine(get_async_database_url(settings.database_url), echo=settings.debug)
sync_engine = create_engine(get_sync_database_url(settings.database_url), echo=settings.debug)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


def get_sync_session() -> Generator[Session, None, None]:
    sync_session = sessionmaker(sync_engine)
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
