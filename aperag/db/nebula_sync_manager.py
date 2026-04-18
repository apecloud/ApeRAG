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

import logging
import os
import re
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    from nebula3.Config import Config, SessionPoolConfig
    from nebula3.gclient.net import ConnectionPool, Session
    from nebula3.gclient.net.SessionPool import SessionPool
except ImportError:
    Config = None
    SessionPoolConfig = None
    ConnectionPool = None
    SessionPool = None
    Session = Any

logger = logging.getLogger(__name__)


def _safe_error_msg(result) -> str:
    """Safely extract error message from Nebula result, handling decode errors."""
    try:
        error_msg = result.error_msg()
        if isinstance(error_msg, bytes):
            for encoding in ["utf-8", "gbk", "latin-1"]:
                try:
                    return error_msg.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return error_msg.decode("utf-8", errors="replace")
        if isinstance(error_msg, str):
            return error_msg
        return str(error_msg)
    except Exception as exc:
        logger.warning(f"Failed to get Nebula error message: {exc}")
        return f"Nebula operation failed (error code: {result.error_code()})"


class NebulaSyncConnectionManager:
    """
    Worker-scoped NebulaGraph sync connection manager with lazy initialization.
    """

    _connection_pool: Optional["ConnectionPool"] = None
    _space_pools: dict[str, "_NebulaSpaceSessionPool"] = {}
    _lock = threading.Lock()
    _config: Optional[Dict[str, Any]] = None
    _prepared_spaces: set[str] = set()

    @classmethod
    def initialize(cls, config: Optional[Dict[str, Any]] = None):
        with cls._lock:
            if cls._connection_pool is not None:
                return

            if ConnectionPool is None or Config is None or SessionPool is None or SessionPoolConfig is None:
                raise RuntimeError(
                    "nebula3-python is not installed. Please install it with: pip install nebula3-python"
                )

            if config:
                cls._config = config
            else:
                cls._config = {
                    "host": os.environ.get("NEBULA_HOST", "127.0.0.1"),
                    "port": int(os.environ.get("NEBULA_PORT", "9669")),
                    "username": os.environ.get("NEBULA_USER", "root"),
                    "password": os.environ.get("NEBULA_PASSWORD", "nebula"),
                    "max_connection_pool_size": int(os.environ.get("NEBULA_MAX_CONNECTION_POOL_SIZE", "10")),
                    "timeout": int(os.environ.get("NEBULA_TIMEOUT", "60000")),
                }

            logger.info(f"Initializing Nebula sync connection pool for worker {os.getpid()}")

            config_obj = Config()
            config_obj.max_connection_pool_size = cls._config["max_connection_pool_size"]
            config_obj.timeout = cls._config["timeout"]

            cls._connection_pool = ConnectionPool()
            if not cls._connection_pool.init([(cls._config["host"], cls._config["port"])], config_obj):
                raise RuntimeError("Failed to initialize Nebula connection pool")

            logger.info(f"Nebula sync connection pool initialized successfully for worker {os.getpid()}")

    @classmethod
    def get_pool(cls) -> "ConnectionPool":
        if cls._connection_pool is None:
            cls.initialize()
        return cls._connection_pool

    @classmethod
    @contextmanager
    def get_session(cls, space: Optional[str] = None) -> Session:
        if space:
            yield cls.get_space_session(space)
            return

        with cls.get_bootstrap_session() as session:
            yield session

    @classmethod
    @contextmanager
    def get_bootstrap_session(cls, space: Optional[str] = None) -> Session:
        pool = cls.get_pool()
        session = pool.get_session(cls._config["username"], cls._config["password"])
        try:
            if space:
                result = session.execute(f"USE {space}")
                if not result.is_succeeded():
                    raise RuntimeError(f"Failed to use space {space}: {_safe_error_msg(result)}")
            yield session
        finally:
            session.release()

    @classmethod
    def get_space_session(cls, space: str) -> "_NebulaSpaceSession":
        cls.get_space_pool(space)
        return _NebulaSpaceSession(cls, space)

    @classmethod
    def get_space_pool(cls, space: str) -> "_NebulaSpaceSessionPool":
        if cls._connection_pool is None:
            cls.initialize()

        with cls._lock:
            pool = cls._space_pools.get(space)
            if pool is not None:
                return pool

            session_pool = SessionPool(
                cls._config["username"],
                cls._config["password"],
                space,
                [(cls._config["host"], cls._config["port"])],
            )
            pool_config = SessionPoolConfig()
            pool_config.min_size = 1
            pool_config.max_size = max(1, cls._config["max_connection_pool_size"])
            pool_config.timeout = cls._config["timeout"]
            pool_config.interval_check = -1
            if not session_pool.init(pool_config):
                raise RuntimeError(f"Failed to initialize Nebula session pool for space {space}")

            wrapped = _NebulaSpaceSessionPool(session_pool)
            cls._space_pools[space] = wrapped
            return wrapped

    @classmethod
    def reset_space_pool(cls, space: str):
        with cls._lock:
            pool = cls._space_pools.pop(space, None)
        if pool is not None:
            pool.close()

    @classmethod
    def execute(cls, space: str, stmt: str):
        return cls._run_with_space_pool(space, lambda pool: pool.execute(stmt))

    @classmethod
    def execute_parameter(cls, space: str, stmt: str, params: dict):
        return cls._run_with_space_pool(space, lambda pool: pool.execute_parameter(stmt, params))

    @classmethod
    def _run_with_space_pool(cls, space: str, action):
        try:
            return action(cls.get_space_pool(space))
        except Exception as exc:
            # Connection resets are recoverable if graphd is still up. Recreate the pool once.
            if "TSocket read 0 bytes" not in str(exc) and "Connection reset by peer" not in str(exc):
                raise
            logger.warning("Nebula session pool for space %s hit a broken connection, recreating once: %s", space, exc)
            cls.reset_space_pool(space)
            return action(cls.get_space_pool(space))

    @classmethod
    def prepare_space(cls, workspace: str, max_wait: int = 30, fail_on_timeout: bool = True) -> str:
        """
        Create and initialize a Nebula space for the given workspace.
        """
        import time

        space_name = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)

        with cls._lock:
            if space_name in cls._prepared_spaces:
                return space_name

        try:
            with cls.get_bootstrap_session() as session:
                result = session.execute("SHOW SPACES")
                if result.is_succeeded():
                    spaces = [row.values()[0].as_string() for row in result]
                    if space_name in spaces:
                        try:
                            with cls.get_bootstrap_session(space=space_name) as test_session:
                                ready = test_session.execute("SHOW TAGS")
                                if ready.is_succeeded():
                                    with cls._lock:
                                        cls._prepared_spaces.add(space_name)
                                    return space_name
                        except Exception:
                            logger.debug(f"Existing space {space_name} is not fully ready yet, continuing setup")
        except Exception as exc:
            logger.warning(f"Nebula fast-path space check failed: {exc}")

        with cls.get_bootstrap_session() as session:
            result = session.execute("SHOW SPACES")
            if not result.is_succeeded():
                raise RuntimeError(f"Failed to list spaces: {_safe_error_msg(result)}")

            spaces = [row.values()[0].as_string() for row in result]
            if space_name not in spaces:
                create_result = session.execute(
                    f"CREATE SPACE IF NOT EXISTS {space_name} "
                    f"(partition_num=10, replica_factor=1, vid_type=FIXED_STRING(256))"
                )
                if not create_result.is_succeeded():
                    raise RuntimeError(f"Failed to create space {space_name}: {_safe_error_msg(create_result)}")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                with cls.get_bootstrap_session(space=space_name) as test_session:
                    result = test_session.execute("SHOW TAGS")
                    if result.is_succeeded():
                        break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            logger.warning(f"Space {space_name} not ready after {max_wait}s, continuing with schema setup")

        with cls.get_bootstrap_session(space=space_name) as session:
            tag_result = session.execute(
                "CREATE TAG IF NOT EXISTS base ("
                "entity_id string, "
                "entity_type string, "
                "description string, "
                "source_id string, "
                "file_path string, "
                "created_at int64"
                ")"
            )
            if not tag_result.is_succeeded():
                logger.warning(f"Failed to create Nebula tag: {_safe_error_msg(tag_result)}")

            edge_result = session.execute(
                "CREATE EDGE IF NOT EXISTS DIRECTED ("
                "weight double, "
                "description string, "
                "keywords string, "
                "source_id string, "
                "file_path string, "
                "created_at int64"
                ")"
            )
            if not edge_result.is_succeeded():
                logger.warning(f"Failed to create Nebula edge: {_safe_error_msg(edge_result)}")

            index_result = session.execute(
                "CREATE TAG INDEX IF NOT EXISTS base_entity_id_index ON base(entity_id(256))"
            )
            if not index_result.is_succeeded():
                logger.warning(f"Failed to create Nebula tag index: {_safe_error_msg(index_result)}")

            edge_index_result = session.execute(
                "CREATE EDGE INDEX IF NOT EXISTS directed_source_id_index ON DIRECTED(source_id(256))"
            )
            if not edge_index_result.is_succeeded():
                logger.warning(f"Failed to create Nebula edge index: {_safe_error_msg(edge_index_result)}")

        time.sleep(2)
        schema_ready = False
        schema_start = time.time()
        while time.time() - schema_start < max_wait:
            try:
                with cls.get_bootstrap_session(space=space_name) as session:
                    result = session.execute(
                        "INSERT VERTEX base(entity_id, entity_type) "
                        "VALUES '__schema_test__':('__schema_test__', 'test')"
                    )
                    if result.is_succeeded():
                        session.execute("DELETE VERTEX '__schema_test__'")
                        schema_ready = True
                        break
            except Exception:
                pass
            time.sleep(0.5)

        if not schema_ready and fail_on_timeout:
            try:
                with cls.get_bootstrap_session(space=space_name) as session:
                    if not session.execute("SHOW TAGS").is_succeeded():
                        raise RuntimeError(f"Schema for space {space_name} is not ready after {max_wait}s")
            except Exception as exc:
                raise RuntimeError(f"Schema for space {space_name} is not ready after {max_wait}s") from exc

        with cls._lock:
            cls._prepared_spaces.add(space_name)
        return space_name

    @classmethod
    def drop_space(cls, space_name: str) -> dict[str, str]:
        cls.reset_space_pool(space_name)
        with cls.get_bootstrap_session() as session:
            result = session.execute(f"DROP SPACE IF EXISTS {space_name}")
        with cls._lock:
            cls._prepared_spaces.discard(space_name)
        if result.is_succeeded():
            return {"status": "success", "message": "data dropped"}
        return {"status": "error", "message": _safe_error_msg(result)}

    @classmethod
    def close(cls):
        with cls._lock:
            pools = list(cls._space_pools.values())
            cls._space_pools.clear()
            cls._prepared_spaces.clear()
            if cls._connection_pool:
                logger.info(f"Closing Nebula connection pool for worker {os.getpid()}")
                cls._connection_pool.close()
                cls._connection_pool = None
                cls._config = None
        for pool in pools:
            pool.close()


def setup_worker_nebula(**kwargs):
    """Legacy no-op kept for compatibility."""
    logger.info(f"Worker {os.getpid()}: Nebula connection will be initialized on-demand (lazy loading)")


def cleanup_worker_nebula(**kwargs):
    """Legacy no-op kept for compatibility."""
    logger.info(f"Worker {os.getpid()}: Nebula connections will be cleaned up automatically")


class _NebulaSpaceSessionPool:
    def __init__(self, pool: "SessionPool"):
        self._pool = pool

    def execute(self, stmt: str):
        return self._pool.execute(stmt)

    def execute_parameter(self, stmt: str, params: dict):
        return self._pool.execute_parameter(stmt, params)

    def close(self):
        self._pool.close()


class _NebulaSpaceSession:
    def __init__(self, manager: type[NebulaSyncConnectionManager], space: str):
        self._manager = manager
        self._space = space

    def execute(self, stmt: str):
        return self._manager.execute(self._space, stmt)

    def execute_parameter(self, stmt: str, params: dict):
        return self._manager.execute_parameter(self._space, stmt, params)

    def release(self):
        return None
