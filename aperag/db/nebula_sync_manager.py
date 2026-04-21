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

from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool, Session

logger = logging.getLogger(__name__)


def _safe_error_msg(result) -> str:
    """Safely extract error message from Nebula result, handling UTF-8 decode errors."""
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
    Worker-level Nebula connection manager using sync driver.

    This avoids event loop issues and provides true connection reuse across worker tasks.
    """

    _connection_pool: Optional["ConnectionPool"] = None
    _lock = threading.Lock()
    _config: Optional[Dict[str, Any]] = None
    _prepared_spaces: set[str] = set()
    _sessions: dict[int, Session] = {}
    _session_spaces: dict[int, Optional[str]] = {}

    @classmethod
    def initialize(cls, config: Optional[Dict[str, Any]] = None):
        """Initialize the connection manager with configuration."""
        with cls._lock:
            if cls._connection_pool is None:
                if config:
                    cls._config = config
                else:
                    cls._config = {
                        "host": os.environ.get("NEBULA_HOST", "127.0.0.1"),
                        "port": int(os.environ.get("NEBULA_PORT", "9669")),
                        "username": os.environ.get("NEBULA_USER", "root"),
                        "password": os.environ.get("NEBULA_PASSWORD", "nebula"),
                        "max_connection_pool_size": int(os.environ.get("NEBULA_MAX_CONNECTION_POOL_SIZE", "50")),
                        "timeout": int(os.environ.get("NEBULA_TIMEOUT", "30000")),
                    }

                logger.info(f"Initializing Nebula sync connection pool for worker {os.getpid()}")

                pool_config = Config()
                pool_config.max_connection_pool_size = cls._config["max_connection_pool_size"]
                pool_config.timeout = cls._config["timeout"]

                cls._connection_pool = ConnectionPool()
                host_port = [(cls._config["host"], cls._config["port"])]
                if not cls._connection_pool.init(host_port, pool_config):
                    raise RuntimeError("Failed to initialize Nebula connection pool")

                logger.info(f"Nebula sync connection pool initialized successfully for worker {os.getpid()}")

    @classmethod
    def get_pool(cls) -> "ConnectionPool":
        """Get the shared connection pool instance."""
        if cls._connection_pool is None:
            cls.initialize()
        return cls._connection_pool

    @classmethod
    @contextmanager
    def get_session(cls, space: Optional[str] = None) -> Session:
        """Get a thread-affine session from the shared connection pool."""
        pool = cls.get_pool()
        thread_id = threading.get_ident()

        with cls._lock:
            session = cls._sessions.get(thread_id)
            current_space = cls._session_spaces.get(thread_id)

            if session is None:
                session = pool.get_session(cls._config["username"], cls._config["password"])
                cls._sessions[thread_id] = session
                cls._session_spaces[thread_id] = None
                current_space = None

        try:
            if space and current_space != space:
                result = session.execute(f"USE {space}")
                if not result.is_succeeded():
                    raise RuntimeError(f"Failed to use space {space}: {_safe_error_msg(result)}")
                with cls._lock:
                    cls._session_spaces[thread_id] = space

            yield session
        except Exception:
            with cls._lock:
                cached_session = cls._sessions.pop(thread_id, None)
                cls._session_spaces.pop(thread_id, None)
            if cached_session is not None:
                cached_session.release()
            raise

    @classmethod
    def prepare_space(cls, workspace: str, max_wait: int = 30, fail_on_timeout: bool = True) -> str:
        """
        Prepare Nebula space and schema, returning the sanitized space name.
        """
        import time

        space_name = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)

        with cls._lock:
            if space_name in cls._prepared_spaces:
                logger.debug(f"Space {space_name} already prepared (cached)")
                return space_name

        try:
            with cls.get_session() as session:
                result = session.execute("SHOW SPACES")
                if not result.is_succeeded():
                    raise RuntimeError(f"Failed to show spaces: {_safe_error_msg(result)}")

                spaces = [row.values()[0].as_string() for row in result]
                if space_name in spaces:
                    try:
                        with cls.get_session(space=space_name) as test_session:
                            test_result = test_session.execute("SHOW TAGS")
                            if test_result.is_succeeded():
                                insert_test = test_session.execute(
                                    "INSERT VERTEX base(entity_id, entity_type) "
                                    "VALUES '__quick_test__':('__quick_test__', 'test')"
                                )
                                if insert_test.is_succeeded():
                                    test_session.execute("DELETE VERTEX '__quick_test__'")
                                    with cls._lock:
                                        cls._prepared_spaces.add(space_name)
                                    logger.info(f"Space {space_name} already exists and ready (fast path)")
                                    return space_name
                    except Exception:
                        logger.debug(f"Quick readiness test failed for space {space_name}, proceeding with full setup")
        except Exception as exc:
            logger.warning(f"Fast path check failed: {exc}, falling back to normal creation")

        with cls.get_session() as session:
            result = session.execute("SHOW SPACES")
            if not result.is_succeeded():
                raise RuntimeError(f"Failed to show spaces: {_safe_error_msg(result)}")

            spaces = [row.values()[0].as_string() for row in result]

            if space_name not in spaces:
                logger.info(f"Creating space {space_name}...")
                create_result = session.execute(
                    f"CREATE SPACE IF NOT EXISTS {space_name} "
                    f"(partition_num=10, replica_factor=1, vid_type=FIXED_STRING(256))"
                )
                if not create_result.is_succeeded():
                    raise RuntimeError(f"Failed to create space {space_name}: {_safe_error_msg(create_result)}")

                start_time = time.time()
                while time.time() - start_time < max_wait:
                    try:
                        with cls.get_session(space=space_name) as test_session:
                            ready_result = test_session.execute("SHOW TAGS")
                            if ready_result.is_succeeded():
                                logger.info(f"Space {space_name} ready after {time.time() - start_time:.1f}s")
                                break
                    except Exception:
                        pass
                    time.sleep(0.5)
                else:
                    logger.warning(f"Space {space_name} not ready after {max_wait}s, but continuing")

        with cls.get_session(space=space_name) as space_session:
            tag_result = space_session.execute(
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
                logger.warning(f"Failed to create tag: {_safe_error_msg(tag_result)}")

            edge_result = space_session.execute(
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
                logger.warning(f"Failed to create edge: {_safe_error_msg(edge_result)}")

            index_result = space_session.execute(
                "CREATE TAG INDEX IF NOT EXISTS base_entity_id_index ON base(entity_id(256))"
            )
            if not index_result.is_succeeded():
                logger.warning(f"Failed to create index: {_safe_error_msg(index_result)}")

        logger.info("Ensuring schema is fully ready...")
        time.sleep(2)

        schema_ready = False
        schema_start = time.time()
        while time.time() - schema_start < max_wait:
            try:
                with cls.get_session(space=space_name) as test_session:
                    test_result = test_session.execute(
                        "INSERT VERTEX base(entity_id, entity_type) "
                        "VALUES '__schema_test__':('__schema_test__', 'test')"
                    )
                    if test_result.is_succeeded():
                        test_session.execute("DELETE VERTEX '__schema_test__'")
                        schema_ready = True
                        logger.info(f"Schema ready after {time.time() - (schema_start - 2):.1f}s")
                        break
            except Exception:
                pass
            time.sleep(0.5)

        if not schema_ready:
            logger.warning(f"Schema may not be fully ready after {max_wait}s")
            validation_passed = False
            try:
                with cls.get_session(space=space_name) as validation_session:
                    validation_result = validation_session.execute("SHOW TAGS")
                    if validation_result.is_succeeded():
                        validation_passed = True
            except Exception as exc:
                logger.error(f"Final validation failed for space {space_name}: {exc}")

            if fail_on_timeout and not validation_passed:
                raise RuntimeError(
                    f"Schema for space {space_name} is not ready after {max_wait}s and failed validation. "
                    f"Set fail_on_timeout=False to continue with potentially incomplete schema."
                )

        with cls._lock:
            cls._prepared_spaces.add(space_name)

        logger.info(f"Space {space_name} prepared successfully")
        return space_name

    @classmethod
    def discard_space(cls, space_name: str) -> None:
        """Remove a dropped space from the local readiness cache."""
        with cls._lock:
            cls._prepared_spaces.discard(space_name)

    @classmethod
    def close(cls):
        """Close the connection pool and clean up resources."""
        with cls._lock:
            for session in cls._sessions.values():
                session.release()
            cls._sessions.clear()
            cls._session_spaces.clear()
            if cls._connection_pool:
                logger.info(f"Closing Nebula connection pool for worker {os.getpid()}")
                cls._connection_pool.close()
                cls._connection_pool = None
                cls._config = None
                cls._prepared_spaces.clear()
