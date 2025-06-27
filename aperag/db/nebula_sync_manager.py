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
        # Ensure the error message is properly handled
        if isinstance(error_msg, bytes):
            # Try different encodings
            for encoding in ["utf-8", "gbk", "latin-1"]:
                try:
                    return error_msg.decode(encoding)
                except UnicodeDecodeError:
                    continue
            # If all fail, use replacement characters
            return error_msg.decode("utf-8", errors="replace")
        elif isinstance(error_msg, str):
            return error_msg
        else:
            return str(error_msg)
    except Exception as e:
        logger.warning(f"Failed to get Nebula error message: {e}")
        return f"Nebula operation failed (error code: {result.error_code()})"


class NebulaSyncConnectionManager:
    """
    Worker-level Nebula connection manager using sync driver.
    This avoids event loop issues and provides true connection reuse across Celery tasks.
    """

    # Class-level storage for worker-scoped connection pool
    _connection_pool: Optional["ConnectionPool"] = None
    _lock = threading.Lock()
    _config: Optional[Dict[str, Any]] = None

    @classmethod
    def initialize(cls, config: Optional[Dict[str, Any]] = None):
        """Initialize the connection manager with configuration."""
        with cls._lock:
            if cls._connection_pool is None:
                # Check if nebula3-python is installed
                if ConnectionPool is None:
                    raise RuntimeError(
                        "nebula3-python is not installed. Please install it with: pip install nebula3-python"
                    )

                # Use provided config or environment variables
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

                # Create connection pool
                cls._connection_pool = ConnectionPool()

                # Initialize connection pool with single host and port
                host_port = [(cls._config["host"], cls._config["port"])]

                # Initialize connection pool
                if not cls._connection_pool.init(host_port, Config()):
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
        """Get a session from the shared connection pool."""
        pool = cls.get_pool()
        session = pool.get_session(cls._config["username"], cls._config["password"])

        try:
            # Set space if provided
            if space:
                result = session.execute(f"USE {space}")
                if not result.is_succeeded():
                    raise RuntimeError(f"Failed to use space {space}: {_safe_error_msg(result)}")

            yield session
        finally:
            session.release()

    @classmethod
    def prepare_space(cls, workspace: str) -> str:
        """Prepare space and return space name."""
        import time

        # Sanitize workspace name for Nebula (only alphanumeric and underscore allowed)
        space_name = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)

        # Check if space exists
        with cls.get_session() as session:
            result = session.execute("SHOW SPACES")
            if not result.is_succeeded():
                raise RuntimeError(f"Failed to show spaces: {_safe_error_msg(result)}")

            spaces = []
            for row in result:
                spaces.append(row.values()[0].as_string())

            if space_name not in spaces:
                logger.info(f"Space {space_name} not found, creating...")

                # Create space with fixed string vid type
                create_result = session.execute(
                    f"CREATE SPACE IF NOT EXISTS {space_name} "
                    f"(partition_num=10, replica_factor=1, vid_type=FIXED_STRING(256))"
                )
                if not create_result.is_succeeded():
                    raise RuntimeError(f"Failed to create space {space_name}: {_safe_error_msg(create_result)}")

                # Wait for space to be ready with fast polling
                max_wait = 30  # Maximum wait time
                start_time = time.time()

                while time.time() - start_time < max_wait:
                    try:
                        with cls.get_session(space=space_name) as test_session:
                            result = test_session.execute("SHOW TAGS")
                            if result.is_succeeded():
                                logger.info(f"Space {space_name} is ready after {time.time() - start_time:.1f}s")
                                break
                    except Exception:
                        pass
                    time.sleep(1)  # Check every second
                else:
                    logger.warning(f"Space {space_name} not ready after {max_wait}s, continuing anyway")

        # Create tags and edges in the space
        with cls.get_session(space=space_name) as space_session:
            # Create base tag for nodes
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

            # Create DIRECTED edge for relationships
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

            # Create indexes
            index_result = space_session.execute(
                "CREATE TAG INDEX IF NOT EXISTS base_entity_id_index ON base(entity_id(256))"
            )
            if not index_result.is_succeeded():
                logger.warning(f"Failed to create index: {_safe_error_msg(index_result)}")

        # Wait for schema to take effect - Nebula docs: wait 2 heartbeat cycles (20s)
        # But we can test early to see if it's ready sooner
        logger.info("Waiting for schema to take effect (Nebula requires ~20 seconds)...")

        # First wait a minimum time to let basic schema creation complete
        time.sleep(5)

        # Then test if schema is actually usable by trying to insert test data
        schema_ready = False
        max_schema_wait = 20
        schema_start = time.time()

        while time.time() - schema_start < max_schema_wait:
            try:
                with cls.get_session(space=space_name) as test_session:
                    # Test if we can actually use the schema
                    test_result = test_session.execute(
                        "INSERT VERTEX base(entity_id, entity_type) VALUES '__schema_test__':('__schema_test__', 'test')"
                    )
                    if test_result.is_succeeded():
                        # Clean up test data
                        test_session.execute("DELETE VERTEX '__schema_test__'")
                        elapsed = time.time() - (schema_start - 5)  # Include initial 5s wait
                        logger.info(f"Schema ready after {elapsed:.1f}s")
                        schema_ready = True
                        break
            except Exception:
                pass
            time.sleep(1)  # Check every second

        if not schema_ready:
            logger.warning("Schema may not be fully ready, but continuing anyway")

        logger.info(f"Space {space_name} created and ready")
        return space_name

    @classmethod
    def close(cls):
        """Close the connection pool and clean up resources."""
        with cls._lock:
            if cls._connection_pool:
                logger.info(f"Closing Nebula connection pool for worker {os.getpid()}")
                cls._connection_pool.close()
                cls._connection_pool = None
                cls._config = None


# Celery signal handlers for worker lifecycle
def setup_worker_nebula(**kwargs):
    """Initialize Nebula when worker starts."""
    NebulaSyncConnectionManager.initialize()
    logger.info(f"Worker {os.getpid()}: Nebula sync connection initialized")


def cleanup_worker_nebula(**kwargs):
    """Cleanup Nebula when worker shuts down."""
    NebulaSyncConnectionManager.close()
    logger.info(f"Worker {os.getpid()}: Nebula sync connection closed")
