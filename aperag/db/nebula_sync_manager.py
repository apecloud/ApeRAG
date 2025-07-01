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
    Nebula connection manager using sync driver with lazy loading.

    This manager provides Worker/Process-level connection reuse through lazy initialization.
    Connections are created on-demand when first used, avoiding unnecessary resource allocation.

    Key features:
    - Lazy loading: connections created only when needed
    - Worker-level reuse: same connection pool shared across all tasks in a worker
    - Thread-safe: uses threading.Lock for initialization
    - Space caching: prepared spaces cached to avoid repeated setup
    - Automatic cleanup: connections closed when process exits
    """

    # Class-level storage for worker-scoped connection pool
    _connection_pool: Optional["ConnectionPool"] = None
    _lock = threading.Lock()
    _config: Optional[Dict[str, Any]] = None
    # Cache for prepared spaces to avoid repeated checks
    _prepared_spaces: set = set()

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
    def prepare_space(cls, workspace: str, max_wait: int = 30, fail_on_timeout: bool = True) -> str:
        """
        Fast space preparation with optimized waiting strategy for improved performance.

        Args:
            workspace: Workspace name to prepare
            max_wait: Maximum time to wait for space readiness (seconds)
            fail_on_timeout: If True, raise exception when schema is not ready after max_wait. If False, log warning and continue.

        Returns:
            Space name

        This optimized version:
        1. Fast-checks if space already exists and is usable
        2. Reduced default wait times compared to original 55s
        3. More aggressive ready detection with 0.5s polling
        4. Always ensures schema readiness for reliable operation
        5. Caches prepared spaces to avoid repeated checks
        6. Configurable timeout behavior for different use cases
        """
        import time

        # Sanitize workspace name for Nebula (only alphanumeric and underscore allowed)
        space_name = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)

        # Ultra-fast path: If we've already prepared this space, return immediately
        with cls._lock:
            if space_name in cls._prepared_spaces:
                logger.debug(f"Space {space_name} already prepared (cached)")
                return space_name

        # Fast path: Check if space exists and is ready
        try:
            with cls.get_session() as session:
                result = session.execute("SHOW SPACES")
                if not result.is_succeeded():
                    raise RuntimeError(f"Failed to show spaces: {_safe_error_msg(result)}")

                spaces = []
                for row in result:
                    spaces.append(row.values()[0].as_string())

                # If space exists, do a quick readiness check
                if space_name in spaces:
                    try:
                        with cls.get_session(space=space_name) as test_session:
                            # Quick test: can we query the schema?
                            test_result = test_session.execute("SHOW TAGS")
                            if test_result.is_succeeded():
                                # Additional quick test: can we use the schema?
                                insert_test = test_session.execute(
                                    "INSERT VERTEX base(entity_id, entity_type) VALUES '__quick_test__':('__quick_test__', 'test')"
                                )
                                if insert_test.is_succeeded():
                                    # Clean up and return immediately
                                    test_session.execute("DELETE VERTEX '__quick_test__'")
                                    logger.info(f"Space {space_name} already exists and ready (fast path)")
                                    # Cache this space as prepared
                                    with cls._lock:
                                        cls._prepared_spaces.add(space_name)
                                    return space_name
                    except Exception:
                        # If quick test fails, continue with normal creation flow
                        logger.debug(f"Quick readiness test failed for space {space_name}, proceeding with full setup")

        except Exception as e:
            logger.warning(f"Fast path check failed: {e}, falling back to normal creation")

        # Normal path: Create or ensure space is properly set up
        with cls.get_session() as session:
            result = session.execute("SHOW SPACES")
            if not result.is_succeeded():
                raise RuntimeError(f"Failed to show spaces: {_safe_error_msg(result)}")

            spaces = []
            for row in result:
                spaces.append(row.values()[0].as_string())

            space_needs_creation = space_name not in spaces

            if space_needs_creation:
                logger.info(f"Creating space {space_name}...")
                # Create space with fixed string vid type
                create_result = session.execute(
                    f"CREATE SPACE IF NOT EXISTS {space_name} "
                    f"(partition_num=10, replica_factor=1, vid_type=FIXED_STRING(256))"
                )
                if not create_result.is_succeeded():
                    raise RuntimeError(f"Failed to create space {space_name}: {_safe_error_msg(create_result)}")

                # Optimized wait for space to be ready - much shorter wait time
                start_time = time.time()
                while time.time() - start_time < max_wait:
                    try:
                        with cls.get_session(space=space_name) as test_session:
                            result = test_session.execute("SHOW TAGS")
                            if result.is_succeeded():
                                logger.info(f"Space {space_name} ready after {time.time() - start_time:.1f}s")
                                break
                    except Exception:
                        pass
                    time.sleep(0.5)  # More frequent polling for faster detection
                else:
                    logger.warning(f"Space {space_name} not ready after {max_wait}s, but continuing")

        # Create or ensure schema exists
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

        # Always ensure schema readiness (formerly controlled by ensure_schema_ready parameter)
        logger.info("Ensuring schema is fully ready...")
        # Initial wait to let basic schema creation complete
        time.sleep(2)

        schema_ready = False
        max_schema_wait = max_wait  # Use the same max_wait for consistency
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
                        elapsed = time.time() - (schema_start - 2)  # Include initial 2s wait
                        logger.info(f"Schema ready after {elapsed:.1f}s")
                        schema_ready = True
                        break
            except Exception:
                pass
            time.sleep(0.5)  # More frequent testing

        if not schema_ready:
            logger.warning(f"Schema may not be fully ready after {max_schema_wait}s")
            # Additional validation: try one more time to use the space
            validation_passed = False
            try:
                with cls.get_session(space=space_name) as validation_session:
                    result = validation_session.execute("SHOW TAGS")
                    if result.is_succeeded():
                        validation_passed = True
                    else:
                        logger.error(f"Space {space_name} is not usable even after timeout - this may cause issues")
            except Exception as e:
                logger.error(f"Final validation failed for space {space_name}: {e}")

            # If fail_on_timeout is enabled and validation failed, raise exception
            if fail_on_timeout and not validation_passed:
                raise RuntimeError(
                    f"Schema for space {space_name} is not ready after {max_schema_wait}s and failed validation. "
                    f"Set fail_on_timeout=False to allow proceeding with potentially incomplete schema."
                )
            elif fail_on_timeout:
                logger.warning("Schema validation passed but schema readiness test failed - continuing")

        logger.info(f"Space {space_name} prepared successfully")

        # Cache this space as prepared
        with cls._lock:
            cls._prepared_spaces.add(space_name)

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


# Legacy Celery signal handlers (now unused with lazy loading)
# These functions are kept for backward compatibility but are no longer used
def setup_worker_nebula(**kwargs):
    """Legacy function - Nebula now uses lazy loading instead of worker signals."""
    logger.info(f"Worker {os.getpid()}: Nebula connection will be initialized on-demand (lazy loading)")


def cleanup_worker_nebula(**kwargs):
    """Legacy function - Nebula cleanup happens automatically on process exit."""
    logger.info(f"Worker {os.getpid()}: Nebula connections will be cleaned up automatically")
