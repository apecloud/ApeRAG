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
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from nebula3.Config import Config
from nebula3.gclient.net.SessionPool import SessionPool

logger = logging.getLogger(__name__)


class NebulaSyncConnectionManager:
    """
    Worker-level NebulaGraph connection manager using sync driver.
    This avoids event loop issues and provides true connection reuse across Celery tasks.
    """

    # Class-level storage for worker-scoped pool
    _session_pool: Optional[SessionPool] = None
    _lock = threading.Lock()
    _config: Optional[Dict[str, Any]] = None
    _space_schemas_created: set = set()  # Track which spaces have schemas created

    @classmethod
    def initialize(cls, config: Optional[Dict[str, Any]] = None):
        """Initialize the connection manager with configuration."""
        with cls._lock:
            if cls._session_pool is None:
                # Use provided config or environment variables
                if config:
                    cls._config = config
                else:
                    cls._config = {
                        "host": os.environ.get("NEBULA_HOST", "localhost"),
                        "port": int(os.environ.get("NEBULA_PORT", "9669")),
                        "username": os.environ.get("NEBULA_USER", "root"),
                        "password": os.environ.get("NEBULA_PASSWORD", "nebula"),
                        "max_connection_pool_size": int(os.environ.get("NEBULA_MAX_CONNECTION_POOL_SIZE", "10")),
                        "timeout": int(os.environ.get("NEBULA_TIMEOUT", "60000")),  # milliseconds
                    }

                logger.info(f"Initializing NebulaGraph sync session pool for worker {os.getpid()}")

                # Create config for connection pool
                pool_config = Config()
                pool_config.max_connection_pool_size = cls._config["max_connection_pool_size"]
                pool_config.timeout = cls._config["timeout"]

                # Create session pool
                cls._session_pool = SessionPool(
                    username=cls._config["username"],
                    password=cls._config["password"],
                    space=None,  # We'll set space dynamically
                    addresses=[(cls._config["host"], cls._config["port"])],
                )

                # Initialize the pool
                ok = cls._session_pool.init([(cls._config["host"], cls._config["port"])], pool_config)
                if not ok:
                    raise RuntimeError("Failed to initialize NebulaGraph session pool")

                logger.info(f"NebulaGraph sync session pool initialized successfully for worker {os.getpid()}")

    @classmethod
    def get_session_pool(cls) -> SessionPool:
        """Get the shared session pool instance."""
        if cls._session_pool is None:
            cls.initialize()
        return cls._session_pool

    @classmethod
    @contextmanager
    def get_session(cls, space: Optional[str] = None):
        """Get a session from the shared pool."""
        pool = cls.get_session_pool()
        session = pool.get_session(cls._config["username"], cls._config["password"])

        if session is None:
            raise RuntimeError("Failed to get session from NebulaGraph pool")

        try:
            # Switch to space if provided
            if space:
                result = session.execute(f"USE `{space}`")
                if not result.is_succeeded():
                    raise RuntimeError(f"Failed to use space {space}: {result.error_msg()}")

            yield session
        finally:
            # Return session to pool
            pool.return_session(session)

    @classmethod
    def sanitize_space_name(cls, workspace: str) -> str:
        """Sanitize workspace name to be valid NebulaGraph space name."""
        # Replace non-alphanumeric characters with underscores
        space_name = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)
        # Ensure it starts with a letter or underscore
        if space_name and not space_name[0].isalpha() and space_name[0] != "_":
            space_name = f"s_{space_name}"
        # Limit length to 64 characters (NebulaGraph limit)
        return space_name[:64]

    @classmethod
    def prepare_space(cls, workspace: str) -> str:
        """Prepare space and return space name."""
        space_name = cls.sanitize_space_name(workspace)

        # Get a session without space to create/check space
        with cls.get_session() as session:
            # Check if space exists
            result = session.execute("SHOW SPACES")
            if result.is_succeeded():
                spaces = [row.values[0].get_sVal().decode() for row in result.get_rows()]
                if space_name not in spaces:
                    # Create space with fixed string VID type
                    create_space_query = f"""
                    CREATE SPACE IF NOT EXISTS `{space_name}` (
                        partition_num=10, 
                        replica_factor=1, 
                        vid_type=FIXED_STRING(256)
                    )
                    """
                    result = session.execute(create_space_query)
                    if not result.is_succeeded():
                        raise RuntimeError(f"Failed to create space: {result.error_msg()}")

                    logger.info(f"Created NebulaGraph space: {space_name}")

                    # Wait for space to be ready
                    time.sleep(2)

                    # Mark that we need to create schema
                    cls._space_schemas_created.discard(space_name)

            # Create schema if not already created
            if space_name not in cls._space_schemas_created:
                cls._create_schema_for_space(space_name)
                cls._space_schemas_created.add(space_name)

        return space_name

    @classmethod
    def _create_schema_for_space(cls, space_name: str):
        """Create schema (TAGs and EDGE types) for the space."""
        with cls.get_session(space=space_name) as session:
            # Create base TAG for all vertices
            create_base_tag = """
            CREATE TAG IF NOT EXISTS `base` (
                entity_id string,
                entity_type string
            )
            """
            result = session.execute(create_base_tag)
            if not result.is_succeeded():
                logger.warning(f"Failed to create base TAG: {result.error_msg()}")

            # Create entity TAG (extends base conceptually)
            create_entity_tag = """
            CREATE TAG IF NOT EXISTS `entity` (
                entity_id string,
                entity_type string,
                entity_name string,
                description string,
                source_id string,
                source_chunk_id string,
                occurrence int,
                created_at int64,
                chunk_ids string
            )
            """
            result = session.execute(create_entity_tag)
            if not result.is_succeeded():
                logger.warning(f"Failed to create entity TAG: {result.error_msg()}")

            # Create DIRECTED edge type for relationships
            create_directed_edge = """
            CREATE EDGE IF NOT EXISTS `DIRECTED` (
                weight double,
                source_id string,
                description string,
                keywords string,
                source_chunk_id string,
                created_at int64
            )
            """
            result = session.execute(create_directed_edge)
            if not result.is_succeeded():
                logger.warning(f"Failed to create DIRECTED edge: {result.error_msg()}")

            # Create indexes for better query performance
            # Index on base.entity_id
            create_base_index = """
            CREATE TAG INDEX IF NOT EXISTS `base_entity_id_idx` 
            ON `base`(entity_id(64))
            """
            result = session.execute(create_base_index)
            if not result.is_succeeded():
                logger.warning(f"Failed to create base index: {result.error_msg()}")

            # Index on entity.entity_name
            create_entity_index = """
            CREATE TAG INDEX IF NOT EXISTS `entity_name_idx` 
            ON `entity`(entity_name(64))
            """
            result = session.execute(create_entity_index)
            if not result.is_succeeded():
                logger.warning(f"Failed to create entity index: {result.error_msg()}")

            # Wait for indexes to be ready
            time.sleep(2)

            logger.info(f"Schema created for space: {space_name}")

    @classmethod
    def close(cls):
        """Close the session pool and clean up resources."""
        with cls._lock:
            if cls._session_pool:
                logger.info(f"Closing NebulaGraph session pool for worker {os.getpid()}")
                cls._session_pool.close()
                cls._session_pool = None
                cls._config = None
                cls._space_schemas_created.clear()


# Celery signal handlers for worker lifecycle
def setup_worker_nebula(**kwargs):
    """Initialize NebulaGraph when worker starts."""
    NebulaSyncConnectionManager.initialize()
    logger.info(f"Worker {os.getpid()}: NebulaGraph sync connection initialized")


def cleanup_worker_nebula(**kwargs):
    """Cleanup NebulaGraph when worker shuts down."""
    NebulaSyncConnectionManager.close()
    logger.info(f"Worker {os.getpid()}: NebulaGraph sync connection closed")
