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

from logging.config import fileConfig

from alembic import context
from pgvector.sqlalchemy import Vector
from sqlalchemy import engine_from_config, pool

from aperag.config import settings as app_config

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# Import the models modules to register all SQLAlchemy tables. Phase 8
# Task #39 carved/deleted the last local ORMs from ``aperag.db.models``,
# which now only re-exports ``Base`` and ``random_id``/``EnumColumn``
# helpers — the per-domain model modules must now be imported explicitly
# here so their mapper classes register against ``Base.metadata`` before
# autogen runs.
import aperag.domains.agent_runtime.db.models  # noqa: F401
import aperag.domains.conversation.db.models  # noqa: F401
import aperag.domains.evaluation.db.models  # noqa: F401
import aperag.domains.governance.db.models  # noqa: F401
import aperag.domains.identity.db.models  # noqa: F401

# Wave 3 T3.1 chunk 2: ``aperag.domains.indexing.db.models`` was hard-
# deleted; the canonical ``DocumentIndex`` ORM now lives at
# ``aperag.indexing.models`` (imported a few lines down for the same
# autogen-registration reason).
import aperag.domains.knowledge_base.db.models  # noqa: F401
import aperag.domains.knowledge_graph.db.models  # noqa: F401
# Wave 7 W7-10: legacy ``aperag.domains.knowledge_graph.graphindex.models``
# (GraphIndexNode / GraphIndexEdge / GraphIndexChunk) was deleted along
# with the rest of the graphindex package; the alembic drop migration
# ``c7e3a1b9f4d6`` removes the corresponding tables. No replacement
# import is needed — ``aperag_lineage_*`` tables are autogen-discovered
# via the ``aperag.indexing.graph_storage.postgres`` module already.
import aperag.domains.marketplace.db.models  # noqa: F401
import aperag.domains.model_platform.db.models  # noqa: F401
import aperag.domains.retrieval.db.models  # noqa: F401

# Phase celery T1.1 — DocumentIndex (the new redesigned indexing state
# table) lives outside any per-domain ``db/models.py`` because the
# indexing redesign deliberately replaces the legacy
# ``aperag.domains.indexing`` Celery surface (Wave 3 hard-delete
# target). The new ``aperag.indexing.models`` is the canonical home;
# import explicitly so autogen sees the table.
import aperag.indexing.models  # noqa: F401

# Phase celery T8 chunk 4a — PostgresLineageGraphStore (T8 chunk 1
# msg=f0571f98) declares its tables on a private ``_LineageGraphBase``
# so the adapter owns ``ensure_schema`` (test/dev fallback) without
# polluting the application's main ``Base.metadata``. Production
# schema is owned by alembic, so the autogen pass needs visibility
# into both metadatas — ``target_metadata`` accepts a list so each
# ``include_object`` / ``compare_metadata`` walk visits every table.
from aperag.indexing.graph_storage.postgres import _LineageGraphBase  # noqa: F401
from aperag.db.models import Base

target_metadata = [Base.metadata, _LineageGraphBase.metadata]

# other values from the config, defined by the needs of env.py,
# can be acquired:
config.set_main_option("sqlalchemy.url", app_config.database_url)


def render_item(type_, obj, autogen_context):
    """Render item with special handling for pgvector types."""
    if type_ == "type" and isinstance(obj, Vector):
        autogen_context.imports.add("from pgvector.sqlalchemy import Vector")
        return "Vector()"
    return False


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=render_item,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, render_item=render_item)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
