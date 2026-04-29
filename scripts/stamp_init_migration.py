#!/usr/bin/env python
# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Reset ``alembic_version`` to the new initial revision after the
migration consolidation in PR #1859.

The PR squashed every prior alembic revision into a single root-only
``930bdb402fc1`` migration. Existing deployments still have
``alembic_version`` pointing at one of the deleted revisions
(e.g. ``b7c9d0e1f2a4``); ``alembic stamp`` refuses to operate on a row
referencing an unknown revision (``Can't locate revision identified by
'<old_id>'``), so the only safe path is a direct SQL update.

Usage:

    uv run python scripts/stamp_init_migration.py
    # or
    make db-stamp-init

The script is **idempotent** — re-running it once the version row has
already been stamped is a no-op. After it completes, run
``alembic check`` to confirm there is no schema drift.

This is a one-off operational helper bound to the PR #1859 cutover; it
will be deleted in a future cleanup once every known deployment has
been stamped.
"""

from __future__ import annotations

import os
import sys

INIT_REVISION = "930bdb402fc1"


def main() -> int:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        # Fall back to the per-component POSTGRES_* env vars the rest of
        # the deployment uses, so the script works under
        # docker-compose / make targets without an extra DATABASE_URL
        # plumbing.
        host = os.getenv("POSTGRES_HOST", "127.0.0.1")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        db = os.getenv("POSTGRES_DB", "postgres")
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"

    # Local import: keep the module fast to ``--help`` and avoid a hard
    # dep on SQLAlchemy when the repo is consumed for non-DB reasons.
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import ProgrammingError

    engine = create_engine(database_url, future=True)
    with engine.connect() as conn:
        # ``alembic_version`` is created the first time alembic touches
        # the DB; a truly fresh deploy does not have the table yet, in
        # which case Postgres raises ``UndefinedTable`` and SQLAlchemy
        # wraps it as ``ProgrammingError``. Convert that into a
        # human-friendly hint instead of a stack trace — the right
        # next step is ``alembic upgrade head`` to build the schema
        # from scratch.
        try:
            current = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        except ProgrammingError:
            print(
                "alembic_version table does not exist — this is a fresh database; "
                "run `alembic upgrade head` (or `make db-migrate`) to build the schema instead.",
                file=sys.stderr,
            )
            return 1
        if current is None:
            print(
                "alembic_version table is empty — this is a fresh database; "
                "run `alembic upgrade head` (or `make db-migrate`) to build the schema instead.",
                file=sys.stderr,
            )
            return 1
        if current == INIT_REVISION:
            print(f"alembic_version already at {INIT_REVISION}; no-op.")
            return 0
        print(f"stamping alembic_version: {current} → {INIT_REVISION}")
        conn.execute(
            text("UPDATE alembic_version SET version_num = :rev"),
            {"rev": INIT_REVISION},
        )
        conn.commit()
        print("done. Run `alembic check` to verify there is no schema drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
