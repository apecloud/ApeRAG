"""add ``Bot.is_system`` + summary bot infrastructure (Wave 10 §K.13)

Wave 10 §K.13 — collection summary regen needs a per-user hidden
"summary bot" to drive Stage 1 agent-runtime free-explore. Schema
changes (per design doc PR #1790):

1. ``Bot.is_system: Boolean default False`` — mirrors existing
   ``ApiKey.is_system`` precedent. UI listings filter out system bots.
2. Partial unique index ``(user, type, is_system)`` over active
   (``gmt_deleted IS NULL``) ``is_system=TRUE`` rows — defends
   against race conditions during register-time creation + lazy
   fallback create.
3. Backfill: insert one ``type='summary', is_system=TRUE`` row for
   every existing user that doesn't already have one. Existing users
   then have a summary bot ready when they next request a regen
   (no first-call latency).

The ``BotType.SUMMARY`` enum value is a Python-only addition; the DB
column is already ``VARCHAR(50)`` so storing the new value needs
no DDL change.

Revision ID: f2c3d4e5b6a8
Revises: e1a2b3c4d5f6
Create Date: 2026-04-28 07:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f2c3d4e5b6a8"
down_revision: Union[str, None] = "e1a2b3c4d5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. New column
    op.add_column(
        "bot",
        sa.Column(
            "is_system",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
    )
    op.create_index("ix_bot_is_system", "bot", ["is_system"], unique=False)

    # 2. Partial unique index for system bots
    op.create_index(
        "uq_bot_user_type_system_active",
        "bot",
        ["user", "type", "is_system"],
        unique=True,
        postgresql_where=sa.text("gmt_deleted IS NULL AND is_system = TRUE"),
    )

    # 3. Backfill: one summary bot per existing user that doesn't
    # already have one. Uses ``substr(md5(...), 1, 16)`` to mint a
    # 16-char random id matching the application's ``_random_id``
    # output shape (the ``bot`` prefix is added by the SELECT).
    op.execute(
        """
        INSERT INTO bot (
            id, "user", title, type, description, status, config,
            is_system, gmt_created, gmt_updated
        )
        SELECT
            'bot' || substr(md5(random()::text || clock_timestamp()::text || u.id), 1, 16),
            u.id,
            'Summary Generation Bot',
            'summary',
            'System-managed bot for collection summary regen (Wave 10).',
            'ACTIVE',
            '{"agent": {"system_prompt_template": null}}',
            TRUE,
            NOW(),
            NOW()
        FROM "user" u
        WHERE NOT EXISTS (
            SELECT 1 FROM bot b
            WHERE b.user = u.id
              AND b.type = 'summary'
              AND b.is_system = TRUE
              AND b.gmt_deleted IS NULL
        );
        """
    )


def downgrade() -> None:
    # Drop backfilled rows first (no-op if upgrade hadn't run on prod
    # — the WHERE clause is precise enough to leave non-system bots
    # alone).
    op.execute(
        """
        DELETE FROM bot
        WHERE type = 'summary'
          AND is_system = TRUE;
        """
    )
    op.drop_index("uq_bot_user_type_system_active", table_name="bot")
    op.drop_index("ix_bot_is_system", table_name="bot")
    op.drop_column("bot", "is_system")
