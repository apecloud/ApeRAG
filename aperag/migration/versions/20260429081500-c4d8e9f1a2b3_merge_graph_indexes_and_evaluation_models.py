"""merge graph index and evaluation model migration heads

Revision ID: c4d8e9f1a2b3
Revises: a1b2c3d4e5f7, a9f4e2c7d6b1
Create Date: 2026-04-29 08:15:00.000000
"""

from __future__ import annotations

from typing import Sequence

revision = "c4d8e9f1a2b3"
down_revision = ("a1b2c3d4e5f7", "a9f4e2c7d6b1")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
