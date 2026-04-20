"""merge agent runtime and celery lease heads

Revision ID: c7d4e2f9b8a1
Revises: b9f1c2d3e4a5, 9f3c2d7b1a4e
Create Date: 2026-04-20 23:25:00.000000

"""

from typing import Sequence, Union

revision: str = "c7d4e2f9b8a1"
down_revision: Union[str, Sequence[str], None] = ("b9f1c2d3e4a5", "9f3c2d7b1a4e")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Merge alembic heads for a single startup path."""


def downgrade() -> None:
    """Downgrade merge revision."""
