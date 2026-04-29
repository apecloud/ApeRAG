"""drop retrieval rerank model-use scenario

Revision ID: a8f4c2d9e1b7
Revises: 930bdb402fc1
Create Date: 2026-04-30 02:40:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a8f4c2d9e1b7"
down_revision: Union[str, None] = "930bdb402fc1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove stale rerank default-model bindings before the enum is hard-cut."""
    op.execute("DELETE FROM model_use WHERE scenario = 'retrieval_rerank'")


def downgrade() -> None:
    """The deleted default-model binding cannot be reconstructed safely."""
    pass
