"""drop rerank capability from model rows

Revision ID: 3c7d2f81b5e9
Revises: a8f4c2d9e1b7
Create Date: 2026-04-30 03:46:00.000000

Task #47 fix-forward to task #35: the previous migration
``a8f4c2d9e1b7`` removed the ``retrieval_rerank`` ``model_use`` rows;
this one removes the underlying ``model.capability='rerank'`` rows so
the next ``ModelCapability`` enum hard-cut (which drops the
``"rerank"`` value entirely) does not fail to deserialise existing
rows.
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3c7d2f81b5e9"
down_revision: Union[str, None] = "a8f4c2d9e1b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove model rows with rerank capability before the enum hard-cut."""
    op.execute("DELETE FROM model WHERE capability = 'rerank'")


def downgrade() -> None:
    """The deleted rerank model rows cannot be reconstructed safely."""
    pass
