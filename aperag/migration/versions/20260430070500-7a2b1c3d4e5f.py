"""extend graph curation suggestion status surface

Revision ID: 7a2b1c3d4e5f
Revises: 3c7d2f81b5e9
Create Date: 2026-04-30 07:05:00.000000

Task #31 Phase A2 reuses the existing ``graph_curation_suggestions``
table. The status column is stored as a string, so adding lifecycle
values is a code/schema change; this migration adds the new
``evidence_refs`` display field used by the review queue.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7a2b1c3d4e5f"
down_revision: Union[str, None] = "3c7d2f81b5e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add display-ready evidence refs to existing graph-curation suggestions."""
    op.add_column("graph_curation_suggestions", sa.Column("evidence_refs", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Drop the display-ready evidence refs column."""
    op.drop_column("graph_curation_suggestions", "evidence_refs")
