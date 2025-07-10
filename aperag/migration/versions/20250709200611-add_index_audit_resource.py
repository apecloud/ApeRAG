"""Add index to auditresource enum

Revision ID: add_index_audit_resource  
Revises: b598e645b2ba
Create Date: 2025-07-09 20:06:11.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_index_audit_resource'
down_revision: Union[str, None] = 'b598e645b2ba'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add 'index' to auditresource enum."""
    # Add the new enum value to the existing type
    op.execute(sa.text("ALTER TYPE auditresource ADD VALUE 'index'"))


def downgrade() -> None:
    """Remove 'index' from auditresource enum."""
    # Note: PostgreSQL doesn't support removing enum values directly
    # This would require recreating the type and updating all references
    # For now, we'll just log a warning
    print("WARNING: Cannot remove enum value 'index' from auditresource. Manual cleanup required.")
    pass 