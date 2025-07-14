"""Add collection summary field and SUMMARY_GENERATING status

Revision ID: add_collection_summary
Revises: 694591d5df94
Create Date: 2025-01-14 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_collection_summary'
down_revision: Union[str, None] = '694591d5df94'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add summary column to collection table
    op.add_column('collection', sa.Column('summary', sa.Text(), nullable=True, comment='LLM-generated summary'))
    
    # First, we need to modify the existing enum type to include SUMMARY_GENERATING
    # We'll use a safe approach: create new enum, alter column, drop old enum
    op.execute("ALTER TYPE collectionstatus ADD VALUE 'SUMMARY_GENERATING'")


def downgrade() -> None:
    """Downgrade schema."""
    # Remove summary column from collection table
    op.drop_column('collection', 'summary')
    
    # Note: Removing enum values is more complex and not directly supported by PostgreSQL
    # For production, you would need to:
    # 1. Create a new enum without SUMMARY_GENERATING
    # 2. Update all existing SUMMARY_GENERATING values to another status
    # 3. Alter the column to use the new enum
    # 4. Drop the old enum
    # For simplicity, we'll leave the enum value in place during downgrade
    pass 