"""add export_task table

Revision ID: a1b2c3d4e5f6
Revises: 8abbaf1aa10d
Create Date: 2026-03-04 12:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '8abbaf1aa10d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'export_task',
        sa.Column('id', sa.String(length=24), nullable=False),
        sa.Column('user', sa.String(length=256), nullable=False),
        sa.Column('collection_id', sa.String(length=24), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('progress', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('object_store_path', sa.Text(), nullable=True),
        sa.Column('file_size', sa.BigInteger(), nullable=True),
        sa.Column('gmt_created', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('gmt_updated', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('gmt_completed', sa.DateTime(timezone=True), nullable=True),
        sa.Column('gmt_expires', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_export_task')),
    )

    op.create_index(op.f('ix_export_task_user'), 'export_task', ['user'], unique=False)
    op.create_index(op.f('ix_export_task_collection_id'), 'export_task', ['collection_id'], unique=False)
    op.create_index('idx_export_task_user_status', 'export_task', ['user', 'status'], unique=False)
    op.create_index('idx_export_task_expires', 'export_task', ['status', 'gmt_expires'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_export_task_expires', table_name='export_task')
    op.drop_index('idx_export_task_user_status', table_name='export_task')
    op.drop_index(op.f('ix_export_task_collection_id'), table_name='export_task')
    op.drop_index(op.f('ix_export_task_user'), table_name='export_task')
    op.drop_table('export_task')
