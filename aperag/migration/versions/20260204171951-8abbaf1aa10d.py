"""add prompt_template table

Revision ID: 8abbaf1aa10d
Revises: 332faa764121
Create Date: 2026-02-04 17:19:51.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '8abbaf1aa10d'
down_revision: Union[str, None] = '332faa764121'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

from aperag.service.prompt_template_service import (
    APERAG_AGENT_INSTRUCTION,
    DEFAULT_AGENT_QUERY_PROMPT,
)


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'prompt_template',
        sa.Column('id', sa.String(length=24), nullable=False),
        sa.Column('prompt_type', sa.String(length=50), nullable=False),
        sa.Column('scope', sa.String(length=20), nullable=False),
        sa.Column('user_id', sa.String(length=256), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('gmt_created', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('gmt_updated', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('gmt_deleted', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_prompt_template'))
    )

    op.create_index(op.f('ix_prompt_template_prompt_type'), 'prompt_template', ['prompt_type'], unique=False)
    op.create_index(op.f('ix_prompt_template_scope'), 'prompt_template', ['scope'], unique=False)
    op.create_index(op.f('ix_prompt_template_user_id'), 'prompt_template', ['user_id'], unique=False)
    op.create_index(op.f('ix_prompt_template_gmt_deleted'), 'prompt_template', ['gmt_deleted'], unique=False)

    op.execute(f"""
        INSERT INTO prompt_template (id, prompt_type, scope, user_id, content, description, gmt_created, gmt_updated)
        VALUES
        ('pt_sys_agent_system', 'agent_system', 'system', NULL, $${APERAG_AGENT_INSTRUCTION}$$, 'System default agent system prompt', NOW(), NOW()),
        ('pt_sys_agent_query', 'agent_query', 'system', NULL, $${DEFAULT_AGENT_QUERY_PROMPT}$$, 'System default agent query prompt template', NOW(), NOW())
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_prompt_template_gmt_deleted'), table_name='prompt_template')
    op.drop_index(op.f('ix_prompt_template_user_id'), table_name='prompt_template')
    op.drop_index(op.f('ix_prompt_template_scope'), table_name='prompt_template')
    op.drop_index(op.f('ix_prompt_template_prompt_type'), table_name='prompt_template')
    op.drop_table('prompt_template')
