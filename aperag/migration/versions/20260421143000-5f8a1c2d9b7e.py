"""refresh agent prompt system defaults

Revision ID: 5f8a1c2d9b7e
Revises: c7d4e2f9b8a1
Create Date: 2026-04-21 14:30:00.000000

"""

from typing import Sequence, Union

from alembic import op

from aperag.service.prompt_template_service import (
    APERAG_AGENT_INSTRUCTION,
    DEFAULT_AGENT_QUERY_PROMPT,
)

revision: str = "5f8a1c2d9b7e"
down_revision: Union[str, None] = "c7d4e2f9b8a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        f"""
        UPDATE prompt_template
        SET
            content = $${APERAG_AGENT_INSTRUCTION}$$,
            description = 'System default agent system prompt',
            gmt_updated = NOW(),
            gmt_deleted = NULL
        WHERE prompt_type = 'agent_system'
          AND scope = 'system'
          AND user_id IS NULL;

        INSERT INTO prompt_template (id, prompt_type, scope, user_id, content, description, gmt_created, gmt_updated)
        SELECT
            'pt_sys_agent_system',
            'agent_system',
            'system',
            NULL,
            $${APERAG_AGENT_INSTRUCTION}$$,
            'System default agent system prompt',
            NOW(),
            NOW()
        WHERE NOT EXISTS (
            SELECT 1
            FROM prompt_template
            WHERE prompt_type = 'agent_system'
              AND scope = 'system'
              AND user_id IS NULL
        );

        UPDATE prompt_template
        SET
            content = $${DEFAULT_AGENT_QUERY_PROMPT}$$,
            description = 'System default agent query prompt template',
            gmt_updated = NOW(),
            gmt_deleted = NULL
        WHERE prompt_type = 'agent_query'
          AND scope = 'system'
          AND user_id IS NULL;

        INSERT INTO prompt_template (id, prompt_type, scope, user_id, content, description, gmt_created, gmt_updated)
        SELECT
            'pt_sys_agent_query',
            'agent_query',
            'system',
            NULL,
            $${DEFAULT_AGENT_QUERY_PROMPT}$$,
            'System default agent query prompt template',
            NOW(),
            NOW()
        WHERE NOT EXISTS (
            SELECT 1
            FROM prompt_template
            WHERE prompt_type = 'agent_query'
              AND scope = 'system'
              AND user_id IS NULL
        );
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
