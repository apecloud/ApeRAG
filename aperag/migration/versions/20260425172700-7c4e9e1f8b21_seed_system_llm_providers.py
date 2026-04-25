"""seed system LLM providers + their default models

Phase 8 task #57 — restore the system-provider seed that was dropped when
Phase 7's destructive baseline regen rebuilt ``aperag/migration/versions``
from scratch. Without this seed the post-cleanup ``llm_provider`` table is
empty on a fresh deployment, so e2e-http-provider hurl tests that ``PUT
/api/v2/providers/{name}`` to set the API key against a system-provided
``alibabacloud`` / ``openrouter`` row receive ``404 Provider not found``
(v2 ``update_llm_provider`` is strict-update only — no upsert).

The actual seed payload lives in ``aperag/migration/sql/model_configs_init.sql``
and is generated from ``models/generate_model_configs.py``. Each row uses
``ON CONFLICT (name) DO UPDATE SET ...`` so re-running this migration on a
DB that already has the providers is a no-op (or a label/base_url refresh
when the seed file is regenerated).

Revision ID: 7c4e9e1f8b21
Revises: a639b0e5515f
Create Date: 2026-04-25 17:27:00.000000
"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "7c4e9e1f8b21"
down_revision: Union[str, None] = "a639b0e5515f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Legacy provider seed retired by the model-system refactor."""
    pass


def downgrade() -> None:
    """No-op.

    The seed is idempotent and the data is product-default content (system
    providers + their model catalogs). Rolling back this migration on a DB
    that has accumulated user-supplied API keys would either lose those keys
    (if we ``DELETE`` the rows) or leave the rows in place anyway — either
    way the safer no-op matches how alembic handles other ``execute_sql_file``
    seed migrations elsewhere in the project.
    """
    pass
