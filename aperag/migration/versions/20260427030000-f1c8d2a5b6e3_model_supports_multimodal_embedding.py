"""add ``supports_multimodal_embedding`` column to ``model``

Wave 5 P2 chunk 3 (per §G.2.5.1 spec amend item 3): a typed
capability flag on ``Model`` for embedding models that accept image
bytes (CLIP / Voyage Multimodal / Jina v3 / OpenAI multimodal
embeddings / etc.) and produce a single vector. Distinct from
``supports_vision`` which describes chat/completion models that
accept images as input. Drives the chunk 4b vision gate's
``EmbeddingService.is_multimodal()`` runtime check — flip on the
collection's embedder spec model and the gate self-disables.

Revision ID: f1c8d2a5b6e3
Revises: e7a3b9c2d1f6
Create Date: 2026-04-27 03:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f1c8d2a5b6e3"
down_revision: Union[str, None] = "e7a3b9c2d1f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "model",
        sa.Column(
            "supports_multimodal_embedding",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("model", "supports_multimodal_embedding")
