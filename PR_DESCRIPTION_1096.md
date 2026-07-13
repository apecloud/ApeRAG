# Summary
This PR updates marketplace collection graph pages to use marketplace-safe graph APIs and improves graph page layout behavior on shorter screens.

## What changed
- Render the marketplace collection `/graph` page with the new graph-hybrid view.
- Add read-only marketplace graph endpoints for `embedding-map` and `entity-search` so published marketplace collections do not call workspace-only graph APIs.
- Let `CollectionGraphHybrid` switch between workspace and marketplace data sources.
- Give graph pages a 720px minimum graph area so shorter screens can scroll instead of compressing the canvas.

## Why
Marketplace collection graph pages should not depend on workspace-only APIs. This change introduces marketplace-safe read-only graph endpoints and updates frontend routing/data-source behavior so graph rendering works correctly for published collections while preserving workspace behavior.

## Validation
- `make openapi-check`
- `uv run ruff check aperag/domains/marketplace/api/routes.py`
- `yarn type-check`
- `yarn lint` (passes with pre-existing unrelated warnings)
