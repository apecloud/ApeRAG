// Shared types between the lab page, the nav, and the renderers.
// Pure module — kept separate so the server-component page.tsx can stay
// import-light (no client-only references).

export type LabMode = 'cosmograph-topology' | 'cosmograph-semantic';

export const DEFAULT_LAB_MODE: LabMode = 'cosmograph-topology';

export const SEMANTIC_SCALE_PRESETS = [1000, 5000, 10000] as const;
export type SemanticScale = (typeof SEMANTIC_SCALE_PRESETS)[number];
