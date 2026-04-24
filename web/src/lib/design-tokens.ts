/**
 * Design tokens — TS SSoT mirror of `src/app/globals.css`.
 *
 * Derived from the frontend-rewrite design bundle at
 * `/Users/earayu/Downloads/aperag/project/src/tokens.jsx` (SSoT).
 * Source of truth for design values consumed from TypeScript
 * (e.g. Graph entity palette, Chat trace inline styles, SVG theming).
 *
 * Do NOT duplicate these values inline inside feature components —
 * import from this module so L2 (Chat) and L3 (Graph) stay canonical.
 */

/**
 * Surface + ink palette. Prefer Tailwind utilities backed by CSS vars
 * (`bg-background`, `text-foreground`, `bg-card`, etc.) whenever possible.
 * Use this TS mirror only when Tailwind classes aren't reachable
 * (SVG fill, inline style, canvas drawing).
 */
export const COLORS = {
  bg: '#FCFBF8',
  card: '#FFFFFF',
  subtle: '#F5F3EE',
  subtleStrong: '#EDEAE2',

  fg: '#0A0A0A',
  fgSecondary: '#3A3A38',
  muted: '#6B6A65',
  mutedSoft: '#9A9893',
  placeholder: '#B8B6B0',

  border: '#EDEAE2',
  borderStrong: '#DDD9CF',

  accent: '#C96442',
  accentSoft: '#FBEDE4',
  accentInk: '#7A3320',

  ok: '#3F7D4A',
  okSoft: '#EAF2E8',
  warn: '#8B6A1A',
  warnSoft: '#F5EED7',
  danger: '#A83530',
  dangerSoft: '#F5E4E2',
  info: '#3A5A8C',
} as const;

/**
 * Knowledge-graph entity palette — muted, not saturated.
 * Consumed by L3 Graph (`collection-graph.tsx`), Inspector, Legend, and
 * any other place that needs to color an entity by its type.
 */
export const ENTITY_PALETTE = {
  person: '#8C5A3D',
  org: '#3F5E4F',
  concept: '#5A4F7A',
  doc: '#6B6A65',
  product: '#8B5A1E',
  event: '#7A3320',
} as const;

export type EntityType = keyof typeof ENTITY_PALETTE;

/**
 * Human-readable labels for entity types (zh-CN).
 * L3 Graph Legend + Inspector header use these instead of raw enum strings.
 */
export const ENTITY_LABELS_ZH: Record<EntityType, string> = {
  person: '人员',
  org: '组织',
  concept: '概念',
  doc: '文档',
  product: '设备',
  event: '事件',
};

export const ENTITY_LABELS_EN: Record<EntityType, string> = {
  person: 'Person',
  org: 'Organization',
  concept: 'Concept',
  doc: 'Document',
  product: 'Product',
  event: 'Event',
};

/**
 * Backend entity_type enum → canonical `EntityType` palette key.
 *
 * The backend emits a wider set of raw enum strings than the 6-tone
 * design palette. This map collapses the wider set onto the 6 canonical
 * keys so that callers never leak raw enum strings into the UI and
 * never have to re-implement the mapping per lane.
 *
 * Unknown / missing types fall back to `doc` (neutral ink gray).
 */
const ENTITY_TYPE_TO_PALETTE: Record<string, EntityType> = {
  person: 'person',
  organization: 'org',
  geo: 'org',
  product: 'product',
  technology: 'product',
  category: 'concept',
  date: 'event',
  event: 'event',
  UNKNOWN: 'doc',
};

/**
 * Resolve a backend `entity_type` string to the canonical palette key.
 * Safe for `null | undefined` input (returns `'doc'`).
 *
 * Usage:
 *   const color = ENTITY_PALETTE[entityTypeToPaletteKey(node.entity_type)];
 *   const label = ENTITY_LABELS_ZH[entityTypeToPaletteKey(node.entity_type)];
 */
export function entityTypeToPaletteKey(
  rawType: string | null | undefined,
): EntityType {
  if (!rawType) return 'doc';
  return ENTITY_TYPE_TO_PALETTE[rawType] ?? 'doc';
}

/**
 * Dark-mode canvas overlays. These mirror token values for imperative
 * drawing contexts (HTML canvas, inline SVG) where Tailwind classes
 * and CSS vars aren't reachable.
 *
 * Pair with the light-mode equivalents from `COLORS`:
 *   stroke: light → `COLORS.bg`     / dark → `CANVAS_DARK.nodeStroke`
 *   link normal:    light → `COLORS.border`       / dark → `CANVAS_DARK.linkNormal`
 *   link highlight: light → `COLORS.borderStrong` / dark → `CANVAS_DARK.linkHighlight`
 */
export const CANVAS_DARK = {
  nodeStroke: '#1A1A18',
  linkNormal: '#3A3A38',
  linkHighlight: '#5A5A58',
} as const;

/**
 * Radius scale (px). Maps to the Tailwind `rounded-*` utilities via
 * the `--radius*` CSS vars in `globals.css`:
 *   rounded-sm → 6px / rounded-md → 8px / rounded-lg → 10px / rounded-xl → 14px
 * Use these values only for inline SVG / canvas / non-Tailwind callsites.
 */
export const RADIUS = {
  xs: 6,
  sm: 8,
  md: 10,
  lg: 14,
  xl: 20,
  pill: 9999,
} as const;

/**
 * Elevation shadows — three restrained tiers.
 * Card shadow-sm is enough for most surfaces; larger tiers reserved
 * for floating inspectors and popovers.
 */
export const SHADOW = {
  xs: '0 1px 2px rgba(10,10,10,0.04), 0 0 0 1px rgba(10,10,10,0.04)',
  sm: '0 2px 6px rgba(10,10,10,0.05), 0 0 0 1px rgba(10,10,10,0.05)',
  md: '0 8px 28px rgba(10,10,10,0.08), 0 0 0 1px rgba(10,10,10,0.05)',
} as const;

/**
 * Font stacks. Must match `next/font/google` wiring in `src/app/layout.tsx`.
 * Prefer Tailwind `font-sans / font-serif / font-mono` utilities over
 * inlining these strings.
 */
export const FONTS = {
  sans: 'var(--font-sans), "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "HarmonyOS Sans SC", "Microsoft YaHei", system-ui, sans-serif',
  serif:
    'var(--font-serif), "Noto Serif SC", "PingFang SC", Georgia, serif',
  mono: 'var(--font-mono), ui-monospace, "SF Mono", Menlo, monospace',
} as const;

/**
 * Layout constants.
 */
export const LAYOUT = {
  sidebarWidth: 232,
  chatCenterMaxWidth: 720,
  chatRightRailWidth: 300,
} as const;
