/**
 * Adapter — translate ApeRAG `KnowledgeGraph` shape to the `points/links`
 * shape that Cosmograph (`@cosmograph/react`) consumes. Pure data
 * transform, no UI dependencies; lives outside the POC page so a future
 * production-graph page can reuse the same mapping if Cosmograph wins
 * the visual bake-off.
 *
 * The two output modes mirror the Cosmograph API surfaces:
 *
 * - `topology`  — graph mode: links carry source/target ids; layout is
 *   computed by Cosmograph's force simulation. Used for the live
 *   `/api/v2/collections/{id}/graphs` data.
 * - `semantic`  — embedding mode: points carry pre-computed `x/y` and a
 *   `cluster` index; no layout simulation. Used for the synthetic
 *   fixture (and, eventually, a backend `/embedding-map` endpoint when
 *   the POC results justify building one).
 */

import type {
  GraphEdge,
  GraphNode,
  KnowledgeGraph,
} from './types';

export type CosmographPoint = {
  id: string;
  label: string;
  /** Numeric cluster id Cosmograph uses for color quantization. */
  cluster: number;
  /** Original entity_type from the API; preserved for filter/legend UI. */
  entity_type?: string | null;
  /** Pre-computed coordinates for `semantic` mode; omitted in `topology`. */
  x?: number;
  y?: number;
  /** Forwarded so the existing detail panel can render rich content. */
  description?: string | null;
};

export type CosmographLink = {
  source: string;
  target: string;
  /** Edge weight; falls back to 1 when the API does not surface one. */
  weight?: number;
  description?: string | null;
};

export type CosmographDataset = {
  points: CosmographPoint[];
  links: CosmographLink[];
  /** Friendly cluster labels keyed by numeric `cluster` id. */
  clusterLabels: Record<number, string>;
};

/**
 * Convert a live `KnowledgeGraph` payload into Cosmograph topology mode
 * data. Cluster id is derived from `entity_type` so points sharing a
 * type land in the same color bucket — the API does not currently
 * surface a real cluster signal.
 */
export function toTopologyDataset(graph: KnowledgeGraph): CosmographDataset {
  const nodes = (graph.nodes ?? []) as GraphNode[];
  const edges = (graph.edges ?? []) as GraphEdge[];

  const typeToCluster = new Map<string, number>();
  const clusterLabels: Record<number, string> = {};

  const points: CosmographPoint[] = nodes.map((node) => {
    const entityType = (node.properties?.entity_type ?? 'unknown') as string;
    let cluster = typeToCluster.get(entityType);
    if (cluster === undefined) {
      cluster = typeToCluster.size;
      typeToCluster.set(entityType, cluster);
      clusterLabels[cluster] = entityType;
    }
    const description =
      (node.properties?.description as string | undefined) ?? null;
    return {
      id: String(node.id),
      label:
        (node.labels?.[0] as string | undefined) ??
        (node.properties?.entity_name as string | undefined) ??
        String(node.id),
      cluster,
      entity_type: entityType,
      description,
    };
  });

  const validIds = new Set(points.map((p) => p.id));
  const links: CosmographLink[] = edges
    .filter((edge) => validIds.has(String(edge.source)) && validIds.has(String(edge.target)))
    .map((edge) => ({
      source: String(edge.source),
      target: String(edge.target),
      weight: 1,
      description:
        (edge.properties?.description as string | undefined) ?? null,
    }));

  return { points, links, clusterLabels };
}

/**
 * Generate a deterministic synthetic semantic map at the requested
 * scale. Points are scattered into `clusterCount` Gaussian blobs across
 * a unit square. No links are produced — semantic mode is point-cloud
 * only, mirroring the arXiv-map reference visual.
 *
 * Seeded with `seed` (default 1) so the same scale renders identically
 * across reloads, which keeps performance benchmarking honest.
 */
export function buildSyntheticSemanticDataset(
  pointCount: number,
  options: { seed?: number; clusterCount?: number } = {},
): CosmographDataset {
  const seed = options.seed ?? 1;
  const clusterCount = options.clusterCount ?? 8;

  // Mulberry32 — small deterministic PRNG; avoids pulling a dep just for fixtures.
  const prng = (() => {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  })();

  const clusterCenters = Array.from({ length: clusterCount }, (_, k) => {
    const angle = (k / clusterCount) * Math.PI * 2;
    return {
      cx: Math.cos(angle) * 0.6,
      cy: Math.sin(angle) * 0.6,
    };
  });

  const clusterLabels: Record<number, string> = {};
  for (let k = 0; k < clusterCount; k += 1) {
    clusterLabels[k] = `Cluster ${k}`;
  }

  const points: CosmographPoint[] = Array.from({ length: pointCount }, (_, i) => {
    const cluster = i % clusterCount;
    const center = clusterCenters[cluster];
    // Box-Muller for a 2D Gaussian — gives the visually pleasing dense
    // cluster cores with sparse outliers that the reference visual has.
    const u1 = Math.max(prng(), 1e-9);
    const u2 = prng();
    const radius = Math.sqrt(-2 * Math.log(u1)) * 0.06;
    const theta = u2 * Math.PI * 2;
    return {
      id: `synthetic-${i}`,
      label: `Document ${i}`,
      cluster,
      x: center.cx + radius * Math.cos(theta),
      y: center.cy + radius * Math.sin(theta),
      description: `Synthetic point ${i} in ${clusterLabels[cluster]} — fixture data for FPS bench.`,
    };
  });

  return { points, links: [], clusterLabels };
}
