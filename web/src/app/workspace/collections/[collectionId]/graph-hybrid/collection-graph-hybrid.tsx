'use client';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/skeleton';
import {
  getGraphHybrid,
  getMarketplaceGraphHybrid,
  searchMarketplaceGraphEntities,
  searchGraphEntities,
} from '@/features/knowledge-graph/client-api';
import type {
  GraphEdge,
  GraphHybridNode,
  GraphSearchEntity,
} from '@/features/knowledge-graph/types';
import { ApiClientError } from '@/lib/api/typed/errors';
import { CANVAS_DARK, COLORS } from '@/lib/design-tokens';
import { cn } from '@/lib/utils';
import {
  ArrowRight,
  ChevronDown,
  ChevronUp,
  ChevronsLeft,
  ChevronsRight,
  GitBranch,
  Loader2,
  Maximize2,
  Minimize2,
  Route,
  Search,
  X,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useTheme } from 'next-themes';
import dynamic from 'next/dynamic';
import { useParams } from 'next/navigation';
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

const ForceGraph2D = dynamic(
  () => import('react-force-graph-2d').then((r) => r),
  { ssr: false },
);

// Hybrid-mode cluster palette — kept local to this file so the
// production /graph entity-type palette stays untouched. Wraps modulo so
// large cluster counts cycle through the palette rather than overflowing.
const HYBRID_CLUSTER_PALETTE = [
  '#3F7F95',
  '#D97757',
  '#7C5DA0',
  '#5BA374',
  '#E0A040',
  '#C44A6E',
  '#4F6D9A',
  '#B07A4F',
  '#5C7C8A',
  '#8B6E2E',
] as const;

const pickClusterColor = (cluster: number) => {
  const len = HYBRID_CLUSTER_PALETTE.length;
  return HYBRID_CLUSTER_PALETTE[((cluster % len) + len) % len];
};

const IMPORTANT_LABEL_SIZE = 13;
const INITIAL_FIT_PADDING = 0;
const INITIAL_FIT_ZOOM_BOOST = 1.22;
const LABEL_RADIUS = 5;
const NODE_MIN = 8;
const NODE_MAX = 22;
const CLUSTER_FILTER_COLLAPSED_LIMIT = 24;
const RESIZE_EPSILON_PX = 2;

const getErrorMessage = (error: unknown, fallback: string) => {
  if (error instanceof ApiClientError || error instanceof Error) {
    return error.message || fallback;
  }
  return fallback;
};

const roundedRect = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) => {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
};

const getNodeSize = (node: unknown) => {
  const value =
    typeof node === 'object' && node !== null && 'value' in node
      ? Number(node.value)
      : 0;
  return Math.min(Number.isFinite(value) ? value : 0, NODE_MAX);
};

const endpointId = (endpoint: unknown) => {
  if (typeof endpoint === 'object' && endpoint !== null && 'id' in endpoint) {
    return String(endpoint.id);
  }
  return String(endpoint || '');
};

// PCA-positioned hybrid node — extends the backend hybrid DTO with the
// d3-force pinned coordinates. fx/fy are
// the d3-force "pinned" coordinates, which short-circuits the
// simulation so the visual reflects the real PCA projection.
type HybridNode = GraphHybridNode & {
  fx: number;
  fy: number;
};

type GraphCamera = {
  x: number;
  y: number;
  zoom: number;
};

export const CollectionGraphHybrid = ({
  marketplace = false,
}: {
  marketplace?: boolean;
}) => {
  const params = useParams();
  const [fullscreen, setFullscreen] = useState<boolean>(false);
  const { resolvedTheme } = useTheme();
  const page_graph = useTranslations('page_graph');

  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);
  const didInitialFitRef = useRef(false);
  const cameraRef = useRef<GraphCamera | null>(null);
  const userInteractedRef = useRef(false);
  const [graphData, setGraphData] = useState<{
    nodes: HybridNode[];
    links: GraphEdge[];
  }>();
  const [graphError, setGraphError] = useState<string>();
  const [clusterLabels, setClusterLabels] = useState<Record<number, string>>(
    {},
  );

  // Keep the canvas at 0×0 until the real graph container exists. The
  // graph is conditionally rendered only after useLayoutEffect measures
  // it, avoiding the old 800×600 fallback that could persist until
  // fullscreen triggered a second measurement.
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  const [activeClusters, setActiveClusters] = useState<number[]>([]);

  const [highlightNodes, setHighlightNodes] = useState<Set<HybridNode>>(
    () => new Set(),
  );
  const [highlightLinks, setHighlightLinks] = useState<Set<GraphEdge>>(
    () => new Set(),
  );
  const [hoverNode, setHoverNode] = useState<HybridNode>();
  const [activeNode, setActiveNode] = useState<HybridNode>();
  const [activeEdge, setActiveEdge] = useState<GraphEdge>();
  const [detailCollapsed, setDetailCollapsed] = useState(false);
  const [clusterFilterExpanded, setClusterFilterExpanded] = useState(false);

  // Path mode — pick two nodes and surface every shortest path between
  // them. Cap at 12 paths so hub-heavy graphs don't explode the
  // highlight set.
  const [interactionMode, setInteractionMode] = useState<'entity' | 'path'>(
    'entity',
  );
  const [pathPicks, setPathPicks] = useState<string[]>([]);
  const [pathPaths, setPathPaths] = useState<string[][]>([]);

  // Vector search — opens an inline input + result overlay when a query
  // is typed. Debounced so we don't hammer the search endpoint.
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState<GraphSearchEntity[]>([]);
  const [searchPending, setSearchPending] = useState(false);

  const getGraphData = useCallback(async () => {
    if (typeof params.collectionId !== 'string') return;
    didInitialFitRef.current = false;
    userInteractedRef.current = false;
    cameraRef.current = null;
    setGraphError(undefined);

    try {
      const hybrid = marketplace
        ? await getMarketplaceGraphHybrid(params.collectionId, 1000)
        : await getGraphHybrid(params.collectionId, 1000);

      if (!hybrid || hybrid.nodes.length === 0) {
        setGraphData({ nodes: [], links: [] });
        setClusterLabels({});
        return;
      }

      const nodes: HybridNode[] = hybrid.nodes.map((node) => ({
        ...node,
        value: Math.max(NODE_MIN, Math.min(node.value, NODE_MAX)),
        fx: node.x,
        fy: node.y,
      }));
      const links: GraphEdge[] = hybrid.edges ?? [];

      setGraphData({ nodes, links });

      const labels: Record<number, string> = {};
      for (const [k, v] of Object.entries(hybrid.cluster_labels)) {
        labels[Number(k)] = v;
      }
      setClusterLabels(labels);
      setActiveClusters(Object.keys(labels).map(Number));
      setClusterFilterExpanded(false);
    } catch (error: unknown) {
      setGraphData({ nodes: [], links: [] });
      setClusterLabels({});
      setActiveClusters([]);
      setClusterFilterExpanded(false);
      setGraphError(getErrorMessage(error, page_graph('load_failed')));
    }
  }, [marketplace, page_graph, params.collectionId]);

  const handleCloseDetail = useCallback(() => {
    setActiveNode(undefined);
    setActiveEdge(undefined);
    setHoverNode(undefined);
    setDetailCollapsed(false);
    setHighlightNodes(new Set());
    setHighlightLinks(new Set());
  }, []);

  // BFS — every shortest path between two node ids. Tracks `parents`
  // for each node at the discovered distance so the enumeration walks
  // the parent DAG and yields each distinct path once. Capped at
  // MAX_PATHS so hub-heavy graphs don't explode the highlight set.
  const bfsAllShortestPaths = useCallback(
    (startId: string, endId: string): string[][] => {
      if (!graphData) return [];
      if (startId === endId) return [[startId]];
      const adjacency = new Map<string, string[]>();
      for (const link of graphData.links) {
        const a = endpointId(link.source);
        const b = endpointId(link.target);
        if (!adjacency.has(a)) adjacency.set(a, []);
        if (!adjacency.has(b)) adjacency.set(b, []);
        adjacency.get(a)!.push(b);
        adjacency.get(b)!.push(a);
      }
      const dist = new Map<string, number>();
      const parents = new Map<string, string[]>();
      dist.set(startId, 0);
      const queue: string[] = [startId];
      let qHead = 0;
      let foundDist: number | null = null;
      while (qHead < queue.length) {
        const cur = queue[qHead];
        qHead += 1;
        const curDist = dist.get(cur)!;
        if (foundDist !== null && curDist >= foundDist) continue;
        for (const nb of adjacency.get(cur) ?? []) {
          if (!dist.has(nb)) {
            dist.set(nb, curDist + 1);
            parents.set(nb, [cur]);
            if (nb === endId) {
              foundDist = curDist + 1;
            } else {
              queue.push(nb);
            }
          } else if (dist.get(nb) === curDist + 1) {
            parents.get(nb)!.push(cur);
          }
        }
      }
      if (!parents.has(endId)) return [];
      const MAX_PATHS = 12;
      const paths: string[][] = [];
      const enumerate = (node: string, suffix: string[]) => {
        if (paths.length >= MAX_PATHS) return;
        if (node === startId) {
          paths.push([startId, ...suffix]);
          return;
        }
        for (const p of parents.get(node) ?? []) {
          if (paths.length >= MAX_PATHS) return;
          enumerate(p, [node, ...suffix]);
        }
      };
      enumerate(endId, []);
      return paths;
    },
    [graphData],
  );

  const handleResizeContainer = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const width = Math.round(rect.width);
    const height = Math.round(rect.height);
    setDimensions((prev) => {
      if (prev.width === width && prev.height === height) return prev;
      if (
        prev.width > 0 &&
        prev.height > 0 &&
        Math.abs(prev.width - width) <= RESIZE_EPSILON_PX &&
        Math.abs(prev.height - height) <= RESIZE_EPSILON_PX
      ) {
        return prev;
      }
      return { width, height };
    });
  }, []);

  const readCamera = useCallback((): GraphCamera | null => {
    const graph = graphRef.current;
    if (!graph?.centerAt || !graph?.zoom) return null;
    const center = graph.centerAt();
    const zoom = graph.zoom();
    if (
      !center ||
      !Number.isFinite(center.x) ||
      !Number.isFinite(center.y) ||
      !Number.isFinite(zoom)
    ) {
      return null;
    }
    return { x: center.x, y: center.y, zoom };
  }, []);

  const rememberCamera = useCallback(() => {
    const camera = readCamera();
    if (camera) cameraRef.current = camera;
    return camera;
  }, [readCamera]);

  const restoreCamera = useCallback((camera: GraphCamera | null) => {
    const graph = graphRef.current;
    if (!graph?.centerAt || !graph?.zoom || !camera) return;
    graph.centerAt(camera.x, camera.y, 0);
    graph.zoom(camera.zoom, 0);
  }, []);

  // Measure synchronously after layout (`useLayoutEffect`) so the
  // first paint already has the right canvas dimensions. Plain
  // `useEffect` runs after paint, leaving a one-frame gap where
  // ForceGraph2D ships a 0×0 canvas → blank screen until the next
  // resize event.
  useLayoutEffect(() => {
    if (!graphData) return;
    handleResizeContainer();
    const frame = requestAnimationFrame(handleResizeContainer);
    return () => cancelAnimationFrame(frame);
  }, [graphData, handleResizeContainer, fullscreen]);

  // Track ongoing layout changes (split-panel resize, sidebar
  // collapse, etc.) — `window.resize` alone misses container-only
  // size changes that don't change the viewport.
  useEffect(() => {
    if (!graphData) return;
    const container = containerRef.current;
    if (!container) return;
    handleResizeContainer();
    const ro = new ResizeObserver(() => handleResizeContainer());
    ro.observe(container);
    window.addEventListener('resize', handleResizeContainer);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', handleResizeContainer);
    };
  }, [graphData, handleResizeContainer]);

  // Pre-index links by both endpoint orders so the path-mode highlight
  // effect can locate the edge between consecutive path nodes in O(1).
  const linksByPair = useMemo(() => {
    const m = new Map<string, GraphEdge>();
    for (const l of graphData?.links ?? []) {
      const a = endpointId(l.source);
      const b = endpointId(l.target);
      m.set(`${a}::${b}`, l);
      m.set(`${b}::${a}`, l);
    }
    return m;
  }, [graphData?.links]);

  // O(1) node lookup by id — avoids the per-frame `nodes.find()` calls
  // that used to make linkVisibility / link painting O(E·N) at 60 FPS.
  const nodesById = useMemo(() => {
    const m = new Map<string, HybridNode>();
    for (const n of graphData?.nodes ?? []) m.set(n.id, n);
    return m;
  }, [graphData?.nodes]);

  // Undirected adjacency list for BFS (path search + connected
  // component preview). Memoised on the graph data so we don't rebuild
  // the map on every state change inside the highlight effect.
  const adjacency = useMemo(() => {
    const m = new Map<string, string[]>();
    for (const link of graphData?.links ?? []) {
      const a = endpointId(link.source);
      const b = endpointId(link.target);
      let arrA = m.get(a);
      if (!arrA) {
        arrA = [];
        m.set(a, arrA);
      }
      arrA.push(b);
      let arrB = m.get(b);
      if (!arrB) {
        arrB = [];
        m.set(b, arrB);
      }
      arrB.push(a);
    }
    return m;
  }, [graphData?.links]);

  useEffect(() => {
    const nextHighlightNodes = new Set<HybridNode>();
    const nextHighlightLinks = new Set<GraphEdge>();

    // Highlight only — no centerAt/zoom. Per user feedback, the canvas
    // should never animate the viewport on selection; the side panel
    // surfaces the data and the user controls pan/zoom themselves.
    if (pathPaths.length > 0) {
      for (const p of pathPaths) {
        for (let i = 0; i < p.length; i += 1) {
          const node = nodesById.get(p[i]);
          if (node) nextHighlightNodes.add(node);
          if (i + 1 < p.length) {
            const edge = linksByPair.get(`${p[i]}::${p[i + 1]}`);
            if (edge) nextHighlightLinks.add(edge);
          }
        }
      }
    } else if (
      interactionMode === 'path' &&
      pathPicks.length === 1 &&
      graphData
    ) {
      // First-pick preview — BFS the connected component reachable
      // from the start node (KG is undirected). Uses the precomputed
      // `adjacency` Map and an index-based queue so this is O(V+E).
      const startId = pathPicks[0];
      const seen = new Set<string>([startId]);
      const queue: string[] = [startId];
      let qHead = 0;
      while (qHead < queue.length) {
        const cur = queue[qHead];
        qHead += 1;
        for (const nb of adjacency.get(cur) ?? []) {
          if (seen.has(nb)) continue;
          seen.add(nb);
          queue.push(nb);
        }
      }
      for (const id of seen) {
        const n = nodesById.get(id);
        if (n) nextHighlightNodes.add(n);
      }
      for (const link of graphData.links) {
        if (
          seen.has(endpointId(link.source)) &&
          seen.has(endpointId(link.target))
        ) {
          nextHighlightLinks.add(link);
        }
      }
    } else if (activeNode) {
      // Walk only the active node's adjacency list — O(deg) instead
      // of scanning every edge in the graph.
      for (const nbId of adjacency.get(activeNode.id) ?? []) {
        const edge = linksByPair.get(`${activeNode.id}::${nbId}`);
        if (edge) nextHighlightLinks.add(edge);
        const nb = nodesById.get(nbId);
        if (nb) nextHighlightNodes.add(nb);
      }
      nextHighlightNodes.add(activeNode);
    } else if (activeEdge) {
      const sId = endpointId(activeEdge.source);
      const tId = endpointId(activeEdge.target);
      const s = nodesById.get(sId);
      const t = nodesById.get(tId);
      nextHighlightLinks.add(activeEdge);
      if (s) nextHighlightNodes.add(s);
      if (t) nextHighlightNodes.add(t);
    }
    setHighlightNodes(nextHighlightNodes);
    setHighlightLinks(nextHighlightLinks);
  }, [
    activeEdge,
    activeNode,
    adjacency,
    graphData,
    interactionMode,
    linksByPair,
    nodesById,
    pathPaths,
    pathPicks,
  ]);

  // ForceGraph may internally adjust its camera when canvas dimensions
  // or paint callbacks update. During node/link/path interactions, keep
  // the user's current camera fixed so selection never nudges the whole
  // map. Real viewport/fullscreen resizes still update dimensions; this
  // guard only restores after interaction state changes.
  useLayoutEffect(() => {
    if (!didInitialFitRef.current) return;
    restoreCamera(cameraRef.current);
  }, [
    activeClusters,
    activeEdge,
    activeNode,
    highlightLinks,
    highlightNodes,
    hoverNode,
    interactionMode,
    pathPaths,
    pathPicks,
    restoreCamera,
  ]);

  useEffect(() => {
    getGraphData();
  }, [getGraphData]);

  // Debounced vector search against the Wave 7 entity-search endpoint.
  useEffect(() => {
    if (!searchOpen || !searchTerm.trim()) {
      setSearchResults([]);
      return;
    }
    if (typeof params.collectionId !== 'string') return;
    const cid = params.collectionId;
    let cancelled = false;
    setSearchPending(true);
    const handle = setTimeout(() => {
      const search = marketplace
        ? searchMarketplaceGraphEntities
        : searchGraphEntities;
      search(cid, searchTerm.trim(), 12)
        .then((results) => {
          if (cancelled) return;
          setSearchResults(results);
          setSearchPending(false);
        })
        .catch(() => {
          if (cancelled) return;
          setSearchResults([]);
          setSearchPending(false);
        });
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [marketplace, params.collectionId, searchOpen, searchTerm]);

  // Disable every d3-force so the simulation can't pull nodes — the
  // PCA layout already places everything via fx/fy. Without this the
  // default charge/center forces still tick once after each render,
  // visibly drifting the cloud whenever React re-renders (every
  // selection click triggers it).
  useEffect(() => {
    if (!graphRef.current || !graphData?.nodes.length) return;
    const ref = graphRef.current;
    ref.d3Force?.('center', null);
    ref.d3Force?.('charge', null);
    ref.d3Force?.('link', null);
    ref.d3Force?.('collide', null);
  }, [graphData]);

  // First-fit only: when data lands AND the canvas has real dimensions,
  // fit once. Subsequent clicks/highlights don't re-fit, so selection
  // never changes the user's viewport.
  useEffect(() => {
    if (didInitialFitRef.current) return;
    if (!graphData?.nodes.length) return;
    if (dimensions.width === 0 || dimensions.height === 0) return;

    const fit = () => {
      if (didInitialFitRef.current) return;
      if (userInteractedRef.current) {
        didInitialFitRef.current = true;
        return;
      }
      const graph = graphRef.current;
      if (!graph?.centerAt || !graph?.zoom) return;

      const xs = graphData.nodes.map((n) => n.x).filter(Number.isFinite);
      const ys = graphData.nodes.map((n) => n.y).filter(Number.isFinite);
      if (!xs.length || !ys.length) return;

      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const rangeX = Math.max(maxX - minX, 1);
      const rangeY = Math.max(maxY - minY, 1);
      const padding = INITIAL_FIT_PADDING;
      const zoom = Math.max(
        0.05,
        Math.min(
          2,
          Math.min(
            Math.max(dimensions.width - padding * 2, 1) / rangeX,
            Math.max(dimensions.height - padding * 2, 1) / rangeY,
          ) * INITIAL_FIT_ZOOM_BOOST,
        ),
      );
      const x = (minX + maxX) / 2;
      const y = (minY + maxY) / 2;
      graph.centerAt(x, y, 0);
      graph.zoom(zoom, 0);
      cameraRef.current = { x, y, zoom };
      didInitialFitRef.current = true;
    };

    const timers = [0, 80, 200, 500].map((delay) => setTimeout(fit, delay));
    return () => timers.forEach(clearTimeout);
  }, [graphData?.nodes, dimensions.width, dimensions.height]);

  const isDark = resolvedTheme === 'dark';
  const nodeStroke = isDark ? CANVAS_DARK.nodeStroke : COLORS.bg;
  const linkNormal = isDark ? CANVAS_DARK.linkNormal : COLORS.border;
  const linkHighlight = isDark
    ? CANVAS_DARK.linkHighlight
    : COLORS.borderStrong;
  const labelFill = isDark ? COLORS.bg : COLORS.fg;

  const totalNodes = graphData?.nodes.length ?? 0;
  const totalEdges = graphData?.links.length ?? 0;

  const activeNodeEdges = useMemo(
    () =>
      activeNode
        ? graphData?.links.filter(
            (l) =>
              endpointId(l.source) === activeNode.id ||
              endpointId(l.target) === activeNode.id,
          ) || []
        : [],
    [activeNode, graphData?.links],
  );

  const activeNodeNeighbors = useMemo(() => {
    if (!activeNode || !graphData?.nodes.length) return [];
    const neighborIds = new Set(
      activeNodeEdges
        .map((e) => {
          const sId = endpointId(e.source);
          const tId = endpointId(e.target);
          return sId === activeNode.id ? tId : sId;
        })
        .filter(Boolean),
    );
    return graphData.nodes.filter((n) => neighborIds.has(n.id));
  }, [activeNode, activeNodeEdges, graphData?.nodes]);

  const allClusterIds = Object.keys(clusterLabels).map(Number);
  const hasOverflowClusters =
    allClusterIds.length > CLUSTER_FILTER_COLLAPSED_LIMIT;
  const visibleClusterIds =
    clusterFilterExpanded || !hasOverflowClusters
      ? allClusterIds
      : allClusterIds.slice(0, CLUSTER_FILTER_COLLAPSED_LIMIT);
  const hiddenClusterCount = Math.max(
    allClusterIds.length - visibleClusterIds.length,
    0,
  );
  const allActive =
    allClusterIds.length > 0 && activeClusters.length === allClusterIds.length;

  const nodeVisibility = useCallback(
    (node: unknown) => {
      const cluster = (node as HybridNode).cluster;
      return activeClusters.length === 0 || activeClusters.includes(cluster);
    },
    [activeClusters],
  );

  const linkVisibility = useCallback(
    (link: unknown) => {
      const edge = link as GraphEdge;
      const sId = endpointId(edge.source);
      const tId = endpointId(edge.target);
      const s = nodesById.get(sId);
      const t = nodesById.get(tId);
      if (!s || !t) return false;
      return (
        activeClusters.length === 0 ||
        (activeClusters.includes(s.cluster) &&
          activeClusters.includes(t.cluster))
      );
    },
    [activeClusters, nodesById],
  );

  if (graphError && !graphData) {
    return (
      <div className="text-destructive flex h-full items-center justify-center px-4 text-center text-sm">
        {graphError}
      </div>
    );
  }

  if (!graphData) {
    return (
      <div className="h-full min-h-0 overflow-hidden">
        <Skeleton className="h-full w-full" />
      </div>
    );
  }

  // Single full-width graph surface. Details float over the canvas so
  // long entity descriptions cannot resize the graph viewport.
  return (
    <div
      className={cn(
        'relative top-0 right-0 bottom-0 left-0 h-full min-h-0 flex-1 overflow-hidden',
        {
          fixed: fullscreen,
          'bg-background': fullscreen,
          'z-49': fullscreen,
          'p-2': fullscreen,
        },
      )}
    >
      <div className="bg-card relative flex h-full min-h-0 flex-col overflow-hidden rounded-lg border">
        {/* Top toolbar — Hybrid badge + counts on the left, mode +
            fullscreen + search controls on the right. */}
        <div className="flex items-center justify-between gap-2 border-b px-3 py-2">
          <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
            <span className="bg-muted/60 rounded-md px-1.5 py-0.5">
              {page_graph('hybrid_entity_count', {
                count: totalNodes.toLocaleString(),
              })}
            </span>
            <span className="bg-muted/60 rounded-md px-1.5 py-0.5">
              {page_graph('hybrid_relation_count', {
                count: totalEdges.toLocaleString(),
              })}
            </span>
            {/* Path-mode status — inlined alongside the counts so the
                hint never takes a separate row. */}
            {interactionMode === 'path' && (
              <span className="text-[11px]">
                {pathPicks.length === 0 && pathPaths.length === 0 && (
                  <span className="text-muted-foreground">
                    · 路径模式: 点击第一个节点作为起点
                  </span>
                )}
                {pathPicks.length === 1 && (
                  <span className="text-foreground">
                    · 起点: <span className="font-medium">{pathPicks[0]}</span>{' '}
                    · 点击第二个节点找路径
                  </span>
                )}
                {pathPaths.length > 0 && (
                  <span className="text-foreground">
                    ·{' '}
                    {pathPaths.length > 1
                      ? `${pathPaths.length} 条最短路径`
                      : '最短路径'}{' '}
                    · {pathPaths[0].length - 1} 跳
                    {pathPaths.length >= 12 && (
                      <span className="text-muted-foreground">
                        {' '}
                        (capped at 12)
                      </span>
                    )}
                  </span>
                )}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="bg-muted flex items-center gap-0.5 rounded-md p-0.5">
              <Button
                size="sm"
                variant={interactionMode === 'entity' ? 'default' : 'ghost'}
                onClick={() => {
                  setInteractionMode('entity');
                  setPathPicks([]);
                  setPathPaths([]);
                }}
                className="h-6 px-2 text-[10px]"
              >
                实体
              </Button>
              <Button
                size="sm"
                variant={interactionMode === 'path' ? 'default' : 'ghost'}
                onClick={() => {
                  setInteractionMode('path');
                  setActiveNode(undefined);
                  setActiveEdge(undefined);
                  setPathPicks([]);
                  setPathPaths([]);
                }}
                className="h-6 px-2 text-[10px]"
              >
                <Route className="mr-1 size-3" /> 路径
              </Button>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setFullscreen(!fullscreen)}
              className="h-7 px-2 text-xs"
              title={fullscreen ? '退出全屏' : '全屏'}
            >
              {fullscreen ? (
                <Minimize2 className="size-3" />
              ) : (
                <Maximize2 className="size-3" />
              )}
            </Button>
            {!searchOpen ? (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setSearchOpen(true)}
                className="h-7 px-2 text-xs"
              >
                <Search className="mr-1 size-3" /> 搜索实体
              </Button>
            ) : (
              <div className="relative">
                <Search className="text-muted-foreground absolute top-1.5 left-2 size-3" />
                <Input
                  autoFocus
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="向量搜索 …"
                  className="h-7 w-56 pr-6 pl-7 text-xs"
                />
                <button
                  type="button"
                  className="text-muted-foreground hover:text-foreground absolute top-1.5 right-1.5"
                  onClick={() => {
                    setSearchOpen(false);
                    setSearchTerm('');
                  }}
                >
                  <X className="size-3" />
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Cluster filter chips — collapsed by default so high-cardinality
            type sets don't push the canvas down. */}
        {allClusterIds.length > 0 && (
          <div className="flex flex-wrap items-center gap-1 border-b px-3 py-2">
            <span className="text-muted-foreground mr-1 text-[10px] tracking-wider uppercase">
              类型
            </span>
            <Button
              size="sm"
              variant={allActive ? 'default' : 'outline'}
              className="h-6 px-2 text-[10px]"
              onClick={() => setActiveClusters(allActive ? [] : allClusterIds)}
            >
              全选
            </Button>
            {visibleClusterIds.map((c) => {
              const enabled = activeClusters.includes(c);
              const label = clusterLabels[c] || `Cluster ${c}`;
              return (
                <Button
                  key={c}
                  size="sm"
                  variant={enabled ? 'default' : 'outline'}
                  className="h-6 px-2 text-[10px]"
                  style={
                    enabled
                      ? {
                          backgroundColor: pickClusterColor(c),
                          color: COLORS.bg,
                          borderColor: 'transparent',
                        }
                      : undefined
                  }
                  onClick={() =>
                    setActiveClusters((prev) =>
                      enabled ? prev.filter((x) => x !== c) : prev.concat(c),
                    )
                  }
                >
                  {label}
                </Button>
              );
            })}
            {hasOverflowClusters && (
              <Button
                size="sm"
                variant="ghost"
                className="text-muted-foreground hover:text-foreground h-6 px-2 text-[10px]"
                onClick={() => setClusterFilterExpanded((prev) => !prev)}
              >
                {clusterFilterExpanded ? (
                  <>
                    <ChevronUp className="mr-1 size-3" /> 收起
                  </>
                ) : (
                  <>
                    <ChevronDown className="mr-1 size-3" /> 展开{' '}
                    {hiddenClusterCount} 个
                  </>
                )}
              </Button>
            )}
          </div>
        )}

        {/* Canvas + overlays */}
        <div ref={containerRef} className="relative min-h-0 flex-1">
          {graphData?.nodes.length === 0 && !graphError && (
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center">
              <div className="text-muted-foreground text-sm">
                {page_graph('no_nodes_found')}
              </div>
            </div>
          )}

          {dimensions.width > 0 && dimensions.height > 0 && (
            <ForceGraph2D
              graphData={graphData}
              width={dimensions.width}
              height={dimensions.height}
              nodeLabel={(nod) => String(nod.id)}
              ref={graphRef}
              backgroundColor="transparent"
              cooldownTicks={0}
              warmupTicks={0}
              d3AlphaDecay={1}
              enableNodeDrag={false}
              nodeVisibility={nodeVisibility}
              onZoom={() => {
                userInteractedRef.current = true;
                rememberCamera();
              }}
              onZoomEnd={() => {
                userInteractedRef.current = true;
                rememberCamera();
              }}
              onBackgroundClick={() => {
                // Click on empty canvas — dismiss any current
                // selection (entity / edge / path pick / path result)
                // so the user can return to the overview without
                // hunting for an X button.
                userInteractedRef.current = true;
                rememberCamera();
                if (interactionMode === 'path') {
                  setPathPicks([]);
                  setPathPaths([]);
                }
                handleCloseDetail();
              }}
              onNodeClick={(node) => {
                userInteractedRef.current = true;
                rememberCamera();
                const n = node as HybridNode;
                if (interactionMode === 'path') {
                  setActiveNode(undefined);
                  setActiveEdge(undefined);
                  setPathPicks((prev) => {
                    const next =
                      prev[0] === undefined ? [n.id] : [prev[0], n.id];
                    if (next.length === 2) {
                      const paths = bfsAllShortestPaths(next[0], next[1]);
                      setPathPaths(paths);
                      return [];
                    }
                    setPathPaths([]);
                    return next;
                  });
                  return;
                }
                if (activeNode?.id === n.id) {
                  handleCloseDetail();
                  return;
                }
                setActiveEdge(undefined);
                setActiveNode(n);
              }}
              onNodeHover={(node) => {
                if (
                  interactionMode === 'path' ||
                  activeNode ||
                  activeEdge ||
                  pathPaths.length > 0
                )
                  return;
                const nextHighlightNodes = new Set<HybridNode>();
                const nextHighlightLinks = new Set<GraphEdge>();
                if (node) {
                  const n = node as HybridNode;
                  for (const nbId of adjacency.get(n.id) ?? []) {
                    const edge = linksByPair.get(`${n.id}::${nbId}`);
                    if (edge) nextHighlightLinks.add(edge);
                  }
                }
                setHoverNode(node ? (node as HybridNode) : undefined);
                setHighlightNodes(nextHighlightNodes);
                setHighlightLinks(nextHighlightLinks);
              }}
              onLinkHover={(link) => {
                if (
                  interactionMode === 'path' ||
                  activeNode ||
                  activeEdge ||
                  pathPaths.length > 0
                )
                  return;
                const nextHighlightLinks = new Set<GraphEdge>();
                if (link) nextHighlightLinks.add(link as GraphEdge);
                setHighlightNodes(new Set());
                setHighlightLinks(nextHighlightLinks);
              }}
              onLinkClick={(link) => {
                userInteractedRef.current = true;
                rememberCamera();
                if (interactionMode === 'path') return;
                if (activeEdge?.id === link.id) {
                  handleCloseDetail();
                  return;
                }
                setActiveNode(undefined);
                setActiveEdge(link as GraphEdge);
              }}
              nodeCanvasObject={(node, ctx, globalScale) => {
                const n = node as HybridNode;
                const x = n.x ?? 0;
                const y = n.y ?? 0;

                let size = Math.min(n.value, NODE_MAX);
                if (n === hoverNode) size += 1;

                const fillColor = pickClusterColor(n.cluster);
                const isDim = highlightNodes.size > 0 && !highlightNodes.has(n);
                const isActive = activeNode?.id === n.id;
                // Path mode: highlight the picked start node the same
                // way as a single click selection so the user always
                // sees which node is "anchored".
                const isPathStart =
                  interactionMode === 'path' &&
                  pathPicks.length === 1 &&
                  pathPicks[0] === n.id;
                const isSelected = isActive || isPathStart;
                // Zoom-aware label gating — at default zoom (~1) only
                // the biggest hubs get labels; as the user zooms in the
                // threshold drops so more labels appear progressively.
                // Clamp the divisor so very high zoom levels show
                // labels for every node and very zoomed-out views
                // suppress them aggressively.
                const labelSizeThreshold =
                  IMPORTANT_LABEL_SIZE / Math.max(globalScale, 0.5);
                const shouldShowLabel =
                  isSelected ||
                  n === hoverNode ||
                  highlightNodes.has(n) ||
                  (!isDim && size >= labelSizeThreshold);

                // Non-highlighted nodes recede so the active selection
                // or path stands out clearly.
                ctx.beginPath();
                ctx.arc(x, y, size, 0, 2 * Math.PI, false);
                ctx.fillStyle = fillColor;
                ctx.globalAlpha = isDim ? 0.12 : 1;
                ctx.fill();
                ctx.globalAlpha = 1;

                ctx.beginPath();
                ctx.arc(x, y, size, 0, 2 * Math.PI, false);
                ctx.lineWidth = 0.6;
                ctx.strokeStyle = nodeStroke;
                ctx.globalAlpha = isDim ? 0.12 : 1;
                ctx.stroke();
                ctx.globalAlpha = 1;

                // Selection halo — paints a soft white ring around any
                // currently-selected node (entity click or path-mode
                // start pick). Sized just outside the fill so it reads
                // as an outer outline without occluding the colour.
                if (isSelected) {
                  const haloOuter = size + 4;
                  ctx.beginPath();
                  ctx.arc(x, y, haloOuter, 0, 2 * Math.PI, false);
                  ctx.lineWidth = 2.5;
                  ctx.strokeStyle = isDark
                    ? 'rgba(255, 255, 255, 0.95)'
                    : 'rgba(255, 255, 255, 0.98)';
                  ctx.globalAlpha = 1;
                  ctx.stroke();
                  // Soft outer glow for extra emphasis.
                  ctx.beginPath();
                  ctx.arc(x, y, haloOuter + 2, 0, 2 * Math.PI, false);
                  ctx.lineWidth = 1;
                  ctx.strokeStyle = isDark
                    ? 'rgba(255, 255, 255, 0.35)'
                    : 'rgba(255, 255, 255, 0.55)';
                  ctx.stroke();
                }

                if (shouldShowLabel) {
                  let fontSize = 13;
                  const offset = 2;
                  const fontFamily =
                    'var(--font-sans), Manrope, system-ui, sans-serif';
                  ctx.font = `500 ${fontSize}px ${fontFamily}`;
                  let textWidth = ctx.measureText(String(n.id)).width - offset;
                  while (textWidth > size * 1.6 && fontSize > 1) {
                    fontSize -= 1;
                    ctx.font = `500 ${fontSize}px ${fontFamily}`;
                    textWidth = ctx.measureText(String(n.id)).width - offset;
                  }
                  const labelX = x - (textWidth + offset) / 2;
                  const labelY = y + size + fontSize + 2;
                  const labelPaddingX = 4;
                  const labelHeight = fontSize + 7;
                  roundedRect(
                    ctx,
                    labelX - labelPaddingX,
                    labelY - fontSize - 4,
                    textWidth + offset + labelPaddingX * 2,
                    labelHeight,
                    LABEL_RADIUS,
                  );
                  ctx.fillStyle = isDark
                    ? 'rgba(17, 24, 39, 0.72)'
                    : 'rgba(255, 255, 255, 0.78)';
                  ctx.globalAlpha = isDim ? 0.12 : 1;
                  ctx.fill();
                  ctx.fillStyle = labelFill;
                  ctx.globalAlpha = isDim ? 0.15 : 1;
                  ctx.fillText(String(n.id), labelX, labelY);
                  ctx.globalAlpha = 1;
                }
              }}
              nodePointerAreaPaint={(node, color, ctx) => {
                const n = node as HybridNode;
                const x = n.x ?? 0;
                const y = n.y ?? 0;
                const size = Math.min(getNodeSize(n), NODE_MAX);
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(x, y, size, 0, 2 * Math.PI, false);
                ctx.fill();
              }}
              linkLabel="id"
              linkColor={(link) => {
                const edge = link as GraphEdge;
                if (highlightLinks.has(edge)) return linkHighlight;
                // When something is highlighted, fade non-participating
                // edges far down so the foreground reads cleanly.
                if (highlightLinks.size > 0)
                  return isDark
                    ? 'rgba(140, 140, 150, 0.10)'
                    : 'rgba(120, 120, 120, 0.08)';
                return linkNormal;
              }}
              linkWidth={(link) => {
                const edge = link as GraphEdge;
                if (highlightLinks.has(edge)) return 1.6;
                return highlightLinks.size > 0 ? 0.4 : 0.8;
              }}
              linkDirectionalParticleWidth={(link) =>
                highlightLinks.has(link as GraphEdge) ? 2.5 : 0
              }
              linkDirectionalParticleSpeed={0.006}
              linkDirectionalParticles={(link) =>
                highlightLinks.has(link as GraphEdge) ? 2 : 0
              }
              linkVisibility={linkVisibility}
            />
          )}

          {/* Search results overlay */}
          {searchOpen && searchTerm.trim() && (
            <div className="bg-card absolute top-2 right-2 z-10 max-h-72 w-72 overflow-y-auto rounded-md border shadow-lg">
              <div className="text-muted-foreground border-b px-3 py-1.5 text-[10px] tracking-wider uppercase">
                {searchPending ? '搜索中…' : `${searchResults.length} 个匹配`}
              </div>
              {searchPending && (
                <div className="text-muted-foreground flex items-center gap-2 p-3 text-xs">
                  <Loader2 className="size-3 animate-spin" /> 向量召回中…
                </div>
              )}
              {!searchPending &&
                searchResults.map((r) => {
                  const n = nodesById.get(r.name);
                  return (
                    <button
                      key={r.name}
                      type="button"
                      className="hover:bg-accent flex w-full flex-col items-start gap-0.5 px-3 py-2 text-left text-xs"
                      onClick={() => {
                        if (n) {
                          setActiveEdge(undefined);
                          setActiveNode(n);
                        }
                        setSearchOpen(false);
                        setSearchTerm('');
                      }}
                    >
                      <div className="font-medium">{r.name}</div>
                      {r.entity_type && (
                        <div className="text-muted-foreground text-[10px]">
                          {r.entity_type}
                        </div>
                      )}
                      {r.description && (
                        <div className="text-muted-foreground line-clamp-2 text-[10px]">
                          {r.description}
                        </div>
                      )}
                    </button>
                  );
                })}
              {!searchPending && searchResults.length === 0 && (
                <div className="text-muted-foreground p-3 text-xs">无结果</div>
              )}
            </div>
          )}
          {(activeNode || activeEdge || pathPaths.length > 0) && (
            <HybridFloatingDetail
              collapsed={detailCollapsed}
              interactionMode={interactionMode}
              activeNode={activeNode}
              activeEdge={activeEdge}
              pathPaths={pathPaths}
              activeNodeEdges={activeNodeEdges}
              activeNodeNeighbors={activeNodeNeighbors}
              nodes={graphData?.nodes ?? []}
              linksByPair={linksByPair}
              onClose={handleCloseDetail}
              onCollapsedChange={setDetailCollapsed}
              onSelectNode={(nodeId) => {
                const match = nodesById.get(nodeId);
                if (!match) return;
                setDetailCollapsed(false);
                setActiveEdge(undefined);
                setActiveNode(match);
                if (interactionMode === 'path') {
                  setInteractionMode('entity');
                  setPathPicks([]);
                  setPathPaths([]);
                }
              }}
              onSelectEdge={(edge) => {
                setDetailCollapsed(false);
                setActiveNode(undefined);
                setActiveEdge(edge);
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
};

const HybridFloatingDetail = ({
  collapsed,
  interactionMode,
  activeNode,
  activeEdge,
  pathPaths,
  activeNodeEdges,
  activeNodeNeighbors,
  nodes,
  linksByPair,
  onClose,
  onCollapsedChange,
  onSelectNode,
  onSelectEdge,
}: {
  collapsed: boolean;
  interactionMode: 'entity' | 'path';
  activeNode?: HybridNode;
  activeEdge?: GraphEdge;
  pathPaths: string[][];
  activeNodeEdges: GraphEdge[];
  activeNodeNeighbors: HybridNode[];
  nodes: HybridNode[];
  linksByPair: Map<string, GraphEdge>;
  onClose: () => void;
  onCollapsedChange: (collapsed: boolean) => void;
  onSelectNode: (nodeId: string) => void;
  onSelectEdge: (edge: GraphEdge) => void;
}) => {
  const showPaths = interactionMode === 'path' && pathPaths.length > 0;
  const nodesById = useMemo(() => {
    const m = new Map<string, HybridNode>();
    for (const n of nodes) m.set(n.id, n);
    return m;
  }, [nodes]);
  const title = showPaths
    ? `${pathPaths.length} 条路径`
    : activeNode?.id ||
      (activeEdge
        ? `${endpointId(activeEdge.source)} → ${endpointId(activeEdge.target)}`
        : '详情');

  if (collapsed) {
    return (
      <button
        type="button"
        className="bg-card/95 hover:bg-card absolute top-3 right-3 z-20 flex max-w-72 items-center gap-2 rounded-lg border px-3 py-2 text-left text-xs shadow-lg backdrop-blur-md transition-colors"
        onClick={() => onCollapsedChange(false)}
        title="展开详情"
      >
        <ChevronsLeft className="text-muted-foreground size-4" />
        <span className="min-w-0 flex-1 truncate font-medium">{title}</span>
      </button>
    );
  }

  return (
    <aside className="bg-card/95 absolute top-3 right-3 bottom-3 z-20 flex w-[22rem] max-w-[calc(100%-1.5rem)] flex-col overflow-hidden rounded-lg border shadow-xl backdrop-blur-md">
      <div className="flex shrink-0 items-center justify-between gap-2 border-b px-3 py-2">
        <div className="min-w-0 truncate text-sm font-medium">{title}</div>
        <div className="flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-7"
            onClick={() => onCollapsedChange(true)}
            title="折叠详情"
          >
            <ChevronsRight className="size-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="size-7"
            onClick={onClose}
            title="关闭详情"
          >
            <X className="size-4" />
          </Button>
        </div>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto p-5">
        {showPaths && (
          <PathListContent
            paths={pathPaths}
            nodesById={nodesById}
            linksByPair={linksByPair}
            onSelectNode={onSelectNode}
          />
        )}

        {!showPaths && activeNode && (
          <EntityContent
            node={activeNode}
            edges={activeNodeEdges}
            neighbors={activeNodeNeighbors}
            onSelectNode={onSelectNode}
            onSelectEdge={onSelectEdge}
          />
        )}

        {!showPaths && !activeNode && activeEdge && (
          <EdgeContent
            edge={activeEdge}
            nodesById={nodesById}
            onSelectNode={onSelectNode}
          />
        )}

        {!showPaths && !activeNode && !activeEdge && (
          <p className="text-muted-foreground text-xs">
            {interactionMode === 'path'
              ? '路径模式: 点击两个节点查看它们之间的最短路径。'
              : '点击画布上的节点或边查看详情。'}
          </p>
        )}
      </div>
    </aside>
  );
};

const EntityTypeBadge = ({
  entityType,
  color,
}: {
  entityType?: string | null;
  color?: string;
}) => (
  <div className="flex items-center gap-2">
    <span
      className="size-2 shrink-0 rounded-full"
      style={{ backgroundColor: color ?? COLORS.border }}
    />
    <span className="text-muted-foreground font-mono text-[10px] tracking-wider uppercase">
      {entityType || 'UNKNOWN'}
    </span>
  </div>
);

const InfoTile = ({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) => (
  <div className="bg-muted/45 rounded-lg px-3 py-2">
    <div className="text-muted-foreground text-[11px]">{label}</div>
    <div className="mt-1 text-sm font-medium tabular-nums">{value}</div>
  </div>
);

const EntityContent = ({
  node,
  edges,
  neighbors,
  onSelectNode,
  onSelectEdge,
}: {
  node: HybridNode;
  edges: GraphEdge[];
  neighbors: HybridNode[];
  onSelectNode: (nodeId: string) => void;
  onSelectEdge: (edge: GraphEdge) => void;
}) => {
  const description =
    (node.properties.description as string | undefined) ?? null;
  const entityType =
    (node.properties.entity_type as string | undefined) ?? null;
  const sourceChunkCount =
    (node.properties.source_chunk_count as number | undefined) ?? 0;

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-col gap-1.5 border-b pb-4">
        <EntityTypeBadge
          entityType={entityType}
          color={pickClusterColor(node.cluster)}
        />
        <h3 className="font-serif text-xl leading-tight font-normal tracking-tight break-words">
          {node.id}
        </h3>
      </header>

      <section className="space-y-2">
        <div className="text-muted-foreground text-xs font-medium">
          实体描述
        </div>
        {description ? (
          <p className="text-foreground/90 text-sm leading-relaxed">
            {description}
          </p>
        ) : (
          <p className="text-muted-foreground text-xs italic">无描述</p>
        )}
      </section>

      <section className="grid grid-cols-3 gap-2">
        <InfoTile label="邻居" value={neighbors.length} />
        <InfoTile label="关系" value={edges.length} />
        <InfoTile label="证据片段" value={sourceChunkCount} />
      </section>

      {neighbors.length > 0 && (
        <section className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="text-muted-foreground text-xs font-medium">
              邻居实体
            </div>
            <Badge variant="secondary">{neighbors.length}</Badge>
          </div>
          <div className="space-y-1.5">
            {neighbors.slice(0, 12).map((n) => (
              <button
                key={n.id}
                type="button"
                className="hover:bg-muted/70 flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors"
                onClick={() => onSelectNode(n.id)}
              >
                <EntityTypeBadge
                  entityType={
                    (n.properties.entity_type as string | undefined) ?? null
                  }
                  color={pickClusterColor(n.cluster)}
                />
                <span className="min-w-0 flex-1 truncate text-sm font-medium">
                  {n.id}
                </span>
                <ArrowRight className="text-muted-foreground size-3.5" />
              </button>
            ))}
          </div>
        </section>
      )}

      {edges.length > 0 && (
        <section className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="text-muted-foreground text-xs font-medium">
              相关关系
            </div>
            <Badge variant="secondary">{edges.length}</Badge>
          </div>
          <div className="space-y-2">
            {edges.slice(0, 16).map((edge, i) => {
              const sId = endpointId(edge.source);
              const tId = endpointId(edge.target);
              const otherId = sId === node.id ? tId : sId;
              return (
                <button
                  key={`${sId}-${tId}-${i}`}
                  type="button"
                  className="hover:bg-muted/70 w-full rounded-lg border p-2 text-left transition-colors"
                  onClick={() => onSelectEdge(edge)}
                >
                  <div className="flex items-center gap-2 text-xs">
                    <GitBranch className="text-muted-foreground size-3.5" />
                    <span className="min-w-0 flex-1 truncate font-medium">
                      {otherId}
                    </span>
                    {edge.properties.weight != null && (
                      <span className="text-muted-foreground tabular-nums">
                        权重 {edge.properties.weight}
                      </span>
                    )}
                  </div>
                  {edge.properties.description && (
                    <div className="text-muted-foreground mt-1 line-clamp-2 text-xs">
                      {edge.properties.description}
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
};

const EdgeContent = ({
  edge,
  nodesById,
  onSelectNode,
}: {
  edge: GraphEdge;
  nodesById: Map<string, HybridNode>;
  onSelectNode: (nodeId: string) => void;
}) => {
  const sId = endpointId(edge.source);
  const tId = endpointId(edge.target);
  const sNode = nodesById.get(sId);
  const tNode = nodesById.get(tId);
  const description = edge.properties.description ?? null;
  const weight = edge.properties.weight;

  return (
    <div className="flex flex-col gap-5">
      <header className="flex flex-col gap-1.5 border-b pb-4">
        <div className="text-muted-foreground flex items-center gap-2 font-mono text-[10px] tracking-wider uppercase">
          <GitBranch className="size-3.5" /> 关系详情
        </div>
        <h3 className="font-serif text-xl leading-tight font-normal tracking-tight break-words">
          {sId}
          <ArrowRight className="mx-1 inline size-4" />
          {tId}
        </h3>
      </header>

      <section className="grid grid-cols-2 gap-2">
        <InfoTile label="权重" value={weight ?? '-'} />
        <InfoTile label="关系类型" value={edge.type ?? '-'} />
      </section>

      <section className="space-y-2">
        <div className="text-muted-foreground text-xs font-medium">
          关系描述
        </div>
        {description ? (
          <p className="text-foreground/90 text-sm leading-relaxed">
            {description}
          </p>
        ) : (
          <p className="text-muted-foreground text-xs italic">无描述</p>
        )}
      </section>

      <section className="space-y-2">
        <div className="text-muted-foreground text-xs font-medium">
          关系端点
        </div>
        <button
          type="button"
          className="hover:bg-muted/70 flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left transition-colors"
          disabled={!sNode}
          onClick={() => sNode && onSelectNode(sNode.id)}
        >
          <span
            className="size-2 shrink-0 rounded-full"
            style={{
              backgroundColor: sNode
                ? pickClusterColor(sNode.cluster)
                : COLORS.border,
            }}
          />
          <span className="text-muted-foreground text-xs">起点</span>
          <span className="min-w-0 flex-1 truncate text-left font-medium">
            {sId}
          </span>
        </button>
        <button
          type="button"
          className="hover:bg-muted/70 flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left transition-colors"
          disabled={!tNode}
          onClick={() => tNode && onSelectNode(tNode.id)}
        >
          <span
            className="size-2 shrink-0 rounded-full"
            style={{
              backgroundColor: tNode
                ? pickClusterColor(tNode.cluster)
                : COLORS.border,
            }}
          />
          <span className="text-muted-foreground text-xs">终点</span>
          <span className="min-w-0 flex-1 truncate text-left font-medium">
            {tId}
          </span>
        </button>
      </section>
    </div>
  );
};

const PathListContent = ({
  paths,
  nodesById,
  linksByPair,
  onSelectNode,
}: {
  paths: string[][];
  nodesById: Map<string, HybridNode>;
  linksByPair: Map<string, GraphEdge>;
  onSelectNode: (nodeId: string) => void;
}) => {
  const start = paths[0]?.[0];
  const end = paths[0]?.[paths[0]?.length - 1];

  return (
    <div className="flex flex-col gap-3">
      <header className="flex flex-col gap-1.5 border-b pb-4">
        <div className="text-muted-foreground flex items-center gap-2 font-mono text-[10px] tracking-wider uppercase">
          <Route className="size-3.5" /> 最短路径 · {paths.length} 条
          {paths.length >= 12 && (
            <span className="text-muted-foreground/70 normal-case">
              (capped at 12)
            </span>
          )}
        </div>
        <h3 className="font-serif text-xl leading-tight font-normal tracking-tight break-words">
          {start ?? '?'}
          <ArrowRight className="mx-1 inline size-4" />
          {end ?? '?'}
        </h3>
      </header>
      {paths.map((path, i) => (
        <PathBlock
          key={i}
          pathIndex={i}
          path={path}
          nodesById={nodesById}
          linksByPair={linksByPair}
          onSelectNode={onSelectNode}
        />
      ))}
    </div>
  );
};

const PathBlock = ({
  pathIndex,
  path,
  nodesById,
  linksByPair,
  onSelectNode,
}: {
  pathIndex: number;
  path: string[];
  nodesById: Map<string, HybridNode>;
  linksByPair: Map<string, GraphEdge>;
  onSelectNode: (nodeId: string) => void;
}) => {
  const hops = Math.max(0, path.length - 1);
  return (
    <div className="rounded-lg border p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-foreground text-xs font-medium">
          路径 {pathIndex + 1}
        </span>
        <span className="text-muted-foreground text-[10px]">
          {hops} 跳 · {path.length} 节点
        </span>
      </div>
      <ol className="flex flex-col gap-2 text-xs">
        {path.map((nodeId, i) => {
          const node = nodesById.get(nodeId);
          const next = path[i + 1];
          const edge =
            next !== undefined ? linksByPair.get(`${nodeId}::${next}`) : null;
          const description =
            (node?.properties.description as string | undefined) ?? null;
          const entityType =
            (node?.properties.entity_type as string | undefined) ?? null;
          const cluster = node?.cluster ?? 0;
          return (
            <li key={`${nodeId}-${i}`} className="flex flex-col gap-1.5">
              <button
                type="button"
                className="hover:bg-muted/70 flex items-start gap-2 rounded-md px-1.5 py-1 text-left transition-colors"
                onClick={() => onSelectNode(nodeId)}
              >
                <span className="bg-muted text-muted-foreground mt-0.5 rounded px-1 text-[9px] tabular-nums">
                  {i + 1}
                </span>
                <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <div className="flex items-center gap-1.5">
                    <span
                      className="size-2 shrink-0 rounded-full"
                      style={{
                        backgroundColor: pickClusterColor(cluster),
                      }}
                    />
                    {entityType && (
                      <span className="text-muted-foreground font-mono text-[9px] tracking-wider uppercase">
                        {entityType}
                      </span>
                    )}
                    <span className="truncate font-medium">
                      {node?.id ?? nodeId}
                    </span>
                  </div>
                  {description && (
                    <p className="text-muted-foreground line-clamp-2 leading-relaxed">
                      {description}
                    </p>
                  )}
                </div>
              </button>
              {edge && (
                <div className="text-muted-foreground border-foreground/20 ml-3 flex flex-col gap-0.5 border-l pl-3 text-[10px]">
                  <div className="flex items-center gap-1.5">
                    <GitBranch className="size-3" />
                    <span className="text-muted-foreground/70">↓</span>
                    {edge.properties.weight != null && (
                      <span className="text-muted-foreground tabular-nums">
                        权重 {edge.properties.weight}
                      </span>
                    )}
                  </div>
                  {edge.properties.description ? (
                    <p className="line-clamp-2 leading-relaxed">
                      {edge.properties.description}
                    </p>
                  ) : (
                    <span className="text-muted-foreground/60 italic">
                      (无关系描述)
                    </span>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
};
