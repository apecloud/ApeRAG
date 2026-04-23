'use client';

import { Markdown } from '@/components/markdown';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getKnowledgeGraph } from '@/features/knowledge-graph/client-api';
import type { GraphEdge, GraphNode } from '@/features/knowledge-graph/types';
import { cn } from '@/lib/utils';
import Color from 'color';
import * as d3 from 'd3';
import {
  ArrowLeft,
  Focus,
  LoaderCircle,
  RefreshCcw,
  Sparkles,
  Waypoints,
} from 'lucide-react';
import dynamic from 'next/dynamic';
import { useParams } from 'next/navigation';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { ForceGraphMethods } from 'react-force-graph-2d';

const ForceGraph2D = dynamic(
  () => import('react-force-graph-2d').then((r) => r),
  {
    ssr: false,
  },
);

type SceneMode = 'overview' | 'focus';

type VisualGraphNode = GraphNode & {
  degree: number;
  value: number;
  x?: number;
  y?: number;
  __labelDimensions?: [number, number];
};

type VisualGraphLink = GraphEdge & {
  source: string | VisualGraphNode;
  target: string | VisualGraphNode;
};

type GraphNodeRef =
  | string
  | number
  | GraphNode
  | VisualGraphNode
  | {
      id?: string | number;
      properties?: {
        entity_type?: string;
      };
    }
  | undefined;

const EMPTY_GRAPH = {
  nodes: [] as VisualGraphNode[],
  links: [] as VisualGraphLink[],
};

const ACCENT_COLORS = [
  '#74c0fc',
  '#63e6be',
  '#ffd166',
  '#ff9671',
  '#c77dff',
  '#90f1ef',
];

const getNodeId = (value: GraphNodeRef): string => {
  if (typeof value === 'string' || typeof value === 'number') {
    return String(value);
  }
  return value?.id ? String(value.id) : '';
};

const getNodeEntityType = (value: GraphNodeRef): string => {
  if (
    value === undefined ||
    typeof value === 'string' ||
    typeof value === 'number'
  ) {
    return '';
  }
  return value.properties?.entity_type || '';
};

export const CollectionGraphShowcase = () => {
  const params = useParams();
  const collectionId =
    typeof params.collectionId === 'string' ? params.collectionId : undefined;

  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<ForceGraphMethods | undefined>(undefined);

  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [graphData, setGraphData] = useState<{
    nodes: VisualGraphNode[];
    links: VisualGraphLink[];
  }>(EMPTY_GRAPH);
  const [sceneMode, setSceneMode] = useState<SceneMode>('overview');
  const [selectedNodeId, setSelectedNodeId] = useState<string>();
  const [hoveredNodeId, setHoveredNodeId] = useState<string>();
  const [loading, setLoading] = useState<boolean>(false);
  const [activeTypes, setActiveTypes] = useState<string[]>([]);
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  const colorScale = useMemo(
    () => d3.scaleOrdinal<string>().range(ACCENT_COLORS),
    [],
  );

  const allTypes = useMemo(() => {
    const types = Array.from(
      new Set(
        graphData.nodes
          .map((node) => node.properties?.entity_type || '')
          .filter(Boolean),
      ),
    );
    return types.sort((a, b) => a.localeCompare(b));
  }, [graphData.nodes]);

  useEffect(() => {
    if (!allTypes.length) return;
    setActiveTypes((current) => (current.length ? current : allTypes));
  }, [allTypes]);

  const degreeMap = useMemo(() => {
    const degrees = new Map<string, number>();
    graphData.nodes.forEach((node) => degrees.set(node.id, node.degree));
    return degrees;
  }, [graphData.nodes]);

  const neighborMap = useMemo(() => {
    const neighbors = new Map<string, Set<string>>();

    const addNeighbor = (sourceId: string, targetId: string) => {
      if (!sourceId || !targetId) return;
      if (!neighbors.has(sourceId)) neighbors.set(sourceId, new Set());
      neighbors.get(sourceId)?.add(targetId);
    };

    graphData.links.forEach((link) => {
      const sourceId = getNodeId(link.source);
      const targetId = getNodeId(link.target);
      addNeighbor(sourceId, targetId);
      addNeighbor(targetId, sourceId);
    });

    return neighbors;
  }, [graphData.links]);

  const selectedNode = useMemo(
    () => graphData.nodes.find((node) => node.id === selectedNodeId),
    [graphData.nodes, selectedNodeId],
  );

  const focusNodeId = selectedNodeId || hoveredNodeId;

  const highlightedNodeIds = useMemo(() => {
    const ids = new Set<string>();
    if (!focusNodeId) return ids;
    ids.add(focusNodeId);
    neighborMap.get(focusNodeId)?.forEach((id) => ids.add(id));
    return ids;
  }, [focusNodeId, neighborMap]);

  const visibleCounts = useMemo(() => {
    const allowedTypes = new Set(activeTypes);
    const visibleNodes = graphData.nodes.filter((node) => {
      const type = node.properties?.entity_type || '';
      return !type || allowedTypes.has(type);
    });

    const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
    const visibleLinks = graphData.links.filter((link) => {
      const sourceId = getNodeId(link.source);
      const targetId = getNodeId(link.target);
      return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId);
    });

    return {
      visibleNodes,
      visibleLinks,
    };
  }, [activeTypes, graphData.links, graphData.nodes]);

  const topNode = useMemo(() => {
    return [...graphData.nodes].sort((a, b) => b.degree - a.degree)[0];
  }, [graphData.nodes]);

  const selectedNeighbors = useMemo(() => {
    if (!selectedNodeId) return [];
    const neighborIds = Array.from(neighborMap.get(selectedNodeId) || []);

    return neighborIds
      .map((id) => graphData.nodes.find((node) => node.id === id))
      .filter((node): node is VisualGraphNode => Boolean(node))
      .sort((a, b) => b.degree - a.degree)
      .slice(0, 8);
  }, [graphData.nodes, neighborMap, selectedNodeId]);

  const loadGraph = useCallback(
    async (mode: SceneMode, centerNodeId?: string) => {
      if (!collectionId) return;

      setLoading(true);
      try {
        const data = await getKnowledgeGraph(collectionId, {
          label: mode === 'focus' && centerNodeId ? centerNodeId : '*',
          maxNodes: mode === 'focus' ? 220 : 800,
          maxDepth: mode === 'focus' ? 2 : undefined,
        });

        if (!data) return;
        const degrees = new Map<string, number>();

        (data.edges || []).forEach((edge) => {
          degrees.set(edge.source, (degrees.get(edge.source) || 0) + 1);
          degrees.set(edge.target, (degrees.get(edge.target) || 0) + 1);
        });

        const nodes =
          data.nodes?.map((node) => {
            const degree = degrees.get(node.id) || 0;
            return {
              ...node,
              degree,
              value: Math.max(Math.sqrt(Math.max(degree, 1)) * 7, 7),
            };
          }) || [];

        setGraphData({
          nodes,
          links: (data.edges as VisualGraphLink[]) || [],
        });
        setSceneMode(mode);
        setSelectedNodeId((current) => {
          if (centerNodeId) return centerNodeId;
          if (current && nodes.some((node) => node.id === current)) return current;
          return nodes[0]?.id;
        });
      } finally {
        setLoading(false);
      }
    },
    [collectionId],
  );

  const fitScene = useCallback(() => {
    graphRef.current?.zoomToFit(800, 80);
  }, []);

  useEffect(() => {
    loadGraph('overview');
  }, [loadGraph]);

  const handleResize = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    setDimensions({
      width: Math.max(container.clientWidth - 2, 0),
      height: Math.max(container.clientHeight - 2, 0),
    });
  }, []);

  useEffect(() => {
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [handleResize]);

  const toggleType = useCallback((type: string) => {
    setActiveTypes((current) => {
      if (current.includes(type)) {
        return current.filter((item) => item !== type);
      }
      return [...current, type];
    });
  }, []);

  const showLabelForNode = useCallback(
    (node: VisualGraphNode, globalScale: number) => {
      if (node.id === selectedNodeId || node.id === hoveredNodeId) return true;
      if (highlightedNodeIds.has(node.id) && globalScale > 1.4) return true;
      if (node.degree >= 6 && globalScale > 1.2) return true;
      return globalScale > 2.3;
    },
    [highlightedNodeIds, hoveredNodeId, selectedNodeId],
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 pb-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.4fr)_22rem]">
        <Card className="overflow-hidden border-slate-800 bg-[#04111b] text-slate-50 shadow-[0_40px_120px_rgba(3,10,18,0.55)]">
          <div className="border-b border-white/10 px-6 py-5">
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge className="border border-cyan-400/30 bg-cyan-400/10 text-cyan-100">
                    Showcase Preview
                  </Badge>
                  <Badge className="border border-amber-300/25 bg-amber-300/10 text-amber-100">
                    Temporary transition page
                  </Badge>
                  <Badge
                    variant="outline"
                    className="border-white/15 bg-white/5 text-slate-300"
                  >
                    {sceneMode === 'overview' ? 'overview scene' : 'focused scene'}
                  </Badge>
                </div>
                <div>
                  <h2 className="max-w-3xl text-3xl font-semibold tracking-[-0.03em] text-white">
                    A more cinematic graph view that still uses your real
                    collection data
                  </h2>
                  <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
                    This preview keeps the current React stack and graph API,
                    but upgrades the scene with stronger hierarchy, zoom-to-fit,
                    contextual labels, and a real neighborhood drill-down mode.
                  </p>
                  <p className="max-w-2xl text-xs leading-5 text-amber-100/85">
                    Temporary transition page for fast visual review. Keep using
                    the existing graph page as the stable path for now.
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="outline"
                  className="border-white/15 bg-white/5 text-slate-100 hover:bg-white/10"
                  onClick={() => loadGraph('overview')}
                >
                  <RefreshCcw />
                  Reload overview
                </Button>
                <Button
                  variant="outline"
                  className="border-white/15 bg-white/5 text-slate-100 hover:bg-white/10"
                  onClick={fitScene}
                >
                  <Focus />
                  Zoom to fit
                </Button>
              </div>
            </div>
          </div>

          <div className="grid gap-4 px-6 pt-5 lg:grid-cols-4">
            {[
              {
                label: 'nodes in scene',
                value: visibleCounts.visibleNodes.length,
              },
              {
                label: 'relationships',
                value: visibleCounts.visibleLinks.length,
              },
              {
                label: 'entity types',
                value: allTypes.length,
              },
              {
                label: 'top hub',
                value: topNode?.id || 'n/a',
              },
            ].map((item) => (
              <div
                key={item.label}
                className="rounded-2xl border border-white/8 bg-white/[0.045] px-4 py-3 backdrop-blur"
              >
                <div className="text-[11px] uppercase tracking-[0.24em] text-slate-400">
                  {item.label}
                </div>
                <div className="mt-2 truncate font-mono text-xl text-white">
                  {item.value}
                </div>
              </div>
            ))}
          </div>

          <CardContent className="px-6 pb-6 pt-4">
            <div
              ref={containerRef}
              className="relative min-h-[72vh] overflow-hidden rounded-[28px] border border-white/10 bg-[#061523]"
            >
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.24),transparent_32%),radial-gradient(circle_at_top_right,rgba(45,212,191,0.16),transparent_28%),linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0))]" />
              <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:28px_28px] opacity-40" />

              <div className="absolute left-4 top-4 z-10 flex max-w-[min(72%,42rem)] flex-wrap gap-2">
                {allTypes.map((type) => {
                  const active = activeTypes.includes(type);
                  return (
                    <button
                      key={type}
                      className={cn(
                        'rounded-full border px-3 py-1 text-xs font-medium transition',
                        active
                          ? 'border-white/15 bg-white/10 text-white'
                          : 'border-white/8 bg-black/15 text-slate-400 hover:bg-white/5',
                      )}
                      onClick={() => toggleType(type)}
                      type="button"
                    >
                      <span
                        className="mr-2 inline-block size-2 rounded-full align-middle"
                        style={{ backgroundColor: colorScale(type) }}
                      />
                      {type}
                    </button>
                  );
                })}
              </div>

              <div className="absolute bottom-4 left-4 z-10 max-w-sm rounded-2xl border border-white/10 bg-slate-950/55 px-4 py-3 text-sm text-slate-200 backdrop-blur-md">
                <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.24em] text-slate-400">
                  <Sparkles className="size-3.5" />
                  scene notes
                </div>
                <p className="mt-2 leading-6 text-slate-300">
                  Click a node to inspect it, then use focus mode to pull a
                  smaller neighborhood graph from the backend. Labels become
                  denser as you zoom in, while inactive relations stay quiet.
                </p>
              </div>

              {loading && (
                <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/45 backdrop-blur-sm">
                  <div className="flex items-center gap-3 rounded-full border border-white/10 bg-black/30 px-4 py-2 text-sm text-white">
                    <LoaderCircle className="size-4 animate-spin" />
                    Loading graph scene…
                  </div>
                </div>
              )}

              <ForceGraph2D
                ref={graphRef}
                graphData={graphData}
                width={dimensions.width}
                height={dimensions.height}
                backgroundColor="transparent"
                cooldownTicks={90}
                onEngineStop={fitScene}
                onZoom={(transform) => {
                  setZoomLevel(transform.k);
                }}
                nodeVisibility={(node) => {
                  const type = node.properties?.entity_type || '';
                  return !type || activeTypes.includes(type);
                }}
                linkVisibility={(link) => {
                  const sourceType = getNodeEntityType(link.source);
                  const targetType = getNodeEntityType(link.target);

                  return (
                    (!sourceType || activeTypes.includes(sourceType)) &&
                    (!targetType || activeTypes.includes(targetType))
                  );
                }}
                onNodeClick={(node) => {
                  setSelectedNodeId(String(node.id));
                }}
                onNodeHover={(node) => {
                  setHoveredNodeId(node ? String(node.id) : undefined);
                }}
                linkColor={(link) => {
                  const sourceId = getNodeId(link.source);
                  const targetId = getNodeId(link.target);
                  const isHighlighted =
                    highlightedNodeIds.has(sourceId) &&
                    highlightedNodeIds.has(targetId);
                  return isHighlighted
                    ? 'rgba(130, 246, 255, 0.82)'
                    : 'rgba(139, 172, 207, 0.16)';
                }}
                linkWidth={(link) => {
                  const sourceId = getNodeId(link.source);
                  const targetId = getNodeId(link.target);
                  const isHighlighted =
                    highlightedNodeIds.has(sourceId) &&
                    highlightedNodeIds.has(targetId);
                  return isHighlighted ? 1.8 : 0.55;
                }}
                linkDirectionalParticles={(link) => {
                  const sourceId = getNodeId(link.source);
                  const targetId = getNodeId(link.target);
                  const isHighlighted =
                    highlightedNodeIds.has(sourceId) &&
                    highlightedNodeIds.has(targetId);
                  return isHighlighted ? 2 : 0;
                }}
                linkDirectionalParticleWidth={(link) => {
                  const sourceId = getNodeId(link.source);
                  const targetId = getNodeId(link.target);
                  const isHighlighted =
                    highlightedNodeIds.has(sourceId) &&
                    highlightedNodeIds.has(targetId);
                  return isHighlighted ? 2.4 : 0;
                }}
                nodeCanvasObject={(node, ctx, globalScale) => {
                  const graphNode = node as VisualGraphNode;
                  const x = graphNode.x || 0;
                  const y = graphNode.y || 0;
                  const size = Math.min(graphNode.value, 26);
                  const isSelected = graphNode.id === selectedNodeId;
                  const isHovered = graphNode.id === hoveredNodeId;
                  const isHighlighted = highlightedNodeIds.has(graphNode.id);
                  const baseColor = colorScale(
                    graphNode.properties?.entity_type || 'unknown',
                  );

                  if (isSelected || isHovered) {
                    ctx.beginPath();
                    ctx.arc(x, y, size + 8, 0, Math.PI * 2, false);
                    ctx.fillStyle = isSelected
                      ? 'rgba(116, 192, 252, 0.16)'
                      : 'rgba(99, 230, 190, 0.12)';
                    ctx.fill();
                  }

                  ctx.beginPath();
                  ctx.arc(x, y, size, 0, Math.PI * 2, false);
                  ctx.fillStyle =
                    isSelected || isHighlighted
                      ? baseColor
                      : Color(baseColor).desaturate(0.25).alpha(0.82).string();
                  ctx.fill();

                  ctx.beginPath();
                  ctx.arc(x, y, size, 0, Math.PI * 2, false);
                  ctx.lineWidth = isSelected ? 2 : 1;
                  ctx.strokeStyle = isSelected
                    ? 'rgba(255,255,255,0.92)'
                    : 'rgba(255,255,255,0.28)';
                  ctx.stroke();

                  if (showLabelForNode(graphNode, globalScale)) {
                    const label = String(graphNode.id);
                    const fontSize = Math.max(10 / globalScale, 4.4);
                    ctx.font = `600 ${fontSize}px Geist, system-ui, sans-serif`;
                    const textWidth = ctx.measureText(label).width;
                    const paddingX = 8 / globalScale;
                    const paddingY = 5 / globalScale;
                    const boxWidth = textWidth + paddingX * 2;
                    const boxHeight = fontSize + paddingY * 2;
                    const labelY = y - size - boxHeight / 2 - 8 / globalScale;

                    ctx.fillStyle = 'rgba(3, 10, 18, 0.78)';
                    ctx.fillRect(
                      x - boxWidth / 2,
                      labelY - boxHeight / 2,
                      boxWidth,
                      boxHeight,
                    );

                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillStyle = 'rgba(240, 249, 255, 0.94)';
                    ctx.fillText(label, x, labelY);
                  }
                }}
                nodePointerAreaPaint={(node, color, ctx) => {
                  const graphNode = node as VisualGraphNode;
                  const x = graphNode.x || 0;
                  const y = graphNode.y || 0;
                  const size = Math.min(graphNode.value, 26) + 4;
                  ctx.fillStyle = color;
                  ctx.beginPath();
                  ctx.arc(x, y, size, 0, Math.PI * 2, false);
                  ctx.fill();
                }}
              />
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card className="border-slate-200/70 bg-white/85 shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Scene controls</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <Button
                className="w-full justify-start"
                disabled={!selectedNode || sceneMode === 'focus'}
                onClick={() => loadGraph('focus', selectedNode?.id)}
                variant="default"
              >
                <Waypoints />
                Explore neighborhood
              </Button>

              <Button
                className="w-full justify-start"
                disabled={sceneMode === 'overview'}
                onClick={() => loadGraph('overview')}
                variant="outline"
              >
                <ArrowLeft />
                Back to overview
              </Button>

              <Button
                className="w-full justify-start"
                onClick={() => setSelectedNodeId(undefined)}
                variant="outline"
              >
                <Focus />
                Clear selection
              </Button>

              <div className="rounded-2xl bg-slate-50 px-4 py-3">
                <div className="text-[11px] uppercase tracking-[0.22em] text-slate-500">
                  current zoom
                </div>
                <div className="mt-2 font-mono text-lg text-slate-900">
                  {zoomLevel.toFixed(2)}x
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-slate-200/70 bg-white/85 shadow-sm">
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Selected node</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {selectedNode ? (
                <>
                  <div>
                    <div className="text-2xl font-semibold tracking-[-0.03em] text-slate-950">
                      {selectedNode.id}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <Badge variant="secondary">
                        {selectedNode.properties?.entity_type || 'untyped'}
                      </Badge>
                      <Badge variant="outline">
                        degree {degreeMap.get(selectedNode.id) || 0}
                      </Badge>
                      {sceneMode === 'focus' && (
                        <Badge
                          variant="outline"
                          className="border-cyan-200 bg-cyan-50 text-cyan-700"
                        >
                          neighborhood mode
                        </Badge>
                      )}
                    </div>
                  </div>

                  {selectedNode.properties?.description ? (
                    <div className="prose prose-sm max-w-none text-slate-700">
                      <Markdown>{selectedNode.properties.description}</Markdown>
                    </div>
                  ) : (
                    <div className="rounded-2xl border border-dashed border-slate-200 px-4 py-3 text-sm text-slate-500">
                      This node does not include a description in the current
                      graph payload.
                    </div>
                  )}

                  <div>
                    <div className="mb-2 text-[11px] uppercase tracking-[0.22em] text-slate-500">
                      strongest neighbors
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {selectedNeighbors.length ? (
                        selectedNeighbors.map((neighbor) => (
                          <button
                            key={neighbor.id}
                            className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-700 transition hover:border-slate-300 hover:bg-slate-100"
                            onClick={() => setSelectedNodeId(neighbor.id)}
                            type="button"
                          >
                            {neighbor.id}
                          </button>
                        ))
                      ) : (
                        <div className="text-sm text-slate-500">
                          No neighbor information available for this node in the
                          current scene.
                        </div>
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-200 px-4 py-5 text-sm leading-6 text-slate-500">
                  Pick a node in the graph to inspect its description, degree,
                  and nearby entities.
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
