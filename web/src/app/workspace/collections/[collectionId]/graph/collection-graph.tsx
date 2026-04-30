'use client';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import {
  getMergeSuggestions as fetchMergeSuggestions,
  getKnowledgeGraph,
  getMarketplaceKnowledgeGraph,
  runMergeSuggestions,
} from '@/features/knowledge-graph/client-api';
import type {
  GraphEdge,
  GraphNode,
  KnowledgeGraph,
  MergeSuggestionsResponse,
} from '@/features/knowledge-graph/types';
import { ApiClientError } from '@/lib/api/typed/errors';
import { cn } from '@/lib/utils';

import { Badge } from '@/components/ui/badge';
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Tooltip, TooltipContent } from '@/components/ui/tooltip';
import {
  CANVAS_DARK,
  COLORS,
  ENTITY_PALETTE,
  entityTypeToPaletteKey,
} from '@/lib/design-tokens';
import { TooltipTrigger } from '@radix-ui/react-tooltip';
import _ from 'lodash';
import {
  Check,
  ChevronDown,
  ChevronUp,
  GitMerge,
  Layers3,
  LoaderCircle,
  Maximize,
  Minimize,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useTheme } from 'next-themes';
import dynamic from 'next/dynamic';
import { useParams } from 'next/navigation';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CollectionGraphEdgeDetail,
  CollectionGraphNodeDetail,
} from './collection-graph-node-detail';
import { CollectionGraphNodeMerge } from './collection-graph-node-merge';

const ForceGraph2D = dynamic(
  () => import('react-force-graph-2d').then((r) => r),
  {
    ssr: false,
  },
);

const resolveEntityColor = (entityType: string | null | undefined): string =>
  ENTITY_PALETTE[entityTypeToPaletteKey(entityType)];

const IMPORTANT_LABEL_SIZE = 16;
const LABEL_RADIUS = 5;
const NODE_SPACING = 11;

const ENTITY_TITLE_KEYS = {
  person: 'entity_person',
  category: 'entity_category',
  date: 'entity_date',
  number: 'entity_number',
  product: 'entity_product',
  organization: 'entity_organization',
  technology: 'entity_technology',
  geo: 'entity_geo',
  event: 'entity_event',
  UNKNOWN: 'entity_UNKNOWN',
} as const;

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

  return Math.min(Number.isFinite(value) ? value : 0, 22);
};

const endpointId = (endpoint: unknown) => {
  if (typeof endpoint === 'object' && endpoint !== null && 'id' in endpoint) {
    return String(endpoint.id);
  }
  return String(endpoint || '');
};

const createCollisionForce = () => {
  let nodes: Array<{
    x?: number;
    y?: number;
    vx?: number;
    vy?: number;
    value?: number;
  }> = [];

  const force = (alpha: number) => {
    for (let i = 0; i < nodes.length; i += 1) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j += 1) {
        const b = nodes[j];
        let dx = (b.x || 0) - (a.x || 0);
        let dy = (b.y || 0) - (a.y || 0);
        let distance = Math.sqrt(dx * dx + dy * dy);
        const minDistance = getNodeSize(a) + getNodeSize(b) + NODE_SPACING;

        if (distance >= minDistance) continue;

        if (distance === 0) {
          distance = 1;
          dx = (Math.random() - 0.5) * 0.01;
          dy = (Math.random() - 0.5) * 0.01;
        }

        const strength = ((minDistance - distance) / distance) * alpha * 0.7;
        const x = dx * strength;
        const y = dy * strength;
        a.vx = (a.vx || 0) - x;
        a.vy = (a.vy || 0) - y;
        b.vx = (b.vx || 0) + x;
        b.vy = (b.vy || 0) + y;
      }
    }
  };

  force.initialize = (
    initializedNodes: Array<{
      x?: number;
      y?: number;
      vx?: number;
      vy?: number;
      value?: number;
    }>,
  ) => {
    nodes = initializedNodes;
  };

  return force;
};

export const CollectionGraph = ({
  marketplace = false,
}: {
  marketplace: boolean;
}) => {
  const params = useParams();
  const [fullscreen, setFullscreen] = useState<boolean>(false);
  const { resolvedTheme } = useTheme();
  const page_graph = useTranslations('page_graph');

  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const graphRef = useRef<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [graphData, setGraphData] = useState<{
    nodes: GraphNode[];
    links: GraphEdge[];
  }>();
  const [graphError, setGraphError] = useState<string>();
  const [mergeSuggestion, setMergeSuggestion] =
    useState<MergeSuggestionsResponse>();
  const [mergeSuggestionOpen, setMergeSuggestionOpen] =
    useState<boolean>(false);
  const [mergeSuggestionLoading, setMergeSuggestionLoading] =
    useState<boolean>(false);
  const [mergeSuggestionRunning, setMergeSuggestionRunning] =
    useState<boolean>(false);
  const emptyMergeSuggestionResponse = useMemo<MergeSuggestionsResponse>(
    () => ({
      suggestions: [],
      total_analyzed_nodes: 0,
      processing_time_seconds: 0,
      from_cache: false,
      generated_at: '',
      total_suggestions: 0,
      pending_count: 0,
      accepted_count: 0,
      rejected_count: 0,
    }),
    [],
  );

  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  const [allEntities, setAllEntities] = useState<{
    [key in string]: GraphNode[];
  }>({});
  const [activeEntities, setActiveEntities] = useState<string[]>([]);

  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  const [hoverNode, setHoverNode] = useState<GraphNode>();
  const [activeNode, setActiveNode] = useState<GraphNode>();
  const [activeEdge, setActiveEdge] = useState<GraphEdge>();
  const [legendCollapsed, setLegendCollapsed] = useState(true);

  const { NODE_MIN, NODE_MAX } = useMemo(
    () => ({
      NODE_MIN: 8,
      NODE_MAX: 22,
    }),
    [],
  );

  const getGraphData = useCallback(async () => {
    if (typeof params.collectionId !== 'string') return;
    setLoading(true);
    setGraphError(undefined);

    try {
      const data: KnowledgeGraph | undefined = marketplace
        ? await getMarketplaceKnowledgeGraph(params.collectionId)
        : await getKnowledgeGraph(params.collectionId);

      if (!data) {
        setGraphData({ nodes: [], links: [] });
        return;
      }

      const edges = data.edges || [];
      const nodes =
        data.nodes?.map((n) => {
          const targetCount = edges.filter((edg) => edg.target === n.id).length;
          const sourceCount = edges.filter((edg) => edg.source === n.id).length;
          return {
            ...n,
            value: Math.max(targetCount, sourceCount, NODE_MIN),
          };
        }) || [];
      const links = edges;

      setGraphData({ nodes, links });

      setAllEntities(_.groupBy(nodes, (n) => n.properties.entity_type));
    } catch (error: unknown) {
      setGraphData({ nodes: [], links: [] });
      setAllEntities({});
      setActiveEntities([]);
      setGraphError(getErrorMessage(error, page_graph('load_failed')));
    } finally {
      setLoading(false);
    }
  }, [NODE_MIN, marketplace, page_graph, params.collectionId]);

  const loadMergeSuggestions = useCallback(async () => {
    if (typeof params.collectionId !== 'string' || marketplace) return;
    setMergeSuggestionLoading(true);
    try {
      const suggestionRes = await fetchMergeSuggestions(params.collectionId);
      setMergeSuggestion(suggestionRes);
    } catch {
      setMergeSuggestion(undefined);
    } finally {
      setMergeSuggestionLoading(false);
    }
  }, [marketplace, params.collectionId]);

  const runMergeSuggestionScan = useCallback(async () => {
    if (typeof params.collectionId !== 'string' || marketplace) return;
    setMergeSuggestionRunning(true);
    try {
      const suggestionRes = await runMergeSuggestions(params.collectionId);
      setMergeSuggestion(suggestionRes);
    } finally {
      setMergeSuggestionRunning(false);
    }
  }, [marketplace, params.collectionId]);

  const handleCloseDetail = useCallback(() => {
    setActiveNode(undefined);
    setActiveEdge(undefined);
    setHoverNode(undefined);
    highlightNodes.clear();
    highlightLinks.clear();
  }, [highlightLinks, highlightNodes]);

  const handleResizeContainer = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const width = container.offsetWidth || 0;
    const height = container.offsetHeight || 0;
    setDimensions({
      width: width - 2,
      height: height - 2,
    });
  }, []);

  useEffect(() => {
    if (activeEntities.length) return;
    setActiveEntities(Object.keys(allEntities));
  }, [activeEntities.length, allEntities]);

  useEffect(() => handleResizeContainer(), [handleResizeContainer]);
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    handleResizeContainer();
    window.addEventListener('resize', handleResizeContainer);
    return () => window.removeEventListener('resize', handleResizeContainer);
  }, [handleResizeContainer, fullscreen]);

  useEffect(() => {
    highlightNodes.clear();
    highlightLinks.clear();

    if (activeNode) {
      const nodeLinks = graphData?.links.filter((link) => {
        return (
          endpointId(link.source) === activeNode.id ||
          endpointId(link.target) === activeNode.id
        );
      });
      nodeLinks?.forEach((link: GraphEdge) => {
        highlightLinks.add(link);
        const sourceNode = graphData?.nodes.find(
          (node) => node.id === endpointId(link.source),
        );
        const targetNode = graphData?.nodes.find(
          (node) => node.id === endpointId(link.target),
        );
        if (sourceNode) highlightNodes.add(sourceNode);
        if (targetNode) highlightNodes.add(targetNode);
      });
      highlightNodes.add(activeNode);
      // @ts-expect-error node.x node.y
      graphRef.current?.centerAt(activeNode.x, activeNode.y, 400);
      graphRef.current?.zoom(3, 600);
    } else if (activeEdge) {
      const sourceId = endpointId(activeEdge.source);
      const targetId = endpointId(activeEdge.target);
      const sourceNode = graphData?.nodes.find((node) => node.id === sourceId);
      const targetNode = graphData?.nodes.find((node) => node.id === targetId);
      highlightLinks.add(activeEdge);
      if (sourceNode) highlightNodes.add(sourceNode);
      if (targetNode) highlightNodes.add(targetNode);

      const sourceX =
        sourceNode && 'x' in sourceNode ? Number(sourceNode.x) : undefined;
      const sourceY =
        sourceNode && 'y' in sourceNode ? Number(sourceNode.y) : undefined;
      const targetX =
        targetNode && 'x' in targetNode ? Number(targetNode.x) : undefined;
      const targetY =
        targetNode && 'y' in targetNode ? Number(targetNode.y) : undefined;

      if (
        Number.isFinite(sourceX) &&
        Number.isFinite(sourceY) &&
        Number.isFinite(targetX) &&
        Number.isFinite(targetY)
      ) {
        graphRef.current?.centerAt(
          ((sourceX as number) + (targetX as number)) / 2,
          ((sourceY as number) + (targetY as number)) / 2,
          400,
        );
        graphRef.current?.zoom(2.4, 600);
      }
    } else {
      graphRef.current?.centerAt(0, 0, 400);
      graphRef.current?.zoom(1.5, 600);
    }
    setHighlightNodes(highlightNodes);
    setHighlightLinks(highlightLinks);
  }, [
    activeEdge,
    activeNode,
    graphData?.links,
    graphData?.nodes,
    highlightLinks,
    highlightNodes,
  ]);

  useEffect(() => {
    getGraphData();
    loadMergeSuggestions();
  }, [getGraphData, loadMergeSuggestions]);

  useEffect(() => {
    if (!graphData || !graphRef.current) return;

    const graph = graphRef.current;
    const linkForce = graph.d3Force('link');
    linkForce?.distance?.((link: GraphEdge) => {
      const source =
        typeof link.source === 'object'
          ? (link.source as GraphNode)
          : undefined;
      const target =
        typeof link.target === 'object'
          ? (link.target as GraphNode)
          : undefined;
      const sourceSize = source ? getNodeSize(source) : NODE_MIN;
      const targetSize = target ? getNodeSize(target) : NODE_MIN;
      return 58 + (sourceSize + targetSize) * 1.05;
    });
    linkForce?.strength?.(0.06);
    graph.d3Force('charge')?.strength?.(-88);
    graph.d3Force('charge')?.distanceMax?.(260);
    graph.d3Force('collide', createCollisionForce());
    graph.d3ReheatSimulation?.();
  }, [NODE_MIN, graphData]);

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
            (link) =>
              endpointId(link.source) === activeNode.id ||
              endpointId(link.target) === activeNode.id,
          ) || []
        : [],
    [activeNode, graphData?.links],
  );
  const activeNodeNeighbors = useMemo(() => {
    if (!activeNode || !graphData?.nodes.length) return [];

    const neighborIds = new Set(
      activeNodeEdges
        .map((edge) => {
          const sourceId = endpointId(edge.source);
          const targetId = endpointId(edge.target);
          return sourceId === activeNode.id ? targetId : sourceId;
        })
        .filter(Boolean),
    );

    return graphData.nodes.filter((node) => neighborIds.has(node.id));
  }, [activeNode, activeNodeEdges, graphData?.nodes]);

  return (
    <div
      className={cn('top-0 right-0 bottom-0 left-0 flex flex-1 flex-col', {
        fixed: fullscreen,
        'bg-background': fullscreen,
        'z-49': fullscreen,
      })}
    >
      <div
        className={cn('mb-2 flex flex-row items-center justify-between gap-2', {
          'px-2': fullscreen,
          'pt-2': fullscreen,
        })}
      >
        <div className="flex min-w-0 flex-row items-baseline gap-3">
          <h1
            className="truncate font-serif text-xl font-normal"
            style={{ letterSpacing: '-0.018em' }}
          >
            {page_graph('metadata.title')}
          </h1>
          {graphData && (
            <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
              <span className="bg-muted/60 rounded-md px-1.5 py-0.5">
                {page_graph('entity_count', {
                  count: totalNodes.toLocaleString(),
                })}
              </span>
              <span className="bg-muted/60 rounded-md px-1.5 py-0.5">
                {page_graph('relation_count', {
                  count: totalEdges.toLocaleString(),
                })}
              </span>
            </div>
          )}
        </div>
        <div className="flex flex-row items-center gap-2">
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="w-40 justify-between"
              >
                {page_graph('node_search')}
                <ChevronDown />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-[240px] p-0" align="end">
              <Command>
                <CommandInput placeholder="Search node..." className="h-9" />
                <CommandList className="max-h-60">
                  <CommandEmpty>{page_graph('no_nodes_found')}</CommandEmpty>
                  <CommandGroup>
                    {_.map(graphData?.nodes, (node, key) => {
                      const isActive = activeNode?.id === node.id;
                      return (
                        <CommandItem
                          key={key}
                          className={cn('capitalize')}
                          value={node.id}
                          onSelect={() => {
                            setActiveNode(isActive ? undefined : node);
                          }}
                        >
                          <div className="truncate">{node.id}</div>
                          <Check
                            className={cn(
                              'ml-auto',
                              isActive ? 'opacity-100' : 'opacity-0',
                            )}
                          />
                        </CommandItem>
                      );
                    })}
                  </CommandGroup>
                </CommandList>
              </Command>
            </PopoverContent>
          </Popover>

          {!marketplace && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  size="icon"
                  variant="outline"
                  className="relative cursor-pointer"
                  onClick={() => setMergeSuggestionOpen(true)}
                >
                  <GitMerge />
                  {mergeSuggestion?.pending_count ? (
                    <Badge
                      variant="outline"
                      className="absolute -top-2 -right-2 h-5 min-w-5 rounded-full px-1 font-mono text-[10px] tabular-nums"
                      style={{
                        backgroundColor: COLORS.accentSoft,
                        color: COLORS.accentInk,
                        borderColor: COLORS.subtleStrong,
                      }}
                    >
                      {mergeSuggestion.pending_count > 10
                        ? '10+'
                        : mergeSuggestion.pending_count}
                    </Badge>
                  ) : null}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {page_graph('merge_infomation', {
                  count: String(mergeSuggestion?.pending_count || 0),
                })}
              </TooltipContent>
            </Tooltip>
          )}

          <Button
            size="icon"
            variant="outline"
            className="cursor-pointer"
            onClick={() => {
              getGraphData();
              loadMergeSuggestions();
            }}
          >
            <LoaderCircle
              className={
                loading || mergeSuggestionLoading ? 'animate-spin' : ''
              }
            />
          </Button>

          <Button
            size="icon"
            variant="outline"
            className="cursor-pointer"
            onClick={() => {
              setFullscreen(!fullscreen);
            }}
          >
            {fullscreen ? <Minimize /> : <Maximize />}
          </Button>
        </div>
      </div>

      <Card
        ref={containerRef}
        className="bg-card/0 relative flex flex-1 gap-0 overflow-hidden py-0"
      >
        {graphData === undefined && !graphError && (
          <div className="absolute top-4/12 left-6/12">
            <div className="flex flex-row gap-2 py-2">
              <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-0"></div>
              <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-200"></div>
              <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-400"></div>
            </div>
          </div>
        )}

        {graphError && (
          <div className="absolute top-4/12 w-full px-6">
            <div className="mx-auto max-w-md text-center">
              <div className="text-foreground text-sm font-medium">
                {page_graph('load_failed')}
              </div>
              <div className="text-muted-foreground mt-2 text-xs">
                {graphError}
              </div>
            </div>
          </div>
        )}

        {!graphError &&
          graphData !== undefined &&
          _.isEmpty(graphData?.nodes) && (
            <div className="absolute top-4/12 w-full">
              <div className="text-muted-foreground text-center">
                {page_graph('no_nodes_found')}
              </div>
            </div>
          )}

        <ForceGraph2D
          graphData={graphData}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel={(nod) => String(nod.id)}
          ref={graphRef}
          backgroundColor="transparent"
          cooldownTicks={140}
          d3AlphaDecay={0.022}
          d3VelocityDecay={0.32}
          linkCurvature={0.025}
          nodeVisibility={(node) => {
            return (
              !node.properties.entity_type ||
              activeEntities.includes(node.properties.entity_type)
            );
          }}
          onNodeClick={(node) => {
            if (activeNode?.id === node.id) {
              handleCloseDetail();
              return;
            }
            setActiveEdge(undefined);
            setActiveNode(node as GraphNode);
          }}
          onNodeHover={(node) => {
            if (activeNode || activeEdge) return;
            highlightNodes.clear();
            highlightLinks.clear();
            if (node) {
              const nodeLinks = graphData?.links.filter((link) => {
                return (
                  endpointId(link.source) === node.id ||
                  endpointId(link.target) === node.id
                );
              });
              nodeLinks?.forEach((link: GraphEdge) => {
                highlightLinks.add(link);
              });
            }
            setHoverNode(
              node
                ? {
                    ...node,
                    id: String(node.id),
                    labels: [],
                    properties: {},
                  }
                : undefined,
            );
            setHighlightNodes(highlightNodes);
            setHighlightLinks(highlightLinks);
          }}
          onLinkHover={(link) => {
            if (activeNode || activeEdge) return;
            highlightNodes.clear();
            highlightLinks.clear();
            if (link) {
              highlightLinks.add(link);
            }
            setHighlightNodes(highlightNodes);
            setHighlightLinks(highlightLinks);
          }}
          onLinkClick={(link) => {
            if (activeEdge?.id === link.id) {
              handleCloseDetail();
              return;
            }
            setActiveNode(undefined);
            setActiveEdge(link as GraphEdge);
          }}
          nodeCanvasObject={(node, ctx) => {
            const x = node.x || 0;
            const y = node.y || 0;

            let size = Math.min(node.value, NODE_MAX);
            if (node === hoverNode) size += 1;

            const entityColor = resolveEntityColor(node.properties.entity_type);
            const isDim = highlightNodes.size > 0 && !highlightNodes.has(node);
            const isActive = activeNode?.id === node.id;
            const shouldShowLabel =
              isActive ||
              node === hoverNode ||
              highlightNodes.has(node) ||
              size >= IMPORTANT_LABEL_SIZE;

            // main fill
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);
            ctx.fillStyle = entityColor;
            ctx.globalAlpha = isDim ? 0.35 : 1;
            ctx.fill();
            ctx.globalAlpha = 1;

            // hairline stroke (bg-colored, blends into surface)
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);
            ctx.lineWidth = 0.6;
            ctx.strokeStyle = nodeStroke;
            ctx.stroke();

            if (shouldShowLabel) {
              // adaptive label
              let fontSize = 13;
              const offset = 2;
              const fontFamily =
                'var(--font-sans), Manrope, system-ui, sans-serif';
              ctx.font = `500 ${fontSize}px ${fontFamily}`;
              let textWidth = ctx.measureText(String(node.id)).width - offset;
              while (textWidth > size * 1.6 && fontSize > 1) {
                fontSize -= 1;
                ctx.font = `500 ${fontSize}px ${fontFamily}`;
                textWidth = ctx.measureText(String(node.id)).width - offset;
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
              ctx.globalAlpha = isDim ? 0.3 : 1;
              ctx.fill();
              ctx.fillStyle = labelFill;
              ctx.globalAlpha = isDim ? 0.4 : 1;
              ctx.fillText(String(node.id), labelX, labelY);
              ctx.globalAlpha = 1;
            }
          }}
          nodePointerAreaPaint={(node, color, ctx) => {
            const x = node.x || 0;
            const y = node.y || 0;
            const size = Math.min(node.value, NODE_MAX);
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);
            ctx.fill();
          }}
          linkLabel="id"
          linkColor={(link) => {
            return highlightLinks.has(link) ? linkHighlight : linkNormal;
          }}
          linkWidth={(link) => {
            return highlightLinks.has(link) ? 1.6 : 0.8;
          }}
          linkDirectionalParticleWidth={(link) => {
            return highlightLinks.has(link) ? 2.5 : 0;
          }}
          linkDirectionalParticleSpeed={0.006}
          linkDirectionalParticles={(link) =>
            highlightLinks.has(link) ? 2 : 0
          }
          linkVisibility={(link) => {
            const sourceNode = graphData?.nodes.find(
              (node) => node.id === endpointId(link.source),
            );
            const targetNode = graphData?.nodes.find(
              (node) => node.id === endpointId(link.target),
            );
            const sourceEntityType = sourceNode?.properties.entity_type || '';
            const targetEntityType = targetNode?.properties.entity_type || '';
            return (
              activeEntities.includes(sourceEntityType) &&
              activeEntities.includes(targetEntityType)
            );
          }}
        />

        {/* Legend — floating card (bottom-left), entity filter */}
        {!_.isEmpty(allEntities) && (
          <div
            className={cn(
              'bg-card/92 absolute bottom-4 left-4 z-10 rounded-xl border shadow-sm backdrop-blur-md transition-all',
              legendCollapsed ? 'p-2' : 'p-3',
            )}
            style={{ minWidth: legendCollapsed ? 0 : 180 }}
          >
            <button
              className={cn(
                'hover:bg-muted/70 flex w-full items-center gap-2 rounded-md text-left transition-colors',
                legendCollapsed ? 'px-2 py-1.5' : 'mb-2 px-1.5 py-1',
              )}
              onClick={() => setLegendCollapsed((value) => !value)}
            >
              <Layers3 className="text-muted-foreground size-3.5" />
              <span className="text-muted-foreground flex-1 font-mono text-[10px] tracking-wider uppercase">
                {page_graph('node_group')}
              </span>
              <span className="text-muted-foreground font-mono text-[10px] tabular-nums">
                {activeEntities.length}/{_.size(allEntities)}
              </span>
              {legendCollapsed ? (
                <ChevronUp className="text-muted-foreground size-3.5" />
              ) : (
                <ChevronDown className="text-muted-foreground size-3.5" />
              )}
            </button>
            {!legendCollapsed && (
              <div className="grid grid-cols-1 gap-1.5">
                {_.map(allEntities, (item, key) => {
                  const isActive = activeEntities.includes(key);
                  const entityTitleKey =
                    ENTITY_TITLE_KEYS[key as keyof typeof ENTITY_TITLE_KEYS];
                  const title = entityTitleKey
                    ? page_graph(entityTitleKey)
                    : key || page_graph('entity_UNKNOWN');
                  return (
                    <button
                      key={key}
                      className={cn(
                        'flex items-center gap-2 rounded-md px-1.5 py-1 text-left text-xs transition-opacity',
                        !isActive && 'opacity-50',
                      )}
                      onClick={() =>
                        setActiveEntities((items) => {
                          if (isActive) {
                            return _.reject(items, (i) => i === key);
                          } else {
                            return _.uniq(items.concat(key));
                          }
                        })
                      }
                    >
                      <span
                        className="size-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: resolveEntityColor(key) }}
                      />
                      <span className="text-foreground/80 truncate">
                        {title}
                      </span>
                      <span className="text-muted-foreground ml-auto font-mono text-[10px] tabular-nums">
                        {item.length}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}

        <CollectionGraphNodeDetail
          open={!mergeSuggestionOpen && Boolean(activeNode)}
          node={activeNode}
          relatedEdges={activeNodeEdges}
          relatedNodes={activeNodeNeighbors}
          onClose={handleCloseDetail}
          onSelectEdge={(edge) => {
            setActiveNode(undefined);
            setActiveEdge(edge);
          }}
          onSelectNode={(node) => {
            setActiveEdge(undefined);
            setActiveNode(node);
          }}
        />
        <CollectionGraphEdgeDetail
          open={!mergeSuggestionOpen && Boolean(activeEdge)}
          edge={activeEdge}
          onClose={handleCloseDetail}
          onSelectNode={(node) => {
            setActiveEdge(undefined);
            setActiveNode(node);
          }}
        />
        {!marketplace && (
          <CollectionGraphNodeMerge
            dataSource={mergeSuggestion ?? emptyMergeSuggestionResponse}
            open={mergeSuggestionOpen}
            onRefresh={loadMergeSuggestions}
            onRun={runMergeSuggestionScan}
            running={mergeSuggestionRunning}
            onClose={() => {
              setActiveNode(undefined);
              setMergeSuggestionOpen(false);
            }}
            onSelectNode={(id: string) => {
              const n = graphData?.nodes.find((nod) => nod.id === id);
              if (n) setActiveNode(n);
            }}
          />
        )}
      </Card>
    </div>
  );
};
