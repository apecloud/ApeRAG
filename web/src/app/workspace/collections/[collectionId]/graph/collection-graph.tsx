'use client';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import {
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
  LoaderCircle,
  Maximize,
  Minimize,
} from 'lucide-react';
import { useTranslations } from 'next-intl';
import { useTheme } from 'next-themes';
import dynamic from 'next/dynamic';
import { useParams } from 'next/navigation';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CollectionGraphNodeDetail } from './collection-graph-node-detail';
import { CollectionGraphNodeMerge } from './collection-graph-node-merge';

const ForceGraph2D = dynamic(
  () => import('react-force-graph-2d').then((r) => r),
  {
    ssr: false,
  },
);

const resolveEntityColor = (entityType: string | null | undefined): string =>
  ENTITY_PALETTE[entityTypeToPaletteKey(entityType)];

const getErrorMessage = (error: unknown, fallback: string) => {
  if (error instanceof ApiClientError || error instanceof Error) {
    return error.message || fallback;
  }
  return fallback;
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

  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  const [allEntities, setAllEntities] = useState<{
    [key in string]: GraphNode[];
  }>({});
  const [activeEntities, setActiveEntities] = useState<string[]>([]);

  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  const [hoverNode, setHoverNode] = useState<GraphNode>();
  const [activeNode, setActiveNode] = useState<GraphNode>();

  const { NODE_MIN, NODE_MAX } = useMemo(
    () => ({
      NODE_MIN: 7,
      NODE_MAX: 24,
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

  const getMergeSuggestions = useCallback(async () => {
    if (typeof params.collectionId !== 'string' || marketplace) return;
    try {
      const suggestionRes = await runMergeSuggestions(params.collectionId);
      setMergeSuggestion(suggestionRes);
    } catch {
      setMergeSuggestion(undefined);
    }
  }, [marketplace, params.collectionId]);

  const handleCloseDetail = useCallback(() => {
    setActiveNode(undefined);
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
          // @ts-expect-error link.source.id link.target.id
          link.source.id === activeNode.id || link.target.id === activeNode.id
        );
      });
      nodeLinks?.forEach((link: GraphEdge) => {
        highlightLinks.add(link);
        highlightNodes.add(link.source);
        highlightNodes.add(link.target);
      });
      highlightNodes.add(activeNode);
      // @ts-expect-error node.x node.y
      graphRef.current?.centerAt(activeNode.x, activeNode.y, 400);
      graphRef.current?.zoom(3, 600);
    } else {
      graphRef.current?.centerAt(0, 0, 400);
      graphRef.current?.zoom(1.5, 600);
    }
    setHighlightNodes(highlightNodes);
    setHighlightLinks(highlightLinks);
  }, [activeNode, graphData?.links, highlightLinks, highlightNodes]);

  useEffect(() => {
    getGraphData();
    getMergeSuggestions();
  }, [getGraphData, getMergeSuggestions]);

  const isDark = resolvedTheme === 'dark';
  const nodeStroke = isDark ? CANVAS_DARK.nodeStroke : COLORS.bg;
  const linkNormal = isDark ? CANVAS_DARK.linkNormal : COLORS.border;
  const linkHighlight = isDark
    ? CANVAS_DARK.linkHighlight
    : COLORS.borderStrong;
  const labelFill = isDark ? COLORS.bg : COLORS.fg;

  const totalNodes = graphData?.nodes.length ?? 0;
  const totalEdges = graphData?.links.length ?? 0;

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
            <span className="text-muted-foreground font-mono text-xs tabular-nums">
              {totalNodes.toLocaleString()} · {totalEdges.toLocaleString()}
            </span>
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

          {!marketplace && !_.isEmpty(mergeSuggestion?.suggestions) && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge
                  variant="outline"
                  className="h-6 min-w-6 cursor-pointer rounded-full px-1.5 font-mono tabular-nums"
                  style={{
                    backgroundColor: COLORS.accentSoft,
                    color: COLORS.accentInk,
                    borderColor: COLORS.subtleStrong,
                  }}
                  onClick={() => setMergeSuggestionOpen(true)}
                >
                  {mergeSuggestion?.suggestions?.length &&
                  mergeSuggestion?.suggestions?.length > 10
                    ? '10+'
                    : mergeSuggestion?.suggestions?.length}
                </Badge>
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
              getMergeSuggestions();
            }}
          >
            <LoaderCircle className={loading ? 'animate-spin' : ''} />
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
            setActiveNode(node as GraphNode);
          }}
          onNodeHover={(node) => {
            if (activeNode) return;
            highlightNodes.clear();
            highlightLinks.clear();
            if (node) {
              const nodeLinks = graphData?.links.filter((link) => {
                //@ts-expect-error link.source.id link.target.id
                return link.source.id === node.id || link.target.id === node.id;
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
            if (activeNode) return;
            highlightNodes.clear();
            highlightLinks.clear();
            if (link) {
              highlightLinks.add(link);
            }
            setHighlightNodes(highlightNodes);
            setHighlightLinks(highlightLinks);
          }}
          nodeCanvasObject={(node, ctx) => {
            const x = node.x || 0;
            const y = node.y || 0;

            let size = Math.min(node.value, NODE_MAX);
            if (node === hoverNode) size += 1;

            const entityColor = resolveEntityColor(node.properties.entity_type);
            const isDim = highlightNodes.size > 0 && !highlightNodes.has(node);
            const isActive = activeNode?.id === node.id;

            // soft halo under large / active nodes
            if (size >= 14 || isActive) {
              ctx.beginPath();
              ctx.arc(x, y, size + 3, 0, 2 * Math.PI, false);
              ctx.fillStyle = entityColor;
              ctx.globalAlpha = isDim ? 0.06 : 0.18;
              ctx.fill();
              ctx.globalAlpha = 1;
            }

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

            // active node: dashed accent ring
            if (isActive) {
              ctx.beginPath();
              ctx.setLineDash([3, 3]);
              ctx.arc(x, y, size + 6, 0, 2 * Math.PI, false);
              ctx.lineWidth = 1.2;
              ctx.strokeStyle = COLORS.accent;
              ctx.globalAlpha = 0.55;
              ctx.stroke();
              ctx.setLineDash([]);
              ctx.globalAlpha = 1;
            }

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
            ctx.fillStyle = labelFill;
            ctx.globalAlpha = isDim ? 0.4 : 1;
            ctx.fillText(
              String(node.id),
              x - (textWidth + offset) / 2,
              y + size + fontSize + 2,
            );
            ctx.globalAlpha = 1;
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
          linkDirectionalParticles={2}
          linkVisibility={(link) => {
            // @ts-expect-error link.source.properties
            const sourceEntityType = link.source?.properties?.entity_type || '';

            // @ts-expect-error link.source.properties
            const tatgetEntityType = link.target?.properties?.entity_type || '';
            return (
              activeEntities.includes(sourceEntityType) &&
              activeEntities.includes(tatgetEntityType)
            );
          }}
        />

        {/* Legend — floating card (bottom-left), entity filter */}
        {!_.isEmpty(allEntities) && (
          <div
            className="bg-card absolute bottom-4 left-4 z-10 rounded-xl border p-3 shadow-sm"
            style={{ minWidth: 180 }}
          >
            <div className="text-muted-foreground mb-2 font-mono text-[10px] tracking-wider uppercase">
              {page_graph('node_group')}
            </div>
            <div className="grid grid-cols-1 gap-1.5">
              {_.map(allEntities, (item, key) => {
                const isActive = activeEntities.includes(key);
                //@ts-expect-error entity i18n key constructed dynamically
                const title = page_graph(`entity_${key}`);
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
                    <span className="text-foreground/80 truncate">{title}</span>
                    <span className="text-muted-foreground ml-auto font-mono text-[10px] tabular-nums">
                      {item.length}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <CollectionGraphNodeDetail
          open={!mergeSuggestionOpen && Boolean(activeNode)}
          node={activeNode}
          onClose={handleCloseDetail}
        />
        {mergeSuggestion && (
          <CollectionGraphNodeMerge
            dataSource={mergeSuggestion}
            open={mergeSuggestionOpen}
            onRefresh={getMergeSuggestions}
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
