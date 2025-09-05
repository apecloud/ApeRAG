'use client';
import {
  GraphEdge,
  GraphNode,
  KnowledgeGraph,
  MergeSuggestionsResponse,
} from '@/api';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { apiClient } from '@/lib/api/client';
import { cn } from '@/lib/utils';

import { Badge } from '@/components/ui/badge';
import Color from 'color';
import * as d3 from 'd3';
import _ from 'lodash';
import { LoaderCircle } from 'lucide-react';
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

export const CollectionGraph = ({
  marketplace = false,
}: {
  marketplace: boolean;
}) => {
  const params = useParams();

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
  const color = useMemo(() => d3.scaleOrdinal(d3.schemeCategory10), []);

  const { NODE_MIN, NODE_MAX } = useMemo(
    () => ({
      NODE_MIN: 6,
      NODE_MAX: 18,
      LINK_MIN: 18,
      LINK_MAX: 36,
    }),
    [],
  );

  const getGraphData = useCallback(async () => {
    if (typeof params.collectionId !== 'string') return;
    setLoading(true);

    let data: KnowledgeGraph;

    if (!marketplace) {
      const res = await apiClient.graphApi.collectionsCollectionIdGraphsGet(
        {
          collectionId: params.collectionId,
        },
        {
          timeout: 1000 * 20,
        },
      );
      data = res.data;
    } else {
      const res =
        await apiClient.defaultApi.marketplaceCollectionsCollectionIdGraphGet(
          {
            collectionId: params.collectionId,
          },
          {
            timeout: 1000 * 20,
          },
        );
      data = res.data as KnowledgeGraph;
    }

    const nodes =
      data.nodes?.map((n) => {
        const targetCount = data.edges.filter(
          (edg) => edg.target === n.id,
        ).length;
        const sourceCount = data.edges.filter(
          (edg) => edg.source === n.id,
        ).length;
        return {
          ...n,
          value: Math.max(targetCount, sourceCount, NODE_MIN),
        };
      }) || [];
    const links = data.edges || [];

    setGraphData({ nodes, links });

    setAllEntities(_.groupBy(nodes, (n) => n.properties.entity_type));

    setLoading(false);
  }, [NODE_MIN, marketplace, params.collectionId]);

  const getMergeSuggestions = useCallback(async () => {
    if (typeof params.collectionId !== 'string' || marketplace) return;
    const suggestionRes =
      await apiClient.graphApi.collectionsCollectionIdGraphsMergeSuggestionsPost(
        {
          collectionId: params.collectionId,
        },
        {
          timeout: 1000 * 20,
        },
      );
    setMergeSuggestion(suggestionRes.data);
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
  }, [handleResizeContainer]);

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
    // init the graph config
    // graphRef.current
    //   ?.d3Force(
    //     'link',
    //     d3.forceLink().distance((link) => {
    //       console.log(link)
    //       return Math.min(
    //         Math.max(link.source.value, link.target.value, LINK_MIN),
    //         LINK_MAX,
    //       );
    //     }),
    //   )
    //   .d3Force('collision', d3.forceCollide().radius(NODE_MIN))
    //   .d3Force('charge', d3.forceManyBody().strength(-40))
    //   .d3Force('x', d3.forceX())
    //   .d3Force('y', d3.forceY());
  }, [getGraphData, getMergeSuggestions]);

  return (
    <>
      <div className="mb-4 flex flex-row items-center justify-between gap-4">
        {/* <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline">
              <Columns3 />
              <span className="hidden lg:inline">
                {page_graph('node_group')}
              </span>
              <ChevronDown />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-56">
            {_.map(allEntities, (item, key) => {
              const isActive = activeEntities.includes(key);
              //@ts-expect-error entity error
              const title = page_graph(`entity_${key}`);
              return (
                <DropdownMenuCheckboxItem
                  key={key}
                  className={cn('capitalize')}
                  checked={isActive}
                  onCheckedChange={(value) =>
                    setActiveEntities((items) => {
                      if (!value) {
                        return _.reject(items, (item) => item === key);
                      } else {
                        return _.uniq(items.concat(key));
                      }
                    })
                  }
                >
                  <div className="flex flex-row items-center gap-1">
                    <div
                      className="size-2 rounded-full"
                      style={{ background: color(key) }}
                    />
                    <span>
                      {title} ({item.length})
                    </span>
                  </div>
                </DropdownMenuCheckboxItem>
              );
            })}
          </DropdownMenuContent>
        </DropdownMenu> */}

        <div className="flex flex-row flex-wrap gap-1">
          {_.map(allEntities, (item, key) => {
            const isActive = activeEntities.includes(key);
            //@ts-expect-error entity error
            const title = page_graph(`entity_${key}`);
            return (
              <Badge
                key={key}
                className={cn(
                  'cursor-pointer capitalize',
                  isActive ? '' : 'border-transparent',
                )}
                style={{
                  backgroundColor: color(key),
                  opacity: isActive ? 1 : 0.7,
                }}
                onClick={() =>
                  setActiveEntities((items) => {
                    if (isActive) {
                      return _.reject(items, (item) => item === key);
                    } else {
                      return _.uniq(items.concat(key));
                    }
                  })
                }
              >
                {title} ({item.length})
              </Badge>
            );
          })}
        </div>

        <div className="flex flex-row items-center gap-2">
          {!marketplace && !_.isEmpty(mergeSuggestion?.suggestions) && (
            <div className="text-sm">
              <span
                className="text-muted-foreground hover:text-primary cursor-pointer"
                onClick={() => setMergeSuggestionOpen(true)}
              >
                {page_graph('merge_infomation', {
                  count: String(mergeSuggestion?.pending_count || 0),
                })}
              </span>
            </div>
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
        </div>
      </div>

      <Card
        ref={containerRef}
        className="bg-card/0 relative flex flex-1 gap-0 py-0"
      >
        {graphData === undefined && (
          <div className="absolute top-4/12 left-6/12">
            <div className="flex flex-row gap-2 py-2">
              <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-0"></div>
              <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-200"></div>
              <div className="bg-muted-foreground animate-caret-blink size-2 rounded-full delay-400"></div>
            </div>
          </div>
        )}

        {graphData !== undefined && _.isEmpty(graphData?.nodes) && (
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
          nodeVisibility={(node) => {
            return (
              !node.properties.entity_type ||
              activeEntities.includes(node.properties.entity_type)
            );
          }}
          minZoom={0.5}
          maxZoom={8}
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
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);

            const colorNormal = color(node.properties.entity_type || '');
            const colorSecondary =
              resolvedTheme === 'dark'
                ? Color(colorNormal).grayscale().darken(0.3)
                : Color(colorNormal).grayscale().lighten(0.6);
            ctx.fillStyle =
              highlightNodes.size === 0 || highlightNodes.has(node)
                ? colorNormal
                : colorSecondary.string();
            ctx.fill();

            // node circle
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);
            ctx.lineWidth = 1;
            ctx.strokeStyle = highlightNodes.has(node)
              ? Color('#FFF').grayscale().string()
              : '#FFF';
            ctx.stroke();

            // node label
            let fontSize = 16;
            const offset = 2;
            ctx.font = `${fontSize}px Arial`;
            let textWidth = ctx.measureText(String(node.id)).width - offset;
            do {
              fontSize -= 2;
              ctx.font = `${fontSize}px Arial`;
              textWidth = ctx.measureText(String(node.id)).width - offset;
            } while (textWidth > size && fontSize > 0);
            ctx.fillStyle = '#fff';
            ctx.fillText(
              String(node.id),
              x - (textWidth + offset) / 2,
              y + fontSize / 2,
            );
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
            if (resolvedTheme === 'dark') {
              return highlightLinks.has(link) ? '#585858' : '#383838';
            } else {
              return highlightLinks.has(link) ? '#BBB' : '#DDD';
            }
          }}
          linkWidth={(link) => {
            return highlightLinks.has(link) ? 2 : 1;
          }}
          linkDirectionalParticleWidth={(link) => {
            return highlightLinks.has(link) ? 3 : 0;
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
    </>
  );
};
