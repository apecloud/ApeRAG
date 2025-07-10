import { GraphEdge, GraphNode } from '@/api';
import { graphApi } from '@/services';
import { Card, Tag, theme } from 'antd';
import Color from 'color';
import * as d3 from 'd3';
import _ from 'lodash';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useModel, useParams } from 'umi';

import ForceGraph2D from 'react-force-graph-2d';
import { NodeDetail } from './NodeDetail';

const color = d3.scaleOrdinal(d3.schemeCategory10);

// type HierarchicalNode = GraphNode & { children: GraphNode[] };

export default () => {
  const params = useParams();
  const { token } = theme.useToken();
  const { themeName } = useModel('global');
  const [graphData, setGraphData] = useState<{
    nodes: GraphNode[];
    links: GraphEdge[];
  }>();
  const containerRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<any>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [labels, setLabels] = useState<string[]>([]);
  const [allEntities, setAllEntities] = useState<{
    [key in string]: GraphNode[];
  }>({});
  const [activeEntities, setActiveEntities] = useState<string[]>([]);

  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  const [hoverNode, setHoverNode] = useState<GraphNode>();
  const [activeNode, setActiveNode] = useState<GraphNode>();

  // const hierarchicalNodes: HierarchicalNode[] = useMemo(() => {
  //   const nodeMap = new Map();
  //   const rootNodes: HierarchicalNode[] = [];
  //   graphData?.nodes.forEach((node) => {
  //     nodeMap.set(node.id, {
  //       ...node,
  //       children: [],
  //     });
  //   });
  //   graphData?.links.forEach((edge) => {
  //     const { source, target } = edge;
  //     const parentNode = nodeMap.get(source);
  //     const childNode = nodeMap.get(target);
  //     if (parentNode && childNode) {
  //       parentNode.children.push(childNode);
  //     }
  //   });
  //   const childNodeIds = new Set();
  //   graphData?.links.forEach((edge) => {
  //     childNodeIds.add(edge.target);
  //   });
  //   graphData?.nodes.forEach((node) => {
  //     if (!childNodeIds.has(node.id)) {
  //       rootNodes.push(nodeMap.get(node.id));
  //     }
  //   });
  //   return rootNodes;
  // }, [graphData]);

  const { NODE_MIN, NODE_MAX, LINK_MIN, LINK_MAX } = useMemo(
    () => ({
      NODE_MIN: 6,
      NODE_MAX: 18,
      LINK_MIN: 18,
      LINK_MAX: 36,
    }),
    [],
  );

  const getData = useCallback(async () => {
    if (!params.collectionId) return;
    const [graphRes, labelsRes] = await Promise.all([
      graphApi.collectionsCollectionIdGraphsGet({
        collectionId: params.collectionId,
      }),
      graphApi.collectionsCollectionIdGraphsLabelsGet({
        collectionId: params.collectionId,
      }),
    ]);
    const nodes =
      graphRes.data.nodes?.map((n) => {
        const targetCount = graphRes.data.edges.filter(
          (edg) => edg.target === n.id,
        ).length;
        const sourceCount = graphRes.data.edges.filter(
          (edg) => edg.source === n.id,
        ).length;
        return {
          ...n,
          value: Math.max(targetCount, sourceCount, NODE_MIN),
        };
      }) || [];
    const links = graphRes.data.edges || [];

    setGraphData({ nodes, links });
    setLabels(labelsRes.data.labels);
    setAllEntities(_.groupBy(nodes, (n) => n.properties.entity_type));
  }, [params.collectionId]);

  const handleCloseDetail = useCallback(() => {
    setActiveNode(undefined);
    setHoverNode(undefined);
    highlightNodes.clear();
    highlightLinks.clear();
  }, []);

  useEffect(() => {
    if (activeEntities.length) return;
    setActiveEntities(Object.keys(allEntities));
  }, [allEntities]);

  useEffect(() => {
    // update size
    const updateSize = () => {
      if (containerRef.current) {
        const width = containerRef.current?.offsetWidth || 0;
        const height = containerRef.current?.offsetHeight || 0;
        setDimensions({
          width: width - 2,
          height: height - 2,
        });
      }
    };
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, [graphData]);

  useEffect(() => {
    if (activeNode) {
      graphRef.current.centerAt(activeNode.x, activeNode.y, 300);
      graphRef.current.zoom(3, 800);
    } else {
      graphRef.current.centerAt(0, 0, 300);
      graphRef.current.zoom(1.5, 800);
    }
  }, [activeNode]);

  useEffect(() => {
    getData();
    graphRef.current
      .d3Force(
        'link',
        d3.forceLink().distance((link) => {
          return Math.min(
            Math.max(link.source.value, link.target.value, LINK_MIN),
            LINK_MAX,
          );
        }),
      )
      .d3Force('collision', d3.forceCollide().radius(NODE_MIN))
      .d3Force('charge', d3.forceManyBody().strength(-40))
      .d3Force('x', d3.forceX())
      .d3Force('y', d3.forceY());
  }, [graphRef]);

  return (
    <Card
      ref={containerRef}
      styles={{
        body: {
          padding: 0,
          position: 'relative',
          height: 'calc(100vh - 180px)',
          cursor: hoverNode ? 'pointer' : 'default',
        },
      }}
    >
      <ForceGraph2D
        graphData={graphData}
        width={dimensions.width}
        height={dimensions.height}
        nodeLabel={(nod) => nod.id}
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
          if (activeNode === node) {
            handleCloseDetail();
            return;
          }

          setActiveNode(node);
          highlightNodes.clear();
          highlightLinks.clear();

          if (node) {
            const nodeLinks = graphData?.links.filter((link: any) => {
              return link.source.id === node.id || link.target.id === node.id;
            });
            nodeLinks?.forEach((link: GraphEdge) => {
              highlightLinks.add(link);
              highlightNodes.add(link.source);
              highlightNodes.add(link.target);
            });
          }
          setHoverNode(node || undefined);
          setHighlightNodes(highlightNodes);
          setHighlightLinks(highlightLinks);
        }}
        onNodeHover={(node) => {
          if (activeNode) return;
          highlightNodes.clear();
          highlightLinks.clear();
          if (node) {
            const nodeLinks = graphData?.links.filter((link: any) => {
              return link.source.id === node.id || link.target.id === node.id;
            });
            nodeLinks?.forEach((link: GraphEdge) => {
              highlightLinks.add(link);
              // highlightNodes.add(link.source);
              // highlightNodes.add(link.target);
            });
          }
          setHoverNode(node || undefined);
          setHighlightNodes(highlightNodes);
          setHighlightLinks(highlightLinks);
        }}
        onLinkHover={(link) => {
          if (activeNode) return;
          highlightNodes.clear();
          highlightLinks.clear();
          if (link) {
            highlightLinks.add(link);
            // highlightNodes.add(link.source);
            // highlightNodes.add(link.target);
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
          let colorSecondary =
            themeName === 'dark'
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
          let fontSize = 15;
          const offset = 2;
          ctx.font = `${fontSize}px Arial`;
          let textWidth = ctx.measureText(node.id).width - offset;
          do {
            fontSize -= 1;
            ctx.font = `${fontSize}px Arial`;
            textWidth = ctx.measureText(node.id).width - offset;
          } while (textWidth > size && fontSize > 0);
          ctx.fillStyle = '#fff';
          ctx.fillText(node.id, x - (textWidth + offset) / 2, y + fontSize / 2);
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
          if (themeName === 'dark') {
            return highlightLinks.has(link) ? '#EEE' : '#55595F';
          } else {
            return highlightLinks.has(link) ? '#999' : '#DDD';
          }
        }}
        linkWidth={(link) => {
          return highlightLinks.has(link) ? 2 : 1;
        }}
        linkDirectionalParticleWidth={(link) => {
          return highlightLinks.has(link) ? 3 : 0;
        }}
        linkDirectionalParticles={2}
        linkVisibility={(link: any) => {
          const sourceEntityType = link.source?.properties?.entity_type || '';
          const tatgetEntityType = link.target?.properties?.entity_type || '';
          return (
            activeEntities.includes(sourceEntityType) &&
            activeEntities.includes(tatgetEntityType)
          );
        }}
      />

      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          top: 0,
          padding: 16,
          background: token.colorBgContainer,
        }}
      >
        {_.map(allEntities, (item, key) => {
          const isActive = activeEntities.includes(key);
          return (
            <Tag
              key={key}
              color={isActive ? color(key) : undefined}
              style={{ cursor: 'pointer' }}
              onClick={() => {
                setActiveEntities((items) => {
                  if (isActive) {
                    return _.reject(items, (item) => item === key);
                  } else {
                    return _.uniq(items.concat(key));
                  }
                });
              }}
            >
              {key}({item.length})
            </Tag>
          );
        })}
      </div>
      <NodeDetail node={activeNode} onClose={handleCloseDetail} />
    </Card>
  );
};
