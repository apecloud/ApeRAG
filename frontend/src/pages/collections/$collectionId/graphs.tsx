import { GraphEdge, GraphNode } from '@/api';
import { graphApi } from '@/services';
import { Tag, theme } from 'antd';
import * as d3 from 'd3';
import _ from 'lodash';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useParams } from 'umi';

import ForceGraph2D from 'react-force-graph-2d';

const color = d3.scaleOrdinal(d3.schemeCategory10);

export default () => {
  const [graphData, setGraphData] = useState<{
    nodes: GraphNode[];
    links: GraphEdge[];
  }>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 400 });

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [labels, setLabels] = useState<string[]>([]);
  const [allEntities, setAllEntities] = useState<{
    [key in string]: GraphNode[];
  }>({});
  const [activeEntities, setActiveEntities] = useState<string[]>([]);

  const [highlightNodes, setHighlightNodes] = useState(new Set());
  const [highlightLinks, setHighlightLinks] = useState(new Set());
  const [hoverNode, setHoverNode] = useState<GraphNode>();

  const params = useParams();
  const { token } = theme.useToken();

  const NODE_R = 6;
  const NODE_MAX = 25;

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
        const targetCount =
          graphRes.data.edges.filter((edg) => edg.target === n.id).length || 0;
        const sourceCount =
          graphRes.data.edges.filter((edg) => edg.source === n.id).length || 0;
        return {
          ...n,
          value: Math.max(targetCount, sourceCount),
        };
      }) || [];
    const links = graphRes.data.edges || [];
    setGraphData({ nodes, links });
    setLabels(labelsRes.data.labels);
    setAllEntities(_.groupBy(nodes, (n) => n.properties.entity_type));
  }, [params.collectionId]);

  useEffect(() => {
    getData();

    const updateSize = () => {
      if (containerRef.current) {
        const size = containerRef.current?.offsetWidth || 0;
        setDimensions({
          width: size,
          height: size,
        });
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  useEffect(() => {
    if (activeEntities.length) return;
    setActiveEntities(Object.keys(allEntities));
  }, [allEntities]);

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
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
      <div ref={containerRef}>
        <ForceGraph2D
          graphData={graphData}
          minZoom={0.1}
          maxZoom={2}
          width={dimensions.width}
          height={dimensions.height}
          d3VelocityDecay={0.6}
          d3AlphaDecay={0.05}
          nodeLabel={(nod) => {
            return `<div style="font-size: 16px; margin-bottom: 8px">${nod.id}</div><div>${nod.properties.description}</div>`;
          }}
          nodeVisibility={(node) => {
            return (
              !node.properties.entity_type ||
              activeEntities.includes(node.properties.entity_type)
            );
          }}
          onNodeHover={(node) => {
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
          onLinkHover={(link) => {
            highlightNodes.clear();
            highlightLinks.clear();
            if (link) {
              highlightLinks.add(link);
              highlightNodes.add(link.source);
              highlightNodes.add(link.target);
            }
            setHighlightNodes(highlightNodes);
            setHighlightLinks(highlightLinks);
          }}
          nodeCanvasObject={(node, ctx) => {
            const x = node.x || 0;
            const y = node.y || 0;
            const size = NODE_R + Math.min(node.value, NODE_MAX);
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);
            ctx.fillStyle = color(node.properties.entity_type || '');
            ctx.fill();

            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);
            ctx.lineWidth = node === hoverNode ? NODE_R : 1;
            ctx.strokeStyle = '#FFF';
            ctx.stroke();
          }}
          nodePointerAreaPaint={(node, color, ctx) => {
            const x = node.x || 0;
            const y = node.y || 0;
            const size = NODE_R + Math.min(node.value, NODE_MAX);
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);
            ctx.fill();
          }}
          linkLabel="id"
          linkColor={(link) => {
            return highlightLinks.has(link)
              ? token.colorText
              : token.colorBorder;
          }}
          linkWidth={(link) => {
            return highlightLinks.has(link) ? 1.5 : 1.5;
          }}
          linkDirectionalParticleWidth={(link) =>
            highlightLinks.has(link) ? 3 : 0
          }
          linkDirectionalParticles={(edg) => {
            return edg.properties.weight ? 3 : 0;
          }}
          linkVisibility={(link: any) => {
            const sourceEntityType = link.source?.properties?.entity_type || '';
            const tatgetEntityType = link.target?.properties?.entity_type || '';
            return (
              activeEntities.includes(sourceEntityType) &&
              activeEntities.includes(tatgetEntityType)
            );
          }}
        />
      </div>
    </div>
  );
};
