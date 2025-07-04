import { GraphEdge, GraphNode } from '@/api';
import { graphApi } from '@/services';
import { Tag, theme } from 'antd';
import * as d3 from 'd3';
import _ from 'lodash';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'umi';

import ForceGraph2D from 'react-force-graph-2d';

const color = d3.scaleOrdinal(d3.schemeCategory10);

export default () => {
  const [graphData, setGraphData] = useState<{
    edges: GraphEdge[];
    nodes: GraphNode[];
  }>();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 400 });

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [labels, setLabels] = useState<string[]>([]);
  const params = useParams();
  const { token } = theme.useToken();

  const NODE_R = 8;

  const entities = useMemo(() => {
    return _.groupBy(graphData?.nodes, (n) => n.properties.entity_type);
  }, [graphData]);

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
    setGraphData(graphRes.data);
    setLabels(labelsRes.data.labels);
    
  }, [params.collectionId]);

  const { nodes, links } = useMemo(() => {
    const _nodes =
      graphData?.nodes?.map((n) => {
        return {
          ...n,
          value:
            graphData?.edges.filter((edg) => edg.target === n.id).length || 0,
        };
      }) || [];
    const data = {
      nodes: _nodes,
      links: graphData?.edges || [],
    };
    return {
      nodes: data.nodes,
      links: data.links,
    };
  }, [graphData]);

  useEffect(() => {
    getData();
  }, []);

  useEffect(() => {
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

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        {_.map(entities, (item, key) => (
          <Tag key={key} color={color(key)}>
            {key}({item.length})
          </Tag>
        ))}
      </div>
      <div ref={containerRef}>
        <ForceGraph2D
          graphData={{ nodes, links }}
          minZoom={0.1}
          maxZoom={2}
          d3VelocityDecay={0.7}
          width={dimensions.width}
          height={dimensions.height}
          nodeLabel={(nod) => {
            return `<div style="font-size: 16px; margin-bottom: 8px">${nod.id}</div><div>${nod.properties.description}</div>`
          }}
          nodeVisibility={true}
          nodeCanvasObject={(node, ctx) => {
            const x = node.x || 0;
            const y = node.y || 0;
            const size = NODE_R + Math.min(node.value, 20);
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);
            ctx.fillStyle = color(node.properties.entity_type || '');
            ctx.fill();

            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);
            ctx.lineWidth = 1;
            ctx.strokeStyle = '#FFF';
            ctx.stroke();
          }}
          nodePointerAreaPaint={(node, color, ctx) => {
            const x = node.x || 0;
            const y = node.y || 0;
            const size = NODE_R + Math.min(node.value, 20);
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, 2 * Math.PI, false);
            ctx.fill();
          }}
          linkLabel="id"
          linkWidth={1.5}
          linkColor={() => token.colorBorder}
          linkDirectionalParticles={(edg) => {
            return edg.properties.weight ? 2 : 0;
          }}
        />
      </div>
    </div>
  );
};
