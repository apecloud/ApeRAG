import { GraphEdge, GraphNode } from '@/api';
import { graphApi } from '@/services';
import { Card, Tag, theme, Typography } from 'antd';
import * as d3 from 'd3';
import _ from 'lodash';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'umi';

const color = d3.scaleOrdinal(d3.schemeCategory10);

export default () => {
  const [graphData, setGraphData] = useState<{
    edges: GraphEdge[];
    nodes: GraphNode[];
  }>();

  const [hoverNode, setHoverNode] = useState<GraphNode>();
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [selectNode, setSelectNode] = useState<GraphNode>();

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [labels, setLabels] = useState<string[]>([]);
  const params = useParams();
  const ref = useRef<HTMLDivElement>(null);

  const { token } = theme.useToken();

  const entityTypes = useMemo(() => {
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

  useEffect(() => {
    getData();
  }, []);

  useEffect(() => {
    const width = ref.current?.clientWidth || 800;
    const height = width;

    const links =
      graphData?.edges.map((d) => ({
        id: d.id,
        source: d.source,
        target: d.target,
      })) || [];

    const nodes =
      graphData?.nodes.map((d) => ({
        ...d,
      })) || [];

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        'link',
        d3
          .forceLink(links)
          .distance(60)
          .id((d) => d.id),
      )
      .force('charge', d3.forceManyBody().strength(-40))
      // .force('center', d3.forceCenter(0, 0).strength(1))
      .force('x', d3.forceX())
      .force('y', d3.forceY());

    const svg = d3
      .create('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [-width / 2, -height / 2, width, height])
      .attr('style', 'max-width: 100%; height: auto;');

    const link = svg
      .append('g')
      .attr('stroke', token.colorText)
      .attr('stroke-opacity', 0.2)
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke-width', 1);
    // .attr("stroke-width", d => Math.sqrt(d.value));

    const node = svg
      .append('g')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1)
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', (d) => {
        const size = _.size(
          graphData?.edges.filter((edge) => edge.target === d.id),
        );
        return 4 + size;
      })
      .style('cursor', 'pointer')
      .attr('fill', (d) => color(d.properties.entity_type || ''))
      .on('mouseover', function (e, data) {
        d3.select(this).transition().duration(300).attr('stroke-width', 2);
        setHoverNode(data);
      })
      .on('mouseout', function () {
        d3.select(this).transition().duration(300).attr('stroke-width', 1);
      })
      .on('click', function (e, data) {
        setSelectNode(data);
      });

    node.append('title').text((d) => d.id);

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);

      node.attr('cx', (d) => d.x).attr('cy', (d) => d.y);
    });

    // Reheat the simulation when drag starts, and fix the subject position.
    function dragstarted(event) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    // Update the subject (dragged node) position during drag.
    function dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    // Restore the target alpha so the simulation cools after dragging ends.
    // Unfix the subject position now that it’s no longer being dragged.
    function dragended(event) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    node.call(
      d3
        .drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended),
    );

    const zoom = d3
      .zoom()
      .scaleExtent([1, 10])
      .on('zoom', (event) => {
        svg.attr('transform', event.transform);
      });

    svg
      .call(zoom)
      .on('mousedown.zoom', null)
      .on('touchstart.zoom', null)
      .on('dblclick.zoom', null);

    ref.current?.replaceChildren(svg.node());
  }, [graphData, token]);

  return (
    <div>
      <div style={{ marginBottom: 20 }}>
        {_.map(entityTypes, (item, key) => (
          <Tag key={key} color={color(key)}>
            {key}({item.length})
          </Tag>
        ))}
      </div>
      <div style={{ display: 'flex', flexDirection: 'row', gap: 20 }}>
        <Card
          ref={ref}
          style={{ overflow: 'hidden', flex: 1, height: 'auto' }}
        ></Card>
        <Card title={hoverNode?.id} style={{ width: 300, right: 0, top: 0 }}>
          <Typography.Text>{hoverNode?.properties.description}</Typography.Text>
        </Card>
      </div>
    </div>
  );
};
