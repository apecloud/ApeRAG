import { ApeNode } from '@/types';
import { CaretRightOutlined } from '@ant-design/icons';
import { applyNodeChanges, NodeChange } from '@xyflow/react';
import { useDebounce } from 'ahooks';
import { Collapse, Form, Select, Switch, theme } from 'antd';
import _ from 'lodash';
import { useCallback, useEffect, useMemo } from 'react';
import { useIntl, useModel } from 'umi';
import { NodeInput } from './_node-input';
import { NodeOutputs } from './_outputs';
import { OutputParams } from './_outputs_params';
import { getCollapsePanelStyle } from './_styles';

export const ApeNodeMerge = ({ node }: { node: ApeNode }) => {
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const { nodes, setNodes, edges } = useModel('bots.$botId.flow.model');

  const values = useMemo(() => node.data.input?.values || [], [node]);
  const applyChanges = useCallback(() => {
    setNodes((nds) => {
      const changes: NodeChange[] = [
        { id: node.id, type: 'replace', item: node },
      ];
      return applyNodeChanges(changes, nds) as ApeNode[];
    });
  }, [node]);

  const { refVectorSearchNode, refFulltextSearchNode, refGraphSearchNode } =
    useMemo(() => {
      const nid = node?.id;
      const connects = edges.filter((edg) => edg.target === nid);
      const sourceNodes = connects.map((edg) =>
        nodes.find((nod) => nod.id === edg.source),
      );
      return {
        refVectorSearchNode: sourceNodes?.find(
          (nod) => nod?.type === 'vector_search',
        ),
        refFulltextSearchNode: sourceNodes?.find(
          (nod) => nod?.type === 'fulltext_search',
        ),
        refGraphSearchNode: sourceNodes?.find(
          (nod) => nod?.type === 'graph_search',
        ),
      };
    }, [edges, nodes]);

  const debouncedRefVectorSearchNode = useDebounce(refVectorSearchNode, {
    wait: 300,
  });
  const debouncedRefFulltextSearchNode = useDebounce(refFulltextSearchNode, {
    wait: 300,
  });
  const debouncedRefGraphSearchNode = useDebounce(refGraphSearchNode, {
    wait: 300,
  });

  useEffect(() => {
    _.set(
      values,
      'vector_search_docs',
      refVectorSearchNode?.id
        ? `{{ nodes.${refVectorSearchNode.id}.output.docs }}`
        : [],
    );
    applyChanges();
  }, [debouncedRefVectorSearchNode]);

  useEffect(() => {
    _.set(
      values,
      'fulltext_search_docs',
      refFulltextSearchNode?.id
        ? `{{ nodes.${refFulltextSearchNode.id}.output.docs }}`
        : [],
    );
    applyChanges();
  }, [debouncedRefFulltextSearchNode]);

  useEffect(() => {
    _.set(
      values,
      'graph_search_docs',
      refGraphSearchNode?.id
        ? `{{ nodes.${refGraphSearchNode.id}.output.docs }}`
        : [],
    );
    applyChanges();
  }, [debouncedRefGraphSearchNode]);

  return (
    <>
      <Collapse
        bordered={false}
        expandIcon={({ isActive }) => {
          return <CaretRightOutlined rotate={isActive ? 90 : 0} />;
        }}
        size="middle"
        defaultActiveKey={['1']}
        style={{ background: 'none' }}
        items={[
          {
            key: '1',
            label: formatMessage({ id: 'flow.input.params' }),
            style: getCollapsePanelStyle(token),
            children: (
              <Form layout="vertical" autoComplete="off">
                <Form.Item
                  required
                  label={formatMessage({ id: 'flow.merge.merge_strategy' })}
                >
                  <Select
                    variant="filled"
                    suffixIcon={null}
                    options={[{ label: 'Union', value: 'union' }]}
                    value={_.get(values, 'merge_strategy')}
                    onChange={(name) => {
                      _.set(values, 'merge_strategy', name);
                      applyChanges();
                    }}
                  />
                </Form.Item>
                <Form.Item
                  required
                  label={formatMessage({ id: 'flow.merge.deduplicate' })}
                  valuePropName="checked"
                  tooltip={formatMessage({ id: 'flow.merge.deduplicate.tips' })}
                >
                  <Switch
                    size="small"
                    checked={_.get(values, 'deduplicate')}
                    onChange={(checked) => {
                      _.set(values, 'deduplicate', checked);
                      applyChanges();
                    }}
                  />
                </Form.Item>
                <Form.Item
                  required
                  label={formatMessage({ id: 'flow.vector_search.source' })}
                >
                  <NodeInput
                    disabled
                    variant="filled"
                    placeholder={formatMessage({
                      id: 'flow.connection.required',
                    })}
                    value={_.get(values, 'vector_search_docs')}
                    onChange={(e) => {
                      _.set(
                        values,
                        'vector_search_docs',
                        e.currentTarget.value,
                      );
                      applyChanges();
                    }}
                  />
                </Form.Item>
                <Form.Item
                  required
                  label={formatMessage({ id: 'flow.fulltext_search.source' })}
                >
                  <NodeInput
                    disabled
                    variant="filled"
                    placeholder={formatMessage({
                      id: 'flow.connection.required',
                    })}
                    value={_.get(values, 'fulltext_search_docs')}
                    onChange={(e) => {
                      _.set(
                        values,
                        'fulltext_search_docs',
                        e.currentTarget.value,
                      );
                      applyChanges();
                    }}
                  />
                </Form.Item>
                <Form.Item
                  required
                  style={{ marginBottom: 0 }}
                  label={formatMessage({ id: 'flow.graph_search.source' })}
                >
                  <NodeInput
                    disabled
                    variant="filled"
                    placeholder={formatMessage({
                      id: 'flow.connection.required',
                    })}
                    value={_.get(values, 'graph_search_docs')}
                    onChange={(e) => {
                      _.set(values, 'graph_search_docs', e.currentTarget.value);
                      applyChanges();
                    }}
                  />
                </Form.Item>
              </Form>
            ),
          },
          {
            key: '2',
            label: formatMessage({ id: 'flow.output.params' }),
            style: getCollapsePanelStyle(token),
            children: <OutputParams node={node} />,
            extra: <NodeOutputs node={node} />,
          },
        ]}
      />
    </>
  );
};
