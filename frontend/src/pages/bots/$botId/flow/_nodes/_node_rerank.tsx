import { ModelSelect } from '@/components';
import { ApeNode } from '@/types';
import { CaretRightOutlined } from '@ant-design/icons';
import { applyNodeChanges, NodeChange } from '@xyflow/react';
import { useDebounce } from 'ahooks';
import { Collapse, Form, Space, theme } from 'antd';
import _ from 'lodash';
import { useCallback, useEffect, useMemo } from 'react';
import { useIntl, useModel } from 'umi';
import { NodeInput } from './_node-input';
import { NodeOutputs } from './_outputs';
import { OutputParams } from './_outputs_params';
import { getCollapsePanelStyle } from './_styles';

export const ApeNodeRerank = ({ node }: { node: ApeNode }) => {
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const { nodes, edges, setNodes } = useModel('bots.$botId.flow.model');
  const { getProviderByModelName } = useModel('models');

  const values = useMemo(() => node.data.input?.values || [], [node]);
  const applyChanges = useCallback(() => {
    setNodes((nds) => {
      const changes: NodeChange[] = [
        { id: node.id, type: 'replace', item: node },
      ];
      return applyNodeChanges(changes, nds) as ApeNode[];
    });
  }, [node]);

  const { refNode } = useMemo(() => {
    const nid = node.id;
    const connects = edges.filter((edg) => edg.target === nid);
    const sourceNodes = connects.map((edg) =>
      nodes.find(
        (nod) =>
          nod.id === edg.source && nod.data.output?.schema?.properties?.docs,
      ),
    );
    const _refNode =
      _.size(sourceNodes) === 1 ? _.first(sourceNodes) : undefined;

    return { refNode: _refNode };
  }, [edges, nodes]);

  const debouncedRefNode = useDebounce(refNode, { wait: 300 });
  useEffect(() => {
    _.set(
      values,
      'docs',
      refNode?.id ? `{{ nodes.${refNode.id}.output.docs }}` : [],
    );
    applyChanges();
  }, [debouncedRefNode]);

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
            label: <Space>{formatMessage({ id: 'flow.input.params' })}</Space>,
            style: getCollapsePanelStyle(token),
            children: (
              <Form layout="vertical" autoComplete="off">
                <Form.Item
                  required
                  tooltip={formatMessage({ id: 'model.rerank.tips' })}
                  label={formatMessage({ id: 'flow.reranker.model' })}
                >
                  <ModelSelect
                    model="rerank"
                    variant="filled"
                    value={_.get(values, 'model')}
                    onChange={(name) => {
                      _.set(values, 'model', name);
                      _.set(
                        values,
                        'custom_llm_provider',
                        getProviderByModelName(name, 'rerank').model
                          ?.custom_llm_provider,
                      );
                      _.set(
                        values,
                        'model_service_provider',
                        getProviderByModelName(name, 'rerank').provider?.name,
                      );
                      applyChanges();
                    }}
                  />
                </Form.Item>

                <Form.Item
                  required
                  style={{ marginBottom: 0 }}
                  label={formatMessage({ id: 'flow.input.source' })}
                >
                  <NodeInput
                    disabled
                    variant="filled"
                    placeholder={formatMessage({
                      id: 'flow.connection.required',
                    })}
                    value={_.get(values, 'docs')}
                    onChange={(e) => {
                      _.set(values, 'docs', e.currentTarget.value);
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
