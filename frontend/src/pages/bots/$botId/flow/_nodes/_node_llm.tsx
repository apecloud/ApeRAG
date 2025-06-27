import { ModelSelect } from '@/components';
import { ApeNode } from '@/types';
import { CaretRightOutlined } from '@ant-design/icons';
import { applyNodeChanges, NodeChange } from '@xyflow/react';
import { useDebounce } from 'ahooks';
import { Collapse, Form, InputNumber, Slider, theme } from 'antd';
import _ from 'lodash';
import { useCallback, useEffect, useMemo } from 'react';
import { useIntl, useModel } from 'umi';
import { NodeInput, NodeInputTextArea } from './_node-input';
import { OutputParams } from './_outputs_params';
import { getCollapsePanelStyle } from './_styles';

export const ApeNodeLlm = ({ node }: { node: ApeNode }) => {
  const { token } = theme.useToken();
  const { nodes, setNodes, edges } = useModel('bots.$botId.flow.model');
  const { getProviderByModelName } = useModel('models');
  const { formatMessage } = useIntl();

  const schema = useMemo(() => node.data.input.schema, [node]);
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
            label: formatMessage({ id: 'flow.input.params' }),
            style: getCollapsePanelStyle(token),
            children: (
              <>
                <Form layout="vertical" autoComplete="off">
                  <Form.Item
                    required
                    label={formatMessage({ id: 'model.name' })}
                    tooltip={formatMessage({ id: 'model.llm.tips' })}
                  >
                    <ModelSelect
                      model="completion"
                      variant="filled"
                      value={_.get(values, 'model_name')}
                      onChange={(name) => {
                        _.set(values, 'model_name', name);
                        _.set(
                          values,
                          'custom_llm_provider',
                          getProviderByModelName(name, 'completion').model
                            ?.custom_llm_provider,
                        );
                        _.set(
                          values,
                          'model_service_provider',
                          getProviderByModelName(name, 'completion').provider
                            ?.name,
                        );
                        applyChanges();
                      }}
                    />
                  </Form.Item>
                  <Form.Item
                    required
                    label={formatMessage({ id: 'flow.temperature' })}
                    tooltip={formatMessage({ id: 'flow.temperature.tips' })}
                  >
                    <Slider
                      style={{ margin: 0 }}
                      min={_.get(schema, 'properties.temperature.minimum')}
                      max={_.get(schema, 'properties.temperature.maximum')}
                      step={0.01}
                      value={_.get(values, 'temperature')}
                      onChange={(value) => {
                        _.set(values, 'temperature', value);
                        applyChanges();
                      }}
                    />
                  </Form.Item>
                  <Form.Item label={formatMessage({ id: 'flow.input.source' })}>
                    <NodeInput
                      variant="filled"
                      disabled
                      value={_.get(values, 'docs')}
                      onChange={(e) => {
                        _.set(values, 'docs', e.currentTarget.value);
                        applyChanges();
                      }}
                    />
                  </Form.Item>
                  <Form.Item
                    required
                    label={formatMessage({ id: 'flow.variable.global' })}
                  >
                    <NodeInput
                      variant="filled"
                      value={_.get(values, 'query')}
                      onChange={(e) => {
                        _.set(values, 'query', e.currentTarget.value);
                        applyChanges();
                      }}
                    />
                  </Form.Item>
                  <Form.Item
                    required
                    style={{ marginBottom: 0 }}
                    label={formatMessage({ id: 'model.prompt_template' })}
                  >
                    <NodeInputTextArea
                      variant="filled"
                      value={_.get(values, 'prompt_template')}
                      style={{ fontSize: 12 }}
                      autoSize
                      onChange={(e) => {
                        _.set(values, 'prompt_template', e.currentTarget.value);
                        applyChanges();
                      }}
                    />
                  </Form.Item>
                </Form>
              </>
            ),
          },
          {
            key: '2',
            label: formatMessage({ id: 'flow.output.params' }),
            style: getCollapsePanelStyle(token),
            children: <OutputParams node={node} />,
          },
        ]}
      />
    </>
  );
};
