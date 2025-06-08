import { ApeNode } from '@/types';
import { CaretRightOutlined } from '@ant-design/icons';
import { Collapse, Form, Select, Slider, theme } from 'antd';

import { applyNodeChanges, NodeChange } from '@xyflow/react';
import _ from 'lodash';
import { useCallback, useMemo } from 'react';
import { useIntl, useModel } from 'umi';
import { NodeOutputs } from './_outputs';
import { OutputParams } from './_outputs_params';
import { getCollapsePanelStyle } from './_styles';

export const ApeNodeGraphSearch = ({ node }: { node: ApeNode }) => {
  const { token } = theme.useToken();
  const { collections } = useModel('collection');
  const { setNodes } = useModel('bots.$botId.flow.model');
  const { formatMessage } = useIntl();

  const schema = useMemo(() => node.data.input.schema, [node]);
  const values = useMemo(() => node.data.input?.values || [], [node]);
  const applyChanges = useCallback(() => {
    setNodes((nds) => {
      const changes: NodeChange[] = [
        {
          id: node.id,
          type: 'replace',
          item: node,
        },
      ];
      return applyNodeChanges(changes, nds) as ApeNode[];
    });
  }, [node]);

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
                    label={formatMessage({ id: 'collection.name' })}
                  >
                    <Select
                      variant="filled"
                      mode="multiple"
                      suffixIcon={null}
                      options={collections?.map((collection) => ({
                        label: collection.title,
                        value: collection.id,
                      }))}
                      value={_.get(values, 'collection_ids')}
                      onChange={(value) => {
                        _.set(values, 'collection_ids', value);
                        applyChanges();
                      }}
                      placeholder={formatMessage({ id: 'collection.select' })}
                    />
                  </Form.Item>
                  <Form.Item
                    required
                    label={formatMessage({ id: 'flow.top_k' })}
                    tooltip={formatMessage({ id: 'flow.top_k.tips' })}
                    style={{ marginBottom: 0 }}
                  >
                    <Slider
                      value={_.get(values, 'top_k')}
                      style={{ margin: 0 }}
                      min={_.get(schema, 'properties.top_k.minimum')}
                      max={_.get(schema, 'properties.top_k.maximum')}
                      step={1}
                      onChange={(value) => {
                        _.set(values, 'top_k', value);
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
            extra: <NodeOutputs node={node} />,
          },
        ]}
      />
    </>
  );
};
