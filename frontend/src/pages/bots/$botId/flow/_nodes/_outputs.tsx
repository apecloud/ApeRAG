import { ApeMarkdown } from '@/components';
import { ApeNode } from '@/types';
import { CaretRightOutlined } from '@ant-design/icons';
import {
  Badge,
  Button,
  Collapse,
  CollapseProps,
  Drawer,
  Progress,
  Space,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import _ from 'lodash';
import { useMemo, useState } from 'react';
import { useIntl, useModel } from 'umi';

type OutputProps = {
  node: ApeNode;
};

export const NodeOutputs = ({ node }: OutputProps) => {
  const { messages } = useModel('bots.$botId.flow.model');
  const { formatMessage } = useIntl();
  const { token } = theme.useToken();
  const [outputsVisible, setOutputsVisible] = useState<boolean>(false);
  const outputs = useMemo(() => {
    let outputKey: 'docs' | undefined;
    switch (node.type) {
      case 'graph_search':
      case 'fulltext_search':
      case 'vector_search':
      case 'merge':
      case 'rerank':
        outputKey = 'docs';
        break;
    }

    if (outputKey) {
      const data = messages.find(
        (msg) => msg.node_id === node.id && msg.event_type === 'node_end',
      )?.data?.outputs;
      return data?.[outputKey];
    }
  }, [messages]);

  const getOutputs: () => CollapseProps['items'] = () =>
    outputs?.map((output, index) => ({
      key: index,
      label: (
        <Typography.Text
          style={{ maxWidth: 400, color: token.colorPrimary }}
          ellipsis
        >
          {index + 1}. {output.metadata?.name || output.metadata?.source}
        </Typography.Text>
      ),
      children: <ApeMarkdown>{output.text}</ApeMarkdown>,
      extra: (
        <Space size="large">
          {output.metadata?.recall_type ? (
            <Typography.Text>
              {formatMessage({
                id: `searchTest.type.${output.metadata.recall_type}`,
              })}
            </Typography.Text>
          ) : null}
          <Tooltip title={`Score: ${output.score}`}>
            <Space align="center">
              <Progress
                type="dashboard"
                percent={((output.score || 0) * 100) / 1}
                showInfo={false}
                size={20}
              />
              <Typography.Text type="secondary">
                {(output.score || 0).toFixed(2)}
              </Typography.Text>
            </Space>
          </Tooltip>
        </Space>
      ),
      style: {
        background: token.colorBgContainer,
        borderRadius: 0,
        borderLeftWidth: 0,
        borderRightWidth: 0,
      },
    }));

  if (!_.size(outputs)) {
    return;
  }

  return (
    <>
      <Button
        size="small"
        type="text"
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
          setOutputsVisible(true);
        }}
        style={{ padding: 0 }}
      >
        <Badge overflowCount={9} count={_.size(outputs)}></Badge>
      </Button>

      <Drawer
        title={formatMessage({ id: 'text.outputs' })}
        open={outputsVisible}
        size="large"
        onClose={(e) => {
          e.stopPropagation();
          setOutputsVisible(false);
        }}
        styles={{
          body: {
            padding: 0,
          },
        }}
        onClick={(e) => {
          e.stopPropagation();
        }}
      >
        <Collapse
          bordered={false}
          defaultActiveKey={['0']}
          expandIcon={({ isActive }) => (
            <CaretRightOutlined rotate={isActive ? 90 : 0} />
          )}
          style={{ background: token.colorBgContainer }}
          items={getOutputs()}
        />
      </Drawer>
    </>
  );
};
