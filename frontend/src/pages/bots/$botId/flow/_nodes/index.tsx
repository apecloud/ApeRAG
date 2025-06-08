import { NodeTypeEnum } from '@/api';
import { ApeFlowNodeStatus, ApeNode, ApeNodeConfig } from '@/types';
import {
  CaretDownOutlined,
  EditOutlined,
  LoadingOutlined,
} from '@ant-design/icons';
import { applyNodeChanges, Handle, NodeChange, Position } from '@xyflow/react';
import { useHover } from 'ahooks';
import {
  Badge,
  Button,
  Form,
  Input,
  Modal,
  Space,
  theme,
  Typography,
} from 'antd';
import { RibbonProps } from 'antd/es/badge/Ribbon';
import _ from 'lodash';
import React, { useCallback, useMemo, useRef, useState } from 'react';
import { useIntl, useModel } from 'umi';
import {
  StyledFlowNode,
  StyledFlowNodeAvatar,
  StyledFlowNodeBody,
  StyledFlowNodeContainer,
  StyledFlowNodeHeader,
  StyledFlowNodeLabel,
} from './_styles';

const nodeStatusMap: {
  [key in ApeFlowNodeStatus]: {
    color: RibbonProps['color'];
    text: string;
  };
} = {
  stopped: {
    color: 'blue',
    text: 'Stopped',
  },
  pending: {
    color: 'pink',
    text: 'Pending',
  },
  running: {
    color: 'volcano',
    text: 'Running',
  },
  complated: {
    color: 'blue',
    text: 'Complated',
  },
};

const FlowNodeWrap = ({
  status,
  children,
}: {
  status: ApeFlowNodeStatus;
  children: React.ReactNode;
}) => {
  if (status === 'stopped') {
    return children;
  }
  return (
    <Badge.Ribbon
      text={nodeStatusMap[status].text}
      color={nodeStatusMap[status].color}
      style={{ top: 20, paddingBlock: 4, transitionDuration: '0.5s' }}
    >
      {children}
    </Badge.Ribbon>
  );
};

const ApeBasicNode = (node: ApeNode) => {
  const [labelModalVisible, setLabelModalVisible] = useState<boolean>(false);
  const { nodes, setNodes, getNodeConfig, messages, flowStatus } = useModel(
    'bots.$botId.flow.model',
  );
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const [form] = Form.useForm<{ ariaLabel: string }>();
  const nodeRef = useRef(null);
  const isHovering = useHover(nodeRef);
  const selected = useMemo(() => node.selected, [node]);
  const collapsed = useMemo(() => Boolean(node.data.collapsed), [node]);
  const originNode = useMemo(() => nodes.find((n) => n.id === node.id), [node]);

  const nodeMessages = useMemo(
    () => messages.filter((msg) => msg.node_id === node.id),
    [messages],
  );

  const nodeStatus: ApeFlowNodeStatus = useMemo(() => {
    const start = Boolean(
      nodeMessages?.find((msg) => msg.event_type === 'node_start'),
    );
    const end = Boolean(
      nodeMessages?.find((msg) => msg.event_type === 'node_end'),
    );
    if (flowStatus === 'stopped') {
      return 'stopped';
    }
    if (!start) {
      return 'pending';
    }
    if (!end) {
      return 'running';
    }
    return 'complated';
  }, [flowStatus, nodeMessages]);

  const nodeLoading = useMemo(
    () => _.includes(['pending', 'running'], nodeStatus),
    [nodeStatus],
  );

  const config: ApeNodeConfig = useMemo(
    () =>
      getNodeConfig(
        originNode?.type,
        originNode?.ariaLabel ||
          formatMessage({ id: `flow.node.type.${originNode?.type}` }),
      ),
    [originNode],
  );

  const onToggleCollapsed = useCallback(() => {
    if (!originNode) return;
    setNodes((nds) => {
      const changes: NodeChange[] = [
        {
          id: originNode.id,
          type: 'replace',
          item: {
            ...originNode,
            data: {
              ...originNode.data,
              collapsed: !collapsed,
            },
          },
        },
      ];
      return applyNodeChanges(changes, nds) as ApeNode[];
    });
  }, [originNode]);

  const onEditNodeLable = useCallback(() => {
    form.setFieldValue('ariaLabel', config.label);
    setLabelModalVisible(true);
  }, [config]);

  const onEditNodeLableClose = useCallback(() => {
    setLabelModalVisible(false);
  }, []);

  const onSaveNodeLable = useCallback(async () => {
    if (!originNode) return;
    const { ariaLabel } = await form.validateFields();
    setNodes((nds) => {
      const changes: NodeChange[] = [
        {
          id: originNode.id,
          type: 'replace',
          item: {
            ...originNode,
            ariaLabel,
          },
        },
      ];
      return applyNodeChanges(changes, nds) as ApeNode[];
    });
    setLabelModalVisible(false);
  }, [originNode]);

  return (
    <>
      <FlowNodeWrap status={nodeStatus}>
        <StyledFlowNodeContainer
          token={token}
          selected={selected || nodeStatus === 'running'}
          isHovering={isHovering}
          color={config.color}
          ref={nodeRef}
        >
          {!config.disableConnectionTarget && (
            <Handle
              type="target"
              position={node.targetPosition || Position.Left}
            />
          )}

          <StyledFlowNode style={{ width: config.width }}>
            <StyledFlowNodeHeader token={token} className="drag-handle">
              <Space>
                <StyledFlowNodeAvatar
                  token={token}
                  color={config.color}
                  shape="square"
                  size="large"
                  src={nodeLoading ? <LoadingOutlined /> : config.icon}
                />
                <div>
                  <div>
                    <StyledFlowNodeLabel>{config.label}</StyledFlowNodeLabel>
                  </div>
                  <Typography.Text copyable type="secondary">
                    {node.id}
                  </Typography.Text>
                </div>
              </Space>
              <Space>
                {isHovering && (
                  <Button
                    type="text"
                    onClick={onEditNodeLable}
                    icon={<EditOutlined />}
                  />
                )}
                <Button
                  type="text"
                  onClick={onToggleCollapsed}
                  icon={
                    <CaretDownOutlined
                      style={{
                        fontSize: '0.8em',
                        transform: `rotate(${collapsed ? -90 : 0}deg)`,
                        transitionDuration: '0.3s',
                      }}
                    />
                  }
                />
              </Space>
            </StyledFlowNodeHeader>
            <StyledFlowNodeBody token={token} collapsed={collapsed}>
              {config.content &&
                originNode &&
                React.createElement(config.content, { node: originNode })}
            </StyledFlowNodeBody>
          </StyledFlowNode>

          {!config.disableConnectionSource && (
            <Handle
              className="node-handler-end"
              type="source"
              position={node.sourcePosition || Position.Right}
              onClick={(e) => {
                e.stopPropagation();
              }}
            />
          )}
        </StyledFlowNodeContainer>
      </FlowNodeWrap>
      <Modal
        title={formatMessage({ id: 'flow.node.custom_label' })}
        open={labelModalVisible}
        onCancel={onEditNodeLableClose}
        onOk={onSaveNodeLable}
        width={380}
      >
        <br />
        <Form form={form} autoComplete="off">
          <Form.Item name="ariaLabel">
            <Input />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
};

export const NodeTypes: { [key in NodeTypeEnum]: any } = {
  start: ApeBasicNode,
  vector_search: ApeBasicNode,
  fulltext_search: ApeBasicNode,
  graph_search: ApeBasicNode,
  merge: ApeBasicNode,
  rerank: ApeBasicNode,
  llm: ApeBasicNode,
};
