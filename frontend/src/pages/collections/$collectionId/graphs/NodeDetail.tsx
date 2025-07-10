import { GraphNode } from '@/api';
import { CloseOutlined } from '@ant-design/icons';
import { Button, Card, theme } from 'antd';
import { useEffect, useState } from 'react';

export const NodeDetail = ({ data }: { data?: GraphNode }) => {
  const [node, setNode] = useState<GraphNode>();
  const { token } = theme.useToken();
  const width = 300;

  useEffect(() => {
    setNode(data)
  }, [data])

  return (
    <Card
      title={node?.labels || node?.id}
      extra={
        <Button shape="circle" type="text" onClick={() => setNode(undefined)}><CloseOutlined /></Button>
      }
      style={{
        width,
        backdropFilter: 'blur(50px)',
        background: token.colorBgContainer,
        position: 'absolute',
        right: 0,
        top: 0,
        bottom: 0,
        borderTop: `none`,
        borderRight: `none`,
        borderBottom: `none`,
        display: 'flex',
        flexDirection: 'column',
        transform: `translateX(${node ? 0 : 2 * width}px)`,
        transition: `0.3s`
      }}
      styles={{
        header: {},
        body: {
          overflow: 'auto'
        }
      }}
    >
      {node?.properties.description}
    </Card>
  );
};
