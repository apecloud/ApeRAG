import { GraphNode } from '@/api';
import { ApeMarkdown } from '@/components';
import { CloseOutlined } from '@ant-design/icons';
import { Button, Card, theme } from 'antd';

export const NodeDetail = ({
  open,
  node,
  onClose,
}: {
  open: boolean;
  node?: GraphNode;
  onClose: () => void;
}) => {
  const { token } = theme.useToken();

  return (
    <Card
      title={node?.labels || node?.id}
      extra={
        <Button shape="circle" type="text" onClick={onClose}>
          <CloseOutlined />
        </Button>
      }
      style={{
        width: '400px',
        backdropFilter: 'blur(50px)',
        background: token.colorBgContainer,
        position: 'absolute',
        right: open ? 0 : '-400px',
        top: 0,
        bottom: 0,
        borderTop: `none`,
        borderRight: `none`,
        borderBottom: `none`,
        display: 'flex',
        flexDirection: 'column',
        transition: `0.2s`,
        zIndex: 100,
      }}
      styles={{
        header: {},
        body: {
          overflow: 'auto',
        },
      }}
    >
      <ApeMarkdown>{node?.properties.description}</ApeMarkdown>
      {/* <pre
        style={{
          whiteSpace: 'pre-wrap',
          fontSize: '0.9rem',
          margin: 0,
          fontFamily: 'inherit',
        }}
      ></pre> */}
    </Card>
  );
};
