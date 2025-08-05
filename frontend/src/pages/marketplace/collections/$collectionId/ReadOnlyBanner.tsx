import {
  HeartOutlined,
  InfoCircleOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { Alert, Button, Space, Typography } from 'antd';
import { useCallback, useState } from 'react';
import { FormattedMessage } from 'umi';

const { Text } = Typography;

interface ReadOnlyBannerProps {
  ownerUsername: string;
  isSubscribed: boolean;
  onSubscribe?: () => Promise<void>;
  loading?: boolean;
}

export const ReadOnlyBanner: React.FC<ReadOnlyBannerProps> = ({
  ownerUsername,
  isSubscribed,
  onSubscribe,
  loading = false,
}) => {
  const [actionLoading, setActionLoading] = useState(false);

  const handleSubscribe = useCallback(async () => {
    if (!onSubscribe) return;

    setActionLoading(true);
    try {
      await onSubscribe();
    } catch (error) {
      // Error handling is done in parent component
    } finally {
      setActionLoading(false);
    }
  }, [onSubscribe]);

  const description = (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Space align="center">
        <UserOutlined style={{ color: '#1890ff' }} />
        <Text>
          <FormattedMessage
            id="collection.marketplace.readonly.description"
            defaultMessage="您正在以只读模式浏览来自 @{ownerUsername} 的共享知识库，无法进行修改操作"
            values={{ ownerUsername }}
          />
        </Text>
      </Space>

      <Space>
        {!isSubscribed && onSubscribe && (
          <Button
            type="primary"
            size="small"
            icon={<HeartOutlined />}
            onClick={handleSubscribe}
            loading={actionLoading || loading}
            style={{
              borderRadius: 6,
              fontSize: '12px',
              height: '28px',
            }}
          >
            <FormattedMessage
              id="collection.marketplace.subscribe"
              defaultMessage="订阅"
            />
          </Button>
        )}
      </Space>
    </Space>
  );

  return (
    <Alert
      type="info"
      icon={<InfoCircleOutlined />}
      message={
        <FormattedMessage
          id="collection.marketplace.readonly.title"
          defaultMessage="只读模式"
        />
      }
      description={description}
      style={{
        marginBottom: 16,
      }}
      closable={false}
    />
  );
};

export default ReadOnlyBanner;
