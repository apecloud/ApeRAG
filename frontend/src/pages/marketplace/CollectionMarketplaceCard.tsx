import { SharedCollection } from '@/api';
import { DATETIME_FORMAT } from '@/constants';
import {
  ClockCircleOutlined,
  HeartFilled,
  HeartOutlined,
  StarOutlined,
  UserOutlined,
} from '@ant-design/icons';
import {
  Avatar,
  Button,
  Card,
  Divider,
  Space,
  Tag,
  theme,
  Typography,
} from 'antd';
import moment from 'moment';
import React from 'react';
import { FormattedMessage, useIntl, useModel } from 'umi';

interface CollectionMarketplaceCardProps {
  /**
   * Shared collection data
   */
  collection: SharedCollection;
  /**
   * Callback when clicking the card
   */
  onClick?: (collection: SharedCollection) => void;
  /**
   * Callback when subscribing to collection
   */
  onSubscribe?: (collectionId: string) => Promise<boolean>;
  /**
   * Callback when unsubscribing from collection
   */
  onUnsubscribe?: (collectionId: string) => Promise<boolean>;
  /**
   * Loading state for subscription operations
   */
  loading?: boolean;
}

export const CollectionMarketplaceCard: React.FC<
  CollectionMarketplaceCardProps
> = ({ collection, onClick, onSubscribe, onUnsubscribe, loading = false }) => {
  const { formatMessage } = useIntl();
  const { user } = useModel('user');
  const { token } = theme.useToken();

  const isOwner = user?.id === collection.owner_user_id;
  const isSubscribed = !!collection.subscription_id;
  const canSubscribe = !isOwner && !isSubscribed;
  const canUnsubscribe = !isOwner && isSubscribed;

  // Truncate description to 120 characters for better layout
  const truncatedDescription = React.useMemo(() => {
    if (!collection.description) return '';
    return collection.description.length > 120
      ? `${collection.description.substring(0, 120)}...`
      : collection.description;
  }, [collection.description]);

  const handleCardClick = () => {
    if (onClick) {
      onClick(collection);
    }
  };

  const handleSubscribe = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onSubscribe) {
      await onSubscribe(collection.id);
    }
  };

  const handleUnsubscribe = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onUnsubscribe) {
      await onUnsubscribe(collection.id);
    }
  };

  const renderActionButton = () => {
    if (isOwner) {
      return (
        <Tag
          color="gold"
          icon={<StarOutlined />}
          style={{
            fontSize: '12px',
            borderRadius: 6,
            margin: 0,
            fontWeight: 500,
          }}
        >
          <FormattedMessage id="collection.marketplace.owner" />
        </Tag>
      );
    }

    if (canSubscribe) {
      return (
        <Button
          type="default"
          size="small"
          onClick={handleSubscribe}
          loading={loading}
          icon={<HeartOutlined />}
          style={{
            borderRadius: 6,
            fontSize: '12px',
            height: '28px',
            fontWeight: 500,
            borderColor: '#1890ff',
            color: '#1890ff',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = '#1890ff';
            e.currentTarget.style.color = '#fff';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
            e.currentTarget.style.color = '#1890ff';
          }}
        >
          <FormattedMessage id="collection.marketplace.subscribe" />
        </Button>
      );
    }

    if (canUnsubscribe) {
      return (
        <Button
          type="default"
          size="small"
          onClick={handleUnsubscribe}
          loading={loading}
          icon={<HeartFilled />}
          style={{
            borderRadius: 6,
            fontSize: '12px',
            height: '28px',
            fontWeight: 500,
            borderColor: '#1890ff',
            color: '#1890ff',
            background: '#f0f9ff',
          }}
        >
          <FormattedMessage id="collection.marketplace.subscribed" />
        </Button>
      );
    }

    return null;
  };

  return (
    <Card size="small" hoverable={!loading} onClick={handleCardClick}>
      <Space
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Typography.Title level={5} style={{ margin: 0 }}>
          {collection.title}
        </Typography.Title>
        {renderActionButton()}
      </Space>

      <Typography.Text type="secondary">
        {truncatedDescription ||
          formatMessage({ id: 'collection.marketplace.no.description' })}
      </Typography.Text>
      <Divider style={{ marginBlock: 8 }} />
      <Space
        direction="horizontal"
        align="center"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <Space size={6} align="center">
          <Avatar size={16} icon={<UserOutlined />} />
          <Typography.Text>{collection.owner_username}</Typography.Text>
        </Space>

        {collection.gmt_subscribed && isSubscribed && (
          <Space align="center">
            <ClockCircleOutlined style={{ color: token.colorTextSecondary }} />
            <Typography.Text type="secondary">
              {moment(collection.gmt_subscribed).format(DATETIME_FORMAT)}
            </Typography.Text>
          </Space>
        )}
      </Space>
    </Card>
  );
};
