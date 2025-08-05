import { SharedCollection } from '@/api';
import { PageContainer } from '@/components';

import { ShareAltOutlined } from '@ant-design/icons';
import { useRequest } from 'ahooks';
import {
  Card,
  Space,
  Tabs,
  TabsProps,
  Typography,
} from 'antd';

import { useCallback, useEffect, useState } from 'react';
import { toast } from 'react-toastify';
import {
  FormattedMessage,
  useIntl,
  useModel,
  useParams,
} from 'umi';
import { MarketplaceDocuments } from './documents';
import { MarketplaceGraphs } from './graphs';
import ReadOnlyBanner from './ReadOnlyBanner';

const { Title, Text } = Typography;

export default () => {
  const { formatMessage } = useIntl();
  const { collectionId } = useParams();
  const { subscribeToCollection } = useModel('collection');
  
  const [marketplaceCollection, setMarketplaceCollection] = useState<SharedCollection | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('documents');

  // Load marketplace collection data
  const {
    data: collectionResponse,
    loading: collectionLoading,
  } = useRequest(
    async () => {
      if (!collectionId) throw new Error('Collection ID is required');
      const { api } = await import('@/services');
      return api.marketplaceCollectionsCollectionIdGet({ collectionId });
    },
    {
      refreshDeps: [collectionId],
    },
  );

  useEffect(() => {
    if (collectionResponse?.data) {
      setMarketplaceCollection(collectionResponse.data);
    }
  }, [collectionResponse]);

  useEffect(() => {
    if (collectionLoading !== undefined) {
      setLoading(collectionLoading);
    }
  }, [collectionLoading]);

  const handleSubscribe = useCallback(async () => {
    if (!collectionId) return;
    
    setActionLoading(true);
    try {
      const success = await subscribeToCollection(collectionId);
      if (success) {
        toast.success(formatMessage({
          id: 'collection.marketplace.subscribe.success',
          defaultMessage: '订阅成功',
        }));
        // Refresh the collection data to update subscription status
        window.location.reload();
      } else {
        toast.error(formatMessage({
          id: 'collection.marketplace.subscribe.failed',
          defaultMessage: '订阅失败',
        }));
      }
    } catch (error) {
      toast.error(formatMessage({
        id: 'collection.marketplace.subscribe.failed',
        defaultMessage: '订阅失败',
      }));
    } finally {
      setActionLoading(false);
    }
  }, [collectionId, subscribeToCollection, formatMessage]);



  const tabItems: TabsProps['items'] = [
    {
      key: 'documents',
      label: (
        <Space>
          <FormattedMessage id="collection.tab.documents" defaultMessage="文档" />
        </Space>
      ),
      children: marketplaceCollection && collectionId ? (
        <MarketplaceDocuments
          collectionId={collectionId}
        />
      ) : null,
    },
    {
      key: 'graphs',
      label: (
        <Space>
          <FormattedMessage id="collection.tab.graphs" defaultMessage="知识图谱" />
        </Space>
      ),
      children: marketplaceCollection && collectionId ? (
        <MarketplaceGraphs
          collectionId={collectionId}
        />
      ) : null,
    },
  ];

  if (loading && !marketplaceCollection) {
    return (
      <PageContainer>
        <Card loading style={{ minHeight: 400 }} />
      </PageContainer>
    );
  }

  if (!marketplaceCollection) {
    return (
      <PageContainer>
        <Card style={{ textAlign: 'center', marginTop: 100 }}>
          <FormattedMessage
            id="collection.marketplace.not.found"
            defaultMessage="知识库不存在或已不可用"
          />
        </Card>
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      {/* Read-Only Banner */}
      <ReadOnlyBanner
        ownerUsername={marketplaceCollection.owner_username}
        isSubscribed={!!marketplaceCollection.subscription_id}
        onSubscribe={handleSubscribe}
        loading={actionLoading}
      />

      {/* Page Header */}
      <Card style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 }}>
          <ShareAltOutlined style={{ color: '#1890ff', fontSize: 24 }} />
          <div style={{ flex: 1 }}>
            <Title level={2} style={{ margin: 0, marginBottom: 8 }}>
              {marketplaceCollection.title}
            </Title>
            <Text type="secondary" style={{ marginBottom: 8, display: 'block' }}>
              <FormattedMessage
                id="collection.marketplace.subtitle"
                defaultMessage="来自 @{ownerUsername} 的共享知识库"
                values={{ ownerUsername: marketplaceCollection.owner_username }}
              />
            </Text>
          </div>
        </div>

        {/* Collection Description */}
        {marketplaceCollection.description && (
          <div style={{ marginTop: 16 }}>
            <Text>{marketplaceCollection.description}</Text>
          </div>
        )}
      </Card>

      {/* Main Content Tabs */}
      <Card>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={tabItems}
          size="large"
        />
      </Card>
    </PageContainer>
  );
};