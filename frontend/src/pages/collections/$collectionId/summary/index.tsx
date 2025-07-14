import { Collection } from '@/api';
import { api } from '@/services';
import { PageContainer } from '@/components';
import { Alert, Button, Card, Col, Divider, Input, Row, Space, Spin, Typography } from 'antd';
import { useEffect, useState } from 'react';
import { toast } from 'react-toastify';
import { FormattedMessage, useIntl, useModel, useParams } from 'umi';

const { Title, Text } = Typography;

export default () => {
  const { collectionId } = useParams<{ collectionId: string }>();
  const { formatMessage } = useIntl();
  const { collection, getCollection } = useModel('collection');
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (collectionId) {
      getCollection(collectionId).finally(() => setLoading(false));
    }
  }, [collectionId]);

  const handleGenerateSummary = async () => {
    if (!collectionId) return;
    
    setIsGeneratingSummary(true);
    try {
      const response = await api.collectionsCollectionIdSummaryGeneratePost({
        collectionId: collectionId
      });
      
      if (response.data.success) {
        toast.success(formatMessage({ id: 'collection.summary.generate.success' }));
        // Refresh collection data to show updated status
        setTimeout(() => {
          getCollection(collectionId);
        }, 1000);
      } else {
        toast.error(formatMessage({ id: 'collection.summary.generate.failed' }));
      }
    } catch (error) {
      console.error('Generate summary error:', error);
      toast.error(formatMessage({ id: 'collection.summary.generate.failed' }));
    } finally {
      setIsGeneratingSummary(false);
    }
  };

  if (loading) {
    return (
      <PageContainer>
        <div style={{ display: 'flex', justifyContent: 'center', padding: '50px' }}>
          <Spin size="large" />
        </div>
      </PageContainer>
    );
  }

  if (!collection) {
    return (
      <PageContainer>
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Text type="secondary">
            <FormattedMessage id="collection.not.found" />
          </Text>
        </div>
      </PageContainer>
    );
  }

  const config = collection.config;
  const isSummaryEnabled = config?.enable_summary;
  const isGenerating = collection.status === 'SUMMARY_GENERATING';

  return (
    <PageContainer>
      {!isSummaryEnabled ? (
        <Alert
          message={formatMessage({ id: 'collection.summary.not.enabled' })}
          description={formatMessage({ id: 'collection.summary.enable.description' })}
          type="warning"
          showIcon
          action={
            <Button type="primary" size="small" href={`/collections/${collectionId}/settings`}>
              <FormattedMessage id="collection.goto.settings" />
            </Button>
          }
          style={{ marginBottom: 24 }}
        />
      ) : (
        <Card>
          <Row gutter={[24, 24]}>
            <Col span={24}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <Title level={4} style={{ margin: 0 }}>
                  <FormattedMessage id="collection.summary.content" />
                </Title>
                <Button
                  type="primary"
                  onClick={handleGenerateSummary}
                  loading={isGeneratingSummary}
                  disabled={isGenerating}
                >
                  {isGenerating 
                    ? formatMessage({ id: 'collection.summary.generating' })
                    : formatMessage({ id: 'collection.summary.generate' })
                  }
                </Button>
              </div>

              {isGenerating && (
                <Alert
                  message={formatMessage({ id: 'collection.summary.generating' })}
                  description={formatMessage({ id: 'collection.summary.generating.description' })}
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                />
              )}

              <Input.TextArea
                rows={8}
                value={collection.summary || ''}
                placeholder={formatMessage({ id: 'collection.summary.empty' })}
                readOnly
                style={{ 
                  fontSize: 14,
                  lineHeight: 1.6,
                }}
              />
            </Col>
          </Row>
        </Card>
      )}
    </PageContainer>
  );
}; 