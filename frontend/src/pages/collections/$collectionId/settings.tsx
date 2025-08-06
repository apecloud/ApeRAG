import { Collection } from '@/api';
import { ShareAltOutlined } from '@ant-design/icons';
import { Form, Card, Switch, Typography, Divider, message } from 'antd';
import { useEffect, useState } from 'react';
import { toast } from 'react-toastify';
import { useIntl, useModel, useParams } from 'umi';
import CollectionForm from '../_form';

const { Title, Text } = Typography;

export default () => {
  const { formatMessage } = useIntl();
  const [form] = Form.useForm<Collection>();
  const { collection, updateCollection } = useModel('collection');
  const { collectionId } = useParams();
  
  // Sharing state
  const [isPublished, setIsPublished] = useState(false);
  const [publishing, setPublishing] = useState(false);
  
  const { 
    sharingStatus,
    sharingLoading,
    getSharingStatus,
    publishCollection,
    unpublishCollection,
  } = useModel('collection');

  // Load sharing status
  useEffect(() => {
    if (collectionId && getSharingStatus) {
      getSharingStatus(collectionId);
    }
  }, [collectionId]);

  // Update local state
  useEffect(() => {
    if (sharingStatus) {
      setIsPublished(sharingStatus.is_published || false);
    }
  }, [sharingStatus]);

  const onFinish = async (values: Collection) => {
    const updated = await updateCollection(values);
    if (updated) {
      toast.success(formatMessage({ id: 'tips.update.success' }));
    }
  };

  const handlePublishToggle = async (checked: boolean) => {
    if (!collectionId) return;
    
    setPublishing(true);
    try {
      if (checked) {
        await publishCollection(collectionId);
        message.success(formatMessage({ id: 'collection.sharing.publish.success' }));
      } else {
        await unpublishCollection(collectionId);
        message.success(formatMessage({ id: 'collection.sharing.unpublish.success' }));
      }
      setIsPublished(checked);
    } catch (error) {
      message.error(formatMessage({ 
        id: checked ? 'collection.sharing.publish.failed' : 'collection.sharing.unpublish.failed' 
      }));
    } finally {
      setPublishing(false);
    }
  };

  return (
    <Card>
      <CollectionForm
        form={form}
        onSubmit={onFinish}
        action="edit"
        values={collection}
      />
      
      <Divider style={{ margin: '40px 0 24px' }} />
      
      <div style={{ marginBottom: 24 }}>
        <Title level={4} style={{ marginBottom: 16 }}>
          {formatMessage({ id: 'collection.sharing.title' })}
        </Title>
        
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          padding: '16px 0'
        }}>
          <div>
            <Text strong>{formatMessage({ id: 'collection.sharing.marketplace.publish' })}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: '14px' }}>
              {formatMessage({ id: 'collection.sharing.marketplace.description' })}
            </Text>
          </div>
          
          <Switch
            checked={isPublished}
            loading={sharingLoading || publishing}
            onChange={handlePublishToggle}
          />
        </div>
      </div>
    </Card>
  );
};