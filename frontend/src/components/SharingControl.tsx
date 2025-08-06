import { SharingStatusResponse } from '@/api';
import { DATETIME_FORMAT } from '@/constants';
import { ExclamationCircleOutlined, ShareAltOutlined } from '@ant-design/icons';
import { Modal, Space, Switch, Tag, Tooltip, Typography } from 'antd';
import moment from 'moment';
import React, { useState } from 'react';
import { FormattedMessage, useIntl } from 'umi';

const { Text } = Typography;

interface SharingControlProps {
  /**
   * Collection sharing status
   */
  sharing?: SharingStatusResponse;
  /**
   * Loading state
   */
  loading?: boolean;
  /**
   * Callback when publishing status changes
   */
  onToggle?: (shouldPublish: boolean) => Promise<void>;
}

export const SharingControl: React.FC<SharingControlProps> = ({
  sharing,
  loading = false,
  onToggle,
}) => {
  const { formatMessage } = useIntl();
  const [modal, contextHolder] = Modal.useModal();
  const [switching, setSwitching] = useState(false);

  const isPublished = sharing?.is_published || false;
  const publishedAt = sharing?.published_at;

  const handleToggle = async (checked: boolean) => {
    if (!onToggle) return;

    const confirmed = await modal.confirm({
      title: formatMessage({
        id: checked
          ? 'collection.sharing.publish.title'
          : 'collection.sharing.unpublish.title',
      }),
      content: formatMessage({
        id: checked
          ? 'collection.sharing.publish.content'
          : 'collection.sharing.unpublish.content',
      }),
      icon: <ExclamationCircleOutlined />,
      okText: formatMessage({
        id: checked
          ? 'collection.sharing.publish.confirm'
          : 'collection.sharing.unpublish.confirm',
      }),
      cancelText: formatMessage({ id: 'action.cancel' }),
      okButtonProps: {
        danger: !checked, // 取消发布时使用危险按钮样式
      },
    });

    if (confirmed) {
      setSwitching(true);
      try {
        await onToggle(checked);
      } finally {
        setSwitching(false);
      }
    }
  };

  return (
    <>
      {contextHolder}
      <Tooltip 
        title={formatMessage({
          id: isPublished 
            ? 'collection.sharing.published.tooltip'
            : 'collection.sharing.unpublished.tooltip'
        })}
      >
        <Space direction="vertical" size={4}>
          <Space size={8}>
            <ShareAltOutlined />
            <Switch
              checked={isPublished}
              onChange={handleToggle}
              loading={loading || switching}
              size="small"
            />
            <Tag
              color={isPublished ? 'success' : 'default'}
              style={{ margin: 0 }}
            >
              <FormattedMessage
                id={
                  isPublished
                    ? 'collection.sharing.status.published'
                    : 'collection.sharing.status.unpublished'
                }
              />
            </Tag>
          </Space>
          {isPublished && publishedAt && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              <FormattedMessage
                id="collection.sharing.published.at"
                values={{
                  time: moment(publishedAt).format(DATETIME_FORMAT),
                }}
              />
            </Text>
          )}
        </Space>
      </Tooltip>
    </>
  );
};