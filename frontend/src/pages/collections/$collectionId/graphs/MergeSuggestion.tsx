import {
  MergeSuggestionItem,
  MergeSuggestionItemStatusEnum,
  MergeSuggestionsResponse,
  SuggestionActionRequestActionEnum,
} from '@/api';
import { graphApi } from '@/services';
import {
  CheckOutlined,
  CloseOutlined,
  Loading3QuartersOutlined,
  TrophyOutlined,
} from '@ant-design/icons';
import {
  Button,
  Card,
  List,
  Space,
  Tabs,
  Tag,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import React, { useCallback, useState } from 'react';
import { useIntl } from 'umi';

const IconText = ({
  icon,
  text,
}: {
  icon: React.ReactNode;
  text: React.ReactNode;
}) => (
  <Space>
    {icon}
    {text}
  </Space>
);

const SuggestionItem = ({
  item,
  onSelectNode,
  afterRejectMergeSuggestion,
  afterAcceptMergeSuggestion,
}: {
  item: MergeSuggestionItem;
  onSelectNode: (name: string) => void;
  afterRejectMergeSuggestion: () => void;
  afterAcceptMergeSuggestion: () => void;
}) => {
  const { token } = theme.useToken();
  const [hover, setHover] = useState<boolean>(false);
  const [loading, setLoading] = useState<{[key in SuggestionActionRequestActionEnum]: boolean}>();

  const handleSuggestionAction = useCallback(
    async (action: SuggestionActionRequestActionEnum) => {
      setLoading({
        accept: action === 'accept',
        reject: action === 'reject',
      })
      const res =
        await graphApi.collectionsCollectionIdGraphsMergeSuggestionsSuggestionIdActionPost(
          {
            suggestionId: item.id,
            collectionId: item.collection_id,
            suggestionActionRequest: {
              action,
              target_entity_data: item.suggested_target_entity,
            },
          },
        );
      if (res.data.status === 'success' && action === 'reject') {
        await afterRejectMergeSuggestion();
      }
      if (res.data.status === 'success' && action === 'accept') {
        await afterAcceptMergeSuggestion();
      }
      setLoading({
        accept: false,
        reject: false,
      })
    },
    [item],
  );

  return (
    <List.Item
      key={item.id}
      style={{ paddingInline: 12 }}
      actions={[
        <IconText
          icon={<TrophyOutlined />}
          text={item.confidence_score}
          key="score"
        />,
        // <IconText icon={<BellOutlined />} text={item.status} key="status" />,
      ]}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <List.Item.Meta
        style={{ marginBottom: 4 }}
        title={
          <Space style={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography.Link
              onClick={() =>
                onSelectNode(item.suggested_target_entity.entity_name)
              }
              color={token.colorPrimary}
            >
              {item.suggested_target_entity.entity_name}
            </Typography.Link>
            <Space
              style={{
                opacity: hover && item.status === 'PENDING' ? 1 : 0,
                transition: 'all 0.3s',
              }}
            >
              <Tooltip title="Accept">
                <Button
                  type="link"
                  size="small"
                  loading={loading?.accept}
                  icon={<CheckOutlined />}
                  onClick={() => handleSuggestionAction('accept')}
                />
              </Tooltip>
              <Tooltip title="Reject">
                <Button
                  type="link"
                  size="small"
                  icon={<CloseOutlined />}
                  loading={loading?.reject}
                  onClick={() => handleSuggestionAction('reject')}
                />
              </Tooltip>
            </Space>
          </Space>
        }
        description={item.merge_reason}
      />
      {item.entity_ids.map((entity) => (
        <Tag
          key={entity}
          style={{ margin: 1, cursor: 'pointer' }}
          onClick={() => onSelectNode(entity)}
        >
          {entity}
        </Tag>
      ))}
    </List.Item>
  );
};

export const MergeSuggestion = ({
  dataSource,
  open,
  onClose,
  onSelectNode,
  onRefresh,
}: {
  dataSource: MergeSuggestionsResponse;
  open: boolean;
  onClose: () => void;
  onSelectNode: (id: string) => void;
  onRefresh: () => void;
}) => {
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const [activeStatus, setActiveStatus] =
    useState<MergeSuggestionItemStatusEnum>('PENDING');

  return (
    <Card
      title={
        <Tabs
          activeKey={activeStatus}
          onChange={(key) =>
            setActiveStatus(key as MergeSuggestionItemStatusEnum)
          }
          tabBarGutter={16}
          size="large"
          items={[
            {
              key: MergeSuggestionItemStatusEnum.PENDING,
              label: formatMessage({
                id: 'collection.merge_suggestion.status.PENDING',
              }),
            },
            {
              key: MergeSuggestionItemStatusEnum.ACCEPTED,
              label: formatMessage({
                id: 'collection.merge_suggestion.status.ACCEPTED',
              }),
            },
            {
              key: MergeSuggestionItemStatusEnum.REJECTED,
              label: formatMessage({
                id: 'collection.merge_suggestion.status.REJECTED',
              }),
            },
          ]}
        />
      }
      extra={
        <Space>
          <Button shape="circle" type="text" onClick={onClose}>
            <CloseOutlined />
          </Button>
        </Space>
      }
      style={{
        width: '460px',
        backdropFilter: 'blur(50px)',
        background: token.colorBgContainer,
        position: 'absolute',
        right: open ? 0 : '-460px',
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
        header: {
          paddingBottom: 0,
          paddingInline: 12,
          // minHeight: 'auto',
        },
        body: {
          overflow: 'auto',
          padding: 0,
        },
      }}
    >
      <List
        size="small"
        itemLayout="vertical"
        dataSource={dataSource.suggestions.filter(
          (s) => s.status === activeStatus,
        )}
        renderItem={(item) => (
          <SuggestionItem
            item={item}
            onSelectNode={onSelectNode}
            afterRejectMergeSuggestion={onRefresh}
            afterAcceptMergeSuggestion={onRefresh}
          />
        )}
      />
    </Card>
  );
};
