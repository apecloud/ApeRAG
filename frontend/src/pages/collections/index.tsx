import { PageContainer, PageHeader, RefreshButton } from '@/components';
import {
  COLLECTION_SOURCE,
  DATETIME_FORMAT,
  MODEL_PROVIDER_ICON,
  UI_COLLECTION_STATUS,
} from '@/constants';
import { CollectionConfigSource } from '@/types';
import { DatabaseOutlined, HeartFilled, PlusOutlined, SearchOutlined } from '@ant-design/icons';
import { useInterval } from 'ahooks';
import {
  Avatar,
  Badge,
  Button,
  Card,
  Col,
  Divider,
  Input,
  Pagination,
  Result,
  Row,
  Select,
  Space,
  Tag,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import _ from 'lodash';
import moment from 'moment';
import { useEffect, useState } from 'react';
import { UndrawEmpty } from 'react-undraw-illustrations';
import { FormattedMessage, Link, useIntl, useModel } from 'umi';

export default () => {
  const [searchParams, setSearchParams] = useState<{
    title?: string;
    source?: CollectionConfigSource;
  }>();
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const { 
    collections, 
    collectionsLoading, 
    collectionsPagination, 
    getCollections,
    setCollectionsPagination
  } = useModel('collection');

  useInterval(() => {
    if (collections?.some((collection) => collection.status !== 'ACTIVE')) {
      getCollections();
    }
  }, 3000);

  const header = (
    <PageHeader
      title={formatMessage({ id: 'collection.name' })}
      description={formatMessage({ id: 'collection.tips' })}
    >
      <Space>
        
        <Input
          placeholder={formatMessage({ id: 'action.search' })}
          prefix={
            <Typography.Text disabled>
              <SearchOutlined />
            </Typography.Text>
          }
          onChange={(e) => {
            setSearchParams({ ...searchParams, title: e.currentTarget.value });
          }}
          allowClear
          value={searchParams?.title}
        />
        <Link to="/collections/new">
          <Tooltip title={<FormattedMessage id="collection.add" />}>
            <Button type="primary" icon={<PlusOutlined />} />
          </Tooltip>
        </Link>
        <RefreshButton
          loading={collectionsLoading}
          onClick={() => getCollections()}
        />
      </Space>
    </PageHeader>
  );

  // Handle pagination change
  const handlePageChange = (page: number, pageSize: number) => {
    setCollectionsPagination({
      ...collectionsPagination,
      current: page,
      pageSize,
    });
    getCollections(page, pageSize);
  };

  useEffect(() => {
    getCollections();
  }, []);

  if (collections === undefined) return;

  const _collections = collections?.filter((item) => {
    const config = item.config;
    const titleMatch = searchParams?.title
      ? item.title?.includes(searchParams.title)
      : true;
    const sourceMatch = searchParams?.source
      ? config?.source === searchParams.source
      : true;
    return titleMatch && sourceMatch;
  });

  return (
    <PageContainer>
      {header}
      {_.isEmpty(_collections) ? (
        <Result
          icon={
            <UndrawEmpty primaryColor={token.colorPrimary} height="200px" />
          }
          subTitle={<FormattedMessage id="text.empty" />}
        />
      ) : (
        <Row gutter={[24, 24]}>
          {_collections?.map((collection) => {
            const config = collection.config;
            const embedding_model_service_provider =
              config?.embedding?.model_service_provider || '';
            const embedding_model_name = config?.embedding?.model;
            const hasProviderIcon = MODEL_PROVIDER_ICON[embedding_model_service_provider];
            return (
              <Col
                key={collection.id}
                xs={24}
                sm={12}
                md={8}
                lg={6}
                xl={6}
                xxl={6}
              >
                <Link to={collection.subscription_id 
                  ? `/marketplace/collections/${collection.id}` 
                  : `/collections/${collection.id}/documents`}>
                  <Card 
                    size="small" 
                    hoverable
                    style={{ position: 'relative' }}
                  >
                    {collection.subscription_id && (
                      <Tag
                        icon={<HeartFilled />}
                        color="blue"
                        style={{
                          position: 'absolute',
                          top: 8,
                          right: 8,
                          zIndex: 1,
                          fontSize: '11px',
                          height: '20px',
                          lineHeight: '18px',
                          borderRadius: '10px',
                          padding: '0 6px'
                        }}
                      >
                        <FormattedMessage id="collection.marketplace.subscribed" />
                      </Tag>
                    )}
                    <div
                      style={{ display: 'flex', gap: 8, alignItems: 'center' }}
                    >
                      <Avatar
                        size={40}
                        src={hasProviderIcon}
                        icon={!hasProviderIcon && <DatabaseOutlined />}
                        shape="square"
                        style={{
                          flex: 'none',
                          backgroundColor: !hasProviderIcon ? '#1890ff' : 'transparent',
                          color: !hasProviderIcon ? '#fff' : 'inherit'
                        }}
                      />
                      <div style={{ flex: 'auto', maxWidth: '75%' }}>
                        <div>
                          <Typography.Text ellipsis>
                            {collection.title}
                          </Typography.Text>
                        </div>
                        <div>
                          <Typography.Text ellipsis type="secondary">
                            {embedding_model_name}
                          </Typography.Text>
                        </div>
                      </div>
                    </div>
                    <Divider style={{ marginBlock: 8 }} />
                    <div
                      style={{
                        display: 'flex',
                        gap: 8,
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}
                    >
                      <Typography.Text
                        ellipsis
                        type="secondary"
                        style={{ fontSize: '0.9em', width: '60%' }}
                      >
                        {moment(collection?.updated).format(DATETIME_FORMAT)}
                      </Typography.Text>
                      <Badge
                        status={
                          collection.status
                            ? UI_COLLECTION_STATUS[collection.status]
                            : 'default'
                        }
                        text={
                          <Typography.Text
                            type="secondary"
                            style={{ fontSize: '0.9em', width: '40%' }}
                          >
                            <FormattedMessage
                              id={`collection.status.${collection.status}`}
                            />
                          </Typography.Text>
                        }
                      />
                    </div>
                  </Card>
                </Link>
              </Col>
            );
          })}
        </Row>
      )}

      {!_.isEmpty(_collections) && collectionsPagination.total > collectionsPagination.pageSize && (
        <div style={{ marginTop: 24, textAlign: 'center' }}>
          <Pagination
            current={collectionsPagination.current}
            total={collectionsPagination.total}
            pageSize={collectionsPagination.pageSize}
            showSizeChanger
            showQuickJumper
            showTotal={(total, range) =>
              formatMessage(
                { id: 'common.pagination.total' },
                {
                  start: range[0],
                  end: range[1],
                  total,
                }
              )
            }
            onChange={handlePageChange}
          />
        </div>
      )}
    </PageContainer>
  );
};
