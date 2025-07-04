import { PageContainer, PageHeader, RefreshButton } from '@/components';
import {
  COLLECTION_SOURCE,
  DATETIME_FORMAT,
  MODEL_PROVIDER_ICON,
  UI_COLLECTION_STATUS,
} from '@/constants';
import { CollectionConfigSource } from '@/types';
import { PlusOutlined, SearchOutlined } from '@ant-design/icons';
import {
  Avatar,
  Badge,
  Button,
  Card,
  Col,
  Divider,
  Input,
  Result,
  Row,
  Select,
  Space,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import _ from 'lodash';
import moment from 'moment';
import { useEffect, useRef, useState } from 'react';
import { UndrawEmpty } from 'react-undraw-illustrations';
import { FormattedMessage, Link, useIntl, useModel } from 'umi';

export default () => {
  const [searchParams, setSearchParams] = useState<{
    title?: string;
    source?: CollectionConfigSource;
  }>();
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const { collections, collectionsLoading, getCollections } =
    useModel('collection');

  // State for polling control
  const [isPolling, setIsPolling] = useState<boolean>(false);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const header = (
    <PageHeader
      title={formatMessage({ id: 'collection.name' })}
      description={formatMessage({ id: 'collection.tips' })}
    >
      <Space>
        <Select
          style={{ width: 180 }}
          placeholder={formatMessage({ id: 'collection.source' })}
          options={Object.keys(COLLECTION_SOURCE).map((key) => {
            return {
              label: formatMessage({ id: `collection.source.${key}` }),
              value: key,
              disabled:
                !COLLECTION_SOURCE[key as CollectionConfigSource].enabled,
            };
          })}
          allowClear
          onChange={(v) => {
            setSearchParams({ ...searchParams, source: v });
          }}
          value={searchParams?.source}
          labelRender={({ label, value }) => {
            return (
              <Space>
                <Avatar
                  size={20}
                  shape="square"
                  src={COLLECTION_SOURCE[value as CollectionConfigSource].icon}
                />
                {label}
              </Space>
            );
          }}
          optionRender={({ label, value }) => {
            return (
              <Space>
                <Avatar
                  size={20}
                  shape="square"
                  src={COLLECTION_SOURCE[value as CollectionConfigSource].icon}
                />
                {label}
              </Space>
            );
          }}
        />
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

  // Check if there are any collections with INACTIVE status
  const hasInactiveCollections = () => {
    return collections?.some((collection) => collection.status === 'INACTIVE');
  };

  // Start polling collections from the backend every 3 seconds
  const startPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
    setIsPolling(true);
    pollingIntervalRef.current = setInterval(() => {
      getCollections();
    }, 3000); // Poll every 3 seconds
  };

  // Stop polling
  const stopPolling = () => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    setIsPolling(false);
  };

  // Initial fetch of collections on component mount
  useEffect(() => {
    getCollections();
  }, []);

  // Watch for changes in collections and polling state
  // If any collection is INACTIVE, start polling; otherwise, stop polling
  useEffect(() => {
    if (collections && !collectionsLoading) {
      if (hasInactiveCollections()) {
        if (!isPolling) {
          startPolling();
        }
      } else {
        if (isPolling) {
          stopPolling();
        }
      }
    }
  }, [collections, collectionsLoading]);

  // Cleanup polling interval on component unmount
  useEffect(() => {
    return () => {
      stopPolling();
    };
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
                <Link to={`/collections/${collection.id}/documents`}>
                  <Card size="small" hoverable>
                    <div
                      style={{ display: 'flex', gap: 8, alignItems: 'center' }}
                    >
                      <Avatar
                        style={{ flex: 'none' }}
                        size={40}
                        src={
                          MODEL_PROVIDER_ICON[embedding_model_service_provider]
                        }
                        shape="square"
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
    </PageContainer>
  );
};
