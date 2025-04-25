import { PageContainer, PageHeader, RefreshButton } from '@/components';
import {
  COLLECTION_SOURCE,
  DATETIME_FORMAT,
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
  const { collections, collectionsLoading, getCollections } =
    useModel('collection');

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

  useEffect(() => {
    getCollections();
  }, []);

  if (collections === undefined) return;

  const _collections = collections?.filter((item) => {
    const titleMatch = searchParams?.title
      ? item.title?.includes(searchParams.title)
      : true;
    const sourceMatch = searchParams?.source
      ? item.config?.source === searchParams.source
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
            const source = collection.config?.source as CollectionConfigSource;
            const sourceIcon = source
              ? COLLECTION_SOURCE[source].icon
              : undefined;
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
                        src={sourceIcon}
                        shape="square"
                      />
                      <div style={{ flex: 'auto', maxWidth: '65%' }}>
                        <div>
                          <Typography.Text ellipsis>
                            {collection.title}
                          </Typography.Text>
                        </div>
                        <div>
                          <Typography.Text ellipsis type="secondary">
                            <FormattedMessage
                              id={`collection.source.${collection.config?.source}`}
                            />
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
