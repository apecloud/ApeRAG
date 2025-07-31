import { SearchRequest, SearchResult } from '@/api';
import { ApeMarkdown, AuthAssetImage } from '@/components';
import { DATETIME_FORMAT } from '@/constants';
import { api } from '@/services';
import { CaretRightOutlined, DeleteOutlined } from '@ant-design/icons';
import {
  Button,
  Card,
  Checkbox,
  Collapse,
  Divider,
  Drawer,
  Form,
  Input,
  Modal,
  Progress,
  Result,
  Slider,
  Space,
  Table,
  TableProps,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import _ from 'lodash';
import moment from 'moment';
import { useEffect, useState } from 'react';
import { toast } from 'react-toastify';
import { UndrawScience } from 'react-undraw-illustrations';
import { FormattedMessage, useIntl, useModel, useParams } from 'umi';

type SearchTypeEnum =
  | 'vector_search'
  | 'fulltext_search'
  | 'graph_search'
  | 'summary_search';

export default () => {
  const { formatMessage } = useIntl();
  const [form] = Form.useForm<SearchRequest>();
  const { collectionId } = useParams();
  const { loading, setLoading } = useModel('global');
  const [modal, contextHolder] = Modal.useModal();

  const [searchType, setSearchType] = useState<SearchTypeEnum[]>([
    'vector_search',
  ]);

  const [records, setRecords] = useState<SearchResult[]>([]);
  const [historyModal, setHistoryModal] = useState<{
    visible: boolean;
    record: SearchResult | undefined;
  }>();
  const { token } = theme.useToken();

  const fetchHistory = async () => {
    if (!collectionId) return;
    const res = await api.collectionsCollectionIdSearchesGet({
      collectionId,
    });
    setRecords(res?.data?.items || []);
  };

  const onSearch = async () => {
    if (!collectionId) return;
    const values = await form.validateFields();
    setLoading(true);
    api
      .collectionsCollectionIdSearchesPost(
        {
          collectionId,
          searchRequest: values,
        },
        {
          timeout: 30 * 1000,
        },
      )
      .then((res) => {
        setLoading(false);
        if (!res.data) {
          toast.error(formatMessage({ id: 'tips.search.failed' }));
          return;
        }
        setHistoryModal({ visible: true, record: res.data });
        fetchHistory();
      })
      .catch(() => {
        setLoading(false);
      });
  };

  const onDeleteRecord = async (record: SearchResult) => {
    if (!collectionId || !record.id) return;
    const confirmed = await modal.confirm({
      title: formatMessage({ id: 'action.confirm' }),
      content: formatMessage({ id: 'search.confirmDeleteHistory' }),
      okButtonProps: {
        danger: true,
        loading,
      },
    });
    if (confirmed) {
      await api.collectionsCollectionIdSearchesSearchIdDelete({
        collectionId,
        searchId: record.id,
      });
      toast.success(formatMessage({ id: 'tips.delete.success' }));
      fetchHistory();
    }
  };

  const columns: TableProps<SearchResult>['columns'] = [
    {
      title: formatMessage({ id: 'search.history_question' }),
      dataIndex: 'query',
      fixed: 'left',
      width: 200,
      render: (_value, record) => {
        return (
          <>
            <Tooltip title={_.truncate(record.query, { length: 200 })}>
              <a
                onClick={() => setHistoryModal({ visible: true, record })}
                style={{ display: 'inline-block', marginBottom: 4 }}
              >
                {_.truncate(record.query, { length: 30 })}
              </a>
            </Tooltip>
            <div style={{ fontSize: 12, color: token.colorTextSecondary }}>
              {record.created
                ? moment(record.created).format(DATETIME_FORMAT)
                : ''}
            </div>
          </>
        );
      },
    },
    {
      title: formatMessage({ id: 'search.searchResults' }),
      align: 'center',
      fixed: 'left',
      width: 120,
      render: (text, record) => (
        <Button
          onClick={() => setHistoryModal({ visible: true, record })}
          type="link"
          style={{ padding: 0 }}
        >
          {_.size(record.items)}
        </Button>
      ),
    },
    {
      title: formatMessage({ id: 'search.vectorTopK' }),
      width: 130,
      render: (text, record) => {
        return (
          <Tooltip title={record.vector_search?.topk}>
            <Progress
              style={{ margin: 0, padding: 0, height: 4, lineHeight: 0 }}
              steps={20}
              percent={((record.vector_search?.topk || 0) * 100) / 20}
              size={[3, 4]}
              showInfo={false}
            />
          </Tooltip>
        );
      },
    },
    {
      title: formatMessage({ id: 'search.similarity' }),
      width: 130,
      render: (text, record) => {
        return (
          <Tooltip title={record.vector_search?.similarity}>
            <Progress
              steps={20}
              percent={((record.vector_search?.similarity || 0) * 100) / 1}
              size={[3, 4]}
              showInfo={false}
            />
          </Tooltip>
        );
      },
    },
    {
      title: formatMessage({ id: 'search.fulltextTopK' }),
      width: 130,
      render: (text, record) => {
        return (
          <Tooltip title={record.fulltext_search?.topk}>
            <Progress
              steps={20}
              percent={((record.fulltext_search?.topk || 0) * 100) / 20}
              size={[3, 4]}
              showInfo={false}
            />
          </Tooltip>
        );
      },
    },
    {
      title: formatMessage({ id: 'search.graphsearchTopK' }),
      width: 130,
      render: (text, record) => {
        return (
          <Tooltip title={record.graph_search?.topk}>
            <Progress
              steps={20}
              percent={((record.graph_search?.topk || 0) * 100) / 20}
              size={[3, 4]}
              showInfo={false}
            />
          </Tooltip>
        );
      },
    },
    {
      title: formatMessage({ id: 'search.summaryTopK' }),
      width: 130,
      render: (text, record) => {
        return (
          <Tooltip title={record.summary_search?.topk}>
            <Progress
              steps={20}
              percent={((record.summary_search?.topk || 0) * 100) / 20}
              size={[3, 4]}
              showInfo={false}
            />
          </Tooltip>
        );
      },
    },
    {
      title: formatMessage({ id: 'search.summarySimilarityThreshold' }),
      width: 130,
      render: (text, record) => {
        return (
          <Tooltip title={record.summary_search?.similarity}>
            <Progress
              steps={20}
              percent={((record.summary_search?.similarity || 0) * 100) / 1}
              size={[3, 4]}
              showInfo={false}
            />
          </Tooltip>
        );
      },
    },
    {
      title: formatMessage({ id: 'action.name' }),
      key: 'action',
      width: 80,
      align: 'center',
      fixed: 'right',
      render: (_value, record) => (
        <Button
          size="small"
          type="link"
          danger
          icon={<DeleteOutlined />}
          onClick={() => onDeleteRecord(record)}
        />
      ),
    },
  ];

  useEffect(() => {
    fetchHistory();
  }, [collectionId]);

  useEffect(() => {
    form.setFieldsValue({
      query: '',
      vector_search: {
        topk: 5,
        similarity: 0.7,
      },
      fulltext_search: {
        topk: 5,
      },
      graph_search: {
        topk: 5,
      },
      summary_search: {
        topk: 5,
        similarity: 0.7,
      },
    });
  }, []);

  return (
    <>
      {contextHolder}
      <Card style={{ marginBottom: 24 }}>
        <Form form={form} onFinish={onSearch}>
          <Form.Item
            name="query"
            required
            style={{ display: 'block' }}
            rules={[
              {
                required: true,
                message: formatMessage({
                  id: 'search.pleaseInputQuestion',
                }),
              },
            ]}
          >
            <Input.TextArea
              placeholder={formatMessage({
                id: 'search.pleaseInputQuestion',
              })}
              rows={3}
            />
          </Form.Item>
          <Space style={{ display: 'flex', justifyContent: 'space-between' }}>
            <Checkbox.Group
              onChange={(checkedValue) => setSearchType(checkedValue)}
              value={searchType}
              options={
                [
                  {
                    label: formatMessage({
                      id: 'search.type.vector_search',
                    }),
                    value: 'vector_search',
                  },
                  {
                    label: formatMessage({
                      id: 'search.type.fulltext_search',
                    }),
                    value: 'fulltext_search',
                  },
                  {
                    label: formatMessage({
                      id: 'search.type.graph_search',
                    }),
                    value: 'graph_search',
                  },
                  {
                    label: formatMessage({
                      id: 'search.type.summary_search',
                    }),
                    value: 'summary_search',
                  },
                ] as { label: string; value: SearchTypeEnum }[]
              }
            />
          </Space>
        </Form>
        <Divider />
        <Space
          direction="horizontal"
          style={{ display: 'flex', justifyContent: 'space-between' }}
        >
          <Form form={form} layout="inline">
            {searchType.find((item) => item === 'vector_search') && (
              <>
                <Form.Item
                  label={formatMessage({ id: 'search.vectorTopK' })}
                  name={['vector_search', 'topk']}
                >
                  <Slider style={{ width: 80 }} min={0} max={20} />
                </Form.Item>

                <Form.Item
                  label={formatMessage({
                    id: 'search.similarityThreshold',
                  })}
                  name={['vector_search', 'similarity']}
                >
                  <Slider style={{ width: 80 }} min={0} max={1} step={0.01} />
                </Form.Item>
              </>
            )}

            {searchType.find((item) => item === 'fulltext_search') && (
              <>
                <Form.Item
                  label={formatMessage({ id: 'search.fulltextTopK' })}
                  name={['fulltext_search', 'topk']}
                >
                  <Slider style={{ width: 80 }} min={0} max={20} />
                </Form.Item>
              </>
            )}

            {searchType.find((item) => item === 'graph_search') && (
              <>
                <Form.Item
                  label={formatMessage({ id: 'search.graphsearchTopK' })}
                  name={['graph_search', 'topk']}
                >
                  <Slider style={{ width: 80 }} min={0} max={20} />
                </Form.Item>
              </>
            )}

            {searchType.find((item) => item === 'summary_search') && (
              <>
                <Form.Item
                  label={formatMessage({ id: 'search.summaryTopK' })}
                  name={['summary_search', 'topk']}
                >
                  <Slider style={{ width: 80 }} min={0} max={20} />
                </Form.Item>

                <Form.Item
                  label={formatMessage({
                    id: 'search.summarySimilarityThreshold',
                  })}
                  name={['summary_search', 'similarity']}
                >
                  <Slider style={{ width: 80 }} min={0} max={1} step={0.01} />
                </Form.Item>
              </>
            )}
          </Form>
          <Button
            disabled={!_.size(searchType)}
            loading={loading}
            onClick={onSearch}
            type="primary"
          >
            {formatMessage({ id: 'search.test' })}
          </Button>
        </Space>
      </Card>

      <Table
        dataSource={records}
        bordered
        columns={columns}
        rowKey="id"
        scroll={{ x: 'max-content' }}
        size="small"
      />

      <Drawer
        open={historyModal?.visible}
        title={formatMessage({ id: 'search.searchResults' })}
        onClose={() => setHistoryModal({ visible: false, record: undefined })}
        footer={false}
        size="large"
        styles={{
          body: {
            padding: 0,
          },
        }}
        width="60%"
        extra={
          <Typography.Text type="secondary">
            {formatMessage(
              { id: 'search.searchResults.detail' },
              { count: _.size(historyModal?.record?.items) },
            )}
          </Typography.Text>
        }
      >
        {_.size(historyModal?.record?.items) ? (
          <>
            <Collapse
              defaultActiveKey={['1', '2', '3', '4', '5']}
              expandIcon={({ isActive }) => (
                <CaretRightOutlined rotate={isActive ? 90 : 0} />
              )}
              style={{
                background: token.colorBgContainer,
                borderRadius: 0,
                borderLeftWidth: 0,
                borderRightWidth: 0,
              }}
              items={historyModal?.record?.items?.map((item) => ({
                key: item.rank,
                label: (
                  <Typography.Text
                    style={{ maxWidth: 400, color: token.colorPrimary }}
                    ellipsis
                    strong
                  >
                    {item.rank}. {item.source}
                  </Typography.Text>
                ),
                children: (
                  <>
                    {item.metadata?.asset_id && (
                      <AuthAssetImage
                        collectionId={item.metadata?.collection_id}
                        documentId={item.metadata?.document_id}
                        src={`asset://${item.metadata?.asset_id}`}
                      />
                    )}
                    <ApeMarkdown>{item.content}</ApeMarkdown>
                  </>
                ),
                extra: (
                  <Space align="center" size="large">
                    {item.recall_type ? (
                      <Typography.Text>
                        {formatMessage({
                          id: `search.type.${item.recall_type}`,
                        })}
                      </Typography.Text>
                    ) : null}
                    <Tooltip title={`Score: ${item.score}`}>
                      <Space>
                        <Progress
                          type="dashboard"
                          percent={((item.score || 0) * 100) / 1}
                          showInfo={false}
                          size={20}
                        />
                        <Typography.Text type="secondary">
                          {(item.score || 0).toFixed(2)}
                        </Typography.Text>
                      </Space>
                    </Tooltip>
                  </Space>
                ),
              }))}
            />
          </>
        ) : (
          <Result
            style={{ paddingTop: 120 }}
            icon={
              <UndrawScience primaryColor={token.colorPrimary} height="200px" />
            }
            subTitle={<FormattedMessage id="search.searchResults.empty" />}
          />
        )}
      </Drawer>
    </>
  );
};
