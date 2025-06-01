import { SearchTestRequest, SearchTestResult } from '@/api';
import { ApeMarkdown } from '@/components';
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

type SearchTypeEnum = 'vector_search' | 'fulltext_search' | 'graph_search';

export default () => {
  const { formatMessage } = useIntl();
  const [form] = Form.useForm<SearchTestRequest>();
  const { collectionId } = useParams();
  const { loading, setLoading } = useModel('global');
  const [modal, contextHolder] = Modal.useModal();

  const [searchType, setSearchType] = useState<SearchTypeEnum[]>([
    'vector_search',
  ]);

  const [records, setRecords] = useState<SearchTestResult[]>([]);
  const [historyModal, setHistoryModal] = useState<{
    visible: boolean;
    record: SearchTestResult | undefined;
  }>();
  const { token } = theme.useToken();

  const fetchHistory = async () => {
    if (!collectionId) return;
    const res = await api.collectionsCollectionIdSearchTestsGet({
      collectionId,
    });
    setRecords(res?.data?.items || []);
  };

  const onSearch = async () => {
    if (!collectionId) return;
    const values = await form.validateFields();
    setLoading(true);
    api
      .collectionsCollectionIdSearchTestsPost(
        {
          collectionId,
          searchTestRequest: values,
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

  const onDeleteRecord = async (record: SearchTestResult) => {
    if (!collectionId || !record.id) return;
    const confirmed = await modal.confirm({
      title: formatMessage({ id: 'action.confirm' }),
      content: formatMessage({ id: 'searchTest.confirmDeleteHistory' }),
      okButtonProps: {
        danger: true,
        loading,
      },
    });
    if (confirmed) {
      await api.collectionsCollectionIdSearchTestsSearchTestIdDelete({
        collectionId,
        searchTestId: record.id,
      });
      toast.success(formatMessage({ id: 'tips.delete.success' }));
      fetchHistory();
    }
  };

  const columns: TableProps<SearchTestResult>['columns'] = [
    {
      title: formatMessage({ id: 'searchTest.history_question' }),
      dataIndex: 'query',
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
      title: formatMessage({ id: 'searchTest.searchResults' }),
      align: 'center',
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
      title: formatMessage({ id: 'searchTest.vectorTopK' }),
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
      title: formatMessage({ id: 'searchTest.similarity' }),
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
      title: formatMessage({ id: 'searchTest.fulltextTopK' }),
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
      title: formatMessage({ id: 'searchTest.graphsearchTopK' }),
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
      title: formatMessage({ id: 'action.name' }),
      key: 'action',
      width: 80,
      align: 'center',
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
                  id: 'searchTest.pleaseInputQuestion',
                }),
              },
            ]}
          >
            <Input.TextArea
              placeholder={formatMessage({
                id: 'searchTest.pleaseInputQuestion',
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
                      id: 'searchTest.type.vector_search',
                    }),
                    value: 'vector_search',
                  },
                  {
                    label: formatMessage({
                      id: 'searchTest.type.fulltext_search',
                    }),
                    value: 'fulltext_search',
                  },
                  {
                    label: formatMessage({
                      id: 'searchTest.type.graph_search',
                    }),
                    value: 'graph_search',
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
                  label={formatMessage({ id: 'searchTest.vectorTopK' })}
                  name={['vector_search', 'topk']}
                >
                  <Slider style={{ width: 80 }} min={0} max={20} />
                </Form.Item>

                <Form.Item
                  label={formatMessage({
                    id: 'searchTest.similarityThreshold',
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
                  label={formatMessage({ id: 'searchTest.fulltextTopK' })}
                  name={['fulltext_search', 'topk']}
                >
                  <Slider style={{ width: 80 }} min={0} max={20} />
                </Form.Item>
              </>
            )}

            {searchType.find((item) => item === 'graph_search') && (
              <>
                <Form.Item
                  label={formatMessage({ id: 'searchTest.graphsearchTopK' })}
                  name={['graph_search', 'topk']}
                >
                  <Slider style={{ width: 80 }} min={0} max={20} />
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
            {formatMessage({ id: 'searchTest.test' })}
          </Button>
        </Space>
      </Card>

      <Table dataSource={records} bordered columns={columns} rowKey="id" />

      <Drawer
        open={historyModal?.visible}
        title={formatMessage({ id: 'searchTest.searchResults' })}
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
              { id: 'searchTest.searchResults.detail' },
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
                children: <ApeMarkdown>{item.content}</ApeMarkdown>,
                extra: (
                  <Space align="center" size="large">
                    {item.recall_type ? (
                      <Typography.Text>
                        {formatMessage({
                          id: `searchTest.type.${item.recall_type}`,
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
            subTitle={<FormattedMessage id="searchTest.searchResults.empty" />}
          />
        )}
      </Drawer>
    </>
  );
};
