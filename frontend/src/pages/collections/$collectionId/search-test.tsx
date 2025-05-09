import React, { useEffect, useState, useCallback } from 'react';
import { Card, Input, Select, Button, List, Tag, Space, Typography, Divider, message, Modal, Descriptions, Popconfirm, Pagination, Table } from 'antd';
import { useParams, useIntl } from 'umi';
import { api } from '@/services';
import moment from 'moment';

const { TextArea } = Input;

const SearchTest: React.FC = () => {
  const { collectionId } = useParams();
  const { formatMessage } = useIntl();
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState<'vector' | 'fulltext' | 'hybrid'>('vector');
  const [vectorParams, setVectorParams] = useState({ topk: 5, similarity: 0.7 });
  const [fulltextParams, setFulltextParams] = useState({ topk: 5 });
  const [result, setResult] = useState<any[]>([]);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [historyModal, setHistoryModal] = useState<{ visible: boolean, record: any | null }>({ visible: false, record: null });
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    fetchHistory();
    setPage(1);
  }, [collectionId]);

  const fetchHistory = async () => {
    if (!collectionId) return;
    try {
      const res = await api.collectionsCollectionIdSearchTestsGet({ collectionId });
      const items = Array.isArray(res?.data?.items) ? res.data.items : [];
      setHistory(items);
      setTotal(items.length);
    } catch (e) {
      setHistory([]);
      setTotal(0);
    }
  };

  const handleTest = async () => {
    if (!query) {
      message.warning(formatMessage({ id: 'searchTest.pleaseInputQuestion' }));
      return;
    }
    setLoading(true);
    try {
      const params: any = {
        query,
        search_type: searchType,
      };
      if (searchType === 'vector') params.vector_search = vectorParams;
      if (searchType === 'fulltext') params.fulltext_search = fulltextParams;
      if (searchType === 'hybrid') {
        params.vector_search = vectorParams;
        params.fulltext_search = fulltextParams;
      }
      const res = await api.collectionsCollectionIdSearchTestsPost({ collectionId: collectionId as string, searchTestRequest: params });
      if (!res || !res.data) {
        message.error(formatMessage({ id: 'searchTest.searchFailed' }) + ' (No response or timeout)');
        setResult([]);
        return;
      }
      setResult(res.data.items || []);
      fetchHistory();
    } catch (e: any) {
      setResult([]);
      message.error(e?.message || formatMessage({ id: 'searchTest.searchFailed' }));
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteHistory = async (searchTestId: string) => {
    console.log('delete', searchTestId, collectionId);
    if (!collectionId || !searchTestId) return;
    try {
      await api.collectionsCollectionIdSearchTestsSearchTestIdDelete({ collectionId, searchTestId });
      message.success(formatMessage({ id: 'searchTest.deleteSuccess' }));
      fetchHistory();
    } catch (e: any) {
      message.error(e?.message || formatMessage({ id: 'searchTest.deleteFailed' }));
    }
  };

  // 历史记录弹窗内容
  const renderHistoryModal = () => {
    const record = historyModal.record;
    if (!record) return null;
    return (
      <Modal
        open={historyModal.visible}
        title={formatMessage({ id: 'searchTest.historyQueryDetails' })}
        onCancel={() => setHistoryModal({ visible: false, record: null })}
        footer={null}
        width={1200}
      >
        <Descriptions column={2} bordered size="small">
          <Descriptions.Item label={formatMessage({ id: 'searchTest.question' })}>{record.query}</Descriptions.Item>
          <Descriptions.Item label={formatMessage({ id: 'searchTest.time' })}>{record.created ? moment(record.created).format('YYYY-MM-DD HH:mm:ss') : ''}</Descriptions.Item>
          <Descriptions.Item label={formatMessage({ id: 'searchTest.searchType' })}>{record.search_type}</Descriptions.Item>
          {record.vector_search && (
            <Descriptions.Item label={formatMessage({ id: 'searchTest.vectorSearchParams' })}>
              {formatMessage({ id: 'searchTest.topK' })}: {record.vector_search.topk}，{formatMessage({ id: 'searchTest.similarity' })}: {record.vector_search.similarity}
            </Descriptions.Item>
          )}
          {record.fulltext_search && (
            <Descriptions.Item label={formatMessage({ id: 'searchTest.fulltextSearchParams' })}>
              {formatMessage({ id: 'searchTest.topK' })}: {record.fulltext_search.topk}
            </Descriptions.Item>
          )}
        </Descriptions>
        <Divider orientation="center">{formatMessage({ id: 'searchTest.searchResults' })}</Divider>
        <Table
          dataSource={record.items || []}
          rowKey="rank"
          columns={[
            {
              title: formatMessage({ id: 'searchTest.rank' }),
              dataIndex: 'rank',
              width: 80,
              align: 'center' as const,
              onHeaderCell: () => ({ style: { whiteSpace: 'nowrap' } }),
            },
            {
              title: formatMessage({ id: 'searchTest.content' }),
              dataIndex: 'content',
              ellipsis: true,
              render: (text: string) => (
                <Typography.Text
                  style={{
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-all',
                  }}
                >
                  {text}
                </Typography.Text>
              ),
            },
            {
              title: formatMessage({ id: 'searchTest.score' }),
              dataIndex: 'score',
              width: 80,
              align: 'center' as const,
              render: (score: number) => score?.toFixed(4),
            },
            {
              title: formatMessage({ id: 'searchTest.source' }),
              dataIndex: 'source',
              width: 180,
              ellipsis: true,
            },
          ]}
          pagination={false}
          style={{ width: '100%' }}
          bordered
        />
      </Modal>
    );
  };

  const columns = [
    {
      title: formatMessage({ id: 'searchTest.question' }),
      dataIndex: 'query',
      ellipsis: true,
      onHeaderCell: () => ({ style: { whiteSpace: 'nowrap' } }),
    },
    {
      title: formatMessage({ id: 'searchTest.time' }),
      dataIndex: 'created',
      width: 180,
      align: 'right' as const,
      onHeaderCell: () => ({ style: { whiteSpace: 'nowrap' } }),
      render: (text: string) => text ? moment(text).format('YYYY-MM-DD HH:mm:ss') : '',
    },
    {
      title: formatMessage({ id: 'searchTest.action' }),
      key: 'action',
      width: 160,
      align: 'center' as const,
      onHeaderCell: () => ({ style: { whiteSpace: 'nowrap' } }),
      render: (_: any, record: any) => (
        <Space size={8} wrap={false}>
          <Button
            size="small"
            type="link"
            onClick={e => {
              e.stopPropagation();
              setHistoryModal({ visible: true, record });
            }}
          >
            {formatMessage({ id: 'searchTest.details' })}
          </Button>
          <Popconfirm
            title={formatMessage({ id: 'searchTest.confirmDeleteHistory' })}
            onConfirm={() => handleDeleteHistory(record.id)}
            okText={formatMessage({ id: 'searchTest.delete' })}
            cancelText={formatMessage({ id: 'searchTest.cancel' })}
          >
            <Button size="small" type="link" danger>
              {formatMessage({ id: 'searchTest.delete' })}
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <>
      <Card title={formatMessage({ id: 'searchTest.title' })} style={{ marginBottom: 24 }}>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <TextArea
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder={formatMessage({ id: 'searchTest.pleaseInputQuestion' })}
            rows={3}
          />
          <Space wrap>
            <Select value={searchType} onChange={setSearchType} style={{ width: 160 }}>
              <Select.Option value="vector">{formatMessage({ id: 'searchTest.vectorSearch' })}</Select.Option>
              <Select.Option value="fulltext">{formatMessage({ id: 'searchTest.fulltextSearch' })}</Select.Option>
              <Select.Option value="hybrid">{formatMessage({ id: 'searchTest.hybridSearch' })}</Select.Option>
            </Select>
            {searchType === 'vector' && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span>{formatMessage({ id: 'searchTest.topK' })}：</span>
                  <Input
                    type="number"
                    min={1}
                    value={vectorParams.topk}
                    onChange={e => setVectorParams({ ...vectorParams, topk: Number(e.target.value) })}
                    placeholder={formatMessage({ id: 'searchTest.topK' })}
                    style={{ width: 80 }}
                  />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span>{formatMessage({ id: 'searchTest.similarityThreshold' })}：</span>
                  <Input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={vectorParams.similarity}
                    onChange={e => setVectorParams({ ...vectorParams, similarity: Number(e.target.value) })}
                    placeholder={formatMessage({ id: 'searchTest.similarityThreshold' })}
                    style={{ width: 100 }}
                  />
                </div>
              </>
            )}
            {searchType === 'fulltext' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span>{formatMessage({ id: 'searchTest.topK' })}：</span>
                <Input
                  type="number"
                  min={1}
                  value={fulltextParams.topk}
                  onChange={e => setFulltextParams({ ...fulltextParams, topk: Number(e.target.value) })}
                  placeholder={formatMessage({ id: 'searchTest.topK' })}
                  style={{ width: 80 }}
                />
              </div>
            )}
            {searchType === 'hybrid' && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span>{formatMessage({ id: 'searchTest.vectorTopK' })}：</span>
                  <Input
                    type="number"
                    min={1}
                    value={vectorParams.topk}
                    onChange={e => setVectorParams({ ...vectorParams, topk: Number(e.target.value) })}
                    placeholder={formatMessage({ id: 'searchTest.vectorTopK' })}
                    style={{ width: 100 }}
                  />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span>{formatMessage({ id: 'searchTest.vectorSimilarity' })}：</span>
                  <Input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={vectorParams.similarity}
                    onChange={e => setVectorParams({ ...vectorParams, similarity: Number(e.target.value) })}
                    placeholder={formatMessage({ id: 'searchTest.vectorSimilarity' })}
                    style={{ width: 120 }}
                  />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span>{formatMessage({ id: 'searchTest.fulltextTopK' })}：</span>
                  <Input
                    type="number"
                    min={1}
                    value={fulltextParams.topk}
                    onChange={e => setFulltextParams({ ...fulltextParams, topk: Number(e.target.value) })}
                    placeholder={formatMessage({ id: 'searchTest.fulltextTopK' })}
                    style={{ width: 100 }}
                  />
                </div>
              </>
            )}
            <Button type="primary" onClick={handleTest} loading={loading}>{formatMessage({ id: 'searchTest.test' })}</Button>
          </Space>
        </Space>
      </Card>
      <Card title={formatMessage({ id: 'searchTest.searchResults' })} style={{ marginBottom: 24 }}>
        <Table
          dataSource={result}
          rowKey="rank"
          columns={[
            {
              title: formatMessage({ id: 'searchTest.rank' }),
              dataIndex: 'rank',
              width: 80,
              align: 'center' as const,
              onHeaderCell: () => ({ style: { whiteSpace: 'nowrap' } }),
            },
            {
              title: formatMessage({ id: 'searchTest.content' }),
              dataIndex: 'content',
              ellipsis: true,
              render: (text: string) => (
                <Typography.Text
                  style={{
                    display: '-webkit-box',
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-all',
                  }}
                >
                  {text}
                </Typography.Text>
              ),
            },
            {
              title: formatMessage({ id: 'searchTest.score' }),
              dataIndex: 'score',
              width: 80,
              align: 'center' as const,
              render: (score: number) => score?.toFixed(4),
            },
            {
              title: formatMessage({ id: 'searchTest.source' }),
              dataIndex: 'source',
              width: 180,
              ellipsis: true,
            },
          ]}
          pagination={false}
          style={{ width: '100%' }}
          bordered
        />
      </Card>
      <Card title={formatMessage({ id: 'searchTest.history' })}>
        <Table
          dataSource={history.slice((page - 1) * pageSize, page * pageSize)}
          columns={columns}
          rowKey="id"
          pagination={false}
          style={{ width: '100%' }}
        />
        <Pagination
          style={{ marginTop: 16, textAlign: 'right' }}
          current={page}
          pageSize={pageSize}
          total={total}
          showSizeChanger
          onChange={(p, ps) => {
            setPage(p);
            setPageSize(ps);
          }}
        />
      </Card>
      {renderHistoryModal()}
    </>
  );
};

export default SearchTest; 