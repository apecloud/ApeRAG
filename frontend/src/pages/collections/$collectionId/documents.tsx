import {
  DocumentFulltextIndexStatusEnum,
  DocumentGraphIndexStatusEnum,
  DocumentVectorIndexStatusEnum,
  RebuildIndexesRequestIndexTypesEnum,
} from '@/api';
import { ChunkViewer, RefreshButton } from '@/components';
import {
  DATETIME_FORMAT,
  SUPPORTED_COMPRESSED_EXTENSIONS,
  SUPPORTED_DOC_EXTENSIONS,
  SUPPORTED_MEDIA_EXTENSIONS,
  UI_INDEX_STATUS,
} from '@/constants';
import { getAuthorizationHeader } from '@/models/user';
import { api } from '@/services';
import { ApeDocument } from '@/types';
import { parseConfig } from '@/utils';
import {
  CopyOutlined,
  DeleteOutlined,
  EyeOutlined,
  MoreOutlined,
  ReloadOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import { useRequest } from 'ahooks';
import {
  Avatar,
  Badge,
  Button,
  Checkbox,
  Drawer,
  Dropdown,
  Input,
  Modal,
  Space,
  Table,
  TableProps,
  theme,
  Tooltip,
  Typography,
  Upload,
  UploadProps,
} from 'antd';
import byteSize from 'byte-size';
import alpha from 'color-alpha';
import _ from 'lodash';
import moment from 'moment';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { defaultStyles, FileIcon } from 'react-file-icon';
import ReactMarkdown from 'react-markdown';
import { toast } from 'react-toastify';
import { FormattedMessage, useIntl, useModel, useParams } from 'umi';

export default () => {
  const [searchParams, setSearchParams] = useState<{
    name?: string;
  }>();
  const { collectionId } = useParams();
  const { collection } = useModel('collection');
  const { setLoading } = useModel('global');
  const { token } = theme.useToken();
  const [modal, contextHolder] = Modal.useModal();
  const { formatMessage } = useIntl();
  const [rebuildModalVisible, setRebuildModalVisible] = useState(false);
  const [rebuildSelectedDocument, setRebuildSelectedDocument] =
    useState<ApeDocument | null>(null);
  const [rebuildSelectedTypes, setRebuildSelectedTypes] = useState<RebuildIndexesRequestIndexTypesEnum[]>(
    [],
  );
  const [viewerVisible, setViewerVisible] = useState(false);
  const [viewingDocument, setViewingDocument] = useState<ApeDocument | null>(
    null,
  );
  const [summaryDrawerVisible, setSummaryDrawerVisible] = useState(false);
  const [summaryContent, setSummaryContent] = useState<string>('');
  const [summaryDoc, setSummaryDoc] = useState<ApeDocument | null>(null);
  const {
    data: documentsRes,
    run: getDocuments,
    loading: documentsLoading,
  } = useRequest(
    () =>
      api.collectionsCollectionIdDocumentsGet({
        collectionId: collectionId || '',
      }),
    {
      refreshDeps: [collectionId],
      pollingInterval: 3000,
    },
  );

  const deleteDocument = useCallback(
    async (documentId?: string) => {
      if (!collectionId || !documentId) return;
      const res = await api.collectionsCollectionIdDocumentsDocumentIdDelete({
        collectionId,
        documentId,
      });
      if (res) {
        toast.success(formatMessage({ id: 'tips.delete.success' }));
        getDocuments();
      }
    },
    [collectionId],
  );

  const handleRebuildIndex = (document: ApeDocument) => {
    setRebuildSelectedDocument(document);
    setRebuildSelectedTypes([RebuildIndexesRequestIndexTypesEnum.VECTOR, RebuildIndexesRequestIndexTypesEnum.FULLTEXT, RebuildIndexesRequestIndexTypesEnum.GRAPH, RebuildIndexesRequestIndexTypesEnum.SUMMARY, RebuildIndexesRequestIndexTypesEnum.VISION]);
    setRebuildModalVisible(true);
  };

  const handleRebuildConfirm = async () => {
    if (!rebuildSelectedDocument || rebuildSelectedTypes.length === 0) return;

    try {
      setLoading(true);
      await api.collectionsCollectionIdDocumentsDocumentIdRebuildIndexesPost({
        collectionId: collectionId!,
        documentId: rebuildSelectedDocument.id!,
        rebuildIndexesRequest: {
          index_types: rebuildSelectedTypes as any,
        },
      });
      toast.success(formatMessage({ id: 'document.index.rebuild.success' }));
      setRebuildModalVisible(false);
      setRebuildSelectedDocument(null);
      setRebuildSelectedTypes([]);
      getDocuments();
    } catch (error) {
      toast.error(formatMessage({ id: 'document.index.rebuild.failed' }));
    } finally {
      setLoading(false);
    }
  };

  // Get documents with failed indexes
  const getDocumentsWithFailedIndexes = () => {
    if (!documents) return [];
    
    return documents.filter((doc) => {
      const hasFailedVector = doc.vector_index_status === 'FAILED';
      const hasFailedFulltext = doc.fulltext_index_status === 'FAILED';
      const hasFailedGraph = doc.graph_index_status === 'FAILED';
      const hasFailedSummary = doc.summary_index_status === 'FAILED';
      const hasFailedVision = doc.vision_index_status === 'FAILED';
      
      return hasFailedVector || hasFailedFulltext || hasFailedGraph || hasFailedSummary || hasFailedVision;
    });
  };

  // Get failed index types for a document
  const getFailedIndexTypes = (document: ApeDocument): RebuildIndexesRequestIndexTypesEnum[] => {
    const failedTypes: RebuildIndexesRequestIndexTypesEnum[] = [];
    
    if (document.vector_index_status === 'FAILED') {
      failedTypes.push(RebuildIndexesRequestIndexTypesEnum.VECTOR);
    }
    if (document.fulltext_index_status === 'FAILED') {
      failedTypes.push(RebuildIndexesRequestIndexTypesEnum.FULLTEXT);
    }
    if (document.graph_index_status === 'FAILED') {
      failedTypes.push(RebuildIndexesRequestIndexTypesEnum.GRAPH);
    }
    if (document.summary_index_status === 'FAILED') {
      failedTypes.push(RebuildIndexesRequestIndexTypesEnum.SUMMARY);
    }
    if (document.vision_index_status === 'FAILED') {
      failedTypes.push(RebuildIndexesRequestIndexTypesEnum.VISION);
    }
    
    return failedTypes;
  };

  // Handle rebuild failed indexes
  const handleRebuildFailedIndexes = async () => {
    const failedDocuments = getDocumentsWithFailedIndexes();
    
    if (failedDocuments.length === 0) {
      toast.info(formatMessage({ id: 'document.index.rebuild.noFailedIndexes' }));
      return;
    }

    const confirmed = await new Promise<boolean>((resolve) => {
      modal.confirm({
        title: formatMessage({ id: 'document.index.rebuild.failed.confirm.title' }),
        content: formatMessage(
          { id: 'document.index.rebuild.failed.confirm.content' },
          { count: failedDocuments.length }
        ),
        onOk: () => resolve(true),
        onCancel: () => resolve(false),
      });
    });

    if (!confirmed) return;

    try {
      setLoading(true);
      let successCount = 0;
      let failureCount = 0;

      // Process documents sequentially to avoid overwhelming the server
      for (const document of failedDocuments) {
        try {
          const failedIndexTypes = getFailedIndexTypes(document);
          if (failedIndexTypes.length > 0) {
            await api.collectionsCollectionIdDocumentsDocumentIdRebuildIndexesPost({
              collectionId: collectionId!,
              documentId: document.id!,
              rebuildIndexesRequest: {
                index_types: failedIndexTypes as any,
              },
            });
            successCount++;
          }
        } catch (error) {
          failureCount++;
          console.error(`Failed to rebuild indexes for document ${document.id}:`, error);
        }
      }

      if (successCount > 0) {
        toast.success(
          formatMessage(
            { id: 'document.index.rebuild.failed.success' },
            { success: successCount, total: failedDocuments.length }
          )
        );
        getDocuments();
      }

      if (failureCount > 0) {
        toast.warning(
          formatMessage(
            { id: 'document.index.rebuild.failed.partial' },
            { failed: failureCount, total: failedDocuments.length }
          )
        );
      }
    } catch (error) {
      toast.error(formatMessage({ id: 'document.index.rebuild.failed' }));
    } finally {
      setLoading(false);
    }
  };

  // 新增：点击查看摘要（直接用record.summary，无需再请求接口）
  const handleViewSummary = (record: ApeDocument) => {
    setSummaryDoc(record);
    setSummaryContent(record.summary || '');
    setSummaryDrawerVisible(true);
  };

  const indexTypeOptions = [
    {
      label: formatMessage({ id: 'document.index.type.vector' }),
      value: 'VECTOR',
    },
    {
      label: formatMessage({ id: 'document.index.type.fulltext' }),
      value: 'FULLTEXT',
    },
    {
      label: formatMessage({ id: 'document.index.type.graph' }),
      value: 'GRAPH',
    },
    {
      label: formatMessage({ id: 'document.index.type.summary' }),
      value: 'SUMMARY',
    },
    {
      label: formatMessage({ id: 'document.index.type.vision' }),
      value: 'VISION',
    },
  ];

  const renderIndexStatus = (
    status?:
      | DocumentVectorIndexStatusEnum
      | DocumentFulltextIndexStatusEnum
      | DocumentGraphIndexStatusEnum,
    updatedTime?: string,
  ) => {
    const statusBadge = (
      <Badge
        status={UI_INDEX_STATUS[status as keyof typeof UI_INDEX_STATUS]}
        text={formatMessage({ id: `document.index.status.${status}` })}
      />
    );

    if (updatedTime) {
      return (
        <Tooltip
          title={`${formatMessage({ id: 'text.updatedAt' })}: ${moment(updatedTime).format(DATETIME_FORMAT)}`}
        >
          {statusBadge}
        </Tooltip>
      );
    }

    return statusBadge;
  };

  const columns: TableProps<ApeDocument>['columns'] = [
    {
      title: formatMessage({ id: 'document.name' }),
      dataIndex: 'name',
      render: (value, record) => {
        const extension =
          record.name?.split('.').pop()?.toLowerCase() ||
          ('unknow' as keyof typeof defaultStyles);
        const iconProps = _.get(defaultStyles, extension);
        const isChunkViewable =
          record.vector_index_status === DocumentVectorIndexStatusEnum.ACTIVE;
        const icon = (
          // @ts-ignore
          <FileIcon
            color={alpha(token.colorPrimary, 0.8)}
            extension={extension}
            {...iconProps}
          />
        );

        const handleViewDocument = () => {
          if (isChunkViewable) {
            setViewingDocument(record);
            setViewerVisible(true);
          } else {
            // Optionally, show a message for non-PDF files
            toast.info(
              formatMessage({ id: 'document.view.unsupportedFormat' }),
            );
          }
        };

        return (
          <Space>
            <Avatar size={36} shape="square" src={icon} />
            <div
              onClick={handleViewDocument}
              style={{ cursor: isChunkViewable ? 'pointer' : 'default' }}
            >
              <Typography.Link disabled={!isChunkViewable}>
                {record.name}
              </Typography.Link>
              <Typography.Text type="secondary" style={{ display: 'block' }}>
                {byteSize(record.size || 0).toString()}
              </Typography.Text>
            </div>
          </Space>
        );
      },
    },
    {
      title: formatMessage({ id: 'document.index.type.vector' }),
      dataIndex: 'vector_index_status',
      width: 120,
      align: 'center',
      render: (value, record) => {
        return renderIndexStatus(
          record.vector_index_status,
          record.vector_index_updated,
        );
      },
    },
    {
      title: formatMessage({ id: 'document.index.type.fulltext' }),
      dataIndex: 'fulltext_index_status',
      width: 120,
      align: 'center',
      render: (value, record) => {
        return renderIndexStatus(
          record.fulltext_index_status,
          record.fulltext_index_updated,
        );
      },
    },
    {
      title: formatMessage({ id: 'document.index.type.graph' }),
      dataIndex: 'graph_index_status',
      width: 120,
      align: 'center',
      render: (value, record) => {
        return renderIndexStatus(
          record.graph_index_status,
          record.graph_index_updated,
        );
      },
    },
    {
      title: formatMessage({ id: 'document.index.type.summary' }),
      dataIndex: 'summary_index_status',
      width: 140,
      align: 'center',
      render: (value, record) => {
        const status = record.summary_index_status;
        const statusBadge = renderIndexStatus(
          status,
          record.summary_index_updated,
        );

        // 只有ACTIVE状态才可以点击查看摘要
        if (status === 'ACTIVE' && record.summary) {
          return (
            <div
              style={{ cursor: 'pointer' }}
              onClick={() => handleViewSummary(record)}
            >
              {statusBadge}
            </div>
          );
        }

        // 其他状态只显示badge，不可点击
        return statusBadge;
      },
    },
    {
      title: formatMessage({ id: 'document.index.type.vision' }),
      dataIndex: 'vision_index_status',
      width: 120,
      align: 'center',
      render: (value, record) => {
        return renderIndexStatus(
          record.vision_index_status,
          record.vision_index_updated,
        );
      },
    },
    {
      title: formatMessage({ id: 'text.updatedAt' }),
      dataIndex: 'updated',
      width: 180,
      render: (value) => {
        return moment(value).format(DATETIME_FORMAT);
      },
    },
    {
      title: formatMessage({ id: 'action.name' }),
      width: 80,
      render: (value, record) => {
        return (
          <Dropdown
            trigger={['click']}
            menu={{
              items: [
                {
                  key: 'rebuild',
                  label: formatMessage({ id: 'document.index.rebuild' }),
                  icon: <ReloadOutlined />,
                  disabled:
                    record.status === 'DELETING' || record.status === 'DELETED',
                  onClick: () => handleRebuildIndex(record),
                },
                {
                  key: 'delete',
                  label: formatMessage({ id: 'action.delete' }),
                  danger: true,
                  icon: <DeleteOutlined />,
                  disabled: record.status === 'DELETING',
                  onClick: async () => {
                    const confirmed = await modal.confirm({
                      title: formatMessage({ id: 'action.confirm' }),
                      content: formatMessage(
                        { id: 'document.delete.confirm' },
                        { name: record.name },
                      ),
                      okButtonProps: {
                        danger: true,
                      },
                    });
                    if (confirmed) {
                      deleteDocument(record.id);
                    }
                  },
                },
              ],
            }}
            overlayStyle={{ width: 160 }}
          >
            <Button type="text" icon={<MoreOutlined />} />
          </Dropdown>
        );
      },
    },
  ];

  const uploadProps = useMemo(
    (): UploadProps => ({
      name: 'files',
      multiple: true,
      // disabled: readonly,
      action: `/api/v1/collections/${collectionId}/documents`,
      data: {},
      showUploadList: false,
      headers: {
        ...getAuthorizationHeader(),
      },
      accept: SUPPORTED_DOC_EXTENSIONS.concat(SUPPORTED_MEDIA_EXTENSIONS)
        .concat(SUPPORTED_COMPRESSED_EXTENSIONS)
        .join(','),
      onChange(info) {
        const { status } = info.file; // todo
        if (status === 'done') {
          if (collectionId) {
            getDocuments();
          }
          setLoading(false);
        } else {
          setLoading(true);
          if (status === 'error') {
            toast.error(formatMessage({ id: 'tips.upload.error' }));
          }
        }
      },
    }),
    [collectionId],
  );

  const documents = useMemo(
    () =>
      documentsRes?.data?.items
        ?.map((document: any) => {
          const item: ApeDocument = {
            ...document,
            config: parseConfig(document.config),
          };
          return item;
        })
        .filter((item: ApeDocument) => {
          const titleMatch = searchParams?.name
            ? item.name?.includes(searchParams.name)
            : true;
          return titleMatch;
        }),
    [documentsRes, searchParams],
  );

  useEffect(() => setLoading(documentsLoading), [documentsLoading]);

  return (
    <>
      <Space
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: 24,
        }}
      >
        <Input
          placeholder={formatMessage({ id: 'action.search' })}
          prefix={
            <Typography.Text disabled>
              <SearchOutlined />
            </Typography.Text>
          }
          onChange={(e) => {
            setSearchParams({ ...searchParams, name: e.currentTarget.value });
          }}
          allowClear
          value={searchParams?.name}
        />
        <Space>
          {collection?.config?.source === 'system' ? (
            <Upload {...uploadProps}>
              <Button type="primary">
                <FormattedMessage id="document.upload" />
              </Button>
            </Upload>
          ) : null}
          <Button
            type="default"
            onClick={handleRebuildFailedIndexes}
            disabled={getDocumentsWithFailedIndexes().length === 0}
            loading={documentsLoading}
            style={{
              borderColor: getDocumentsWithFailedIndexes().length > 0 ? '#ff4d4f' : undefined,
              color: getDocumentsWithFailedIndexes().length > 0 ? '#ff4d4f' : undefined,
            }}
          >
            <ReloadOutlined />
            <FormattedMessage id="document.index.rebuild.failed.button" />
            {getDocumentsWithFailedIndexes().length > 0 && (
              <span style={{ marginLeft: 4, fontSize: '12px', opacity: 0.8 }}>
                ({getDocumentsWithFailedIndexes().length})
              </span>
            )}
          </Button>
          <RefreshButton
            loading={documentsLoading}
            onClick={() => collectionId && getDocuments()}
          />
        </Space>
      </Space>
      <Table rowKey="id" bordered columns={columns} dataSource={documents} />
      {contextHolder}

      <Drawer
        title={formatMessage(
          { id: 'document.view.title' },
          { name: viewingDocument?.name },
        )}
        placement="right"
        width={'90vw'}
        onClose={() => {
          setViewerVisible(false);
          setViewingDocument(null);
        }}
        open={viewerVisible}
        destroyOnClose
      >
        {viewingDocument && collectionId && (
          <ChunkViewer document={viewingDocument} collectionId={collectionId} />
        )}
      </Drawer>

      <Drawer
        title={formatMessage({ id: 'document.summary.view' })}
        placement="right"
        width={600}
        onClose={() => {
          setSummaryDrawerVisible(false);
          setSummaryContent('');
          setSummaryDoc(null);
        }}
        open={summaryDrawerVisible}
        destroyOnClose
        extra={
          summaryContent ? (
            <Button
              type="text"
              icon={<CopyOutlined />}
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(summaryContent);
                  toast.success(
                    formatMessage({ id: 'document.summary.copy.success' }),
                  );
                } catch (error) {
                  toast.error(
                    formatMessage({ id: 'document.summary.copy.failed' }),
                  );
                }
              }}
            >
              {formatMessage({ id: 'document.summary.copy' })}
            </Button>
          ) : null
        }
      >
        {summaryDoc && (
          <div
            style={{
              marginBottom: 16,
              paddingBottom: 16,
              borderBottom: '1px solid #f0f0f0',
            }}
          >
            <Typography.Text strong style={{ fontSize: '16px' }}>
              {summaryDoc.name}
            </Typography.Text>
          </div>
        )}
        {summaryContent ? (
          <ReactMarkdown>{summaryContent}</ReactMarkdown>
        ) : (
          <Typography.Text type="secondary">
            {formatMessage({ id: 'document.summary.empty' })}
          </Typography.Text>
        )}
      </Drawer>

      <Modal
        title={formatMessage({ id: 'document.index.rebuild.title' })}
        open={rebuildModalVisible}
        onCancel={() => {
          setRebuildModalVisible(false);
          setRebuildSelectedDocument(null);
          setRebuildSelectedTypes([]);
        }}
        onOk={handleRebuildConfirm}
        okText={formatMessage({ id: 'document.index.rebuild.confirm' })}
        cancelText={formatMessage({ id: 'action.cancel' })}
        okButtonProps={{
          disabled: rebuildSelectedTypes.length === 0,
        }}
      >
        <div style={{ marginBottom: 16 }}>
          <Typography.Text type="secondary">
            {formatMessage({ id: 'document.index.rebuild.description' })}
          </Typography.Text>
        </div>

        {rebuildSelectedDocument && (
          <div style={{ marginBottom: 16 }}>
            <Typography.Text strong>
              {rebuildSelectedDocument.name}
            </Typography.Text>
          </div>
        )}

        <div style={{ marginBottom: 16 }}>
          <Checkbox
            indeterminate={
              rebuildSelectedTypes.length > 0 &&
              rebuildSelectedTypes.length < indexTypeOptions.length
            }
            checked={rebuildSelectedTypes.length === indexTypeOptions.length}
            onChange={(e) => {
              if (e.target.checked) {
                setRebuildSelectedTypes(
                  indexTypeOptions.map((option) => option.value as RebuildIndexesRequestIndexTypesEnum),
                );
              } else {
                setRebuildSelectedTypes([]);
              }
            }}
          >
            {formatMessage({ id: 'document.index.rebuild.select.all' })}
          </Checkbox>
        </div>

        <Checkbox.Group
          options={indexTypeOptions}
          value={rebuildSelectedTypes}
          onChange={(values) => setRebuildSelectedTypes(values as RebuildIndexesRequestIndexTypesEnum[])}
          style={{ display: 'flex', flexDirection: 'row', gap: 16 }}
        />
      </Modal>
    </>
  );
};
