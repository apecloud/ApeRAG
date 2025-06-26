import {
  DocumentFulltextIndexStatusEnum,
  DocumentGraphIndexStatusEnum,
  DocumentVectorIndexStatusEnum,
} from '@/api';
import { RefreshButton } from '@/components';
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
  DeleteOutlined,
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
  const [rebuildSelectedTypes, setRebuildSelectedTypes] = useState<string[]>(
    [],
  );
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
    setRebuildSelectedTypes(['VECTOR', 'FULLTEXT', 'GRAPH']);
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
          index_types: rebuildSelectedTypes as (
            | 'VECTOR'
            | 'FULLTEXT'
            | 'GRAPH'
          )[],
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
        const icon = (
          // @ts-ignore
          <FileIcon
            color={alpha(token.colorPrimary, 0.8)}
            extension={extension}
            {...iconProps}
          />
        );

        return (
          <Space>
            <Avatar size={36} shape="square" src={icon} />
            <div>
              <div>{record.name}</div>
              <Typography.Text type="secondary">
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
          <RefreshButton
            loading={documentsLoading}
            onClick={() => collectionId && getDocuments()}
          />
        </Space>
      </Space>
      <Table rowKey="id" bordered columns={columns} dataSource={documents} />
      {contextHolder}

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
                  indexTypeOptions.map((option) => option.value),
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
          onChange={(values) => setRebuildSelectedTypes(values as string[])}
          style={{ display: 'flex', flexDirection: 'row', gap: 16 }}
        />
      </Modal>
    </>
  );
};
