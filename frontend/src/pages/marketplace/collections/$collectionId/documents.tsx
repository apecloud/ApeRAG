import {
  DocumentFulltextIndexStatusEnum,
  DocumentGraphIndexStatusEnum,
  DocumentVectorIndexStatusEnum,
} from '@/api';
import { ChunkViewer } from '@/components';
import {
  DATETIME_FORMAT,
  UI_INDEX_STATUS,
} from '@/constants';
import { SearchOutlined } from '@ant-design/icons';
import { useRequest } from 'ahooks';
import {
  Avatar,
  Badge,
  Drawer,
  Input,
  Space,
  Table,
  TableProps,
  theme,
  Tooltip,
  Typography,
} from 'antd';
import byteSize from 'byte-size';
import alpha from 'color-alpha';
import _ from 'lodash';
import moment from 'moment';
import { useMemo, useState } from 'react';
import { defaultStyles, FileIcon } from 'react-file-icon';
import ReactMarkdown from 'react-markdown';
import { useIntl } from 'umi';
import { ApeDocument } from '@/types';
import { parseConfig } from '@/utils';
import { api } from '@/services';

interface MarketplaceDocumentsProps {
  collectionId: string;
}

export const MarketplaceDocuments = ({ collectionId }: MarketplaceDocumentsProps) => {
  const [searchParams, setSearchParams] = useState<{
    name?: string;
  }>();
  const { token } = theme.useToken();
  const { formatMessage } = useIntl();
  const [viewerVisible, setViewerVisible] = useState(false);
  const [viewingDocument, setViewingDocument] = useState<ApeDocument | null>(
    null,
  );
  const [summaryDrawerVisible, setSummaryDrawerVisible] = useState(false);
  const [summaryContent, setSummaryContent] = useState<string>('');
  const [summaryDoc, setSummaryDoc] = useState<ApeDocument | null>(null);

  const {
    data: documentsRes,
    loading: documentsLoading,
  } = useRequest(
    () =>
      api.marketplaceCollectionsCollectionIdDocumentsGet({
        collectionId: collectionId || '',
      }),
    {
      refreshDeps: [collectionId],
      pollingInterval: 3000,
    },
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

  // 新增：点击查看摘要（直接用record.summary，无需再请求接口）
  const handleViewSummary = (record: ApeDocument) => {
    setSummaryDoc(record);
    setSummaryContent(record.summary || '');
    setSummaryDrawerVisible(true);
  };

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
  ];

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
      </Space>
      <Table 
        rowKey="id" 
        bordered 
        columns={columns} 
        dataSource={documents}
        loading={documentsLoading}
      />

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
          <ChunkViewer 
            document={viewingDocument} 
            collectionId={collectionId}
            isMarketplace={true}
          />
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
    </>
  );
};