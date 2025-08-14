import {
  Button,
  Result,
  Table,
  Typography,
  Card,
  Steps,
} from 'antd';
import { 
  CloseOutlined,
  FileOutlined,
  InboxOutlined,
  CheckCircleOutlined 
} from '@ant-design/icons';
import { useState, useEffect } from 'react';
import { useNavigate, useParams, useLocation, FormattedMessage, useIntl } from 'umi';

interface FailedDocument {
  documentId: string;
  name: string;
  error: string;
}

// Error code to i18n key mapping
const ERROR_CODE_MAP: Record<string, string> = {
  'DOCUMENT_EXPIRED': 'document.error.expired',
  'DOCUMENT_NOT_UPLOADED': 'document.error.notUploaded',
  'DOCUMENT_NOT_FOUND': 'document.error.notFound',
  'CONFIRMATION_FAILED': 'document.error.confirmationFailed',
};

interface ConfirmResultState {
  confirmedCount: number;
  failedDocuments: FailedDocument[];
  isLoading: boolean;
}

export default () => {
  const { collectionId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const intl = useIntl();

  const [state, setState] = useState<ConfirmResultState>({
    confirmedCount: 0,
    failedDocuments: [],
    isLoading: false,
  });

  useEffect(() => {
    const result = (location.state as any)?.result;
    if (result) {
      setState({
        confirmedCount: result.confirmed_count || 0,
        failedDocuments: result.failed_documents || [],
        isLoading: false,
      });
    }
  }, []);

  const columns = [
    {
      title: intl.formatMessage({ id: 'document.name' }),
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => name || intl.formatMessage({ id: 'document.unknown' }),
    },
    {
      title: intl.formatMessage({ id: 'document.error.reason' }),
      dataIndex: 'error',
      key: 'error',
      render: (error: string) => {
        // Map error code to i18n message
        const i18nKey = ERROR_CODE_MAP[error];
        if (i18nKey) {
          return intl.formatMessage({ id: i18nKey });
        }
        // Fallback to the original error message if not a known code
        return error;
      },
    },
  ];

  return (
    <>
      <Card>
        <Result
          status={state.failedDocuments.length === 0 ? "success" : "warning"}
          title={intl.formatMessage(
            { id: 'document.upload.result.success' },
            { count: state.confirmedCount }
          )}
          subTitle={
            state.failedDocuments.length > 0 
              ? intl.formatMessage(
                  { id: 'document.upload.result.partialFailed' },
                  { count: state.failedDocuments.length }
                )
              : intl.formatMessage({ id: 'document.upload.result.allSuccess' })
          }
          extra={[
            <Button 
              type="primary" 
              key="back"
              size="large"
              onClick={() => navigate(`/collections/${collectionId}/documents`)}
            >
              <FormattedMessage id="document.upload.result.backToList" />
            </Button>
          ]}
        />

        {state.failedDocuments.length > 0 && (
          <div style={{ marginTop: 32 }}>
            <Typography.Title level={4}>
              <FormattedMessage id="document.upload.result.failedDetails" />
            </Typography.Title>
            <Table
              dataSource={state.failedDocuments}
              columns={columns}
              rowKey="documentId"
              pagination={false}
            />
          </div>
        )}
      </Card>
    </>
  );
};
