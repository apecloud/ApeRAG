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
import { useNavigate, useParams, useLocation, FormattedMessage } from 'umi';

interface FailedDocument {
  documentId: string;
  name: string;
  error: string;
}

interface ConfirmResultState {
  confirmedCount: number;
  failedDocuments: FailedDocument[];
  isLoading: boolean;
}

export default () => {
  const { collectionId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

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
      title: '文档名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '失败原因',
      dataIndex: 'error',
      key: 'error',
    },
  ];

  return (
    <>
      <Card>
        <Result
          status={state.failedDocuments.length === 0 ? "success" : "warning"}
          title={`成功添加 ${state.confirmedCount} 个文档到知识库`}
          subTitle={
            state.failedDocuments.length > 0 
              ? `${state.failedDocuments.length} 个文档添加失败`
              : "所有文档已成功添加到知识库，系统正在后台建立索引"
          }
          extra={[
            <Button 
              type="primary" 
              key="back"
              size="large"
              onClick={() => navigate(`/collections/${collectionId}/documents`)}
            >
              返回文档列表
            </Button>
          ]}
        />

        {state.failedDocuments.length > 0 && (
          <div style={{ marginTop: 32 }}>
            <Typography.Title level={4}>失败文档详情</Typography.Title>
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
