import { api } from '@/services';
import { getAuthorizationHeader } from '@/models/user';
import {
  Button,
  Checkbox,
  Progress,
  Table,
  Typography,
  Space,
  Card,
  Statistic,
  Tabs,
  Tag,
  Modal,
  Tooltip,
  Steps,
} from 'antd';
import { 
  ReloadOutlined, 
  DeleteOutlined, 
  CloseOutlined,
  FileOutlined,
  InboxOutlined,
  CheckCircleOutlined 
} from '@ant-design/icons';
import { useState, useEffect } from 'react';
import { useNavigate, useParams, useLocation, FormattedMessage, useIntl } from 'umi';
import { toast } from 'react-toastify';
import byteSize from 'byte-size';

interface UploadTask {
  id: string;
  name: string;
  size: number;
  path: string;
  status: 'pending' | 'uploading' | 'success' | 'failed';
  progress: number;
  error?: string;
  documentId?: string;
  uploadSpeed?: number;
  remainingTime?: number;
  file: File;
}

interface UploadTaskState {
  tasks: UploadTask[];
  currentTab: 'all' | 'uploading' | 'success' | 'failed';
  isUploading: boolean;
  isConfirming: boolean;
  selectedTaskIds: string[];
  statistics: {
    total: number;
    uploading: number;
    success: number;
    failed: number;
  };
}

const formatFileSize = (size: number) => byteSize(size).toString();

export default () => {
  const { collectionId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const { formatMessage } = useIntl();
  const [modal, contextHolder] = Modal.useModal();

  const [state, setState] = useState<UploadTaskState>({
    tasks: [],
    currentTab: 'all',
    isUploading: false,
    isConfirming: false,
    selectedTaskIds: [],
    statistics: { total: 0, uploading: 0, success: 0, failed: 0 },
  });

  useEffect(() => {
    // 从路由状态获取选中的文件
    const files = (location.state as any)?.files || [];
    const tasks: UploadTask[] = files.map((file: any) => ({
      id: file.id,
      name: file.name,
      size: file.size,
      path: file.path,
      status: 'pending' as const,
      progress: 0,
      file: file.file,
    }));
    
    setState(prev => ({
      ...prev,
      tasks,
      statistics: { ...prev.statistics, total: tasks.length },
    }));
    
    // 自动开始上传
    if (tasks.length > 0) {
      startUpload(tasks);
    }
  }, []);

  const updateStatistics = (tasks: UploadTask[]) => {
    const stats = tasks.reduce(
      (acc, task) => {
        if (task.status === 'uploading') acc.uploading++;
        else if (task.status === 'success') acc.success++;
        else if (task.status === 'failed') acc.failed++;
        else if (task.status === 'pending') acc.pending++;
        return acc;
      },
      { total: tasks.length, uploading: 0, success: 0, failed: 0, pending: 0 }
    );
    
    setState(prev => ({ ...prev, statistics: stats }));
  };

  const startUpload = async (tasks: UploadTask[]) => {
    setState(prev => ({ ...prev, isUploading: true }));
    
    // 并发上传，限制并发数为3
    const concurrency = 3;
    const chunks = [];
    for (let i = 0; i < tasks.length; i += concurrency) {
      chunks.push(tasks.slice(i, i + concurrency));
    }
    
    for (const chunk of chunks) {
      await Promise.all(chunk.map(task => uploadSingleFile(task)));
    }
    
    setState(prev => ({ ...prev, isUploading: false }));
  };

  const uploadSingleFile = async (task: UploadTask) => {
    try {
      setState(prev => {
        const newTasks = prev.tasks.map(t => 
          t.id === task.id ? { ...t, status: 'uploading' as const, progress: 0 } : t
        );
        // Update statistics when starting upload
        const stats = newTasks.reduce(
          (acc, t) => {
            if (t.status === 'uploading') acc.uploading++;
            else if (t.status === 'success') acc.success++;
            else if (t.status === 'failed') acc.failed++;
            else if (t.status === 'pending') acc.pending++;
            return acc;
          },
          { total: newTasks.length, uploading: 0, success: 0, failed: 0, pending: 0 }
        );
        return { ...prev, tasks: newTasks, statistics: stats };
      });
      
      const formData = new FormData();
      formData.append('file', task.file);  // Changed from 'files' to 'file' to match backend API
      
      const xhr = new XMLHttpRequest();
      
      return new Promise<void>((resolve, reject) => {
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const progress = Math.round((event.loaded * 100) / event.total);
            setState(prev => ({
              ...prev,
              tasks: prev.tasks.map(t => 
                t.id === task.id ? { ...t, progress } : t
              ),
            }));
          }
        };
        
        xhr.onload = () => {
          if (xhr.status === 200) {
            try {
              const response = JSON.parse(xhr.responseText);
              console.log('Upload response:', response); // Debug log
              setState(prev => {
                const newTasks = prev.tasks.map(t => 
                  t.id === task.id ? { 
                    ...t, 
                    status: 'success' as const, 
                    progress: 100,
                    documentId: response.document_id || response.data?.document_id
                  } : t
                );
                // Calculate statistics directly here
                const stats = newTasks.reduce(
                  (acc, t) => {
                    if (t.status === 'uploading') acc.uploading++;
                    else if (t.status === 'success') acc.success++;
                    else if (t.status === 'failed') acc.failed++;
                    else if (t.status === 'pending') acc.pending++;
                    return acc;
                  },
                  { total: newTasks.length, uploading: 0, success: 0, failed: 0, pending: 0 }
                );
                return { ...prev, tasks: newTasks, statistics: stats };
              });
              resolve();
            } catch (error) {
              reject(new Error(formatMessage({ id: 'document.upload.error.parseResponse' })));
            }
          } else {
            // Parse error response
            let errorMessage = formatMessage({ id: 'document.upload.error.httpError' }, { status: xhr.status });
            try {
              const errorResponse = JSON.parse(xhr.responseText);
              if (errorResponse.detail) {
                errorMessage = errorResponse.detail;
              } else if (errorResponse.message) {
                errorMessage = errorResponse.message;
              }
            } catch (e) {
              // If response is not JSON, use status text
              if (xhr.statusText) {
                errorMessage = formatMessage({ id: 'document.upload.error.httpError' }, { status: xhr.statusText });
              }
            }
            
            // Handle specific error codes
            if (xhr.status === 422) {
              // Unprocessable Entity - usually validation errors
              if (errorMessage.includes('unsupported file type')) {
                errorMessage = formatMessage({ id: 'document.upload.error.unsupportedFileType' });
              } else if (errorMessage.includes('file size is too large')) {
                errorMessage = formatMessage({ id: 'document.upload.error.fileSizeTooLarge' });
              }
            } else if (xhr.status === 404) {
              errorMessage = formatMessage({ id: 'document.upload.error.collectionNotFound' });
            } else if (xhr.status === 403) {
              errorMessage = formatMessage({ id: 'document.upload.error.noPermission' });
            } else if (xhr.status === 401) {
              errorMessage = formatMessage({ id: 'document.upload.error.authFailed' });
            }
            
            reject(new Error(errorMessage));
          }
        };
        
        xhr.onerror = () => {
          reject(new Error(formatMessage({ id: 'document.upload.error.networkError' })));
        };
        
        xhr.open('POST', `/api/v1/collections/${collectionId}/documents/upload`);
        
        // 添加认证头
        const authHeaders = getAuthorizationHeader();
        if (authHeaders) {
          Object.entries(authHeaders).forEach(([key, value]) => {
            xhr.setRequestHeader(key, value);
          });
        }
        
        xhr.send(formData);
      });
      
    } catch (error) {
      setState(prev => {
        const newTasks = prev.tasks.map(t => 
          t.id === task.id ? { 
            ...t, 
            status: 'failed' as const, 
            error: error instanceof Error ? error.message : formatMessage({ id: 'document.upload.error.uploadFailed' })
          } : t
        );
        // Calculate statistics directly here
        const stats = newTasks.reduce(
          (acc, t) => {
            if (t.status === 'uploading') acc.uploading++;
            else if (t.status === 'success') acc.success++;
            else if (t.status === 'failed') acc.failed++;
            else if (t.status === 'pending') acc.pending++;
            return acc;
          },
          { total: newTasks.length, uploading: 0, success: 0, failed: 0, pending: 0 }
        );
        return { ...prev, tasks: newTasks, statistics: stats };
      });
    }
  };

  const handleRetryFailed = () => {
    const failedTasks = state.tasks.filter(t => t.status === 'failed');
    if (failedTasks.length > 0) {
      startUpload(failedTasks);
    }
  };


  const handleConfirmUpload = async () => {
    const successTasks = state.tasks.filter(t => t.status === 'success');
    const documentIds = successTasks.map(t => t.documentId!);
    
    if (documentIds.length === 0) {
      toast.warning(formatMessage({ id: 'document.upload.error.noDocumentsToConfirm' }));
      return;
    }
    
    try {
      setState(prev => ({ ...prev, isConfirming: true }));
      
      const response = await api.collectionsCollectionIdDocumentsConfirmPost({
        collectionId: collectionId!,
        confirmDocumentsRequest: { document_ids: documentIds }
      });
      
      // 跳转到确认结果页面
      navigate(`/collections/${collectionId}/documents/upload/result`, {
        state: { result: response.data }
      });
    } catch (error) {
      toast.error(formatMessage({ id: 'document.upload.error.confirmFailed' }) + ': ' + (error instanceof Error ? error.message : formatMessage({ id: 'document.upload.error.unknown' })));
    } finally {
      setState(prev => ({ ...prev, isConfirming: false }));
    }
  };

  const filteredTasks = state.tasks.filter(task => {
    switch (state.currentTab) {
      case 'uploading': return task.status === 'uploading';
      case 'success': return task.status === 'success';
      case 'failed': return task.status === 'failed';
      default: return true;
    }
  });

  const columns = [
    {
      title: formatMessage({ id: 'document.name' }),
      dataIndex: 'name',
      key: 'name',
      ellipsis: true,
    },
    {
      title: formatMessage({ id: 'document.path' }),
      dataIndex: 'path',
      key: 'path',
      ellipsis: true,
    },
    {
      title: formatMessage({ id: 'document.size' }),
      dataIndex: 'size',
      key: 'size',
      width: 120,
      render: (size: number) => formatFileSize(size),
    },
    {
      title: formatMessage({ id: 'document.status' }),
      dataIndex: 'status',
      key: 'status',
      width: 150,
      render: (status: string, task: UploadTask) => {
        switch (status) {
          case 'uploading':
            return <Progress percent={task.progress} size="small" />;
          case 'success':
            return <Tag color="green">{formatMessage({ id: 'document.upload.status.success' })}</Tag>;
          case 'failed':
            return (
              <Tooltip title={task.error || formatMessage({ id: 'document.upload.status.failed' })}>
                <Tag color="red">{formatMessage({ id: 'document.upload.status.failed' })}</Tag>
              </Tooltip>
            );
          default:
            return <Tag>{formatMessage({ id: 'document.upload.status.pending' })}</Tag>;
        }
      },
    },
    {
      title: formatMessage({ id: 'action.name' }),
      key: 'action',
      width: 80,
      render: (_: any, task: UploadTask) => (
        task.status === 'failed' ? (
          <Button 
            size="small" 
            onClick={() => uploadSingleFile(task)}
            loading={state.isUploading}
          >
            <FormattedMessage id="document.upload.action.retry" />
          </Button>
        ) : null
      ),
    },
  ];

  return (
    <>
      {/* Header with steps and exit button */}
      <div style={{ 
        marginBottom: '24px'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
          <Typography.Title level={4} style={{ margin: 0 }}>
            <FormattedMessage id="document.upload" />
          </Typography.Title>
          <Button 
            icon={<CloseOutlined />}
            onClick={() => navigate(`/collections/${collectionId}/documents`)}
          >
            <FormattedMessage id="action.back" />
          </Button>
        </div>
        
        <Steps current={1}>
          <Steps.Step 
            title={<FormattedMessage id="document.upload.step.select" />} 
            icon={<FileOutlined />}
          />
          <Steps.Step 
            title={<FormattedMessage id="document.upload.step.upload" />} 
            icon={<InboxOutlined />}
          />
        </Steps>
      </div>

      <Card style={{ marginBottom: 24 }}>
        <Space size="large">
          <Statistic title={formatMessage({ id: 'document.upload.progress.totalFiles' })} value={state.statistics.total} />
          <Statistic title={formatMessage({ id: 'document.upload.progress.uploading' })} value={state.statistics.uploading} />
          <Statistic title={formatMessage({ id: 'document.upload.progress.completed' })} value={state.statistics.success} />
          <Statistic title={formatMessage({ id: 'document.upload.progress.failed' })} value={state.statistics.failed} />
        </Space>
      </Card>

      <Tabs 
        activeKey={state.currentTab}
        onChange={(key) => setState(prev => ({ ...prev, currentTab: key as any }))}
        style={{ marginBottom: 16 }}
      >
        <Tabs.TabPane tab={formatMessage({ id: 'document.upload.progress.tab.all' }, { count: state.statistics.total })} key="all" />
        <Tabs.TabPane tab={formatMessage({ id: 'document.upload.progress.tab.uploading' }, { count: state.statistics.uploading })} key="uploading" />
        <Tabs.TabPane tab={formatMessage({ id: 'document.upload.progress.tab.completed' }, { count: state.statistics.success })} key="success" />
        <Tabs.TabPane tab={formatMessage({ id: 'document.upload.progress.tab.failed' }, { count: state.statistics.failed })} key="failed" />
      </Tabs>

      <div style={{ marginBottom: 16 }}>
        <Button 
          onClick={handleRetryFailed}
          disabled={state.statistics.failed === 0}
          icon={<ReloadOutlined />}
        >
          <FormattedMessage id="document.upload.progress.retryFailed" />
        </Button>
      </div>

      <Card>
        <Table
          dataSource={filteredTasks}
          columns={columns}
          rowKey="id"
          pagination={{ pageSize: 20 }}
        />

        <div style={{ 
          marginTop: '24px', 
          padding: '16px 0',
          borderTop: '1px solid #f0f0f0',
          display: 'flex',
          justifyContent: 'space-between'
        }}>
          <Button 
            size="large"
            onClick={() => navigate(`/collections/${collectionId}/documents/upload`)}
          >
            <FormattedMessage id="action.back" />
          </Button>
          <Button 
            type="primary" 
            size="large"
            onClick={handleConfirmUpload}
            loading={state.isConfirming}
            disabled={state.statistics.success === 0}
          >
            <FormattedMessage id="document.upload.confirm" /> ({state.statistics.success})
          </Button>
        </div>
      </Card>

      {contextHolder}
    </>
  );
};
