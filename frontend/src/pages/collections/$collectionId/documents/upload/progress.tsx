import { api } from '@/services';
import { getAuthorizationHeader } from '@/models/user';
import {
  Button,
  Progress,
  Table,
  Typography,
  Space,
  Card,
  Tabs,
  Tag,
  Modal,
  Tooltip,
  Steps,
  message,
} from 'antd';
import { 
  ReloadOutlined, 
  DeleteOutlined, 
  CloseOutlined,
  FileOutlined,
  InboxOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useState, useEffect, useRef } from 'react';
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
    pending: number;
  };
  shouldContinue: boolean; // Flag to control whether to continue uploading
  pagination: {
    current: number;
    pageSize: number;
  };
}

const formatFileSize = (size: number) => byteSize(size).toString();

export default () => {
  const { collectionId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const { formatMessage } = useIntl();
  const [modal, contextHolder] = Modal.useModal();
  const uploadAbortRef = useRef(false); // Ref to track if upload should be aborted
  const startUploadRef = useRef<() => void>(); // Ref to store startUpload function

  const [state, setState] = useState<UploadTaskState>({
    tasks: [],
    currentTab: 'all',
    isUploading: false,
    isConfirming: false,
    selectedTaskIds: [],
    statistics: { total: 0, uploading: 0, success: 0, failed: 0, pending: 0 },
    shouldContinue: true,
    pagination: {
      current: 1,
      pageSize: 20,
    },
  });

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
    
    return stats;
  };

  const startUpload = async () => {
    setState(prev => ({ ...prev, isUploading: true, shouldContinue: true }));
    uploadAbortRef.current = false;
    
    // 并发上传，限制并发数为3
    const concurrency = 3;
    const uploadQueue: UploadTask[] = [];
    
    // Get all pending tasks from current state
    state.tasks.forEach(task => {
      if (task.status === 'pending' || task.status === 'failed') {
        uploadQueue.push(task);
      }
    });
    
    const uploadNext = async (): Promise<void> => {
      // Check if we should stop uploading
      if (uploadAbortRef.current || uploadQueue.length === 0) {
        return;
      }
      
      const task = uploadQueue.shift();
      if (task) {
        try {
          await uploadSingleFile(task);
        } catch (error) {
          console.error(`Failed to upload ${task.name}:`, error);
        }
        // Continue with next file regardless of success/failure
        await uploadNext();
      }
    };
    
    // Start concurrent uploads
    const promises = [];
    for (let i = 0; i < Math.min(concurrency, uploadQueue.length); i++) {
      promises.push(uploadNext());
    }
    
    await Promise.all(promises);
    
    setState(prev => ({ ...prev, isUploading: false }));
    
    // Get the latest statistics after upload completes
    setState(prev => {
      const stats = updateStatistics(prev.tasks);
      return { ...prev, statistics: stats };
    });
  };

  const uploadSingleFile = async (task: UploadTask) => {
    // Check if we should stop uploading
    if (uploadAbortRef.current) {
      return;
    }
    
    try {
      setState(prev => {
        const newTasks = prev.tasks.map(t => 
          t.id === task.id ? { ...t, status: 'uploading' as const, progress: 0, error: undefined } : t
        );
        const stats = updateStatistics(newTasks);
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
          if (xhr.status === 200 || xhr.status === 201) {
            try {
              const response = JSON.parse(xhr.responseText);
              console.log('Upload response:', response); // Debug log
              setState(prev => {
                const newTasks = prev.tasks.map(t => 
                  t.id === task.id ? { 
                    ...t, 
                    status: 'success' as const, 
                    progress: 100,
                    documentId: response.document_id || response.data?.document_id,
                    error: undefined
                  } : t
                );
                const stats = updateStatistics(newTasks);
                return { ...prev, tasks: newTasks, statistics: stats };
              });
              resolve();
            } catch (error) {
              const errorMsg = formatMessage({ id: 'document.upload.error.parseResponse' });
              setState(prev => {
                const newTasks = prev.tasks.map(t => 
                  t.id === task.id ? { 
                    ...t, 
                    status: 'failed' as const, 
                    error: errorMsg
                  } : t
                );
                const stats = updateStatistics(newTasks);
                return { ...prev, tasks: newTasks, statistics: stats };
              });
              reject(new Error(errorMsg));
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
                errorMessage = xhr.statusText;
              }
            }
            
            // Handle specific error codes and messages
            if (xhr.status === 400) {
              // Bad Request - usually validation errors
              if (errorMessage.includes('unsupported file type')) {
                const fileExt = task.name.split('.').pop()?.toLowerCase() || '';
                errorMessage = formatMessage({ id: 'document.upload.error.unsupportedFileType' }, { fileType: `.${fileExt}` });
              } else if (errorMessage.includes('file size is too large')) {
                errorMessage = formatMessage({ id: 'document.upload.error.fileSizeTooLarge' });
              }
            } else if (xhr.status === 422) {
              // Unprocessable Entity - validation errors
              if (errorMessage.includes('unsupported file type')) {
                const fileExt = task.name.split('.').pop()?.toLowerCase() || '';
                errorMessage = formatMessage({ id: 'document.upload.error.unsupportedFileType' }, { fileType: `.${fileExt}` });
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
            
            // Update task status to failed
            setState(prev => {
              const newTasks = prev.tasks.map(t => 
                t.id === task.id ? { 
                  ...t, 
                  status: 'failed' as const, 
                  error: errorMessage
                } : t
              );
              const stats = updateStatistics(newTasks);
              return { ...prev, tasks: newTasks, statistics: stats };
            });
            
            // Don't show toast for individual failures - user can see in Failed tab
            reject(new Error(errorMessage));
          }
        };
        
        xhr.onerror = () => {
          const errorMsg = formatMessage({ id: 'document.upload.error.networkError' });
          setState(prev => {
            const newTasks = prev.tasks.map(t => 
              t.id === task.id ? { 
                ...t, 
                status: 'failed' as const, 
                error: errorMsg
              } : t
            );
            const stats = updateStatistics(newTasks);
            return { ...prev, tasks: newTasks, statistics: stats };
          });
          reject(new Error(errorMsg));
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
      // This catch block handles any errors not caught by the xhr handlers
      const errorMsg = error instanceof Error ? error.message : formatMessage({ id: 'document.upload.error.uploadFailed' });
      setState(prev => {
        const newTasks = prev.tasks.map(t => 
          t.id === task.id ? { 
            ...t, 
            status: 'failed' as const, 
            error: errorMsg
          } : t
        );
        const stats = updateStatistics(newTasks);
        return { ...prev, tasks: newTasks, statistics: stats };
      });
    }
  };

  // Store startUpload in ref so it can be called from useEffect
  startUploadRef.current = startUpload;

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
      statistics: { ...prev.statistics, total: tasks.length, pending: tasks.length },
    }));
    
    // 自动开始上传
    if (tasks.length > 0) {
      // Use setTimeout to ensure state is updated before starting upload
      setTimeout(() => {
        if (startUploadRef.current) {
          startUploadRef.current();
        }
      }, 100);
    }
  }, []);

  const handleRetryFailed = () => {
    // Reset failed tasks to pending before retrying
    setState(prev => {
      const newTasks = prev.tasks.map(t => 
        t.status === 'failed' ? { ...t, status: 'pending' as const, error: undefined, progress: 0 } : t
      );
      const stats = updateStatistics(newTasks);
      return { ...prev, tasks: newTasks, statistics: stats };
    });
    
    // Start upload for all tasks (will only process pending ones)
    setTimeout(() => startUpload(), 100);
  };

  // Reset pagination when switching tabs
  useEffect(() => {
    setState(prev => ({
      ...prev,
      pagination: {
        ...prev.pagination,
        current: 1,
      },
    }));
  }, [state.currentTab]);

  const handleStopUpload = () => {
    uploadAbortRef.current = true;
    setState(prev => ({ ...prev, shouldContinue: false }));
    message.info(formatMessage({ id: 'document.upload.stopped' }));
  };

  const handleConfirmUpload = async () => {
    const successTasks = state.tasks.filter(t => t.status === 'success');
    const documentIds = successTasks.map(t => t.documentId!).filter(id => id);
    
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

  const handlePaginationChange = (page: number, pageSize?: number) => {
    setState(prev => ({
      ...prev,
      pagination: {
        current: page,
        pageSize: pageSize || prev.pagination.pageSize,
      },
    }));
  };

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
      width: 120,
      align: 'center' as const,
      render: (status: string, task: UploadTask) => {
        switch (status) {
          case 'uploading':
            return <Progress percent={task.progress} size="small" status="active" />;
          case 'success':
            return <Tag color="green">Success</Tag>;
          case 'failed':
            return (
              <Tooltip title={task.error}>
                <Tag color="red">Failed</Tag>
              </Tooltip>
            );
          case 'pending':
            return <Tag>Pending</Tag>;
          default:
            return <Tag>{status}</Tag>;
        }
      },
    },
  ];

  // Calculate overall progress
  const overallProgress = state.statistics.total > 0 
    ? Math.round(((state.statistics.success + state.statistics.failed) / state.statistics.total) * 100)
    : 0;

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

      <Tabs 
        activeKey={state.currentTab}
        onChange={(key) => setState(prev => ({ ...prev, currentTab: key as any }))}
        style={{ marginBottom: 16 }}
        tabBarExtraContent={
          <Space>
            {state.isUploading && (
              <Button 
                onClick={handleStopUpload}
                danger
              >
                <FormattedMessage id="action.stop" />
              </Button>
            )}
            <Button 
              onClick={handleRetryFailed}
              disabled={state.statistics.failed === 0}
              icon={<ReloadOutlined />}
            >
              <FormattedMessage id="document.upload.progress.retryFailed" />
              {state.statistics.failed > 0 && ` (${state.statistics.failed})`}
            </Button>
          </Space>
        }
      >
        <Tabs.TabPane tab={formatMessage({ id: 'document.upload.progress.tab.all' }, { count: state.statistics.total })} key="all" />
        <Tabs.TabPane tab={formatMessage({ id: 'document.upload.progress.tab.uploading' }, { count: state.statistics.uploading })} key="uploading" />
        <Tabs.TabPane tab={formatMessage({ id: 'document.upload.progress.tab.completed' }, { count: state.statistics.success })} key="success" />
        <Tabs.TabPane 
          tab={
            <span style={{ color: state.statistics.failed > 0 ? '#ff4d4f' : undefined }}>
              {formatMessage({ id: 'document.upload.progress.tab.failed' }, { count: state.statistics.failed })}
            </span>
          } 
          key="failed" 
        />
      </Tabs>

      <Card>
        <Table
          dataSource={filteredTasks}
          columns={columns}
          rowKey="id"
          pagination={{ 
            current: state.pagination.current,
            pageSize: state.pagination.pageSize,
            showSizeChanger: true,
            pageSizeOptions: ['10', '20', '50', '100'],
            onChange: handlePaginationChange,
            onShowSizeChange: handlePaginationChange,
            total: filteredTasks.length,
          }}
          rowClassName={(record) => {
            if (record.status === 'failed') return 'error-row';
            if (record.status === 'success') return 'success-row';
            return '';
          }}
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
