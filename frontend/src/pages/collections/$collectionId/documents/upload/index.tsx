import React, { useState, useRef, useEffect } from 'react';
import { Button, Table, Progress, Space, Typography, Card, Checkbox, message, Steps } from 'antd';
import { InboxOutlined, FolderOpenOutlined, FileOutlined, CloseOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { useNavigate, useParams, FormattedMessage, useIntl } from 'umi';
import byteSize from 'byte-size';
import { nanoid } from 'nanoid';

interface ScannedFile {
  id: string;
  name: string;
  path: string;
  size: number;
  type: string;
  selected: boolean;
  file: File;
}

interface FileSelectionState {
  selectedFiles: File[];
  scannedFiles: ScannedFile[];
  isScanning: boolean;
  scanProgress: number;
  totalSize: number;
  totalCount: number;
}

const FileSelectionPage: React.FC = () => {
  const navigate = useNavigate();
  const { collectionId } = useParams();
  const { formatMessage } = useIntl();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const [state, setState] = useState<FileSelectionState>({
    selectedFiles: [],
    scannedFiles: [],
    isScanning: false,
    scanProgress: 0,
    totalSize: 0,
    totalCount: 0
  });

  // Prevent default drag and drop behavior on the entire document
  useEffect(() => {
    const preventDefaults = (e: Event) => {
      e.preventDefault();
      e.stopPropagation();
    };

    // Add event listeners to prevent default behavior
    const events = ['dragenter', 'dragover', 'dragleave', 'drop'];
    events.forEach(eventName => {
      document.addEventListener(eventName, preventDefaults, false);
    });

    // Cleanup
    return () => {
      events.forEach(eventName => {
        document.removeEventListener(eventName, preventDefaults, false);
      });
    };
  }, []);

  const handleFileSelect = async (files: FileList | File[]) => {
    setState(prev => ({ ...prev, isScanning: true, scanProgress: 0 }));
    
    const scannedFiles: ScannedFile[] = [];
    let totalSize = 0;
    
    const fileArray = Array.from(files);
    
    for (let i = 0; i < fileArray.length; i++) {
      const file = fileArray[i];
      
      // Skip hidden files and system files
      if (file.name.startsWith('.')) continue;
      
      scannedFiles.push({
        id: nanoid(),
        name: file.name,
        path: (file as any).webkitRelativePath || file.name,
        size: file.size,
        type: file.type || 'application/octet-stream',
        selected: true,
        file: file
      });
      totalSize += file.size;
      
      setState(prev => ({ 
        ...prev, 
        scanProgress: ((i + 1) / fileArray.length) * 100 
      }));
    }
    
    setState(prev => ({
      ...prev,
      scannedFiles,
      totalSize,
      totalCount: scannedFiles.length,
      isScanning: false,
      scanProgress: 100
    }));
  };

  // Handle drag events
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    // Check if we're leaving the drop zone entirely
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX;
    const y = e.clientY;
    
    if (x < rect.left || x >= rect.right || y < rect.top || y >= rect.bottom) {
      setIsDragging(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const items = e.dataTransfer.items;
    const files: File[] = [];

    // Process dropped items
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      
      if (item.kind === 'file') {
        const entry = item.webkitGetAsEntry();
        
        if (entry) {
          // Handle both files and directories
          await processEntry(entry, files);
        } else {
          // Fallback for browsers that don't support webkitGetAsEntry
          const file = item.getAsFile();
          if (file) {
            files.push(file);
          }
        }
      }
    }

    if (files.length > 0) {
      handleFileSelect(files);
    }
  };

  // Recursively process file system entries
  const processEntry = async (entry: any, files: File[]): Promise<void> => {
    if (entry.isFile) {
      return new Promise((resolve) => {
        entry.file((file: File) => {
          files.push(file);
          resolve();
        });
      });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      return new Promise((resolve) => {
        reader.readEntries(async (entries: any[]) => {
          for (const childEntry of entries) {
            await processEntry(childEntry, files);
          }
          resolve();
        });
      });
    }
  };

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files);
    }
  };

  const handleFolderInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files);
    }
  };

  const toggleFileSelection = (fileId: string) => {
    setState(prev => {
      const updatedFiles = prev.scannedFiles.map(file =>
        file.id === fileId ? { ...file, selected: !file.selected } : file
      );
      
      const selectedFiles = updatedFiles.filter(f => f.selected);
      const totalSize = selectedFiles.reduce((sum, f) => sum + f.size, 0);
      
      return {
        ...prev,
        scannedFiles: updatedFiles,
        totalSize,
        totalCount: selectedFiles.length
      };
    });
  };

  const toggleSelectAll = (checked: boolean) => {
    setState(prev => {
      const updatedFiles = prev.scannedFiles.map(file => ({ ...file, selected: checked }));
      const totalSize = checked ? updatedFiles.reduce((sum, f) => sum + f.size, 0) : 0;
      
      return {
        ...prev,
        scannedFiles: updatedFiles,
        totalSize,
        totalCount: checked ? updatedFiles.length : 0
      };
    });
  };

  const handleStartUpload = () => {
    const selectedFiles = state.scannedFiles.filter(f => f.selected);
    if (selectedFiles.length === 0) {
      message.warning(formatMessage({ id: 'document.upload.noFilesSelected' }));
      return;
    }
    
    // Navigate to upload progress page with selected files
    navigate(`/collections/${collectionId}/documents/upload/progress`, {
      state: { files: selectedFiles }
    });
  };

  const formatFileSize = (size: number) => {
    return byteSize(size).toString();
  };

  const columns = [
    {
      title: (
        <Checkbox
          checked={state.scannedFiles.length > 0 && state.scannedFiles.every(f => f.selected)}
          indeterminate={state.scannedFiles.some(f => f.selected) && !state.scannedFiles.every(f => f.selected)}
          onChange={(e) => toggleSelectAll(e.target.checked)}
        />
      ),
      key: 'select',
      width: 50,
      render: (_: any, file: ScannedFile) => (
        <Checkbox
          checked={file.selected}
          onChange={() => toggleFileSelection(file.id)}
        />
      )
    },
    {
      title: formatMessage({ id: 'document.name' }),
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => (
        <Space>
          <FileOutlined />
          <span>{name}</span>
        </Space>
      )
    },
    {
      title: formatMessage({ id: 'document.path' }),
      dataIndex: 'path',
      key: 'path',
      ellipsis: true
    },
    {
      title: formatMessage({ id: 'document.size' }),
      dataIndex: 'size',
      key: 'size',
      width: 120,
      render: (size: number) => formatFileSize(size)
    },
    {
      title: formatMessage({ id: 'document.type' }),
      dataIndex: 'type',
      key: 'type',
      width: 150,
      ellipsis: true
    },
    {
      title: formatMessage({ id: 'action.name' }),
      key: 'action',
      width: 80,
      render: (_: any, file: ScannedFile) => (
        <Button 
          size="small" 
          danger
          onClick={() => {
            setState(prev => ({
              ...prev,
              scannedFiles: prev.scannedFiles.filter(f => f.id !== file.id),
              totalSize: prev.scannedFiles
                .filter(f => f.id !== file.id && f.selected)
                .reduce((sum, f) => sum + f.size, 0),
              totalCount: prev.scannedFiles.filter(f => f.id !== file.id && f.selected).length
            }));
          }}
        >
          <FormattedMessage id="action.delete" />
        </Button>
      )
    }
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
        
        <Steps current={0}>
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

      <div>
        
        {/* File selection area */}
        <div 
          style={{ 
            border: `2px dashed ${isDragging ? '#1890ff' : '#d9d9d9'}`,
            borderRadius: '8px',
            padding: '40px',
            textAlign: 'center',
            marginBottom: '24px',
            backgroundColor: isDragging ? '#e6f7ff' : '#fafafa',
            transition: 'all 0.3s ease'
          }}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <InboxOutlined style={{ fontSize: '48px', color: '#999', marginBottom: '16px' }} />
          
          <Typography.Paragraph>
            <FormattedMessage id="document.upload.dragTip" />
          </Typography.Paragraph>
          
          <Space size="large">
            <Button
              type="primary"
              icon={<FileOutlined />}
              size="large"
              onClick={() => fileInputRef.current?.click()}
            >
              <FormattedMessage id="document.upload.selectFiles" />
            </Button>
            
            <Button
              icon={<FolderOpenOutlined />}
              size="large"
              onClick={() => folderInputRef.current?.click()}
            >
              <FormattedMessage id="document.upload.selectFolder" />
            </Button>
          </Space>
          
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileInputChange}
            style={{ display: 'none' }}
            accept=".pdf,.doc,.docx,.txt,.md,.ppt,.pptx,.xls,.xlsx"
          />
          
          <input
            ref={folderInputRef}
            type="file"
            // @ts-ignore
            webkitdirectory=""
            directory=""
            multiple
            onChange={handleFolderInputChange}
            style={{ display: 'none' }}
          />
        </div>

        {/* Scanning progress */}
        {state.isScanning && (
          <div style={{ marginBottom: '24px' }}>
            <Progress percent={Math.round(state.scanProgress)} status="active" />
            <Typography.Text type="secondary">
              <FormattedMessage id="document.upload.scanning" />
            </Typography.Text>
          </div>
        )}

        {/* File list */}
        {state.scannedFiles.length > 0 && (
          <>
            <div style={{ marginBottom: '16px' }}>
              <Typography.Text strong>
                <FormattedMessage 
                  id="document.upload.fileCount" 
                  values={{ 
                    count: state.totalCount,
                    total: state.scannedFiles.length,
                    size: formatFileSize(state.totalSize)
                  }}
                />
              </Typography.Text>
            </div>
            
            <Table
              dataSource={state.scannedFiles}
              columns={columns}
              rowKey="id"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showTotal: (total) => `${formatMessage({ id: 'text.total' })} ${total} ${formatMessage({ id: 'text.items' })}`
              }}
              scroll={{ y: 400 }}
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
                onClick={() => navigate(`/collections/${collectionId}/documents`)}
              >
                <FormattedMessage id="action.cancel" />
              </Button>
              <Button
                type="primary"
                size="large"
                onClick={handleStartUpload}
                disabled={state.scannedFiles.filter(f => f.selected).length === 0}
              >
                <FormattedMessage id="document.upload.next" />
                {state.totalCount > 0 && ` (${state.totalCount})`}
              </Button>
            </div>
          </>
        )}
      </div>
    </>
  );
};

export default FileSelectionPage;
