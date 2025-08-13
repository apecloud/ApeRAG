import React, { useState, useRef } from 'react';
import { Button, Table, Progress, Space, Typography, Card, Checkbox, message } from 'antd';
import { InboxOutlined, FolderOpenOutlined, FileOutlined } from '@ant-design/icons';
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
  
  const [state, setState] = useState<FileSelectionState>({
    selectedFiles: [],
    scannedFiles: [],
    isScanning: false,
    scanProgress: 0,
    totalSize: 0,
    totalCount: 0
  });

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
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Card>
        <Typography.Title level={4}>
          <FormattedMessage id="document.upload.selectFiles" />
        </Typography.Title>
        
        {/* File selection area */}
        <div style={{ 
          border: '2px dashed #d9d9d9',
          borderRadius: '8px',
          padding: '40px',
          textAlign: 'center',
          marginBottom: '24px',
          backgroundColor: '#fafafa'
        }}>
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
            
            <div style={{ marginTop: '24px', textAlign: 'right' }}>
              <Space>
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
                  <FormattedMessage id="document.upload.start" />
                  {state.totalCount > 0 && ` (${state.totalCount})`}
                </Button>
              </Space>
            </div>
          </>
        )}
      </Card>
    </div>
  );
};

export default FileSelectionPage;
