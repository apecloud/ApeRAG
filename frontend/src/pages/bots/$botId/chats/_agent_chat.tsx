import {
  ChatMessage,
  Feedback,
  Collection,
} from '@/api';
import { ApeMarkdown } from '@/components';
import { TypingAnimate } from '@/components/typing-animate';
import { MODEL_PROVIDER_ICON } from '@/constants';
import { api } from '@/services';
import {
  SendOutlined,
  SearchOutlined,
  CloseOutlined,
  RobotOutlined,
  UserOutlined,
} from '@ant-design/icons';
import { ReadyState } from 'ahooks/lib/useWebSocket';
import {
  Button,
  Input,
  Select,
  Space,
  Tag,
  Dropdown,
  Checkbox,
  List,
  Avatar,
  Typography,
  Switch,
  message,
  Spin,
  Card,
  Tooltip,
  GlobalToken,
  theme,
} from 'antd';
import styled from '@emotion/styled';
import { css } from '@emotion/react';
import _ from 'lodash';
import { useState, useRef, useEffect, useCallback } from 'react';
import { useModel } from 'umi';
import { ChatMessageItem } from './_chat_message';

const { Text } = Typography;
const { TextArea } = Input;

// Agent Chat Container Styles
const AgentChatContainer = styled.div<{ token: GlobalToken }>`
  display: flex;
  flex-direction: column;
  height: calc(100vh - 140px);
  
  .agent-header {
    padding: 16px 0;
    border-bottom: 1px solid ${props => props.token.colorBorderSecondary};
    margin-bottom: 16px;
  }
  
  .agent-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    margin-bottom: 16px;
  }
  
  .agent-messages {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 16px;
  }
  
  .agent-input-area {
    border-top: 1px solid ${props => props.token.colorBorderSecondary};
    padding-top: 16px;
  }
  
  .collection-selector {
    min-width: 200px;
  }
  
  .model-selector {
    min-width: 180px;
  }
  
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: ${props => props.token.colorTextSecondary};
  }
  
  .empty-icon {
    font-size: 48px;
    margin-bottom: 16px;
  }
`;

interface AgentChatProps {
  messages: ChatMessage[];
  loading: boolean;
  onSubmit: (message: {
    query: string;
    collection_ids: string[];
    model_name: string;
    web_search_enabled: boolean;
  }) => void;
  onCancel: () => void;
  onVote: (message: ChatMessage, feedback: Feedback) => void;
  readyState: ReadyState;
}

export const AgentChat: React.FC<AgentChatProps> = ({
  messages,
  loading,
  onSubmit,
  onCancel,
  onVote,
  readyState,
}) => {
  const { token } = theme.useToken();
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);
  const [collectionDropdownOpen, setCollectionDropdownOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState('gpt-4');
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [searchKeyword, setSearchKeyword] = useState('');
  
  // Get collections and models from API
  const [collections, setCollections] = useState<Collection[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load collections on mount
  useEffect(() => {
    const loadCollections = async () => {
      setCollectionsLoading(true);
      try {
        const res = await api.collectionsGet();
        setCollections(res.data.items || []);
      } catch (error) {
        message.error('Failed to load collections');
      }
      setCollectionsLoading(false);
    };
    
    loadCollections();
  }, []);

  // Load models on mount
  useEffect(() => {
    const loadModels = async () => {
      setModelsLoading(true);
      try {
        const res = await api.llmProviderModelsGet();
        const allModels = res.data.items || [];
        // Filter for completion models
        const completionModels = allModels.filter((model: any) => model.api === 'completion');
        setModels(completionModels);
        if (completionModels.length > 0) {
          setSelectedModel(completionModels[0].name || 'gpt-4');
        }
      } catch (error) {
        message.error('Failed to load models');
      }
      setModelsLoading(false);
    };
    
    loadModels();
  }, []);

  // Filter collections based on search
  const filteredCollections = collections.filter(collection =>
    (collection as any).title?.toLowerCase().includes(searchKeyword.toLowerCase())
  );

  const handleCollectionToggle = (collectionId: string, checked: boolean) => {
    if (checked) {
      setSelectedCollections(prev => [...prev, collectionId]);
    } else {
      setSelectedCollections(prev => prev.filter(id => id !== collectionId));
    }
  };

  const removeCollection = (collectionId: string) => {
    setSelectedCollections(prev => prev.filter(id => id !== collectionId));
  };

  const getCollectionName = (id: string) => {
    return collections.find(c => c.id === id)?.title || id;
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const agentMessage = {
      query: inputValue,
      collection_ids: selectedCollections,
      model_name: selectedModel,
      web_search_enabled: webSearchEnabled,
    };

    onSubmit(agentMessage);
    setInputValue('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (inputValue.trim() && !loading && readyState === ReadyState.Open) {
        handleSendMessage();
      }
    }
  };

  const collectionDropdownItems = {
    items: [
      {
        key: 'search',
        label: (
          <Input
            placeholder="Search collections..."
            prefix={<SearchOutlined />}
            value={searchKeyword}
            onChange={(e) => setSearchKeyword(e.target.value)}
            onClick={(e) => e.stopPropagation()}
          />
        ),
      },
      {
        type: 'divider' as const,
      },
      ...filteredCollections.map(collection => ({
        key: collection.id!,
        label: (
          <div onClick={(e) => e.stopPropagation()}>
            <Checkbox
              checked={selectedCollections.includes(collection.id!)}
              onChange={(e) => handleCollectionToggle(collection.id!, e.target.checked)}
            >
              <Space direction="vertical" size={0}>
                <Text strong>{(collection as any).title}</Text>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {(collection as any).documentCount || (collection as any).document_count || 0} documents
                </Text>
              </Space>
            </Checkbox>
          </div>
        ),
      })),
    ],
  };

  return (
    <AgentChatContainer token={token}>
      {/* Messages Area */}
      <div className="agent-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <RobotOutlined className="empty-icon" />
            <Text type="secondary">Start chatting with your AI agent</Text>
            <Text type="secondary" style={{ fontSize: '12px', marginTop: 8 }}>
              Select collections, choose a model, and ask questions
            </Text>
          </div>
        )}
        {messages.map((item, index) => (
          <ChatMessageItem
            key={index}
            onVote={onVote}
            loading={
              item.role === 'ai' &&
              _.size(messages) === index + 1 &&
              loading &&
              _.isEmpty(item.data)
            }
            item={item}
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - 输入框+下方按钮区布局（左右分布，风格统一） */}
      <div
        className="input-bar"
        style={{
          background: '#fff',
          borderRadius: 24,
          boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
          padding: '12px 16px 8px 16px',
          margin: '12px 0 0 0',
          minHeight: 0,
        }}
      >
        <TextArea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask your question..."
          autoSize={{ minRows: 1, maxRows: 4 }}
          disabled={readyState !== ReadyState.Open}
          style={{
            border: 'none',
            background: 'transparent',
            resize: 'none',
            fontSize: 16,
            minHeight: 40,
            outline: 'none',
            boxShadow: 'none',
            width: '100%',
            padding: 0,
            lineHeight: 1.5,
          }}
        />
        {/* 按钮区 - flex两端对齐 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 8, marginBottom: 0 }}>
          {/* 左侧：collection选择、websearch */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {/* @collection 选择 */}
            <Dropdown
              menu={collectionDropdownItems}
              open={collectionDropdownOpen}
              onOpenChange={setCollectionDropdownOpen}
              trigger={['click']}
              placement="topLeft"
            >
              <Button
                icon={<span>@</span>}
                style={{
                  height: 36,
                  minWidth: 36,
                  maxWidth: 120,
                  borderRadius: '50%',
                  background: 'none',
                  border: 'none',
                  boxShadow: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'background 0.2s',
                  overflow: 'visible',
                  textOverflow: 'clip',
                  whiteSpace: 'normal',
                  padding: '0 8px',
                }}
                className="input-bar-btn"
              />
            </Dropdown>
            {/* Web Search 开关 */}
            <Button
              icon={<SearchOutlined />}
              type={webSearchEnabled ? 'primary' : 'default'}
              onClick={() => setWebSearchEnabled((v) => !v)}
              style={{
                height: 36,
                minWidth: 36,
                borderRadius: '50%',
                background: 'none',
                border: 'none',
                boxShadow: 'none',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'background 0.2s',
              }}
              className="input-bar-btn"
            />
          </div>
          {/* 右侧：模型选择、发送按钮 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {/* 模型下拉 */}
            <Select
              className="input-bar-select"
              value={selectedModel}
              onChange={setSelectedModel}
              loading={modelsLoading}
              placeholder="Model"
              style={{ minWidth: 120, maxWidth: 400, height: 36, borderRadius: 18, background: '#f7f7f8', border: 'none', outline: 'none', overflow: 'visible', textOverflow: 'clip', whiteSpace: 'normal' }}
              dropdownStyle={{ minWidth: 120, maxWidth: 400, overflow: 'visible', whiteSpace: 'normal' }}
              bordered={false}
            >
              {(models as any[]).map((model) => (
                <Select.Option key={model['name'] || model['model']} value={model['name'] || model['model']} style={{ maxWidth: 400, overflow: 'visible', textOverflow: 'clip', whiteSpace: 'normal', wordBreak: 'break-all' }}>
                  {model['name'] || model['model']}
                </Select.Option>
              ))}
            </Select>
            {/* 发送/停止按钮 */}
            <Button
              type="primary"
              icon={loading ? <CloseOutlined /> : <SendOutlined />}
              onClick={loading ? onCancel : handleSendMessage}
              disabled={readyState !== ReadyState.Open || (!inputValue.trim() && !loading)}
              style={{
                height: 36,
                minWidth: 36,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 18,
                boxShadow: 'none',
              }}
              className="input-bar-send"
            />
          </div>
        </div>
        {/* 已选 collection tag 展示 */}
        {selectedCollections.length > 0 && (
          <div style={{ marginTop: 8, marginBottom: 0 }}>
            {selectedCollections.map(id => (
              <Tag
                key={id}
                closable
                onClose={() => removeCollection(id)}
                style={{ marginBottom: 4 }}
              >
                {getCollectionName(id)}
              </Tag>
            ))}
          </div>
        )}
      </div>
    </AgentChatContainer>
  );
}; 