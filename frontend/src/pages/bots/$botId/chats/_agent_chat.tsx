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
import { css, styled } from '@emotion/react';
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
        const res = await api.llm_provider_modelsGet();
        const allModels = res.data.items || [];
        // Filter for completion models
        const completionModels = allModels.filter(model => model.api === 'completion');
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
    collection.title?.toLowerCase().includes(searchKeyword.toLowerCase())
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
                <Text strong>{collection.title}</Text>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {collection.document_count || 0} documents
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
      <div className="agent-header">
        <div className="agent-controls">
          {/* Collection Selector */}
          <div>
            <Text type="secondary" style={{ marginRight: 8 }}>Collections:</Text>
            <Dropdown
              menu={collectionDropdownItems}
              open={collectionDropdownOpen}
              onOpenChange={setCollectionDropdownOpen}
              trigger={['click']}
              placement="bottomLeft"
            >
              <Button className="collection-selector">
                <Space>
                  📁 {selectedCollections.length > 0 ? `${selectedCollections.length} selected` : 'Select collections'}
                  <SearchOutlined />
                </Space>
              </Button>
            </Dropdown>
          </div>

          {/* Selected Collections Tags */}
          {selectedCollections.length > 0 && (
            <div>
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

          {/* Model Selector */}
          <div>
            <Text type="secondary" style={{ marginRight: 8 }}>Model:</Text>
            <Select
              className="model-selector"
              value={selectedModel}
              onChange={setSelectedModel}
              loading={modelsLoading}
              placeholder="Select model"
            >
              {models.map(model => (
                <Select.Option key={model.name} value={model.name}>
                  <Space>
                    <Avatar
                      size="small"
                      src={MODEL_PROVIDER_ICON[model.provider]}
                    />
                    {model.name}
                  </Space>
                </Select.Option>
              ))}
            </Select>
          </div>

          {/* Web Search Toggle */}
          <div>
            <Text type="secondary" style={{ marginRight: 8 }}>Web Search:</Text>
            <Switch
              checked={webSearchEnabled}
              onChange={setWebSearchEnabled}
              checkedChildren="ON"
              unCheckedChildren="OFF"
            />
          </div>
        </div>
      </div>

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

      {/* Input Area */}
      <div className="agent-input-area">
        <Space.Compact style={{ width: '100%' }}>
          <TextArea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask your question..."
            autoSize={{ minRows: 2, maxRows: 6 }}
            disabled={readyState !== ReadyState.Open}
            style={{ resize: 'none' }}
          />
          <Button
            type="primary"
            icon={loading ? <CloseOutlined /> : <SendOutlined />}
            onClick={loading ? onCancel : handleSendMessage}
            disabled={readyState !== ReadyState.Open || (!inputValue.trim() && !loading)}
            style={{ height: 'auto', alignSelf: 'stretch' }}
          >
            {loading ? 'Stop' : 'Send'}
          </Button>
        </Space.Compact>
      </div>
    </AgentChatContainer>
  );
}; 