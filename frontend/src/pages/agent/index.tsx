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
} from 'antd';
import {
  SendOutlined,
  SearchOutlined,
  CloseOutlined,
  RobotOutlined,
} from '@ant-design/icons';
import { useState, useRef, useEffect } from 'react';
import { MODEL_PROVIDER_ICON } from '@/constants';
import styles from './index.less';

const { Text } = Typography;
const { TextArea } = Input;

// Mock data for collections
const mockCollections = [
  { id: '1', name: '技术文档', documentCount: 1234 },
  { id: '2', name: 'API文档', documentCount: 567 },
  { id: '3', name: '产品手册', documentCount: 89 },
  { id: '4', name: '用户指南', documentCount: 445 },
  { id: '5', name: '运维手册', documentCount: 321 },
  { id: '6', name: '架构设计', documentCount: 156 },
];

// Mock data for models
const mockModels = [
  { id: 'claude-3-5-sonnet', name: 'claude-3.5-sonnet', provider: 'anthropic' },
  { id: 'gpt-4', name: 'GPT-4', provider: 'openai' },
  { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'openai' },
  { id: 'glm-4', name: 'GLM-4', provider: 'glm-4' },
];

// Mock message type
interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  collections?: string[];
  sources?: Array<{
    collection: string;
    document: string;
    score: number;
  }>;
  webSearch?: boolean;
  timestamp: Date;
}

export default function AgentPage() {
  const [selectedCollections, setSelectedCollections] = useState<string[]>([]);
  const [collectionDropdownOpen, setCollectionDropdownOpen] = useState(false);
  const [selectedModel, setSelectedModel] = useState('claude-3-5-sonnet');
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [searchKeyword, setSearchKeyword] = useState('');

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Filter collections based on search
  const filteredCollections = mockCollections.filter(collection =>
    collection.name.toLowerCase().includes(searchKeyword.toLowerCase())
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
    return mockCollections.find(c => c.id === id)?.name || id;
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      collections: selectedCollections,
      webSearch: webSearchEnabled,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simulate API call
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `基于${selectedCollections.length > 0 ? selectedCollections.map(getCollectionName).join('、') : 'AI分析'}，我来回答您的问题：\n\n${inputValue}\n\n这是一个模拟的回答。在实际实现中，这里会调用真实的ApeRAG搜索API和模型API来生成回答。`,
        sources: selectedCollections.length > 0 ? [
          {
            collection: getCollectionName(selectedCollections[0]),
            document: '相关文档1.pdf',
            score: 0.95,
          },
          {
            collection: getCollectionName(selectedCollections[0]),
            document: '相关文档2.md',
            score: 0.88,
          },
        ] : undefined,
        webSearch: webSearchEnabled,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 2000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const renderMessage = (message: Message) => {
    const isUser = message.type === 'user';
    
    return (
      <div
        key={message.id}
        className={`${styles.message} ${isUser ? styles.userMessage : styles.assistantMessage}`}
      >
        <div className={styles.messageContent}>
          <div className={styles.messageText}>
            {message.content}
          </div>
          
          {isUser && message.collections && message.collections.length > 0 && (
            <div className={styles.messageCollections}>
              📁 {message.collections.map(getCollectionName).join(' ')}
            </div>
          )}
          
          {!isUser && message.sources && (
            <div className={styles.messageSources}>
              <Text type="secondary">📚 来源：</Text>
              <ul>
                {message.sources.map((source, index) => (
                  <li key={index}>
                    <Text type="secondary">
                      • {source.collection}：{source.document} (相关度: {source.score})
                    </Text>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {!isUser && message.webSearch && (
            <div className={styles.messageWebSearch}>
              <Text type="secondary">🔍 网络信息：已包含最新网络搜索结果</Text>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className={styles.agentContainer}>
      {/* Chat Messages Area */}
      <div className={styles.messagesArea}>
        {messages.length === 0 && (
          <div className={styles.emptyState}>
            <RobotOutlined className={styles.emptyIcon} />
            <Text type="secondary">开始与ApeRAG智能助手对话</Text>
          </div>
        )}
        
        {messages.map(renderMessage)}
        
        {isLoading && (
          <div className={styles.loadingMessage}>
            <div className={styles.messageContent}>
              <Spin size="small" />
              <Text type="secondary" style={{ marginLeft: 8 }}>
                🤔 分析问题中...正在搜索相关知识库
              </Text>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Collection Selection Area */}
      <div className={styles.collectionArea}>
        <Dropdown
          open={collectionDropdownOpen}
          onOpenChange={setCollectionDropdownOpen}
          placement="topLeft"
          trigger={['click']}
          dropdownRender={() => (
            <div className={styles.collectionDropdown}>
              <div className={styles.dropdownHeader}>
                <Input
                  placeholder="🔍 搜索collection..."
                  value={searchKeyword}
                  onChange={(e) => setSearchKeyword(e.target.value)}
                  allowClear
                  size="small"
                />
              </div>
              <div className={styles.dropdownContent}>
                {filteredCollections.map((collection) => (
                  <div key={collection.id} className={styles.collectionItem}>
                    <Checkbox
                      checked={selectedCollections.includes(collection.id)}
                      onChange={(e) => handleCollectionToggle(collection.id, e.target.checked)}
                    >
                      <div className={styles.collectionInfo}>
                        <Text strong>{collection.name}</Text>
                        <Text type="secondary" className={styles.collectionCount}>
                          ({collection.documentCount} docs)
                        </Text>
                      </div>
                    </Checkbox>
                  </div>
                ))}
              </div>
            </div>
          )}
        >
          <Button
            type="text"
            className={styles.addCollectionBtn}
          >
            @
          </Button>
        </Dropdown>
        
        {selectedCollections.length === 0 ? (
          <Text type="secondary" className={styles.placeholder}>
            点击 @ 选择knowledge collection
          </Text>
        ) : (
          <Space wrap>
            {selectedCollections.map(id => (
              <Tag
                key={id}
                closable
                onClose={() => removeCollection(id)}
                className={styles.collectionTag}
              >
                @{getCollectionName(id)}
              </Tag>
            ))}
          </Space>
        )}
      </div>

      {/* Input Area */}
      <div className={styles.inputArea}>
        <div className={styles.inputContainer}>
          <TextArea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Message ApeRAG Agent..."
            autoSize={{ minRows: 3, maxRows: 8 }}
            className={styles.messageInput}
            disabled={isLoading}
          />
          {inputValue.trim() && !isLoading && (
            <Button
              type="text"
              icon={<SendOutlined />}
              onClick={handleSendMessage}
              className={styles.sendIcon}
              size="small"
            />
          )}
        </div>
      </div>

      {/* Bottom Control Bar */}
      <div className={styles.controlBar}>
        <div className={styles.leftControls}>
          <div className={styles.webSearchToggle}>
            <Space>
              <SearchOutlined />
              <Switch
                checked={webSearchEnabled}
                onChange={setWebSearchEnabled}
                size="small"
              />
              <Text type={webSearchEnabled ? 'default' : 'secondary'}>
                Web Search
              </Text>
            </Space>
          </div>
        </div>
        
        <div className={styles.rightControls}>
          <div className={styles.modelSelector}>
            <Select
              value={selectedModel}
              onChange={setSelectedModel}
              style={{ minWidth: 200 }}
              size="small"
            >
              {mockModels.map(model => (
                <Select.Option key={model.id} value={model.id}>
                  <Space align="center">
                    {MODEL_PROVIDER_ICON[model.provider] && (
                      <Avatar
                        size={16}
                        shape="square"
                        src={MODEL_PROVIDER_ICON[model.provider]}
                      />
                    )}
                    {model.name}
                  </Space>
                </Select.Option>
              ))}
            </Select>
          </div>
        </div>
      </div>


    </div>
  );
} 