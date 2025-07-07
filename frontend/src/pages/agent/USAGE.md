# ApeRAG Agent 页面

## 概述

这是一个基于MCP-Agent架构的ApeRAG智能对话助手页面，提供类似Cursor/Google Gemini的聊天体验。

## 访问地址

启动前端服务后，访问：`http://localhost:3001/web/agent`

⚠️ **注意**：ApeRAG项目的路由配置了 `basename="/web"`，所以需要在路径前加上 `/web` 前缀

## 功能特性

### 1. 智能对话
- 支持多轮对话
- 基于选择的collections进行知识检索
- 显示搜索来源和相关度分数

### 2. Collection管理
- 点击 `@` 符号选择knowledge collections  
- 类似Cursor的下拉交互体验，在@符号上方展开
- 支持多选collections，即时生效
- 可以随时添加/移除collections
- 内置搜索框快速过滤collections

### 3. 模型选择
- 支持多种LLM模型切换
- 显示模型提供商图标
- 当前支持：Claude-3.5-Sonnet, GPT-4, GPT-3.5-Turbo, GLM-4

### 4. Web搜索增强
- 位于底部控制栏左侧的Web搜索开关
- 可选的网络搜索功能补充
- 搜索结果会标注来源

## 界面布局

```
┌─────────────────────────────────────────┐
│                                         │
│     Chat Messages Area                  │
│     (显示对话历史和搜索结果)               │
│                                         │
├─────────────────────────────────────────┤
│ [@] @collection1  @collection2  ✕       │ ← Collection选择区域
├─────────────────────────────────────────┤
│ Type your message...              [Send]│ ← 消息输入框
├─────────────────────────────────────────┤
│ [🔍Web Search]         claude-4-sonnet ▼│ ← 底部控制栏
└─────────────────────────────────────────┘
```

## 使用方法

### 基本对话
1. 在输入框中输入问题
2. 按Enter或点击Send发送
3. 等待AI回答

### 选择Collections
1. 点击左上角的 `@` 按钮
2. 在@符号上方弹出的下拉菜单中选择collections
3. 可以在搜索框中输入关键词过滤collections
4. 勾选/取消勾选需要的collections，选择即时生效

### 切换模型
1. 在底部控制栏中点击模型下拉框
2. 选择合适的模型
3. 后续对话将使用新选择的模型

### 启用Web搜索
1. 在底部控制栏左侧切换Web Search开关
2. 开启后会在回答中包含网络搜索信息

## 当前状态

⚠️ **当前为演示版本，使用Mock数据**

- Collections数据：使用预设的6个示例collections
- 模型列表：使用预设的4个示例模型
- API调用：使用模拟的延迟响应
- 搜索结果：返回模拟的搜索结果

## 待实现功能

### 后端集成
- [ ] 集成真实的ApeRAG MCP API
- [ ] 集成真实的模型配置API
- [ ] 集成Web搜索API
- [ ] 实现流式响应

### 功能增强
- [ ] AI自动选择collections
- [ ] 推理过程显示
- [ ] 对话历史持久化
- [ ] 导出对话记录
- [ ] 深色主题切换

### 性能优化
- [ ] 虚拟滚动优化长对话
- [ ] 图片/文件上传支持
- [ ] 响应式设计优化

## 技术栈

- **框架**: React + UmiJS
- **UI组件**: Ant Design
- **样式**: Less
- **状态管理**: React Hooks
- **类型**: TypeScript

## 开发说明

### 文件结构
```
frontend/src/pages/agent/
├── index.tsx          # 主组件
├── index.less         # 样式文件
└── README.md         # 说明文档
```

### Mock数据位置
- Collections: `mockCollections` 数组
- Models: `mockModels` 数组
- API响应: `handleSendMessage` 函数中的setTimeout

### 样式定制
所有样式都在 `index.less` 中定义，支持：
- 深色主题
- 响应式设计
- 动画效果
- 自定义滚动条

## 浏览器兼容性

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 已知问题

1. 移动端体验需要进一步优化
2. 长消息的折叠/展开功能待实现
3. 键盘快捷键支持待添加

## 反馈和建议

如有问题或建议，请提交Issue或直接联系开发团队。 