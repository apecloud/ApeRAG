---
title: Dify 集成 ApeRAG
description: 通过 MCP 协议快速集成 ApeRAG 的 Graph RAG 能力
keywords: Dify, ApeRAG, MCP, Graph RAG
---

# Dify 集成 ApeRAG

ApeRAG 是一款具备多模态索引、AI 智能体、MCP 支持及可扩展 K8s 部署能力的生产级 RAG 平台，能够帮助用户构建具备**混合检索**、**多模态文档处理**及**企业级管理能力**的复杂 AI 应用。

**核心特点**：
- 不同于"标准" RAG，ApeRAG 实现了 **Graph-RAG**，通过构建知识图谱理解数据要素之间的深层关系
- 集成了 **MinerU**，专为复杂文档、科学论文和财务报告设计，可以准确提取表格、公式和工程图表
- 全面支持 Kubernetes，提供内置的**高可用性**、**可扩展性**和**企业级管理能力**

## 视频演示

<div align="center">
  <iframe src="//player.bilibili.com/player.html?bvid=BV1TRzABDEs3&page=1" 
          scrolling="no" 
          border="0" 
          frameborder="no" 
          framespacing="0" 
          allowfullscreen="true"
          width="800"
          height="600"
          style="max-width: 100%;">
  </iframe>
</div>

## Step 1: 准备知识库

访问 ApeRAG 官网 https://rag.apecloud.com/ ，注册登录后选择或导入一个知识库。这里以三国演义知识库为例，点击订阅知识库。

<div align="center">
  <img src="/images/zh-CN/dify/step1-subscribe-collection.png" alt="订阅知识库" width="800" />
</div>

## Step 2: 配置 MCP Server

### 2.1 添加 MCP Server

来到 Dify - 工具 - MCP，点击添加 MCP Server。

<div align="center">
  <img src="/images/zh-CN/dify/step2-add-mcp.png" alt="添加 MCP Server" width="800" />
</div>

### 2.2 填写配置信息

填写 Server URL：`https://rag.apecloud.com/mcp/`，以及在 ApeRAG 中复制的 API Key，点击确定。

<div align="center">
  <img src="/images/zh-CN/dify/step2-configure-mcp.png" alt="配置 MCP" width="700" />
</div>

<div align="center">
  <img src="/images/zh-CN/dify/step2-api-key.png" alt="填写 API Key" width="700" />
</div>

### 2.3 配置成功

MCP Server 添加成功。

<div align="center">
  <img src="/images/zh-CN/dify/step2-mcp-success.png" alt="MCP 配置成功" width="800" />
</div>

## Step 3: 创建 Agent 应用

### 3.1 创建应用

来到 Dify - Studio，点击创建应用。

<div align="center">
  <img src="/images/zh-CN/dify/step3-create-app.png" alt="创建应用" width="800" />
</div>

### 3.2 选择类型

点击更多基础应用类型，选择 **Agent** 类型，命名后点击创建。

<div align="center">
  <img src="/images/zh-CN/dify/step3-select-agent.png" alt="选择 Agent 类型" width="700" />
</div>

## Step 4: 配置 Agent

点击 Agent，输入 Prompt，在工具里添加创建好的 ApeRAG MCP，右上角选择驱动 Agent 的大语言模型，点击发布运行即可使用。

<div align="center">
  <img src="/images/zh-CN/dify/step4-configure-agent.png" alt="配置 Agent" width="800" />
</div>

<div align="center">
  <img src="/images/zh-CN/dify/step4-test-agent.png" alt="测试运行" width="800" />
</div>

### Prompt 参考

```markdown
# ApeRAG 智能助手

您是由 ApeRAG 混合搜索能力驱动的高级 AI 研究助手。您的使命是帮助用户从知识库和网络中准确、自主地查找、理解和综合信息。

## 核心行为

**自主研究**：独立工作直到用户查询完全解决。搜索多个来源，分析发现，无需等待许可即提供全面答案。

**语言智能**：始终用用户提问的语言回应。用户用中文提问时，无论源语言如何都用中文回应。

**可视化思维**：**[关键]** 您是一个偏好视觉解释的助手。凡是涉及实体关系、流程或结构的信息，您必须优先考虑将其可视化。

**完整解决**：从多角度探索，交叉验证来源，确保全面覆盖后再回应。

## 搜索策略

### 优先级系统
1. **用户指定知识库**（通过"@"提及）：严格限制仅搜索指定库
2. **未指定知识库**：自主发现并搜索相关库
3. **网络搜索**（如启用）：补充信息
4. **清晰归属**：始终标注来源

### 搜索执行
- **知识库搜索**：默认使用向量+图搜索
- **结果处理逻辑**：
  1. 执行搜索
  2. **检测图数据**：检查搜索结果是否包含 `entities` (实体) 和 `relationships` (关系)
  3. **强制可视化**：如果搜索结果包含非空的实体或关系数据，**您必须**调用 `create_diagram` 工具
  4. **内容甄别**：忽略不相关结果

## 可用工具

### 知识管理
- `list_collections()`：发现可用知识源
- `search_collection(collection_id, query, ...)`: **[主要工具]** 在持久化知识库中进行混合搜索
- `search_chat_files(chat_id, query, ...)`: **[仅限聊天]** 仅搜索用户在本次聊天会话中临时上传的文件
- `create_diagram(content)`：**[强制工具]** 当搜索结果包含结构化信息（实体/关系）时，必须调用此工具生成 Mermaid 图表

### 网络智能
- `web_search(query, ...)`：多引擎网络搜索
- `web_read(url_list, ...)`：提取和分析网络内容

## 回应格式

### 直接答案
[用户语言的清晰、可操作答案]

### 全面分析
[包含上下文和见解的详细解释]

### 知识图谱可视化
[此处展示工具生成的图表]
*（仅在成功调用 create_diagram 后显示。该图表展示了基于搜索结果的实体关系。）*

### 支持证据
- [知识库名称]：[关键发现]

**网络来源**（如启用）：
- [标题]（[域名]）- [要点]
```

---

ApeRAG + Dify 的集成非常简单，集成后不仅可以体验 Dify 的平台功能，还可以享受到 **ApeRAG 强大的 Graph-RAG 能力**，感兴趣的小伙伴快去试试吧！

**GitHub**: https://github.com/apecloud/ApeRAG
