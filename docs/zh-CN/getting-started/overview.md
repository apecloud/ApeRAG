---
title: ApeRAG 概览
description: 面向首次接触 ApeRAG 用户的产品定位、核心流程和文档入口。
---

# ApeRAG 概览

ApeRAG 是一个面向私有化部署的 RAG 与智能体平台。它把知识库、混合检索、图谱检索、文档解析、模型管理、对话智能体、评估和外部集成放在同一套系统中，目标是让团队可以在自己的环境里构建可维护的知识问答与智能体应用。

这份文档回答三个问题：ApeRAG 解决什么问题、第一次使用时应该按什么路径走、后续应该读哪些文档。

## 适合什么场景

ApeRAG 适合以下场景：

- **企业知识库问答**：上传文档、网页或文本内容后，用自然语言检索和问答。
- **私有数据上的智能体**：让智能体在企业知识库、Web 搜索和模型能力之间编排任务。
- **混合检索实验与落地**：同时使用向量、全文、图谱、摘要和视觉索引，提高不同类型问题的召回质量。
- **模型与权限集中管理**：在一个系统里配置 LLM Provider、API Key、Quota、审计日志和 Marketplace 分享流程。
- **外部工具接入**：通过 MCP、Dify 或 OpenAI-compatible API 接入已有工作流。

如果你只是想快速验证系统，可以先走 Docker Compose 快速开始；如果要生产部署，应继续阅读部署与运维相关文档。

## 核心概念

| 概念 | 作用 |
| --- | --- |
| Collection / 知识库 | 文档、索引和检索配置的主要容器。 |
| Document / 文档 | 上传、导入或解析后的知识来源。 |
| Index / 索引 | 支持向量、全文、图谱、摘要、视觉等检索能力。 |
| Chat / 对话 | 面向用户的问答入口，可结合知识库和智能体能力。 |
| Agent Runtime | 执行智能体 Turn、工具调用、Prompt 解析和 SSE 时间线。 |
| Model Provider | OpenAI-compatible 或其他模型服务的接入配置。 |
| Marketplace | 将知识库发布、订阅并以只读方式共享给其他用户。 |

## 推荐上手路径

1. **启动系统**：按 [快速启动](./quickstart.md) 在本机启动 ApeRAG。
2. **配置模型**：如果使用本地 Ollama，按 [配置 Ollama 模型提供商](../deployment/llm-providers-ollama.md) 添加 provider 和模型。
3. **创建知识库**：在 Web 界面创建 Collection，并选择适合的模型和索引配置。
4. **导入内容**：上传文件，或通过 URL / 文本导入内容。
5. **等待索引完成**：确认文档解析和索引任务完成后再开始问答。
6. **开始对话**：在 Chat 中选择知识库，验证检索、回答和引用效果。
7. **按需接入外部工具**：使用 MCP、Dify 或 OpenAI-compatible 接口接入其他客户端。

## 文档导航

| 你要做什么 | 推荐阅读 |
| --- | --- |
| 第一次本地启动 | [快速启动](./quickstart.md) |
| 配置本地 Ollama 模型 | [配置 Ollama 模型提供商](../deployment/llm-providers-ollama.md) |
| 构建自定义镜像 | [构建 Docker 镜像](../deployment/build-docker-image.md) |
| 调试后端和异步任务 | [调试指南](../reference/how-to-debug.md) |
| 了解当前后端模块结构 | [模块化 current-state 架构](../../modularization/architecture.md) |
| 查看中文文档重写蓝图 | [中文文档重写计划](../rewrite-plan.md) |

## 当前文档口径

这轮中文文档只描述 current state：今天 main 分支上真实存在的产品能力、模块结构和操作路径。历史迁移过程、旧设计方案和 TODO 文档不再作为主线文档维护。

如果你需要理解模块化重构后的后端结构，以 `docs/modularization/architecture.md` 为准；中文文档只在必要处引用它，不重复定义 G1-G19 gate、Protocol + DI seam 或 domain canonical path。
