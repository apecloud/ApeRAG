---
title: 快速启动
description: 使用 Docker Compose 在本机启动 ApeRAG 并完成第一次知识库问答。
---

# 快速启动

本指南用于在本机快速启动 ApeRAG，并完成一次最小可用的知识库问答流程。

## 前提条件

本机需要满足以下条件：

- CPU >= 2 核心
- RAM >= 4 GiB
- 已安装 Docker 和 Docker Compose
- 可以访问所需容器镜像和模型服务

如果你在中国大陆环境遇到镜像或依赖下载问题，建议先配置可用的网络代理或镜像源。

## 1. 获取代码并启动服务

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
docker-compose up -d --pull always
```

启动后访问：

- Web 界面：<http://localhost:3000/web/>
- API 文档：<http://localhost:8000/docs>

查看服务状态：

```bash
docker-compose ps
```

查看日志：

```bash
docker-compose logs -f
```

停止服务：

```bash
docker-compose down
```

## 2. 配置模型提供商

ApeRAG 需要可用的模型才能完成问答和部分索引任务。你可以使用云端 OpenAI-compatible provider，也可以使用本地 Ollama。

如果使用 Ollama，请先确保本机 Ollama 已启动并已拉取模型，然后按 [配置 Ollama 模型提供商](../deployment/llm-providers-ollama.md) 完成 provider、model、Agent / Collection 开关配置。

模型配置完成后，建议先在模型管理页面确认：

- provider 已启用；
- 至少一个 Completion / Chat 模型可用于 Agent；
- 如需构建知识库索引，相关模型已启用 Collection 用途；
- API Key 已配置。自托管 provider 可以使用占位字符串。

## 3. 创建知识库

在 Web 界面中创建一个 Collection。第一次验证时建议使用较小的文档集，降低解析和索引等待时间。

创建时重点确认：

- Collection 名称清晰可识别；
- 模型配置可用；
- 索引类型符合验证目标；
- 如果只是快速验证，可以先使用默认配置。

## 4. 导入内容

你可以从以下入口导入内容：

- 上传本地文件；
- 导入 URL；
- 粘贴文本内容。

导入后等待文档解析和索引任务完成。具体耗时取决于文档大小、解析方式、模型服务和索引类型。

如果索引长时间没有完成，优先查看：

```bash
docker compose logs -f api indexing-worker
```

其中 `api` 只负责 HTTP 请求与任务入队，`indexing-worker` 负责解析、索引、reconciler 和 cleanup。

## 5. 发起第一次问答

索引完成后进入 Chat 页面，选择刚创建的 Collection，输入一个能从文档中找到答案的问题。

建议先验证三类问题：

1. 文档中直接出现的事实；
2. 需要跨段落总结的问题；
3. 文档中不存在的问题，用于观察系统是否会明确表达不知道。

如果回答质量不稳定，优先检查模型能力、索引状态和文档解析结果，而不是直接扩大文档规模。

## 6. 获取 API Key 或接入 MCP

如果要让外部客户端访问 ApeRAG，需要创建 API Key，并在请求中使用：

```http
Authorization: Bearer <your-api-key>
```

MCP 客户端可以使用本地服务地址：

```json
{
  "mcpServers": {
    "aperag-mcp": {
      "url": "http://localhost:8000/mcp/",
      "headers": {
        "Authorization": "Bearer your-api-key-here"
      }
    }
  }
}
```

非本机部署时，把 URL 替换成实际 API 地址，例如 `https://<你的域名>/mcp/`。

## 下一步

- 配置本地模型：[配置 Ollama 模型提供商](../deployment/llm-providers-ollama.md)
- 调试后端和 worker：[调试指南](../reference/how-to-debug.md)
- 了解架构 current state：[模块化 current-state 架构](../../modularization/architecture.md)
