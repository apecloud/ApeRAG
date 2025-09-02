---
title: 构建指南
description:
keywords:
---

# 构建指南

本节介绍如何构建 ApeRAG 容器镜像。这主要适用于需要创建自己的构建或部署到"快速开始"中未涵盖的环境的用户。

## 构建容器镜像

项目使用 Docker 和 `make` 命令来构建容器镜像。

- **本地平台构建**：这些命令为您当前机器的架构构建镜像。

  ```bash
  # 为本地平台构建所有必要的镜像
  make build-local

  # 仅为本地平台构建后端镜像
  make build-aperag-local

  # 仅为本地平台构建前端镜像
  make build-aperag-frontend-local
  ```

- **多平台构建**：这些命令为多种架构（例如 amd64、arm64）构建镜像。这需要设置和配置 Docker Buildx。

  ```bash
  # 为多个平台构建所有必要的镜像
  make build

  # 仅为多个平台构建后端镜像
  make build-aperag

  # 仅为多个平台构建前端镜像
  make build-aperag-frontend
  ```

  您可以使用 `PLATFORMS` 变量指定目标平台，例如：

  ```bash
  make build PLATFORMS=linux/amd64,linux/arm64
  ```

## 部署

有关常见的部署方法，请参考主 README 中的"快速开始"部分：

- [Kubernetes 快速开始](../README-zh.md#kubernetes-部署推荐生产环境)
- [Docker Compose 快速开始](../README-zh.md#快速开始)

对于自定义部署，您需要调整这些方法或使用构建的容器镜像与您选择的编排平台配合使用。确保所有必需的服务（数据库、后端、前端、Celery worker）都正确配置并能够相互通信。
