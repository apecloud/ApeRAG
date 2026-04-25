---
title: 构建 Docker 镜像
description: 构建 ApeRAG 后端与前端容器镜像、本地调试与多平台发布的完整指南
position: 1
---

# 构建 Docker 镜像

本文覆盖从本地单平台构建到多平台发布的完整流程。适合两类读者：

- 想把改动后的代码打进自己镜像，在 Docker Compose 或 Kubernetes 上自验的开发者
- 准备把 ApeRAG 集成到自己 CI 流水线里，做内部发布的运维/平台团队

如果你只想跑起 ApeRAG 而不改代码，直接用官方镜像 + 根目录的 [`docker-compose.yml`](https://github.com/apecloud/ApeRAG/blob/main/docker-compose.yml)（见 [快速开始](../getting-started/quickstart.md)），不需要本文。

## 镜像构成

ApeRAG 由两个独立镜像组成，`docker-compose.yml` 里分别对应 `api` / `celeryworker` / `celerybeat` / `flower`（共用后端镜像）和 `frontend`：

| 镜像 | 用途 | Dockerfile | 基础镜像 |
|------|------|-----------|---------|
| `apecloud/aperag` | FastAPI 主进程 + Celery worker/beat/flower | [`Dockerfile`](https://github.com/apecloud/ApeRAG/blob/main/Dockerfile) | `python:3.11.13-slim`（多阶段 + uv） |
| `apecloud/aperag-frontend` | Next.js standalone 构建产物 + PM2 runtime | [`web/Dockerfile`](https://github.com/apecloud/ApeRAG/blob/main/web/Dockerfile) | `node:20.18.0-alpine` |

默认镜像仓库是 `apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com`（阿里云张家口）；`docker-compose.yml` 在本机使用时会回退到 Docker Hub（`${REGISTRY:-docker.io}`）。

## 前置条件

- Docker 20.10 或更新（需要 `docker buildx` 子命令）
- 前端构建需要 **Node.js 20+** 和 **Yarn classic** — 这是为了在宿主机先生成 `web/build/` 输出后再打包进镜像
- 多平台构建需要可用的 QEMU（macOS Docker Desktop 默认自带；Linux 需先 `docker run --privileged --rm tonistiigi/binfmt --install all`）
- 可选：`make`（所有命令都有对应 Makefile target，推荐用 Makefile）

本地构建不需要在宿主机安装 Python/uv — 这些都在镜像构建阶段的容器里完成。

## 本地单平台构建

开发自验推荐走本地构建，只构建当前机器架构、不推送仓库。镜像直接 `--load` 到本机 Docker daemon。

```bash
# 同时构建后端 + 前端
make build-local

# 只构建后端
make build-aperag-local

# 只构建前端
make build-aperag-frontend-local
```

默认版本标签是 `nightly`（见 Makefile 顶部 `VERSION ?= nightly`），想覆盖传 `VERSION` 即可：

```bash
make build-local VERSION=v1.0.0-dev
```

构建完成后本地 Docker 里就会出现 `apecloud/aperag:v1.0.0-dev` 和 `apecloud/aperag-frontend:v1.0.0-dev`，可以直接用在 `docker-compose.yml` 里：

```bash
REGISTRY='' VERSION=v1.0.0-dev docker compose up -d
```

（`REGISTRY=''` 让 compose 用无前缀的镜像名，匹配本机打好的 tag。）

## 多平台构建

发布到仓库时要同时产出 `linux/amd64` 和 `linux/arm64`。多平台构建走 `docker buildx`，并且**必须携带 `--push`**（`docker buildx build` 不支持把多平台 manifest `--load` 到本机）。

Makefile 已经包装好了：

```bash
# 构建并推送后端 + 前端到 ${REGISTRY}
make build VERSION=v1.2.3 REGISTRY=docker.io

# 只发后端
make build-aperag VERSION=v1.2.3 REGISTRY=docker.io

# 只发前端
make build-aperag-frontend VERSION=v1.2.3 REGISTRY=docker.io
```

默认目标架构是 `linux/amd64,linux/arm64`，可通过 `BUILDX_PLATFORM` 覆盖：

```bash
make build VERSION=v1.2.3 BUILDX_PLATFORM=linux/amd64
```

首次运行时会自动创建名为 `multi-platform` 的 buildx builder（`make setup-builder`）；排查 builder 本身的问题可以 `make clean-builder` 后重跑。

## 版本与镜像标签

构建时参与的 Make 变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VERSION` | `nightly` | 镜像 tag；发版时建议传 Git tag（如 `v1.2.3`） |
| `REGISTRY` | `apecloud-registry.cn-zhangjiakou.cr.aliyuncs.com` | 推送目标仓库 |
| `APERAG_IMAGE` | `apecloud/aperag` | 后端镜像仓库名 |
| `APERAG_FRONTEND_IMG` | `apecloud/aperag-frontend` | 前端镜像仓库名 |
| `BUILDX_PLATFORM` | `linux/amd64,linux/arm64` | 多平台目标 |
| `BUILDX_ARGS` | `--sbom=false --provenance=false` | 额外 buildx 参数 |

镜像内部会烧入一份 `aperag/version/__init__.py`（由 `release-version` 目标在每次 build 前重新生成），包含 `VERSION` 字符串和当前 `HEAD` 的 7 位短 commit hash，便于线上反查当前运行的代码版本。

## 后端镜像细节

[`Dockerfile`](https://github.com/apecloud/ApeRAG/blob/main/Dockerfile) 采用 2-stage 构建：

1. **Builder 阶段**：基于 `python:3.11.13-slim`，安装 `uv` 并 `uv sync --active` 把所有依赖安装进 `/opt/venv`。
2. **Final 阶段**：全新的 `python:3.11.13-slim`，只从 builder 拷贝 `/opt/venv`，再 `COPY . /app` 并 `pip install --no-deps -e .`，把项目代码以可编辑模式挂上。

这种结构的好处：

- Runtime 镜像里不保留 `build-essential` / `git` / `uv` 等构建工具，面向最终用户的镜像保持较小体积
- 依赖层 (`uv sync`) 与代码层 (`COPY . /app`) 分离，只改业务代码不会使依赖缓存失效

Entrypoint 是 [`scripts/entrypoint.sh`](https://github.com/apecloud/ApeRAG/blob/main/scripts/entrypoint.sh)，会先等 PostgreSQL 起来、确保 `pgvector` 扩展存在，再 `exec` 真正的启动命令（`scripts/start-api.sh` / `scripts/start-celery-worker.sh` 等）。

## 前端镜像细节

[`web/Dockerfile`](https://github.com/apecloud/ApeRAG/blob/main/web/Dockerfile) 是一个 runtime-only 的镜像：它假设 `web/build/` 已经在宿主机上构建好，直接把它拷进容器，再装 PM2、用 `pm2-runtime start server.js` 启动。

所以完整的前端构建流程是两步：

1. `make build-aperag-frontend-assets` 在宿主机上跑 `yarn install && yarn build`，产出 `web/build/`（Next.js standalone 输出）
2. `docker buildx build` 打包成镜像

`make build-aperag-frontend[-local]` 已经把这两步串起来了，手工调试时也可以单独跑 step 1 观察前端构建日志。

镜像暴露端口 3000，环境变量 `PORT=3000`、`HOSTNAME=0.0.0.0`；Docker Compose 里前端通过 `API_SERVER_ENDPOINT=http://api:8000` 反向代理到后端。

## Docker Compose

打完镜像后，根目录的 [`docker-compose.yml`](https://github.com/apecloud/ApeRAG/blob/main/docker-compose.yml) 是最常用的 orchestration 入口。除了 `api` / `celeryworker` / `celerybeat` / `flower` / `frontend`，还包含所有必需的基础设施服务：`postgres`（含 pgvector）、`redis`、`qdrant`、`es`（Elasticsearch 带 IK 分词器）。

两个可选图数据库服务通过 Docker Compose profile 控制，默认不启动：

- `--profile neo4j`：启动 Neo4j 5.26 enterprise 作为 Graph 后端（可选，默认用 PostgreSQL 存储图数据）
- `--profile nebula`：启动 Nebula Graph 3.8 作为 Graph 后端（与 `neo4j` profile 互斥）

常用命令：

```bash
# 只启动默认服务（不带 profile）
docker compose up -d

# 带 Neo4j 图库
docker compose --profile neo4j up -d

# 关掉并清卷（危险：会删除 PostgreSQL / Qdrant / ES 数据）
docker compose down -v
```

## Kubernetes 部署入口

Helm chart 位于 [`deploy/aperag/`](https://github.com/apecloud/ApeRAG/tree/main/deploy/aperag)，`values.yaml` 里的 image 相关字段直接对应本文档的构建产物：

- `image.repository`（默认 `docker.io/apecloud/aperag`）
- `image.tag`（默认 `v0.0.0-nightly`）
- 前端字段同理（`frontend.image.repository` / `frontend.image.tag`）

如果要在本地 KinD / minikube 里用刚 build 好的本地镜像，可以跳过推送直接 side-load：

```bash
# minikube
make load-images-to-minikube VERSION=v1.0.0-dev

# KinD
make load-images-to-kind VERSION=v1.0.0-dev KIND_CLUSTER_NAME=aperag
```

## CI 发布流程

仓库里用 [`.github/workflows/release-image.yml`](https://github.com/apecloud/ApeRAG/blob/main/.github/workflows/release-image.yml) 自动发布镜像。两种触发方式：

- **GitHub Release published**：自动用 release tag 作为镜像 tag，同时发布后端与前端两个镜像，成功后触发 E2E workflow
- **Workflow dispatch**：手动触发，支持只发某一个镜像（`image_name` 下拉），tag 来自 `image_tag` 输入

工作流本身不直接写 `docker buildx`，而是复用 `apecloud/apecloud-cd/.github/workflows/release-image-cache-sync.yml` 做多平台构建 + 仓库缓存同步；具体镜像名称、Dockerfile 路径、pre-build 钩子（`MAKE_OPS_PRE: build-aperag-frontend-assets`）都通过 inputs 传入。

## 常见问题

**Q：只改了后端代码，前端不想重新 build？**
只跑 `make build-aperag-local`。前端镜像可以继续用上次的；`docker-compose.yml` 里两个镜像的 tag 是各自独立的环境变量控制。

**Q：`yarn build` 报错 `ENOMEM`？**
前端 standalone build 会占用比较多内存（~2GB）。在 CI 或小内存容器里构建时，通过 `NODE_OPTIONS=--max-old-space-size=4096 yarn build` 提高 heap 限制。

**Q：构建多平台镜像时报 `no match for platform in manifest`？**
一般是基础镜像在该 registry mirror 下没有 arm64 manifest。尝试 `docker pull` 基础镜像时带 `--platform=linux/arm64` 确认可用；不行就切换回 Docker Hub。

**Q：可不可以只发后端不发前端？**
可以。`make build-aperag VERSION=... REGISTRY=...` 只发后端；前端可以继续用官方镜像或者上次发的版本。`docker-compose.yml` 里两个 tag 独立，拆版发布不会冲突。

**Q：`Dockerfile` 里为什么要 `pip install --no-deps -e .`？**
`uv sync` 已经把所有依赖装进 `/opt/venv`，这一步只负责把 `aperag/` 包以 editable 模式挂到 venv 里（主要是为了让 `importlib.metadata` 能正确返回包信息），不再安装依赖。

## 相关链接

- [`docker-compose.yml`](https://github.com/apecloud/ApeRAG/blob/main/docker-compose.yml) — 单机编排入口
- [`deploy/aperag/`](https://github.com/apecloud/ApeRAG/tree/main/deploy/aperag) — Helm chart
- [快速开始](../getting-started/quickstart.md) — 仅使用官方镜像的最短路径
- [如何调试](../reference/how-to-debug.md) — 启动失败 / 日志排查
