---
title: ApeRAG 企业版仓库迁移方案
description: 企业版主仓 + 开源版子集 + 双向同步链路 - 完整详细中文方案
---

# ApeRAG 企业版仓库迁移方案

> earayu2 directive (`#企业版` msg=c2c8d8f2 + msg=c0b009e2): 团队主开发要迁到企业版仓库, 但开源版要继续维护, 而且企业版必须是开源版的超集。10 lane 团队讨论收口后由总架构师起草完整中文方案。

## 0. 这件事在做什么

简单说: 我们要把日常开发主仓库从现在的开源版 (`apecloud/ApeRAG`) 迁到一个新的企业版仓库 (`apecloud/aperag-enterprise`)。但我们不能丢掉开源版 — 开源版必须继续公开、继续接受社区贡献、必须是企业版功能的一个干净子集。

简单的事情可能不简单的地方:
- **企业版必须是开源版的超集**: 企业版有的开源版可能没有, 但开源版有的企业版必须有
- **企业版功能可以双向流动**: 企业版做的某些通用改进, 应该能流回开源版给社区用
- **以后日常开发主战场在企业版**: 开发者每天 push 的是企业版仓库, 不是开源版

整个事情的核心矛盾: 怎么在「开发主仓只有一个」和「开源版仍然是活的公开仓」之间找平衡, 不让两边互相打架, 也不让企业代码意外泄漏到开源仓。

## 1. 为什么现在做

- earayu2 想让后续大量功能 (license / SSO / 企业 dashboard / 私有部署 connector / 客户定制等) 在企业版迭代, 不污染开源版的 scope
- 现在还没开始迁, 越往后拖越难: 现在 `apecloud/ApeRAG` 还相对干净, 没有大量企业代码混在里面; 等企业代码开始铺开后再分就要做大手术
- 团队 10 lane 充分讨论 (Weston / dongdong / 符炫炜 / ziang / huangzhangshu / 不穷 / huangheng / 明书 / Bryce / Planetegg) 已经收口, 现在欠的就是 earayu2 拍板 + 启动时机

## 2. 整体方案 (一句话)

**企业版仓库 (`apecloud/aperag-enterprise`) 作为日常开发主战场, 但保留 `apecloud/ApeRAG` 开源版的完整历史和社区 PR 流程, 通过 PR 标签 + 代码目录硬隔离 + 双向同步机制, 让两边自然协同, 不互相覆盖, 不泄漏企业代码到开源仓。**

具体落实成 4 件事:

1. **代码怎么组织**: 现有开源代码目录不动, 企业代码新增到独立目录
2. **PR 怎么写**: 每个 PR 必须打标签, 说清楚改动属于「开源可回流」还是「企业独占」还是「混合需要拆」
3. **企业改动怎么流到开源**: 标了「开源可回流」的 PR, 自动 cherry-pick 到开源仓开正常 PR, 走开源仓的 review + merge
4. **开源改动怎么流到企业**: 每周一次自动把开源主分支 merge 进企业主分支, 保证企业版包含所有开源最新修复

## 3. 代码具体怎么组织

### 3.1 仓库关系

```
apecloud/aperag-enterprise (新建, 私有, 团队日常开发主战场)
  ↑
  日常 push, 主分支永远 = 开源全部 + 企业功能

apecloud/ApeRAG (维持现有, 公开, 社区入口)
  ↑
  保留完整历史 + 社区 PR + issue + fork
  日常用 cherry-pick 接收企业仓里「开源可回流」的改动
```

### 3.2 目录结构 (非常重要 — 硬约束)

企业版仓库内部:

```
aperag-enterprise/
├── aperag/                    ← 开源后端代码 (现有, 不动)
├── web/                       ← 开源前端代码 (现有, 不动)
├── deploy/aperag/             ← 开源部署文件 (现有, 不动)
├── docs/                      ← 开源文档 (现有, 不动)
├── tests/                     ← 开源测试 (现有, 不动)
│
├── enterprise/                ← 企业后端代码 (新加, 商业许可)
│   ├── license/               ← 授权管理
│   ├── connectors/            ← 客户定制连接器
│   ├── modules/               ← 企业功能模块
│   └── migrations/            ← (可选) 企业表 migration
├── web-enterprise/            ← 企业前端代码 (新加)
├── deploy/enterprise/         ← 企业部署 overlay (新加)
├── docs/enterprise/           ← 企业文档 (新加)
└── tests/enterprise/          ← 企业测试 (新加)

└── docs/zh-CN/architecture/enterprise-sync-policy.md  ← 同步策略文档
```

**核心约束 (用 CI 强制, 不靠人记)**:

1. **方向单向**: 开源代码不能引用企业代码, 但企业代码可以引用开源代码
   - 比如 `aperag/llm/embedding.py` 不能 `from enterprise.license import ...`
   - 反过来 `enterprise/connectors/customer_x.py` 可以 `from aperag.llm import EmbeddingService` (这是合法的, 企业功能基于开源核心扩展)
   - CI 用静态分析检查每个 import 语句, 违反就 build 红, 不允许合 PR

2. **License header 标记**: 每个企业目录里的文件首行必须有商业许可注释, 比如:
   ```python
   # License: Commercial - apecloud (NOT Apache 2.0)
   ```
   开源目录里的文件保持原有 Apache 2.0 header。CI 检查文件路径和 header 一致性, 不一致就 build 红。

3. **数据库表分层** (per Weston msg=fedaba61 BLOCKER 修正 — 防止 EE-only migration 隐式混进 OSS 主链):
   - **开源 alembic chain 只放开源 schema 变更**, EE-only 不进
   - **企业独占表默认放 `enterprise/migrations/`**, 用独立版本表或独立迁移工具 (跟 OSS alembic 完全分开)
   - **部署顺序**: 先跑开源 alembic migration → 再跑企业 migration (两个独立链, 各自线性)
   - **如果某个 schema 改动两边都需要**: 必须作为 `oss-safe` PR 进开源主链, 再同步回企业仓 — 不允许把 EE-only revision 混进 OSS 主链, 也不允许 OSS revision `down_revision` 接到 EE revision
   - 企业新加的表用 `enterprise_` 前缀 (比如 `enterprise_license_keys`) 防表名冲突
   - 「不开 alembic 多 head」准确表述: 开源链和企业链**各自线性**, 不在同一个 alembic head 里交叉 (不是说两个仓共用一个 head, 而是各自独立 head 各自线性)

4. **配置文件分层**: 企业版独占的环境变量 / Helm values / connector 配置, 都放企业目录, 开源 deploy 路径不带任何企业字段。

### 3.3 build 双套

CI 跑两套 build:

- `EDITION=oss` build: 把 `enterprise/` `web-enterprise/` `deploy/enterprise/` `docs/enterprise/` `tests/enterprise/` 临时挪走, 跑完整开源 lint + test + build, **必须能通过**
- `EDITION=enterprise` build: 全代码跑企业 CI, 含开源 + 企业测试

第一种 build 的存在保证了「开源版独立可运行」这个硬契约。任何 PR 让开源 build 跑不过, 不允许合。

## 4. PR 工作流

### 4.1 PR 必打标签

企业仓里每个 PR 创建时, 模板强制选一个标签:

| 标签 | 含义 | 改动范围 | 同步行为 |
|------|------|---------|---------|
| `oss-safe` | 开源可回流 | 只动开源目录 (`aperag/` `web/` `deploy/aperag/` `docs/` `tests/`) | 合并后 bot 自动到开源仓开 cherry-pick PR |
| `enterprise-only` | 企业独占 | 只动企业目录 (`enterprise/` `web-enterprise/` `deploy/enterprise/` etc.) | 不流到开源仓 |
| `mixed` | 混合 | 同时动开源和企业 | CI 拒绝, 必须拆成两个 PR (一个 oss-safe + 一个 enterprise-only) |

**为什么强制拆 mixed**:

如果允许同一个 PR 既改开源又改企业, 后期回流到开源时只能拿这个 PR 的部分内容, 拆 commit 很难做干净。如果开发时就拆开, 流程自然 — 哪些进开源哪些不进, 一眼看清楚。

### 4.2 自动同步流程 (减少人工纪律消耗)

每个 enterprise PR 在 CI 阶段自动跑一个「同步预演」:

- 如果 PR 标 `oss-safe`: CI 模拟把改动应用到开源仓, 显示「这个 PR 会让开源仓增加哪些改动」, reviewer 一眼看清楚, 没问题就 merge
- 如果 PR 标 `enterprise-only`: CI 验证「这个 PR 应用到开源仓后开源仓 0 改动」, 防止开发者误把企业代码写进开源路径

PR merge 之后:

- `oss-safe` 标签的 PR: bot 自动到开源仓 (`apecloud/ApeRAG`) 开一个对应的 cherry-pick PR (草稿状态), 作者点 publish 就发布给开源仓 reviewer 看, 走正常开源 PR 流程
- `enterprise-only` 标签的 PR: 留在企业仓不动

### 4.3 开源贡献流回企业

社区贡献者还是在 `apecloud/ApeRAG` 提 PR (开源仓继续是社区入口), maintainer review + merge 到开源主分支。

之后每周一次 bot 自动跑一个任务:

```bash
git fetch upstream main   # upstream = apecloud/ApeRAG
git merge upstream/main   # 合并到企业仓主分支
# 跑企业全量 CI, 通过则推送企业仓 main
```

这样开源仓的所有改动 (社区 PR + cherry-pick 回来的企业改动) 都自动同步到企业仓, 不会丢。

### 4.4 防泄漏 (4 道闸, 不是 1 道)

不靠任何单一机制, 4 道闸叠加:

1. **路径闸**: CI 静态分析每个 PR, 检查开源路径不能 import 企业路径 — 违反 build 红
2. **License header 闸**: CI 检查企业文件首行必须有商业许可 header, 开源文件不能有 — 违反 build 红
3. **关键词扫描闸**: CI 跑 grep 扫描 PR 改动, 如果开源路径里出现客户名 / 私有 endpoint / API key / 商业许可关键词等 — 警告 + 可能 block (whitelist 化, 不依赖 commit message)
4. **同步预演闸**: 每个 PR 跑 dry-run sync, 显示对开源仓的实际影响给 reviewer 看, reviewer 是最后一道人审

任何一道闸 fail 都不允许 merge。

## 5. 启动时机和时间表

### 5.1 启动前提 (双 gate)

启动企业版迁移之前必须满足两个条件:

1. **earayu2 确认渐进式方向** (本文档 ratify)
2. **当前 in-flight 工作收口** (per Weston msg=fedaba61 NIT + PM msg=9dd572f0 修正 — POC 是独立 lane 不需等 X1-X5 全完成):
   - PR #1951 (可观测性 P0 spec) ratify + merge
   - PR #1952 v2 (可观测性需求对齐文档) 重写 + ratify + merge
   - task #89 telemetry P0 子任务 X1-X5 **dispatch** (派单完成即可启动企业版 POC, 不需等子任务全 merge)
   - 不阻塞: task #11 GC orphan vector follow-up + huangheng sediment fold-in queue (这些 backlog lane 不阻 POC 启动)

为什么要等 in-flight 收口: 这些 epic 都在 `apecloud/ApeRAG` 主仓上正常推进, 如果同时切主仓写权 + 启动迁移, 会让所有 in-flight PR 不知道往哪个仓提, 造成混乱。huangheng 在讨论里专门提了这个时序约束 (msg=1320ff5a), 团队一致同意。

### 5.2 时间表 (启动后约 1-1.5 周)

| 步骤 | 内容 | 工期 | 主导人 |
|------|------|------|------|
| Step 0 | freeze in-flight, 等所有正在跑的 PR / task 收口 | 1-3 天 | PM @不穷 |
| Step 1 | 建仓: `git push --mirror` 把 `apecloud/ApeRAG` 完整历史推到 `apecloud/aperag-enterprise` + 配 remote | ~1 天 | @Planetegg (SRE) |
| Step 2 | 加企业目录 + import 闸 + License header 闸 + 关键词闸 + 同步预演闸 + PR 模板 + 同步策略文档 | ~3 天 | @符炫炜 (代码边界 + 策略) + @明书 (CI gate 工具) |
| Step 3 | 双向同步 bot + 团队培训 PR 标签纪律 + 一次完整 round-trip 验证 | ~3 天 | @Planetegg (CI) + @符炫炜 (培训) |

最后一次 round-trip 验证: 把企业仓主分支跑一次抽取 → 跟当前 `apecloud/ApeRAG` 主分支对比, **必须 0 diff**。如果有 diff 说明哪里出问题, 修完再来一次, 直到 0 diff 才上。

### 5.3 启动后

- 团队日常开发全切到 `apecloud/aperag-enterprise`
- `apecloud/ApeRAG` 仍正常接受社区 PR (不锁写, 不 force-push)
- 周期性 bot 双向 sync 跑起来
- 1-2 个月后看运行数据, 决定是否需要进一步升级 (见下面 § 8)

## 6. 角色分工

| 角色 | 谁 | 工作 |
|------|------|------|
| 仓库搭建 + CI workflow + 同步 bot | @Planetegg (SRE) | step 1 + step 3 主导 |
| 代码边界设计 + `enterprise-sync-policy.md` 起草 + PR 标签纪律 spec | @符炫炜 (架构) | step 2 主导 |
| CI gate 工具 (import 闸 / 同步预演闸 / round-trip 测试) | @明书 (资深程序员) | step 2 协作, 起草 `tools/sync_oss.py` + `.ossignore` + GH Action |
| 架构 cross-CR boundary 一致性 | @Weston | step 2 完成后审 + step 3 round-trip 验证 |
| cr-checklist sediment 入仓 | @huangheng | step 完成后 fold 进 cr-checklist (Lesson 应用 demo) |
| 测试边界 + 防泄漏 lint | @huangzhangshu | step 2 + step 3 协助 |
| 前端开源边界 verify | @dongdong | step 2 web 目录边界审 |
| index/worker domain verify | @ziang | step 2 backend 目录边界审 |

PM @不穷 推进节奏。

## 7. 风险 + 应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 现有 in-flight PR 一直没 drain | 启动时机延后 | freeze + 加速现有 epic 收口, 不并发 |
| 团队适应 PR 标签纪律慢 | 早期可能有 mixed PR 提交 | CI 强制拒, 自然倒逼; 第一周做培训 |
| 法务边界没拍板就启动 | 后期反复改目录 | step 0 之前 earayu2 + 法务先拍板「哪些功能开源 / 哪些企业独占」, 边界一次定 |
| 开源用户问「为什么没有 X 功能」 | 社区体验 | docs/enterprise/ 写清楚开源版 / 企业版差异表; 开源 README 不出现企业字眼 |
| 同步 bot 出 bug 让企业代码漏到开源仓 | 严重泄漏事故 | 4 道闸叠加 + round-trip 测试 + dry-run sync 预演给 reviewer 看, 单点失误不会直接漏 |
| 企业代码越来越多, 开源版 scope 萎缩 | 开源版失去吸引力, 影响品牌 | 每月 review 一次开源版 / 企业版功能比例, 关键能力守住开源 (检索 / 索引 / 图谱 / agent 框架等核心) |

## 8. 之后什么时候重评

启动后 1-3 个月数据驱动重评。触发条件 (任何一个就启动重评讨论):

- **企业代码总行数 > 开源代码总行数的 30%** → 评估是否要做更深的代码分层 (比如 packages 化)
- **手工 cherry-pick 回流的 PR 月 > 30 个** → 评估是否要把 cherry-pick 流程自动化加强
- **同步预演闸 false-positive 月 > 5 次** → 加强 import 闸的静态分析精度
- **社区抱怨「开源版功能缩水」反馈频率 > 1 次/周** → 重新评估开源 / 企业边界

到时候团队再开会讨论, 不预先承诺升级时机和方案。

## 9. 不做什么 (避免折腾)

- **不做** 一次性把现有 OSS 代码重组到 `packages/community-core/` 这种新目录 — 风险太大, 牵动太多 (Weston msg=846de8f2 push back 了我 msg=f5d2cd8c 的 `packages/community-core` 重组方案; 我 msg=9cabf57c 撤回)
- **不做** 用 `git filter-repo --subdirectory-filter` 自动重写开源仓主分支历史 — 会破坏现有社区 fork / PR 评论里的 commit hash 引用, 信誉成本高 (Weston msg=3b42599f)
- **不做** 把开源仓锁成 bot-only 写权限 — 切断社区 PR 入口 (Weston msg=30f96387)
- **不做** 单仓多分支 / 单仓 build flag 编译切换 — 开源用户 git clone 能看到企业代码, 知识产权 / 法律风险大
- **不做** 把 commit message 黑名单作为防泄漏主要防线 — 测试 / fixture / docs / 配置 / 注释都可能泄漏, 黑名单 cover 不全 (Bryce msg=d6120b07 自己撤回了这个方案)
- **不做** 给所有现有外部贡献者发邮件强制迁移 — 让社区在公开仓继续提 PR 是更自然的协作方式
- **不做** 短期内大改 alembic migration chain 拆多 head — 双仓维护下多 head 很脆弱, 必要时 EE 表用前缀 + 共享单链 (huangheng msg=1320ff5a)

## 10. 法务边界 (启动前必须拍板)

技术方案不能解决法务边界问题。启动前 earayu2 + 法务必须先拍板:

**开源必有** (`apecloud/ApeRAG` 永远包含):
- 现有 ApeRAG 全部基础功能 (向量索引 / 全文索引 / 图谱索引 / 检索 / 重排 / agent 框架 / admin 基础)
- 文档解析 / chunk / embedding 基础流程
- 现有部署 (Docker / Helm / 基础 SRE)

**企业独占** (候选, 待 earayu2 + 法务确认):
- License 管理 / Edition gate
- SSO / 高级权限管理
- **高级审计 / 合规报表 / 审计日志导出** (现有 `aperag/domains/governance/AuditLog` 是开源版基础能力, 企业版只增强: 高级审计 dashboard / 合规报表 / 长期归档 / 审计导出 — per dongdong msg=4d773716 cite 修正, 防误读现有 OSS audit log 被移走)
- 客户定制 connectors (specific customer integrations)
- 私有部署高级配置 (HA / 多 region / 混合云)
- 企业 dashboard (cost / token / SLA UI)
- 高级 LLM 账本和 cost 归因 UI (per task #89 P2 LLM ledger 上层 UI 部分)

边界一次定下来, 后期不要反复改。如果未来要把某个企业功能开源, 走单独流程: 该功能从 `enterprise/` 移到 `aperag/` + 改 License header + 在企业仓里走 `oss-safe` PR + cherry-pick 回开源。

## 11. 关联文档

- earayu2 directives: `#企业版` msg=c2c8d8f2 + msg=c0b009e2
- 团队讨论 trail: `#企业版` 10 lane converge (Weston / dongdong / 符炫炜 / ziang / huangzhangshu / 不穷 / huangheng / 明书 / Bryce / Planetegg)
- 同步策略详细 spec: `enterprise-sync-policy.md` (启动后 step 2 起草)
- 当前 in-flight 收口 list: PR #1951 / PR #1952 / task #89 X1-X5 / task #11 / sediment fold-in queue
- 现有 license 模式: `apecloud/ApeRAG` Apache 2.0

---

**起草**: @符炫炜 (总架构师)
**日期**: 2026-04-30
**版本**: v1 (10 lane 团队讨论收口 + earayu2 directive 后起草; 待 earayu2 ratify + 法务边界拍板后启动)
**讨论模式**: 跟 task #31 Phase A 自然中文实施方案 (PR #1934) 同 directive — 用自然中文回答业务问题, 不堆内部术语
