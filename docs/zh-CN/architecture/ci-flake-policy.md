# CI Flake 处理策略 — Provider Matrix 单 shape 失败放行规则

**生效日期**：2026-04-30（task #33 Layer 2 P0 codify）

**适用范围**：所有 PR 的 `.github/workflows/e2e-http-{lite,qdrant-nebula,qdrant-neo4j}.yml` 三 variant CI。

**编辑历史**：
- 2026-04-30 chenyexuan 初版 — fold-in Planetegg msg=46a0c5de gate wording + huangheng / Weston / ziang 多 PR cross-check forensics 实证。

---

## 1. 现状（数据驱动）

最近 20 次 PR-trigger workflow run 实证（gh run list 拉数据）：

| Workflow | pass | fail | cancel | fail rate |
|---|---|---|---|---|
| `e2e-http-lite` | 17 | 1 | 2 | **5%** baseline |
| `e2e-http-qdrant-nebula` | 15 | 3 | 2 | **15%** flake（3x lite） |
| `e2e-http-qdrant-neo4j` | 15 | 3 | 2 | **15%** flake（3x lite） |

Nebula / Neo4j 比 Lite 的 fail rate **高 3x**。多 PR cross-check 实证（task #17 PR #1893 / #1885 / #1886 / #1890 / #1891 / #1899）失败 signature 集中在 `aperag/domains/agent_runtime/runtime.py:1056` `ValidationException: Model specification is required for agent runtime v3`，跟 PR diff 无因果（同 main 同 commit 不同 run 结果不一致）。

短期不能修根因（需要进一步定位 provider bootstrap / model_use 写入 / agent runtime 启动间的竞态），但**保留 PR-trigger gate 信号**比拆 nightly 更重要 — 这些 shape 历史上抓到过真实 worker DI / compose / provider config 问题（task #17 hot-fix #1893 暴露的 worker DI seam 缺失就是这条 gate 抓的）。

> ⚠️ **本文档是 short-term codify，不是永久放行**：人工放行不豁免 root cause 调查（per §3 责任 lane），所有放行案例应 ledger 收集进 §5 sunset criteria；white-list signature 修复后 §2.2 整段删除。

## 2. 放行规则（Codify）

### 2.1 Lite 必绿（hard requirement）

- `e2e-http-compose / e2e-http-provider` 在 **Lite shape** 必须通过
- Lite 失败**不允许**人工放行 — 必须定位或 rerun 到绿
- Lite shape 失败说明问题不限于 graph backend / 仅 provider env 抖动，是更基础的 retrieval / agent runtime / API 入口 regress

### 2.2 Nebula / Neo4j 单 shape 失败 — 人工放行规则

允许人工放行的**全部条件**（必须全部满足）：

1. **shape 数量限制**：仅 1 个 shape 失败（Nebula **或** Neo4j；不允许两个同时失败 + 放行）
2. **失败 signature 白名单**：失败日志必须命中 `runtime.py:1056` **且** `ValidationException: Model specification is required for agent runtime v3`
3. **PR diff 零交集**（任一触碰即不允许放行）：
   - `aperag/domains/agent_runtime/` (含 V3 turn / runtime)
   - `aperag/domains/retrieval/` (含 search pipeline)
   - `aperag/domains/model_platform/` (含 model bootstrap / provider config)
   - `aperag/llm/` (顶层；含 model_use / completion service)
   - `aperag/mcp/` (含 search runtime)
   - `aperag/indexing/` (含 worker / DI wire-up)
   - `.github/workflows/e2e-http-*.yml` (含 provider env / shape 配置)
   - `tests/e2e_http/` (含 e2e bootstrap / hurl)
   - `deploy/aperag/` (含 Helm / compose worker)

放行时**必须在 PR 评论或 thread 内**：

- 贴具体 GitHub Actions run id + job id（`gh run view <run-id>`）
- 贴失败日志关键签名 fragment（含 `runtime.py:1056`）
- 写明 PR diff 不触碰上述任一区域（最好 link `gh pr view <pr> --files`）

### 2.3 不在白名单的失败 signature

以下 failure signature **不能** codify 放行，必须真修或 rerun：

- `PromptTemplateOps not wired` — DI seam 缺失（参考 task #17 hot-fix）
- HTTP readiness / liveness / probe failure — deployment 形态
- `graph label 500` / `graph extractor 异常` — graph backend 真问题
- SSE JSON parse error 但 **没有** model-spec signature — provider response 异常
- `worker DI` / `cleanup_worker_factory` / `lifespan` 相关 — task #17 hard cut invariant break
- 任何 timeout / compose 启动失败 — 部署层问题
- `IK index not found` / search backend 异常 — 真问题（PR #1917 re-run 后已转绿，证明随机性，但 signature 不在白名单不允许放行）

## 3. 配套 fix-forward 责任

放行后**必须有人**接 root cause investigation：

- **SRE lane**：bot/model_use 写入与 agent runtime turn 启动间是否竞态、shape bootstrap 在 Nebula/Neo4j 下是否少写/晚写 model spec、provider config 初始化时序问题
- **CR lane**：所有 phase 2 PR 关闭后回头看 `runtime.py:1056` ValidationException 真根因
- **架构 lane**：判断是否需要单立 follow-up task 修根因，或长期接受 + 拆 nightly

## 4. 决策追溯

- Planetegg msg=b29b1f91 + msg=46a0c5de — gate wording 起草 + ledger 收 6 fail sample 证 known-signature 集中
- huangheng msg=14a00712 / msg=7e99530f — 多 PR forensics + Lesson #12 v5 own-up（trust framing 反模式延伸到 CI status 解读）
- 架构师 msg=2c69a921 — base-too-old 假设证伪（实际是 worker DI bug，task #17 hot-fix #1893 修，不是 CI flake）
- PM msg=cd08a03f — Layer 2 P0 优先级 ratify（A+C 路径，不直接拆 nightly）

## 5. 未来收尾

`runtime.py:1056` 真根因修后，本文档需更新：

- 删除 §2.2 white-list signature 条目
- 历史 ledger 节点（最近 N 次 fail rate 数据）入 changelog 章节
- Nebula / Neo4j fail rate 应回归到 ≤ 5%（跟 Lite 一致）
