# Evaluation 当前产品状态与使用说明

本文档面向产品、测试、交付和一线支持同学，说明 ApeRAG 当前主线里的 Evaluation 是什么、现在有哪些能力、推荐怎么使用，以及目前仍有哪些边界。

注意：

- 本文以当前主线实现为准。
- 如果你同时看到了 [evaluation-design](../design/evaluation-design.md)，请把它理解为一份较早期的设计稿，而不是当前产品说明。

## 1. 现状与当前能力

当前主线里的 Evaluation 采用的是一条更偏“数据集 + 运行任务”的工作流，而不是从聊天页面临时发起评测。

它的核心思路是：

1. 先在 **Collection** 侧整理要怎么测。
2. 把它发布成一个可运行的 **dataset version**。
3. 再到 **Bot** 侧为某个具体 bot 发起 **run**。
4. 最后在 run 详情里看进度、结果和逐条 item。

一句话概括：

- **Collection > Evaluation** 负责准备评测数据。
- **Bot > Evaluation** 负责运行和查看结果。
- **Chat 页面不是 Evaluation 的起点。**

### 1.1 入口在哪里

当前有 3 个主要页面：

#### Collection 侧入口

路径：

- `/workspace/collections/{collectionId}/benchmarks`

作用：

- 这是整个 Evaluation 流程的起点。
- 在这里创建 benchmark dataset。
- 在这里发布第一个可运行的 dataset version。
- 在这里查看某个 collection 下有哪些 dataset、版本状态和用例数。

#### Bot 侧入口

路径：

- `/workspace/bots/{botId}/evaluation`

作用：

- 这里只负责 **run 管理**。
- 用一个已发布的 `dataset version id` 为当前 bot 发起 run。
- 查看这个 bot 已经跑过哪些 run、当前进度如何。

#### Run 详情页

路径：

- `/workspace/bots/{botId}/evaluation/runs/{runId}`

作用：

- 查看单次 run 的总体进度。
- 查看每个 case 对应的 item 状态、分数、最近一次 attempt、trace/chat 入口。
- 对单个失败项执行重试。

### 1.2 当前核心对象

为了更容易理解页面和接口，先统一几个名词。

#### Dataset

Dataset 表示“这个 collection 应该如何被测试”的一个数据集定义。

它通常包含：

- 名称
- 描述
- 来源类型
- 版本数量
- 最新版本状态

#### Dataset Version

Version 是某个 dataset 的一个可运行快照。

当前产品语义上，只有 **published version** 才适合拿去跑 bot run。

文档里你会经常看到这句话：

- 先准备 **published dataset version**

可以简单理解为：

- dataset 是“测试集合”
- version 是“当前拿去跑的那一版”

#### Run

Run 是“让某个 bot 使用某个 dataset version 跑一遍评测”的一次执行任务。

每条 run 都属于：

- 一个 bot
- 一个 dataset version

#### Run Item

Run item 是 run 里的逐条用例执行结果。

每个 item 会记录类似信息：

- case key
- 执行状态
- 分数
- 最近一次 attempt
- trace / chat
- 错误信息

### 1.3 当前已经能做什么

#### Collection > Evaluation 页面

当前已经具备的能力包括：

1. 查看当前 collection 下的 benchmark datasets 列表
2. 查看每个 dataset 的：
   - 最新版本
   - 版本数
   - 用例数
   - 创建时间
   - 版本状态
3. 搜索 dataset 或 version
4. 创建 dataset
5. 为 dataset 发布首个 version
6. 在 version 发布后展示 `dataset version id`
7. 复制 version id，供后续去 bot 侧创建 run

当前这个页面强调的是：

- **先把评测数据准备好**
- 而不是立刻从 bot/chat 侧开始跑

#### Bot > Evaluation 页面

当前已经具备的能力包括：

1. 查看当前 bot 的 runs 列表
2. 按 run id / dataset version id / 状态搜索
3. 创建新 run
4. 查看各类 summary：
   - 总数
   - 运行中
   - 已完成
   - 失败
5. 在从 Collection 侧跳转过来时预填 `dataset version id`
6. 打开某个 run 的详情页

这个页面当前明确只承接一件事：

- **为当前 bot 跑评测**

它不再承担：

- 从零开始准备 dataset
- 替用户自动挑 collection
- 把 chat 当成评测入口

#### Run 详情页

当前已经具备的能力包括：

1. 查看 run 级别信息：
   - run id
   - dataset version id
   - 状态
   - 创建/更新时间
   - 进度
   - 总数
   - 已完成数
   - 平均分
2. 查看 item 列表
3. 查看最近一次 attempt
4. 打开 trace / chat
5. 取消正在运行的 run
6. 重试单个 item

## 2. 怎么使用

当前最推荐的产品使用路径如下。

### 第一步：从 Collection 页面进入 Evaluation

进入：

- `Collection > Evaluation`

先做一件小事：

- 创建一个真实但范围很小的 dataset

建议第一批不要做大而全，先用一个明确场景打通整条链路，例如：

- 客服升级问题检查
- 检索准确率 smoke
- 某份 SOP 文档问答回归

### 第二步：发布第一个 version

dataset 创建完成后，立刻发布第一个 version。

当前页面更偏向“先完成最小闭环”，所以你可以先放一个代表性 case，把链路跑通。

建议至少填写：

- 输入消息

如果有条件，也建议补上：

- 期望答案
- 参考上下文

发布成功后，页面会给出：

- 已发布的 version id

### 第三步：显式选择目标 bot

当前产品不再默认替用户选 bot。

也就是说，version 发布完之后，需要你自己明确：

- 这次是要跑哪个 bot

然后进入：

- `Bot > Evaluation`

### 第四步：为当前 bot 创建 run

在 bot 的 Evaluation 页面里：

1. 填入或使用预填的 `dataset version id`
2. 按需填写 run 名称
3. 点击创建 run

这时系统会为“当前 bot + 当前 version”创建一条 run。

### 第五步：查看进度与结果

run 创建后，你可以：

1. 在 bot 的 Evaluation 页面看 run 列表和总体状态
2. 进入 run 详情页看逐条 item
3. 对失败项执行重试
4. 通过 trace / chat 继续排查某条 case

## 3. 适合用来做什么

当前这版 Evaluation 更适合以下场景：

1. **Smoke test**
   - 新 bot 上线前，用少量代表性 case 快速确认链路可跑

2. **回归检查**
   - prompt、检索配置、bot 配置调整后，用固定 version 重新跑一遍

3. **Collection 侧数据准备**
   - 先围绕某个 collection 准备测试数据，再把它交给多个 bot 分别验证

4. **问题定位**
   - 对失败 run item 继续 drill down 到 trace / chat

## 4. 当前边界与已知限制

当前这版已经能形成最小闭环，但还没有扩成一个“大而全”的评测平台。以下限制需要明确告诉用户。

### 4.1 Chat 不是起点

当前不建议从 chat 心智发起 Evaluation。

推荐心智是：

- **Collection 负责准备数据**
- **Bot 负责运行结果**

### 4.2 当前更适合先做“小而真实”的第一版

当前 UI 对“首个 version”做了较强引导，更适合先用少量代表性 case 跑通流程。

如果你想一次性做大规模数据编辑、批量导入、复杂版本管理，当前这版还不是最终形态。

### 4.3 Bot 侧依然需要明确的 dataset version id

Bot 页现在不会替你自动猜 dataset，也不会静默替你选 collection。

这意味着用户需要明确知道：

- 这次 run 要用哪个 published dataset version

### 4.4 当前是单 bot 的 run 视角

当前 Bot > Evaluation 页面是按 bot 维度组织的。

已经支持：

- 查看这个 bot 跑过哪些 run

但还没有扩成：

- 多 bot 横向对比视图
- 更高级的聚合报表
- 更完整的导出分析能力

### 4.5 详情数据会随着执行逐步出现

run 创建之后，worker 会逐步分发和执行 item。

因此：

- 刚创建 run 时，详情页可能先看到基础结构
- item、attempt、trace 等信息会随着执行逐步补齐

如果只用一句话概括当前状态，可以这样说：

> **ApeRAG 现在已经具备一条可用的 Evaluation 最小闭环：从 Collection 准备 dataset 和 version，再到 Bot 侧发起 run 并查看结果。**

如果要对第一次接触的人快速介绍当前这版，可以直接说：

> 当前 ApeRAG 的 Evaluation 分两部分：  
> 第一部分在 Collection 里，负责准备评测数据集和可运行版本；  
> 第二部分在 Bot 里，负责拿这个版本去跑 run、看进度和看结果。  
> 推荐你先从一个小而真实的 dataset 开始，发布首个 version 后，再到目标 bot 的 Evaluation 页面发起 run。这样最快能把整条链路跑通。
