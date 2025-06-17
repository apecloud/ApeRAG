# ApeRAG 评估系统

一个基于Ragas的RAG系统评估工具，支持对ApeRAG Bot进行全面的性能评测。

## 🚀 快速开始

### 1. 配置评估环境

**复制并编辑配置文件：**

```bash
# 复制示例配置文件
cp aperag/evaluation/config.example.yaml aperag/evaluation/config.yaml
```

**必须修改的配置项：**

1. **API密钥配置** - 设置环境变量或直接在配置文件中修改：
   ```yaml
   api:
     base_url: "http://localhost:8000/api/v1"
     api_token: "${APERAG_API_KEY}"  # 替换为你的 ApeRAG API Key

   llm_for_eval:
     api_key: "${OPENROUTER_API_KEY}"  # 替换为你的 LLM API Key

   embeddings_for_eval:
     api_key: "${SILICONFLOW_API_KEY}"  # 替换为你的 Embedding API Key
   ```

2. **Bot ID配置** - 替换为你创建的Bot ID：
   ```yaml
   evaluations:
     - bot_id: "your-bot-id-here"  # 🔴 必须替换为实际的Bot ID
   ```

3. **样本数量配置** - 根据需要调整测试样本数量：
   ```yaml
   evaluations:
     - max_samples: 10  # 🔴 建议先用小数量测试，如3-10个样本
   ```

### 2. 创建Bot和Collection

⚠️ **重要前置条件：** 运行评估前，你需要先在ApeRAG系统中创建：

1. **创建Collection（知识库）**：
   - 在ApeRAG Web界面中创建Collection
   - 上传相关文档并完成索引

2. **创建Bot**：
   - 基于上述Collection创建Bot
   - 配置Bot的对话参数
   - 记录Bot ID并更新到配置文件中

### 3. 准备评估数据集

确保你的数据集是CSV格式，包含 `question` 和 `answer` 两列：

```csv
question,answer
"什么是检索增强生成？","检索增强生成（RAG）是一种结合信息检索和文本生成的AI技术..."
"如何优化RAG系统性能？","可以通过改进检索策略、优化chunk size、使用更好的embedding模型等方式..."
```

### 4. 运行评估

使用Make命令（推荐）：

```bash
# 运行评估
make evaluate
```

或者直接使用Python命令：

```bash
# 使用默认配置
python -m aperag.evaluation.run

# 指定配置文件
python -m aperag.evaluation.run --config aperag/evaluation/config.yaml
```

## 📊 评估指标说明

系统使用Ragas框架的5个核心指标：

| 指标 | 说明 | 范围 | 依赖 |
|------|------|------|------|
| **Faithfulness** | 答案对检索内容的忠实度 | 0-1 | LLM |
| **Answer Relevancy** | 答案与问题的相关性 | 0-1 | LLM + Embedding |
| **Context Precision** | 检索上下文的精确度 | 0-1 | LLM |
| **Context Recall** | 检索上下文的召回率 | 0-1 | LLM |
| **Answer Correctness** | 答案的整体正确性 | 0-1 | LLM |

## 📁 输出报告

评估完成后会生成多种格式的报告：

```
evaluation/reports/
├── evaluation_summary_YYYYMMDD_HHMMSS.json    # 详细统计数据
├── evaluation_report_YYYYMMDD_HHMMSS.md       # 可读性报告
└── evaluation_report_YYYYMMDD_HHMMSS.csv      # 原始评估数据
```

## ⚙️ 高级配置

### 性能调优

在 `config.yaml` 中调整以下参数：

```yaml
advanced:
  request_timeout: 30    # API超时时间（秒）
  request_delay: 3       # 请求间延迟（秒）
  batch_size: 3          # 批处理大小
  save_intermediate: true # 保存中间结果
```

### 环境变量

设置环境变量以避免在配置文件中硬编码密钥：

```bash
export APERAG_API_KEY="your-aperag-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export SILICONFLOW_API_KEY="your-siliconflow-api-key"
```
