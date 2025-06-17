# ApeRAG Evaluation Module

一个低成本、配置驱动的RAG评测系统，用于评估ApeRAG中的Bot性能。

## 功能特点

- 📊 **配置驱动**: 通过YAML配置文件定义评测任务
- 🤖 **Bot评测**: 支持对已创建的Bot进行批量问答测试
- 📈 **Ragas指标**: 集成Ragas库计算多维度评测指标
- 📁 **多格式支持**: 支持CSV和JSON格式的评测数据集
- 📝 **丰富报告**: 生成CSV、JSON和Markdown格式的评测报告
- ⚡ **批量处理**: 支持并发调用API，提高评测效率

## 快速开始

### 1. 准备环境

确保已安装必要的依赖：

```bash
# 在项目根目录运行
make install
```

### 2. 配置文件

编辑 `aperag/evaluation/config.yaml`：

```yaml
# API配置
api:
  base_url: "http://localhost:8000/api/v1"
  # api_token: "your-api-token"  # 或通过环境变量APERAG_API_TOKEN设置

# 评测任务
evaluations:
  - task_name: "我的Bot评测"
    bot_id: "1"  # 替换为您的Bot ID
    dataset_path: "./my_dataset.csv"
    max_samples: 10  # 可选：限制样本数量
    report_dir: "./evaluation_reports/my_bot"
    metrics:
      - faithfulness
      - answer_relevancy
      - context_precision
      - context_recall
      - answer_correctness
```

### 3. 准备数据集

创建CSV或JSON格式的数据集，必须包含 `question` 和 `answer` 列：

```csv
question,answer
"什么是RAG？","RAG是检索增强生成的缩写..."
"如何使用ApeRAG？","首先需要创建一个知识库..."
```

### 4. 运行评测

```bash
# 使用默认配置文件
python -m aperag.evaluation.run

# 或指定配置文件
python -m aperag.evaluation.run --config /path/to/config.yaml
```

## 评测指标说明

### Ragas指标

- **Faithfulness (忠实度)**: 衡量生成答案的事实准确性，答案是否基于检索到的上下文
- **Answer Relevancy (答案相关性)**: 评估答案对问题的相关程度
- **Context Precision (上下文精确度)**: 衡量检索到的上下文与问题的相关性
- **Context Recall (上下文召回率)**: 评估检索到的上下文是否包含回答问题所需的信息
- **Answer Correctness (答案正确性)**: 使用LLM评估答案与标准答案的语义相似度

## 输出报告

评测完成后，会在指定的 `report_dir` 目录生成以下文件：

1. **evaluation_report_YYYYMMDD_HHMMSS.csv**: 详细的评测结果，包含每个问题的得分
2. **evaluation_summary_YYYYMMDD_HHMMSS.json**: 评测摘要，包含各指标的统计信息
3. **evaluation_report_YYYYMMDD_HHMMSS.md**: Markdown格式的可读报告
4. **intermediate_results_YYYYMMDD_HHMMSS.json**: 中间结果（可选）

## 高级配置

### 环境变量

- `APERAG_API_TOKEN`: API认证令牌
- `OPENAI_API_BASE`: OpenAI API基础URL（用于Ragas评测）
- `OPENAI_API_KEY`: OpenAI API密钥

### 批处理配置

```yaml
advanced:
  request_timeout: 30      # 请求超时时间（秒）
  batch_size: 5           # 批处理大小
  request_delay: 1        # 批次间延迟（秒）
  save_intermediate: true # 是否保存中间结果
```

## 注意事项

1. **API兼容性**: 当前版本假设Bot的聊天API返回标准的OpenAI格式响应。如果您的API返回格式不同，可能需要修改 `_call_bot_api` 方法。

2. **Context字段**: 评测需要Bot返回检索到的上下文信息。如果您的API不返回context字段，可能需要：
   - 修改Bot的API实现，添加context返回
   - 或在评测代码中调整context的获取逻辑

3. **性能考虑**: 
   - 大数据集评测可能需要较长时间
   - 建议先用小样本测试配置是否正确
   - 可以通过 `max_samples` 限制样本数量

4. **API限流**: 如果遇到API限流，可以调整 `batch_size` 和 `request_delay`

## 扩展开发

### 添加新的评测指标

1. 在 `run.py` 的 `_get_metrics` 方法中添加新指标的映射
2. 确保新指标与Ragas兼容或实现自定义指标

### 自定义报告格式

修改 `_save_results` 和 `_generate_markdown_report` 方法来生成自定义格式的报告。

### 集成到CI/CD

可以将评测集成到持续集成流程中：

```bash
# 在CI脚本中
python -m aperag.evaluation.run --config ci_evaluation_config.yaml
# 检查返回码判断是否成功
```

## 故障排除

遇到问题？请查看[详细的故障排除指南](./TROUBLESHOOTING.md)。

### 常见问题快速解答

1. **"Dataset missing required columns"**: 确保数据集包含 `question` 和 `answer` 列
2. **API认证失败**: 检查API token是否正确设置
3. **Ragas评测失败**: 确保设置了OPENAI_API_KEY环境变量（即使使用其他LLM提供商）
4. **连接超时**: 增加 `request_timeout` 值或检查网络连接

### 调试模式

设置日志级别为DEBUG获取更多信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 示例

项目包含一个示例配置，使用三国演义问答数据集：

```bash
# 运行示例评测
python -m aperag.evaluation.run
```

这将使用默认的 `config.yaml` 运行评测，并在 `./evaluation_reports/demo_eval` 目录生成报告。 