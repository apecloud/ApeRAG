# ApeRAG Evaluation System

An RAG system evaluation tool based on Ragas, supporting comprehensive performance assessment for ApeRAG Bot.

## 🚀 Quick Start

### 1\. Configure Evaluation Environment

**Copy and edit the configuration file:**

```bash
# Copy the example configuration file
cp aperag/evaluation/config.example.yaml aperag/evaluation/config.yaml
```

**Required configuration items to modify:**

1.  **API Key Configuration** - Set environment variables or modify directly in the configuration file:

    ```yaml
    api:
      base_url: "http://localhost:8000/api/v1"
      api_token: "${APERAG_API_KEY}"  # Replace with your ApeRAG API Key

    llm_for_eval:
      api_key: "${OPENROUTER_API_KEY}"  # Replace with your LLM API Key

    embeddings_for_eval:
      api_key: "${SILICONFLOW_API_KEY}"  # Replace with your Embedding API Key
    ```

2.  **Bot ID Configuration** - Replace with your created Bot ID:

    ```yaml
    evaluations:
      - bot_id: "your-bot-id-here"  # 🔴 Must replace with the actual Bot ID
    ```

3.  **Sample Count Configuration** - Adjust the number of test samples as needed:

    ```yaml
    evaluations:
      - max_samples: 10  # 🔴 Recommended to start with a small number, e.g., 3-10 samples
    ```

### 2\. Create Bot and Collection

⚠️ **Important Prerequisites:** Before running the evaluation, you need to first create the following in the ApeRAG system:

1.  **Create Collection (Knowledge Base)**:

      * Create a Collection in the ApeRAG Web interface
      * Upload relevant documents and complete indexing

2.  **Create Bot**:

      * Create a Bot based on the above Collection
      * Configure the Bot's conversation parameters
      * Record the Bot ID and update it in the configuration file

### 3\. Prepare Evaluation Dataset

Ensure your dataset is in CSV format, containing `question` and `answer` columns:

```csv
question,answer
"What is Retrieval Augmented Generation?","Retrieval Augmented Generation (RAG) is an AI technique that combines information retrieval with text generation..."
"How to optimize RAG system performance?","You can optimize by improving retrieval strategies, optimizing chunk size, using better embedding models, etc..."
```

### 4\. Run Evaluation

Using the Make command (recommended):

```bash
# Run evaluation
make evaluate
```

Or directly using the Python command:

```bash
# Use default configuration
python -m aperag.evaluation.run

# Specify configuration file
python -m aperag.evaluation.run --config aperag/evaluation/config.yaml
```

## 📊 Evaluation Metrics Description

The system uses 5 core metrics from the Ragas framework:

| Metric               | Description                               | Range | Dependencies    |
| :------------------- | :---------------------------------------- | :---- | :-------------- |
| **Faithfulness** | The faithfulness of the answer to the retrieved content | 0-1   | LLM             |
| **Answer Relevancy** | The relevance of the answer to the question | 0-1   | LLM + Embedding |
| **Context Precision** | The precision of the retrieved context      | 0-1   | LLM             |
| **Context Recall** | The recall of the retrieved context         | 0-1   | LLM             |
| **Answer Correctness** | The overall correctness of the answer       | 0-1   | LLM             |

## 📁 Output Reports

After evaluation, reports in various formats will be generated:

```
evaluation_reports/
├── evaluation_summary_YYYYMMDD_HHMMSS.json    # Detailed statistics
├── evaluation_report_YYYYMMDD_HHMMSS.md       # Readable report
└── evaluation_report_YYYYMMDD_HHMMSS.csv      # Raw evaluation data
```

## ⚙️ Advanced Configuration

### Performance Tuning

Adjust the following parameters in `config.yaml`:

```yaml
advanced:
  request_timeout: 30    # API timeout in seconds
  request_delay: 3       # Delay between requests in seconds
  batch_size: 3          # Batch size
  save_intermediate: true # Save intermediate results
```

### Environment Variables

Set environment variables to avoid hardcoding keys in the configuration file:

```bash
export APERAG_API_KEY="your-aperag-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export SILICONFLOW_API_KEY="your-siliconflow-api-key"
```
