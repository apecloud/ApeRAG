#!/usr/bin/env python3
# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ApeRAG Evaluation Runner

This script runs evaluation tasks defined in config.yaml.
It loads datasets, calls bot APIs, and generates comprehensive reports using Ragas.
"""

import asyncio
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import yaml
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Main class for running RAG evaluations"""

    def __init__(self, config_path: str = None):
        """Initialize evaluation runner with configuration"""
        self.config_path = config_path or Path(__file__).parent / "config.yaml"
        self.config = self._load_config()
        self.llm_for_eval = self._setup_llm_for_eval()
        self.embeddings_for_eval = self._setup_embeddings_for_eval()

    def _is_valid_number(self, value: Any) -> bool:
        """Check if a value is a valid finite number (not NaN or infinity)"""
        if not isinstance(value, (int, float)):
            return False
        return math.isfinite(value) and not math.isnan(value)

    def _safe_metric_calculation(self, values: List[float]) -> Dict[str, Any]:
        """Safely calculate statistics from a list of values, filtering out NaN and invalid values"""
        # Filter out NaN and invalid values
        valid_values = [v for v in values if self._is_valid_number(v)]

        if not valid_values:
            return {
                "mean": None,
                "min": None,
                "max": None,
                "count": 0,
                "valid_count": 0,
                "invalid_count": len(values),
                "std": None,
            }

        mean_val = sum(valid_values) / len(valid_values)
        std_val = 0.0
        if len(valid_values) > 1:
            variance = sum((x - mean_val) ** 2 for x in valid_values) / len(valid_values)
            std_val = math.sqrt(variance)

        return {
            "mean": mean_val,
            "min": min(valid_values),
            "max": max(valid_values),
            "count": len(values),
            "valid_count": len(valid_values),
            "invalid_count": len(values) - len(valid_values),
            "std": std_val,
        }

    def _clean_for_json(self, obj: Any) -> Any:
        """Clean an object for JSON serialization by handling NaN, infinity, and other problematic values"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj):
                return None
            elif math.isinf(obj):
                return None
            else:
                return obj
        else:
            return obj

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        logger.info(f"Loading configuration from {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Replace environment variables
        config = self._replace_env_vars(config)
        return config

    def _replace_env_vars(self, obj: Any) -> Any:
        """Recursively replace ${VAR} with environment variables"""
        if isinstance(obj, str):
            if obj.startswith("${") and obj.endswith("}"):
                var_name = obj[2:-1]
                return os.environ.get(var_name, obj)
            return obj
        elif isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        return obj

    def _setup_llm_for_eval(self) -> ChatOpenAI:
        """Setup LLM for Ragas evaluation"""
        llm_config = self.config["llm_for_eval"]
        return ChatOpenAI(
            base_url=llm_config["api_base"],
            api_key=llm_config["api_key"],
            model=llm_config["model"],
            temperature=llm_config["temperature"],
        )

    def _setup_embeddings_for_eval(self) -> Optional[OpenAIEmbeddings]:
        """Setup embeddings for Ragas evaluation"""
        # Check if embeddings configuration exists
        embeddings_config = self.config.get("embeddings_for_eval")

        if not embeddings_config or not embeddings_config.get("api_base"):
            logger.warning("No embeddings configuration found. Embedding-based metrics will be skipped.")
            return None

        try:
            return OpenAIEmbeddings(
                openai_api_base=embeddings_config["api_base"],
                openai_api_key=embeddings_config["api_key"],
                model=embeddings_config.get("model", "text-embedding-3-small"),
                check_embedding_ctx_length=False,
            )
        except Exception as e:
            logger.warning(f"Failed to setup embeddings: {e}. Embedding-based metrics will be skipped.")
            return None

    def _load_dataset(self, dataset_path: str, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Load dataset from CSV or JSON file"""
        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)

            # Handle different column names
            question_col = None
            answer_col = None

            # Look for question column
            for col in df.columns:
                if col.lower() in ["question", "query", "input", "q"]:
                    question_col = col
                    break

            # Look for answer column
            for col in df.columns:
                if col.lower() in ["answer", "response", "output", "a", "ground_truth"]:
                    answer_col = col
                    break

            if question_col is None:
                raise ValueError(f"No question column found in CSV. Available columns: {list(df.columns)}")
            if answer_col is None:
                raise ValueError(f"No answer column found in CSV. Available columns: {list(df.columns)}")

            logger.info(f"Using question column: '{question_col}', answer column: '{answer_col}'")

            # Convert to standard format
            dataset = []
            for _, row in df.iterrows():
                dataset.append({"question": str(row[question_col]), "answer": str(row[answer_col])})

        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                dataset = data
            else:
                raise ValueError("JSON file should contain a list of question-answer pairs")

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Only CSV and JSON are supported.")

        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset[:max_samples]
            logger.info(f"Limited dataset to {max_samples} samples")

        return dataset

    async def _call_bot_api(self, bot_id: str, question: str) -> Dict[str, Any]:
        """Call bot API and get response with context"""
        import time

        start_time = time.perf_counter()  # Use more precise timing

        try:
            # Use direct HTTP request instead of OpenAI client to handle ApeRAG's response format
            api_config = self.config["api"]
            base_url = api_config["base_url"]
            api_token = api_config.get("api_token") or os.environ.get("APERAG_API_TOKEN")

            # Configure timeout
            advanced_config = self.config.get("advanced", {})
            timeout = advanced_config.get("request_timeout", 30)

            # The chat/completions endpoint is at /v1, not /api/v1
            if base_url.endswith("/api/v1"):
                # Remove /api/v1 and add /v1 for chat completions
                host = base_url[:-7]  # Remove "/api/v1"
                chat_url = f"{host}/v1/chat/completions"
            else:
                chat_url = f"{base_url}/v1/chat/completions"

            headers = {}
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"
            headers["Content-Type"] = "application/json"

            request_body = {"messages": [{"role": "user", "content": question}], "model": "aperag", "stream": False}

            params = {"bot_id": bot_id}

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(chat_url, headers=headers, json=request_body, params=params)
                response.raise_for_status()

                # Calculate response time with higher precision
                end_time = time.perf_counter()
                response_time = round(end_time - start_time, 3)  # Round to 3 decimal places

                logger.debug(f"API call took {response_time:.3f} seconds")

                # Parse response
                response_data = response.json()

                # Check for API error format
                if "error" in response_data:
                    error_msg = response_data.get("error", {}).get("message", "Unknown API error")
                    logger.error(f"API returned error: {error_msg}")
                    return {"response": "", "context": [], "error": error_msg, "response_time": response_time}

                # Extract content from OpenAI-compatible response
                choices = response_data.get("choices", [])
                if not choices:
                    logger.warning("No choices in API response")
                    return {
                        "response": "",
                        "context": [],
                        "error": "No response content",
                        "response_time": response_time,
                    }

                raw_content = choices[0].get("message", {}).get("content", "")
                logger.debug(f"Raw API response content: {raw_content[:200]}...")

                # Parse content with |DOC_QA_REFERENCES| separator
                if "|DOC_QA_REFERENCES|" in raw_content:
                    parts = raw_content.split("|DOC_QA_REFERENCES|", 1)
                    response_text = parts[0].strip()
                    context_text = parts[1].strip() if len(parts) > 1 else ""

                    # Parse context JSON
                    context_data = []
                    if context_text:
                        try:
                            # Try to parse as JSON
                            parsed_context = json.loads(context_text)

                            # Handle both single object and array
                            if isinstance(parsed_context, list):
                                context_data = parsed_context
                            elif isinstance(parsed_context, dict):
                                context_data = [parsed_context]

                            logger.debug(f"Successfully parsed context: {len(context_data)} items")

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse context JSON: {e}")
                            # Keep as string if parsing fails
                            context_data = [{"text": context_text}]

                    return {
                        "response": response_text,
                        "context": context_data,  # Keep as parsed object, not string
                        "error": None,
                        "response_time": response_time,
                    }
                else:
                    # No separator, treat whole content as response
                    return {"response": raw_content, "context": [], "error": None, "response_time": response_time}

        except httpx.HTTPStatusError as e:
            end_time = time.perf_counter()
            response_time = round(end_time - start_time, 3)
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(f"API HTTP error: {error_msg}")
            return {"response": "", "context": [], "error": error_msg, "response_time": response_time}
        except Exception as e:
            end_time = time.perf_counter()
            response_time = round(end_time - start_time, 3)
            error_msg = f"API call failed: {str(e)}"
            logger.error(f"API call error: {error_msg}")
            return {"response": "", "context": [], "error": error_msg, "response_time": response_time}

    async def _process_dataset(self, bot_id: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process dataset and get bot responses"""
        results = []
        total = len(df)

        # Get advanced settings
        advanced_config = self.config.get("advanced", {})
        request_delay = advanced_config.get("request_delay", 0)

        for idx, row in df.iterrows():
            question = row["question"]
            ground_truth = row["answer"]

            logger.info(f"Processing question {idx + 1}/{total}: {question[:50]}...")

            # Call bot API
            api_result = await self._call_bot_api(bot_id, question)

            # Log response time
            response_time = api_result.get("response_time", 0)
            if api_result.get("error"):
                logger.info(f"Question {idx + 1}/{total} failed in {response_time:.2f}s: {api_result.get('error')}")
            else:
                logger.info(f"Question {idx + 1}/{total} completed in {response_time:.2f}s")

            # Build result record
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "response": api_result.get("response", ""),
                "context": api_result.get("context", []),  # Keep as parsed object/list
                "error": api_result.get("error"),
                "response_time": response_time,
            }

            results.append(result)

            # Add delay between requests if configured
            if request_delay > 0:
                await asyncio.sleep(request_delay)

        return results

    def _prepare_ragas_dataset(self, results: List[Dict]) -> Dataset:
        """Prepare dataset for Ragas evaluation"""
        # Filter out failed requests
        valid_results = [r for r in results if not r.get("error") and r.get("response")]

        if not valid_results:
            logger.warning("No valid results found for Ragas evaluation")
            return Dataset.from_dict({"question": [], "answer": [], "contexts": [], "ground_truth": []})

        # Prepare data for Ragas
        ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        for result in valid_results:
            ragas_data["question"].append(result["question"])
            ragas_data["answer"].append(result["response"])
            ragas_data["ground_truth"].append(result["ground_truth"])

            # Handle context - extract text fields from JSON context
            contexts_list = []
            raw_context_data = result.get("context", [])

            if raw_context_data:
                # Extract text from parsed context data
                for item in raw_context_data:
                    if isinstance(item, dict) and "text" in item:
                        text_content = item["text"]
                        if text_content and text_content.strip():
                            contexts_list.append(text_content.strip())
                    elif isinstance(item, str) and item.strip():
                        # If it's already a string, use it directly
                        contexts_list.append(item.strip())

            # If no contexts extracted from raw_context_data, try the context string
            if not contexts_list:
                context_str = result.get("context", "")
                if context_str and context_str.strip():
                    try:
                        # Try to parse context string as JSON
                        import json

                        context_json = json.loads(context_str)

                        # Handle different JSON structures
                        if isinstance(context_json, list):
                            for item in context_json:
                                if isinstance(item, dict) and "text" in item:
                                    text_content = item["text"]
                                    if text_content and text_content.strip():
                                        contexts_list.append(text_content.strip())
                        elif isinstance(context_json, dict) and "text" in context_json:
                            text_content = context_json["text"]
                            if text_content and text_content.strip():
                                contexts_list.append(text_content.strip())
                    except json.JSONDecodeError:
                        # If JSON parsing fails, use the string directly
                        contexts_list.append(context_str.strip())

            # Ensure we have at least an empty string for Ragas
            if not contexts_list:
                contexts_list = [""]

            ragas_data["contexts"].append(contexts_list)

            logger.debug(f"Extracted {len(contexts_list)} context items for question: {result['question'][:50]}...")

        logger.info(f"Prepared Ragas dataset with {len(valid_results)} valid samples")
        return Dataset.from_dict(ragas_data)

    def _get_metrics(self, metric_names: List[str]):
        """Get Ragas metric objects from names"""
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }

        # Metrics that require embeddings
        embedding_required_metrics = {"answer_relevancy", "context_precision", "context_recall"}

        metrics = []
        for name in metric_names:
            if name in metric_map:
                # Skip embedding-based metrics if embeddings are not available
                if name in embedding_required_metrics and self.embeddings_for_eval is None:
                    logger.warning(f"Skipping metric '{name}' as it requires embeddings which are not available")
                    continue
                metrics.append(metric_map[name])
            else:
                logger.warning(f"Unknown metric: {name}")

        if not metrics:
            logger.warning("No valid metrics available for evaluation")

        return metrics

    def _save_results(
        self,
        results: List[Dict[str, Any]],
        eval_results: Optional[Dataset],
        task_config: Dict[str, Any],
        report_dir: Path,
    ):
        """Save evaluation results to files"""
        logger.info(f"Saving results to {report_dir}")

        # Create report directory
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Save detailed CSV report
        df_results = pd.DataFrame(results)
        if eval_results:
            # Add Ragas scores to the dataframe
            eval_df = eval_results.to_pandas()
            # Merge on index (assuming same order)
            df_results = pd.concat([df_results, eval_df], axis=1)

        csv_path = report_dir / f"evaluation_report_{timestamp}.csv"
        df_results.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Saved detailed report to {csv_path}")

        # 2. Save JSON summary
        summary = {
            "task_name": task_config["task_name"],
            "bot_id": task_config["bot_id"],
            "dataset_path": task_config["dataset_path"],
            "timestamp": timestamp,
            "total_samples": len(results),
            "metrics": {},
        }

        if eval_results:
            # Calculate average scores for each metric
            eval_df = eval_results.to_pandas()
            for col in eval_df.columns:
                if col not in ["question", "answer", "contexts", "ground_truth"]:
                    summary["metrics"][col] = self._safe_metric_calculation(eval_df[col].tolist())

        json_path = report_dir / f"evaluation_summary_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._clean_for_json(summary), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved summary to {json_path}")

        # 3. Save Markdown report
        md_path = report_dir / f"evaluation_report_{timestamp}.md"
        self._generate_markdown_report(summary, df_results, md_path)
        logger.info(f"Saved markdown report to {md_path}")

        # 4. Save intermediate results if configured
        if self.config.get("advanced", {}).get("save_intermediate", True):
            intermediate_path = report_dir / f"intermediate_results_{timestamp}.json"
            with open(intermediate_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved intermediate results to {intermediate_path}")

    async def _fetch_bot_details(self, bot_id: str) -> Dict[str, Any]:
        """Fetch bot details from API"""
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.config['api']['api_token']}"}
                url = f"{self.config['api']['base_url']}/bots/{bot_id}"

                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Failed to fetch bot details: {response.status_code} - {response.text}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching bot details: {e}")
            return {}

    async def _fetch_collection_details(self, collection_id: str) -> Dict[str, Any]:
        """Fetch collection details from API"""
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.config['api']['api_token']}"}
                url = f"{self.config['api']['base_url']}/collections/{collection_id}"

                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Failed to fetch collection details: {response.status_code} - {response.text}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching collection details: {e}")
            return {}

    def _generate_ragas_metrics_explanation(self) -> str:
        """Generate explanation for Ragas evaluation metrics"""
        return """
## Ragas Evaluation Metrics Explanation

Ragas (Retrieval Augmented Generation Assessment) is a framework specifically designed for evaluating RAG systems, providing the following core metrics:

### 1. Faithfulness
- **Definition**: Measures the consistency between generated answers and retrieved context information
- **Calculation**: Analyzes whether statements in the answer can find support in the retrieved context
- **Score Range**: 0-1, higher scores indicate answers are more faithful to source materials
- **Significance**: Ensures AI doesn't generate hallucinated content that contradicts facts

### 2. Answer Relevancy  
- **Definition**: Evaluates how relevant the generated answer is to the user's question
- **Calculation**: Analyzes whether the answer directly addresses the user's specific question
- **Score Range**: 0-1, higher scores indicate more relevant answers
- **Significance**: Avoids irrelevant responses and ensures targeted answers

### 3. Context Precision
- **Definition**: Measures the proportion of relevant content in retrieved context information
- **Calculation**: Evaluates the ratio of useful information vs irrelevant information in retrieval results
- **Score Range**: 0-1, higher scores indicate more precise retrieval
- **Significance**: Optimizes retrieval strategy and reduces noise

### 4. Context Recall
- **Definition**: Evaluates whether the retrieval system can find all relevant information needed to answer the question
- **Calculation**: Checks if all information the answer depends on can be found in the retrieved context
- **Score Range**: 0-1, higher scores indicate more complete retrieval coverage
- **Significance**: Ensures important information is not missed

### 5. Answer Correctness
- **Definition**: Comprehensively evaluates answer correctness, combining factual accuracy and semantic similarity
- **Calculation**: Compares generated answers with standard answers at semantic and factual levels
- **Score Range**: 0-1, higher scores indicate more accurate answers
- **Significance**: Comprehensively measures RAG system output quality
"""

    async def _generate_markdown_report(
        self,
        task_name: str,
        bot_id: str,
        dataset_path: str,
        timestamp: str,
        results: List[Dict],
        ragas_results: Optional[List[Dict]],
        output_path: Path,
    ) -> None:
        """Generate a markdown evaluation report"""

        # Fetch bot and collection details
        logger.info("Fetching bot and collection details...")
        bot_details = await self._fetch_bot_details(bot_id)

        # Parse bot config to get collection IDs
        collection_details = []
        if bot_details.get("collection_ids"):
            for collection_id in bot_details["collection_ids"]:
                collection_detail = await self._fetch_collection_details(collection_id)
                if collection_detail:
                    collection_details.append(collection_detail)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# ApeRAG Evaluation Report\n\n")
            f.write(f"**Task Name:** {task_name}\n\n")
            f.write(f"**Dataset:** {dataset_path}\n\n")
            f.write(f"**Timestamp:** {timestamp}\n\n")
            f.write(f"**Total Samples:** {len(results)}\n\n")

            # Bot Configuration Section
            f.write("## Bot Configuration\n\n")
            f.write(f"**Bot ID:** {bot_id}\n\n")
            if bot_details:
                f.write(f"**Bot Title:** {bot_details.get('title', 'N/A')}\n\n")
                f.write(f"**Bot Type:** {bot_details.get('type', 'N/A')}\n\n")
                f.write(f"**Bot Description:** {bot_details.get('description', 'N/A')}\n\n")

                # Parse bot config for model information
                try:
                    import json

                    if bot_details.get("config"):
                        bot_config = json.loads(bot_details["config"])
                        if bot_config.get("model_name"):
                            f.write(f"**LLM Model:** {bot_config['model_name']}\n\n")
                        if bot_config.get("model_service_provider"):
                            f.write(f"**LLM Provider:** {bot_config['model_service_provider']}\n\n")
                except json.JSONDecodeError:
                    logger.warning("Failed to parse bot config JSON")

            # Collection Configuration Section
            if collection_details:
                f.write("## Collection Configuration\n\n")
                for i, collection in enumerate(collection_details, 1):
                    f.write(f"### Collection {i}: {collection.get('title', 'N/A')}\n\n")
                    f.write(f"**Collection ID:** {collection.get('id', 'N/A')}\n\n")
                    f.write(f"**Description:** {collection.get('description', 'N/A')}\n\n")

                    # Parse collection config for model information
                    config = collection.get("config", {})
                    if config:
                        f.write("**Model Configuration:**\n\n")

                        # Embedding model
                        embedding_config = config.get("embedding", {})
                        if embedding_config:
                            f.write(f"- **Embedding Model:** {embedding_config.get('model', 'N/A')}\n")
                            f.write(
                                f"- **Embedding Provider:** {embedding_config.get('model_service_provider', 'N/A')}\n"
                            )
                            if embedding_config.get("custom_llm_provider"):
                                f.write(
                                    f"- **Embedding Custom Provider:** {embedding_config.get('custom_llm_provider', 'N/A')}\n"
                                )

                        # Completion model
                        completion_config = config.get("completion", {})
                        if completion_config:
                            f.write(f"- **Completion Model:** {completion_config.get('model', 'N/A')}\n")
                            f.write(
                                f"- **Completion Provider:** {completion_config.get('model_service_provider', 'N/A')}\n"
                            )
                            if completion_config.get("custom_llm_provider"):
                                f.write(
                                    f"- **Completion Custom Provider:** {completion_config.get('custom_llm_provider', 'N/A')}\n"
                                )

                        # Knowledge graph setting
                        if "enable_knowledge_graph" in config:
                            f.write(f"- **Knowledge Graph Enabled:** {config.get('enable_knowledge_graph', False)}\n")

                        f.write("\n")

            # Ragas Evaluation Metrics Section
            if ragas_results:
                f.write("## Ragas Evaluation Results\n\n")
                if isinstance(ragas_results, list) and len(ragas_results) > 0:
                    # Calculate average scores with NaN handling
                    metrics = {}
                    for result in ragas_results:
                        for key, value in result.items():
                            if isinstance(value, (int, float)) and key not in [
                                "question",
                                "answer",
                                "contexts",
                                "ground_truth",
                                "user_input",
                                "response",
                                "retrieved_contexts",
                                "reference",
                            ]:
                                if key not in metrics:
                                    metrics[key] = []
                                # Only append valid numbers
                                if self._is_valid_number(value):
                                    metrics[key].append(value)

                    for metric, values in metrics.items():
                        if values:  # Only calculate if we have valid values
                            stats = self._safe_metric_calculation(values)
                            avg_score = stats["mean"]
                            min_score = stats["min"]
                            max_score = stats["max"]
                            valid_count = stats["valid_count"]
                            total_count = len([r for r in ragas_results if metric in r])

                            if avg_score is not None:
                                f.write(
                                    f"- **{metric.replace('_', ' ').title()}**: {avg_score:.3f} "
                                    f"(Range: {min_score:.3f} - {max_score:.3f}, "
                                    f"Valid: {valid_count}/{total_count})\n"
                                )
                            else:
                                f.write(
                                    f"- **{metric.replace('_', ' ').title()}**: N/A "
                                    f"(No valid values, Total samples: {total_count})\n"
                                )
                        else:
                            # No valid values found for this metric
                            total_count = len([r for r in ragas_results if metric in r])
                            f.write(
                                f"- **{metric.replace('_', ' ').title()}**: N/A "
                                f"(No valid values, Total samples: {total_count})\n"
                            )
                    f.write("\n")

                # Add metrics explanation
                f.write(self._generate_ragas_metrics_explanation())

            # Response Time Statistics Section
            f.write("## Performance Statistics\n\n")

            # Calculate response time statistics
            response_times = [
                result.get("response_time", 0)
                for result in results
                if result.get("response_time") is not None and result.get("response_time") > 0
            ]
            successful_response_times = [
                result.get("response_time", 0)
                for result in results
                if not result.get("error")
                and result.get("response_time") is not None
                and result.get("response_time") > 0
            ]

            response_time_stats = {}
            if response_times:
                response_time_stats = {
                    "total_calls": len(results),
                    "calls_with_response_time": len(response_times),
                    "successful_calls": len(successful_response_times),
                    "failed_calls": len(results) - len([r for r in results if not r.get("error")]),
                    "average_response_time": sum(response_times) / len(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "total_time": sum(response_times),
                }
                if successful_response_times:
                    response_time_stats["average_successful_response_time"] = sum(successful_response_times) / len(
                        successful_response_times
                    )
                    response_time_stats["min_successful_response_time"] = min(successful_response_times)
                    response_time_stats["max_successful_response_time"] = max(successful_response_times)

            f.write(f"**Response Time Statistics:**\n\n{json.dumps(response_time_stats)}\n\n")

            # Sample Results Section
            f.write("## Sample Results\n\n")

            for i, result in enumerate(results[:10], 1):  # Show first 10 samples
                f.write(f"### Sample {i}\n\n")
                f.write(f"**Question:** {result['question']}\n\n")
                f.write(f"**Ground Truth:** {result['ground_truth']}\n\n")
                f.write(f"**Bot Response:** \n```\n{result['response']}\n```\n\n")

                # Add response time
                response_time = result.get("response_time", 0)
                f.write(f"**Response Time:** {response_time:.2f} seconds\n\n")

                # Add error information if present
                if result.get("error"):
                    f.write(f"**Error:** {result['error']}\n\n")

                # Add Ragas metrics for this sample if available
                if result.get("ragas_metrics"):
                    f.write("**Evaluation Metrics:**\n\n")
                    for metric, value in result["ragas_metrics"].items():
                        if isinstance(value, (int, float)):
                            if self._is_valid_number(value):
                                f.write(f"- {metric.replace('_', ' ').title()}: {value:.3f}\n")
                            else:
                                f.write(f"- {metric.replace('_', ' ').title()}: N/A (Invalid value)\n")
                    f.write("\n")

                f.write("---\n\n")

            if len(results) > 10:
                f.write(f"*({len(results) - 10} more samples not shown)*\n\n")

        logger.info(f"Enhanced markdown report saved to {output_path}")

    async def _run_ragas_evaluation(self, results: List[Dict], metrics: List[str]) -> Optional[List[Dict]]:
        """Run Ragas evaluation on the results"""
        if not metrics:
            logger.warning("No metrics specified for Ragas evaluation")
            return None

        try:
            # Prepare data for Ragas
            logger.info("Preparing data for Ragas evaluation")
            ragas_dataset = self._prepare_ragas_dataset(results)

            # Debug: Check dataset content
            logger.info(f"Ragas dataset prepared with {len(ragas_dataset)} samples")
            if len(ragas_dataset) > 0:
                logger.debug(f"Sample dataset entry: {dict(ragas_dataset[0])}")

            ragas_metrics = self._get_metrics(metrics)
            logger.info(f"Using Ragas metrics: {[str(m) for m in ragas_metrics]}")

            # Check if we have valid data
            if len(ragas_dataset) == 0:
                logger.warning("Empty dataset for Ragas evaluation")
                return None

            # Run evaluation
            logger.info("Starting Ragas evaluation...")
            if self.embeddings_for_eval is not None:
                # Use both LLM and embeddings
                eval_results = evaluate(
                    dataset=ragas_dataset,
                    metrics=ragas_metrics,
                    llm=self.llm_for_eval,
                    embeddings=self.embeddings_for_eval,
                    raise_exceptions=False,
                )
            else:
                # Use only LLM, skip embedding-based metrics
                logger.info("Using LLM-only evaluation (no embeddings available)")
                eval_results = evaluate(
                    dataset=ragas_dataset, metrics=ragas_metrics, llm=self.llm_for_eval, raise_exceptions=False
                )
            logger.info("Ragas evaluation completed successfully")

            # Convert to dict and return
            results_dict = eval_results.to_pandas().to_dict("records")
            logger.info(f"Ragas results converted to {len(results_dict)} records")

            # Debug: Check for NaN values in results
            nan_count = 0
            total_metrics = 0
            for record in results_dict:
                for key, value in record.items():
                    if isinstance(value, (int, float)):
                        total_metrics += 1
                        if math.isnan(value) or math.isinf(value):
                            nan_count += 1
                            logger.warning(
                                f"Found NaN/Inf value for metric '{key}' in question: {record.get('user_input', record.get('question', 'Unknown'))[:50]}..."
                            )

            if nan_count > 0:
                logger.warning(
                    f"Found {nan_count} NaN/Inf values out of {total_metrics} total metric values. These will be handled safely in statistics calculation."
                )
            else:
                logger.info("No NaN/Inf values found in Ragas results")

            return results_dict

        except Exception as e:
            logger.error(f"Ragas evaluation failed with exception: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def _generate_reports(
        self,
        task_name: str,
        bot_id: str,
        dataset_path: str,
        timestamp: str,
        results: List[Dict],
        ragas_results: Optional[List[Dict]],
        report_dir: Path,
    ) -> None:
        """Generate evaluation reports in multiple formats"""

        # Ensure report directory exists
        report_dir.mkdir(parents=True, exist_ok=True)

        # Create a mapping from question to ragas results for easier lookup
        ragas_lookup = {}
        if ragas_results:
            for ragas_row in ragas_results:
                # Ragas uses 'user_input' field for questions
                question = ragas_row.get("user_input", "") or ragas_row.get("question", "")
                if question:
                    ragas_lookup[question] = ragas_row

        # Merge results with ragas scores
        enriched_results = []
        for result in results:
            enriched_result = {
                "question": result.get("question", ""),
                "ground_truth": result.get("ground_truth", ""),
                "response": result.get("response", ""),
                "context": result.get("context", []),
                "response_time": result.get("response_time", 0),
                "error": result.get("error"),
            }

            # Add Ragas metrics if available for this question
            question = result.get("question", "")
            if question in ragas_lookup:
                ragas_row = ragas_lookup[question]
                ragas_metrics = {}
                for metric_name, metric_value in ragas_row.items():
                    if metric_name not in [
                        "user_input",
                        "response",
                        "retrieved_contexts",
                        "reference",
                        "question",
                        "answer",
                        "contexts",
                        "ground_truth",
                    ]:
                        # Clean the metric value for JSON serialization
                        ragas_metrics[metric_name] = self._clean_for_json(metric_value)
                enriched_result["ragas_metrics"] = ragas_metrics
            else:
                enriched_result["ragas_metrics"] = {}

            enriched_results.append(enriched_result)

        # Calculate overall Ragas statistics
        overall_ragas_stats = {}
        if ragas_results:
            # Collect all metric values with NaN filtering
            all_metrics = {}
            for ragas_row in ragas_results:
                for metric_name, metric_value in ragas_row.items():
                    if metric_name not in [
                        "user_input",
                        "response",
                        "retrieved_contexts",
                        "reference",
                        "question",
                        "answer",
                        "contexts",
                        "ground_truth",
                    ] and isinstance(metric_value, (int, float)):
                        # Only include valid numbers (not NaN or infinity)
                        if self._is_valid_number(metric_value):
                            if metric_name not in all_metrics:
                                all_metrics[metric_name] = []
                            all_metrics[metric_name].append(metric_value)

            # Calculate statistics for each metric using safe calculation
            for metric_name, values in all_metrics.items():
                if values:  # Make sure we have values
                    overall_ragas_stats[metric_name] = self._safe_metric_calculation(values)

        # Calculate response time statistics
        response_times = [
            result.get("response_time", 0)
            for result in results
            if result.get("response_time") is not None and result.get("response_time") > 0
        ]
        successful_response_times = [
            result.get("response_time", 0)
            for result in results
            if not result.get("error") and result.get("response_time") is not None and result.get("response_time") > 0
        ]

        response_time_stats = {}
        if response_times:
            response_time_stats = {
                "total_calls": len(results),
                "calls_with_response_time": len(response_times),
                "successful_calls": len(successful_response_times),
                "failed_calls": len(results) - len([r for r in results if not r.get("error")]),
                "average_response_time": sum(response_times) / len(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "total_time": sum(response_times),
            }
            if successful_response_times:
                response_time_stats["average_successful_response_time"] = sum(successful_response_times) / len(
                    successful_response_times
                )
                response_time_stats["min_successful_response_time"] = min(successful_response_times)
                response_time_stats["max_successful_response_time"] = max(successful_response_times)

        # Generate comprehensive JSON summary
        summary = {
            "task_name": task_name,
            "bot_id": bot_id,
            "dataset_path": dataset_path,
            "timestamp": timestamp,
            "total_samples": len(results),
            "samples_with_ragas": len(ragas_results) if ragas_results else 0,
            "overall_ragas_metrics": overall_ragas_stats,
            "response_time_statistics": response_time_stats,
            "results": enriched_results,
        }

        summary_path = report_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self._clean_for_json(summary), f, indent=2, ensure_ascii=False)
        logger.info(f"Comprehensive evaluation summary saved to {summary_path}")

        # Generate Markdown report (optional, for human readability)
        md_path = report_dir / f"evaluation_report_{timestamp}.md"
        await self._generate_markdown_report(
            task_name, bot_id, dataset_path, timestamp, enriched_results, ragas_results, md_path
        )

    async def run_evaluation(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single evaluation task"""
        task_name = task_config["task_name"]
        bot_id = task_config["bot_id"]
        dataset_path = task_config["dataset_path"]
        max_samples = task_config.get("max_samples")
        report_dir = task_config["report_dir"]
        metrics = task_config.get("metrics", ["faithfulness", "answer_relevancy"])

        logger.info(f"Starting evaluation task: {task_name}")

        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = self._load_dataset(dataset_path, max_samples)
        logger.info(f"Loaded {len(dataset)} samples from dataset")

        # Process questions
        logger.info(f"Processing {len(dataset)} questions with bot {bot_id}")

        # Get advanced config for batch processing and delays
        advanced_config = self.config.get("advanced", {})
        batch_size = advanced_config.get("batch_size", 5)
        request_delay = advanced_config.get("request_delay", 1)

        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            batch_tasks = [self._call_bot_api(bot_id, item["question"]) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)

            for j, result in enumerate(batch_results):
                item = batch[j]
                results.append(
                    {
                        "question": item["question"],
                        "ground_truth": item["answer"],
                        "response": result["response"],
                        "context": result["context"],
                        "response_time": result.get("response_time", 0),
                        "error": result.get("error"),
                    }
                )

            logger.info(f"Processed {min(i + batch_size, len(dataset))}/{len(dataset)} questions")

            # Add delay between batches to respect rate limits
            if i + batch_size < len(dataset) and request_delay > 0:
                logger.debug(f"Waiting {request_delay} seconds before next batch...")
                await asyncio.sleep(request_delay)

        # Run Ragas evaluation
        logger.info("Running Ragas evaluation")
        ragas_results = await self._run_ragas_evaluation(results, metrics)

        # Generate reports
        logger.info(f"Saving results to {report_dir}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        await self._generate_reports(
            task_name, bot_id, dataset_path, timestamp, results, ragas_results, Path(report_dir)
        )

        logger.info(f"Evaluation task completed: {task_name}")
        return {"task_name": task_name, "results": results, "ragas_results": ragas_results}

    async def run_all(self):
        """Run all evaluation tasks defined in configuration"""
        evaluations = self.config.get("evaluations", [])

        if not evaluations:
            logger.warning("No evaluation tasks defined in configuration")
            return

        logger.info(f"Found {len(evaluations)} evaluation tasks")

        for task_config in evaluations:
            await self.run_evaluation(task_config)

        logger.info("All evaluation tasks completed")


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run ApeRAG evaluations")
    parser.add_argument(
        "--config", type=str, help="Path to configuration file (default: config.yaml in module directory)"
    )
    args = parser.parse_args()

    runner = EvaluationRunner(args.config)
    await runner.run_all()


if __name__ == "__main__":
    asyncio.run(main())
