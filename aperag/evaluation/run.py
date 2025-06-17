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
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import yaml
from datasets import Dataset
from langchain_openai import ChatOpenAI
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
        """Initialize the evaluation runner with configuration"""
        self.config_path = config_path or Path(__file__).parent / "config.yaml"
        self.config = self._load_config()
        self.llm_for_eval = self._setup_llm_for_eval()

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
                api_url = f"{host}/v1/chat/completions"
            else:
                # Assume it's already the correct base
                api_url = f"{base_url.rstrip('/')}/v1/chat/completions"

            logger.debug(f"Using API URL: {api_url}")

            # Prepare request
            headers = {"Content-Type": "application/json"}
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"

            request_body = {"messages": [{"role": "user", "content": question}], "model": "aperag", "stream": False}

            params = {"bot_id": bot_id}

            # Make HTTP request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url, json=request_body, headers=headers, params=params, timeout=timeout
                )
                response.raise_for_status()

            result = response.json()
            logger.debug(f"API Response JSON: {result}")

            # Check if it's an error response
            if "error" in result:
                error_msg = result["error"].get("message", "Unknown error")
                logger.error(f"API returned error: {error_msg}")
                return {"response": f"Error: {error_msg}", "context": [], "error": error_msg}

            # Parse OpenAI-style response
            if "choices" in result and len(result["choices"]) > 0:
                bot_response = result["choices"][0]["message"]["content"]
                logger.debug(f"Bot response content: '{bot_response[:100]}...' (length: {len(bot_response)})")
            else:
                logger.warning("No choices found in API response")
                bot_response = ""

            # Parse references from response if embedded
            context = []
            references_marker = "参考文档："  # DOC_QA_REFERENCES constant

            if references_marker in bot_response:
                # Split response and references
                parts = bot_response.split(references_marker)
                if len(parts) > 1:
                    bot_response = parts[0].strip()
                    try:
                        references_json = parts[1].strip()
                        references = json.loads(references_json)
                        # Extract text from references as context
                        context = [ref.get("text", "") for ref in references if ref.get("text")]
                        logger.debug(f"Extracted {len(context)} context items from references")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse references JSON")

            # Ensure context is a list
            if not isinstance(context, list):
                context = [str(context)] if context else []

            return {"response": bot_response, "context": context, "raw_result": result}

        except Exception as e:
            logger.error(f"Error calling bot API: {e}")
            return {"response": f"Error: {str(e)}", "context": [], "error": str(e)}

    async def _process_dataset(self, bot_id: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process dataset by calling bot API for each question"""
        results = []
        advanced_config = self.config.get("advanced", {})
        batch_size = advanced_config.get("batch_size", 5)
        request_delay = advanced_config.get("request_delay", 1)

        logger.info(f"Processing {len(df)} questions with bot {bot_id}")

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            batch_results = []

            # Process batch concurrently
            tasks = []
            for _, row in batch.iterrows():
                task = self._call_bot_api(bot_id, row["question"])
                tasks.append(task)

            batch_responses = await asyncio.gather(*tasks)

            # Combine results
            for (_, row), response in zip(batch.iterrows(), batch_responses):
                result = {
                    "question": row["question"],
                    "ground_truth": row["answer"],
                    "response": response["response"],
                    "context": response["context"],
                }
                batch_results.append(result)
                results.append(result)

            logger.info(f"Processed {min(i + batch_size, len(df))}/{len(df)} questions")

            # Delay between batches
            if i + batch_size < len(df):
                await asyncio.sleep(request_delay)

        return results

    def _prepare_ragas_dataset(self, results: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for Ragas evaluation"""
        logger.info("Preparing data for Ragas evaluation")

        # Extract data for Ragas format
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for result in results:
            questions.append(result["question"])
            answers.append(result["response"])
            # Ensure contexts is a list of strings
            context = result["context"]
            if isinstance(context, str):
                contexts.append([context])
            elif isinstance(context, list):
                contexts.append([str(c) for c in context])
            else:
                contexts.append([str(context)])
            ground_truths.append(result["ground_truth"])

        # Create Ragas dataset
        dataset_dict = {"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}

        return Dataset.from_dict(dataset_dict)

    def _get_metrics(self, metric_names: List[str]):
        """Get Ragas metric objects from names"""
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }

        metrics = []
        for name in metric_names:
            if name in metric_map:
                metrics.append(metric_map[name])
            else:
                logger.warning(f"Unknown metric: {name}")

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
                    summary["metrics"][col] = {
                        "mean": float(eval_df[col].mean()),
                        "std": float(eval_df[col].std()),
                        "min": float(eval_df[col].min()),
                        "max": float(eval_df[col].max()),
                    }

        json_path = report_dir / f"evaluation_summary_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
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

    def _generate_markdown_report(
        self,
        task_name: str,
        bot_id: str,
        dataset_path: str,
        timestamp: str,
        results: List[Dict],
        ragas_results: Optional[Dict],
        output_path: Path,
    ) -> None:
        """Generate a markdown evaluation report"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# ApeRAG Evaluation Report\n\n")
            f.write(f"**Task Name:** {task_name}\n\n")
            f.write(f"**Bot ID:** {bot_id}\n\n")
            f.write(f"**Dataset:** {dataset_path}\n\n")
            f.write(f"**Timestamp:** {timestamp}\n\n")
            f.write(f"**Total Samples:** {len(results)}\n\n")

            if ragas_results:
                f.write("## Ragas Evaluation Metrics\n\n")
                if isinstance(ragas_results, list) and len(ragas_results) > 0:
                    # Calculate average scores
                    metrics = {}
                    for result in ragas_results:
                        for key, value in result.items():
                            if isinstance(value, (int, float)) and key not in [
                                "question",
                                "answer",
                                "contexts",
                                "ground_truth",
                            ]:
                                if key not in metrics:
                                    metrics[key] = []
                                metrics[key].append(value)

                    for metric, values in metrics.items():
                        avg_score = sum(values) / len(values)
                        f.write(f"- **{metric.replace('_', ' ').title()}**: {avg_score:.3f}\n")
                    f.write("\n")

            f.write("## Sample Results\n\n")

            for i, result in enumerate(results[:10], 1):  # Show first 10 samples
                f.write(f"### Sample {i}\n\n")
                f.write(f"**Question:** {result['question']}\n\n")
                f.write(f"**Ground Truth:** {result['ground_truth']}\n\n")
                f.write(f"**Bot Response:** {result['response']}\n\n")
                if result.get("context"):
                    f.write(f"**Context:** {result['context']}\n\n")
                f.write("---\n\n")

            if len(results) > 10:
                f.write(f"*({len(results) - 10} more samples not shown)*\n\n")

    async def _run_ragas_evaluation(self, results: List[Dict], metrics: List[str]) -> Optional[Dict]:
        """Run Ragas evaluation on the results"""
        if not metrics:
            return None

        try:
            # Prepare data for Ragas
            logger.info("Preparing data for Ragas evaluation")
            ragas_dataset = self._prepare_ragas_dataset(results)
            ragas_metrics = self._get_metrics(metrics)

            # Run evaluation
            eval_results = evaluate(
                dataset=ragas_dataset, metrics=ragas_metrics, llm=self.llm_for_eval, raise_exceptions=False
            )
            logger.info("Ragas evaluation completed")
            return eval_results.to_pandas().to_dict("records")

        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}")
            return None

    def _generate_reports(
        self,
        task_name: str,
        bot_id: str,
        dataset_path: str,
        results: List[Dict],
        ragas_results: Optional[Dict],
        report_dir: str,
        timestamp: str,
    ) -> Dict[str, str]:
        """Generate evaluation reports in multiple formats"""
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        reports = {}

        # CSV Report
        csv_path = report_dir / f"evaluation_report_{timestamp}.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        reports["csv"] = str(csv_path)
        logger.info(f"Saved detailed report to {csv_path}")

        # JSON Summary
        summary = {
            "task_name": task_name,
            "bot_id": bot_id,
            "dataset_path": dataset_path,
            "timestamp": timestamp,
            "total_samples": len(results),
            "ragas_results": ragas_results,
            "results": results,
        }

        json_path = report_dir / f"evaluation_summary_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        reports["json"] = str(json_path)
        logger.info(f"Saved summary to {json_path}")

        # Markdown Report
        md_path = report_dir / f"evaluation_report_{timestamp}.md"
        self._generate_markdown_report(task_name, bot_id, dataset_path, timestamp, results, ragas_results, md_path)
        reports["markdown"] = str(md_path)
        logger.info(f"Saved markdown report to {md_path}")

        # Intermediate Results (for debugging)
        intermediate_path = report_dir / f"intermediate_results_{timestamp}.json"
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        reports["intermediate"] = str(intermediate_path)
        logger.info(f"Saved intermediate results to {intermediate_path}")

        return reports

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
        reports = self._generate_reports(task_name, bot_id, dataset_path, results, ragas_results, report_dir, timestamp)

        logger.info(f"Evaluation task completed: {task_name}")
        return {"task_name": task_name, "results": results, "ragas_results": ragas_results, "reports": reports}

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
