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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from aperag.api import api

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Main class for running RAG evaluations"""

    def __init__(self, config_path: str = None):
        """Initialize the evaluation runner with configuration"""
        self.config_path = config_path or Path(__file__).parent / "config.yaml"
        self.config = self._load_config()
        self.api_client = self._setup_api_client()
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

    def _setup_api_client(self) -> api.DefaultApi:
        """Setup API client for calling bot endpoints"""
        api_config = self.config["api"]
        base_url = api_config["base_url"]

        # Configure API client
        configuration = api.Configuration()
        configuration.host = base_url

        # Set API token if available
        api_token = api_config.get("api_token") or os.environ.get("APERAG_API_TOKEN")
        if api_token:
            configuration.api_key["Authorization"] = api_token
            configuration.api_key_prefix["Authorization"] = "Bearer"

        api_client = api.ApiClient(configuration)
        return api.DefaultApi(api_client)

    def _setup_llm_for_eval(self) -> ChatOpenAI:
        """Setup LLM for Ragas evaluation"""
        llm_config = self.config["llm_for_eval"]
        return ChatOpenAI(
            base_url=llm_config["api_base"],
            api_key=llm_config["api_key"],
            model=llm_config["model"],
            temperature=llm_config["temperature"],
        )

    def _load_dataset(self, dataset_path: str, max_samples: Optional[int] = None) -> pd.DataFrame:
        """Load dataset from CSV or JSON file"""
        logger.info(f"Loading dataset from {dataset_path}")

        if dataset_path.endswith(".csv"):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(".json"):
            df = pd.read_json(dataset_path)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path}")

        # Validate required columns
        required_columns = ["question", "answer"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}")

        # Limit samples if specified
        if max_samples:
            df = df.head(max_samples)
            logger.info(f"Limited dataset to {max_samples} samples")

        logger.info(f"Loaded {len(df)} samples from dataset")
        return df

    async def _call_bot_api(self, bot_id: str, question: str) -> Dict[str, Any]:
        """Call bot API and get response with context"""
        try:
            # Prepare request body
            request_body = {"messages": [{"role": "user", "content": question}], "stream": False}

            # Call API
            advanced_config = self.config.get("advanced", {})
            timeout = advanced_config.get("request_timeout", 30)

            # Make HTTP request directly since we need custom parameters
            import httpx

            api_config = self.config["api"]
            base_url = api_config["base_url"]
            url = f"{base_url}/chat/completions?bot_id={bot_id}"

            headers = {}
            api_token = api_config.get("api_token") or os.environ.get("APERAG_API_TOKEN")
            if api_token:
                headers["Authorization"] = f"Bearer {api_token}"

            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=request_body, headers=headers, timeout=timeout)
                response.raise_for_status()

            result = response.json()

            # Extract response and context
            bot_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")

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
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse references JSON")

            # If no context found, check other possible fields
            if not context:
                context = result.get("context", [])
            if not context and "sources" in result:
                context = result["sources"]
            if not context and "references" in result:
                refs = result["references"]
                if isinstance(refs, list):
                    context = [ref.get("text", "") for ref in refs if isinstance(ref, dict) and ref.get("text")]

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

    def _generate_markdown_report(self, summary: Dict[str, Any], df_results: pd.DataFrame, output_path: Path):
        """Generate a markdown report"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# ApeRAG Evaluation Report\n\n")
            f.write(f"**Task Name:** {summary['task_name']}\n\n")
            f.write(f"**Bot ID:** {summary['bot_id']}\n\n")
            f.write(f"**Dataset:** {summary['dataset_path']}\n\n")
            f.write(f"**Timestamp:** {summary['timestamp']}\n\n")
            f.write(f"**Total Samples:** {summary['total_samples']}\n\n")

            if summary["metrics"]:
                f.write("## Metrics Summary\n\n")
                f.write("| Metric | Mean | Std | Min | Max |\n")
                f.write("|--------|------|-----|-----|-----|\n")
                for metric, stats in summary["metrics"].items():
                    f.write(
                        f"| {metric} | {stats['mean']:.3f} | {stats['std']:.3f} | "
                        f"{stats['min']:.3f} | {stats['max']:.3f} |\n"
                    )
                f.write("\n")

            f.write("## Sample Results\n\n")
            # Show first 5 results as examples
            for i, row in df_results.head(5).iterrows():
                f.write(f"### Sample {i + 1}\n\n")
                f.write(f"**Question:** {row['question']}\n\n")
                f.write(f"**Ground Truth:** {row['ground_truth']}\n\n")
                f.write(f"**Bot Response:** {row['response']}\n\n")
                if "context" in row and row["context"]:
                    f.write(f"**Context:** {row['context']}\n\n")
                f.write("---\n\n")

    async def run_evaluation(self, task_config: Dict[str, Any]):
        """Run a single evaluation task"""
        task_name = task_config["task_name"]
        logger.info(f"Starting evaluation task: {task_name}")

        try:
            # Load dataset
            df = self._load_dataset(task_config["dataset_path"], task_config.get("max_samples"))

            # Process dataset with bot API
            results = await self._process_dataset(task_config["bot_id"], df)

            # Evaluate with Ragas if metrics are specified
            eval_results = None
            if "metrics" in task_config and task_config["metrics"]:
                logger.info("Running Ragas evaluation")
                ragas_dataset = self._prepare_ragas_dataset(results)
                metrics = self._get_metrics(task_config["metrics"])

                try:
                    eval_results = evaluate(
                        dataset=ragas_dataset, metrics=metrics, llm=self.llm_for_eval, raise_exceptions=False
                    )
                    logger.info("Ragas evaluation completed")
                except Exception as e:
                    logger.error(f"Ragas evaluation failed: {e}")

            # Save results
            report_dir = Path(task_config["report_dir"])
            self._save_results(results, eval_results, task_config, report_dir)

            logger.info(f"Evaluation task completed: {task_name}")

        except Exception as e:
            logger.error(f"Evaluation task failed: {task_name}")
            logger.exception(e)
            raise

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
