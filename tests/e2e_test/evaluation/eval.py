import csv
import json
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,  # 用LLM评估答案正确性
    answer_relevancy,  # 答案相关性评估生成答案对输入问题的相关程度，关注答案是否直接回答问题，避免冗余或无关信息。
    context_precision,  # 上下文精确度衡量检索到的上下文是否与问题高度相关，评估检索系统是否将最相关的文档或片段排在前面。
    context_recall,  # 上下文召回率评估检索到的上下文是否涵盖了回答问题所需的所有必要信息，通常与人工标注的“地面真相”（ground truth）答案比较。
    faithfulness,  # 忠实度衡量生成答案的事实准确性，即生成的答案是否忠实于检索到的上下文内容。它主要用于评估生成阶段，避免生成模型产生“幻觉”（hallucination），即生成未被上下文支持的信息。
)

# ---------------------------------------------
# Context Precision
# 依赖于 question 和 context。它衡量context与question的相关性，即上下文中有多少内容是对回答问题真正有用的。
# 适合优化检索系统的精确性，减少无关信息的干扰。例如，在法律文档检索中，避免返回无关条款。

# Context Recall
# 依赖于 question、context 和 ground truth。它衡量context是否包含了ground truth中所有必要的信息，以正确回答问题。
# 适合确保检索系统不遗漏关键信息。例如，在医疗问答中，确保返回所有相关诊断信息。

# Answer Relevance
# 依赖于 question 和 answer
# 用于衡量模型生成的答案与用户提问的相关性。评估模型生成的答案是否直接回答了用户的问题，而不是偏离主题或提供无关信息。

# Faithfulness
# 依赖于 context 和 answer
# 用于衡量模型生成的答案是否忠实于提供的上下文（context），即答案是否基于上下文中的信息，而没有引入未在上下文中出现的内容（幻觉）。

# Answer Correctness
# 依赖 answer、ground truth（有时也包括 question）。
# 使用LLM计算 answer 和 ground truth 之间的语义相似度。

# Answer Semantic Similarity
# 依赖 answer、ground truth
# 使用embedding + 余弦相似度，计算 answer 和 ground truth 之间的语义相似度。

# ---------------------------------------------


llm_for_eval = ChatOpenAI(
    base_url=os.environ["OPENAI_API_BASE"], api_key=os.environ["OPENAI_API_KEY"], model="gpt-3.5-turbo", temperature=0
)


# --- Step 2: Define Your RAG System Interface ---
def my_rag_system(question: str) -> Dict[str, Any]:
    """
    TODO: Dummy RAG implementation. 生产环境请替换为真实检索 + 生成逻辑
    """
    example_answer = f"This is a placeholder answer generated for the question: '{question}'."
    example_contexts = [
        f"Placeholder context relevant to '{question}'. Context 1.",
        f"Another placeholder context for '{question}'. Context 2.",
        "A generic piece of context that might or might not be relevant.",
    ]
    return {"answer": example_answer, "contexts": example_contexts}


# --- Step 3: Create a Simple Test Dataset ---
def create_tiny_testset() -> Dataset:
    print("Creating test dataset...")
    df = pd.read_csv("./datasets/qa-1300.csv")
    df = df.head(10)
    print(f"Load dataset: len={len(df)}")
    return Dataset.from_pandas(df)


# --- Step 4: Prepare Data for Ragas Evaluation ---
def prepare_evaluation_data(ds: Dataset) -> Dataset:
    print("Preparing data for Ragas evaluation...\n")

    question_list = []
    answer_list = []
    contexts_list = []
    ground_truth_list = []

    for test_case in ds:
        question = test_case["question"]
        ground_truth = test_case["answer"]

        rag_output = my_rag_system(question)

        question_list.append(question)
        answer_list.append(rag_output["answer"])
        contexts_list.append(rag_output["contexts"])
        ground_truth_list.append(ground_truth)

    return Dataset.from_dict(
        {
            "question": question_list,
            "answer": answer_list,
            "contexts": contexts_list,
            "ground_truth": ground_truth_list,
        }
    )


def print_evaluation_output(ds: Dataset, output_dir: str = "."):
    def preprocess_row(row):
        # 处理 contexts 列表，转换为 JSON 字符串格式
        if "contexts" in row and isinstance(row["contexts"], list):
            row["contexts"] = json.dumps(row["contexts"], ensure_ascii=False)
        # 处理其他字段中的换行符
        for k, v in row.items():
            if isinstance(v, str):
                row[k] = v.replace("\n", "\\n")
        return row

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = f"eval-output-{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)

    # 预处理数据集
    processed_ds = ds.map(preprocess_row)

    # 导出CSV
    processed_ds.to_csv(
        output_path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", doublequote=True, encoding="utf-8"
    )

    print(f"评估结果已导出到: {output_path}")


# --- Step 5: Run Ragas Evaluation ---
def evaluate_rag_with_ragas(dataset: Dataset):
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
    results = evaluate(dataset=dataset, metrics=metrics, llm=llm_for_eval, raise_exceptions=False)
    return results


def print_metrics(evaluation_results: Dataset, output_dir: str = "."):
    if not evaluation_results:
        print("[Error] Ragas evaluation failed.")
        return

    print("\n--- RAGAS 评分汇总 ---")
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    filename = f"eval-metrics-{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)
    df = evaluation_results.to_pandas()
    print(df)
    df.to_csv(output_path, index=False)


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- RAG System Evaluation Script ---")
    ds = create_tiny_testset()
    eval_dataset = prepare_evaluation_data(ds)
    # print_evaluation_output(eval_dataset)

    eval_results = evaluate_rag_with_ragas(eval_dataset)
    print_metrics(eval_results)

    print("\n--- Script execution finished ---")
