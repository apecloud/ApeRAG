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
Test script for ApeRAG evaluation module

This script tests basic functionality without requiring a full evaluation run.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from aperag.evaluation.run import EvaluationRunner


async def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")

    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.example.yaml"

    runner = EvaluationRunner(str(config_path))
    print("✓ Configuration loaded successfully")
    print(f"  API base URL: {runner.config['api']['base_url']}")
    print(f"  Number of evaluation tasks: {len(runner.config.get('evaluations', []))}")


async def test_dataset_loading():
    """Test dataset loading"""
    print("\nTesting dataset loading...")

    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.example.yaml"

    runner = EvaluationRunner(str(config_path))

    # Test with the example dataset
    dataset_path = "./tests/evaluation/datasets/qa-1300.csv"
    if Path(dataset_path).exists():
        df = runner._load_dataset(dataset_path, max_samples=5)
        print("✓ Dataset loaded successfully")
        print(f"  Number of samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First question: {df.iloc[0]['question'][:50]}...")
    else:
        print(f"✗ Dataset not found at {dataset_path}")


async def test_api_connection():
    """Test API connection (without actually calling bot)"""
    print("\nTesting API connection setup...")

    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent / "config.example.yaml"

    runner = EvaluationRunner(str(config_path))

    # Just test that the API client is configured
    if runner.api_client:
        print("✓ API client configured successfully")
        print(f"  Host: {runner.api_client.api_client.configuration.host}")
    else:
        print("✗ Failed to configure API client")


async def main():
    """Run all tests"""
    print("=== ApeRAG Evaluation Module Test ===\n")

    try:
        await test_config_loading()
        await test_dataset_loading()
        await test_api_connection()

        print("\n=== All tests completed ===")
        print("\nTo run a full evaluation, use:")
        print("  python -m aperag.evaluation.run")
        print("\nOr use the Makefile:")
        print("  make evaluate")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
