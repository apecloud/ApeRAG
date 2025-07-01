#!/usr/bin/env python3
"""
Simplified script to export knowledge graph data using LightRAG.export_for_kg_eval method.

Environment Variables:
    Required Database Connection:
    - POSTGRES_HOST: PostgreSQL host (default: localhost)
    - POSTGRES_PORT: PostgreSQL port (default: 5432)
    - POSTGRES_USER: PostgreSQL username (default: postgres)
    - POSTGRES_PASSWORD: PostgreSQL password (default: postgres)
    - POSTGRES_DB: PostgreSQL database name (default: postgres)

    Required Workspace Configuration:
    - KG_WORKSPACE_ID: The workspace ID to extract data from (required)
    - KG_MODEL_NAME: Model name for output filename (optional, default: unknown)

    Optional Extraction Parameters:
    - KG_SAMPLE_SIZE: Number of entities to sample (default: 50)
    - KG_OUTPUT_FILE: Output filename (optional, auto-generated if not provided)
    - KG_INCLUDE_SOURCE_TEXTS: Whether to include source texts (default: true)

Example usage:
    export KG_WORKSPACE_ID="col44d209a5925b9405"
    export KG_MODEL_NAME="google/gemini-2.5-flash"
    export KG_SAMPLE_SIZE="30"
    python export_kg_eval.py
"""

import asyncio
import json
import os
from typing import Any, Dict

from aperag.graph.lightrag.lightrag import LightRAG


def get_env_config() -> Dict[str, Any]:
    """Get configuration from environment variables."""

    # Required workspace configuration
    workspace_id = os.getenv("KG_WORKSPACE_ID", "col44d209a5925b9405")
    if not workspace_id:
        raise ValueError("KG_WORKSPACE_ID environment variable is required")

    # Optional parameters
    model_name = os.getenv("KG_MODEL_NAME", "unknown")
    sample_size = int(os.getenv("KG_SAMPLE_SIZE", "30000"))
    output_file = os.getenv("KG_OUTPUT_FILE")
    include_source_texts = os.getenv("KG_INCLUDE_SOURCE_TEXTS", "true").lower() in ("true", "1", "yes")

    # Auto-generate output filename if not provided
    if not output_file:
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        output_file = f"kg_eval_{safe_model_name}.json"

    return {
        "workspace_id": workspace_id,
        "model_name": model_name,
        "sample_size": sample_size,
        "output_file": output_file,
        "include_source_texts": include_source_texts,
    }


async def main():
    """Main function to export data using LightRAG.export_for_kg_eval."""

    # Load environment variables
    import dotenv

    dotenv.load_dotenv(".env")

    try:
        config = get_env_config()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nRequired environment variables:")
        print("  KG_WORKSPACE_ID - The workspace ID to extract data from")
        print("\nOptional environment variables:")
        print("  KG_MODEL_NAME - Model name for output filename")
        print("  KG_SAMPLE_SIZE - Number of entities to sample (default: 50)")
        print("  KG_OUTPUT_FILE - Output filename (auto-generated if not provided)")
        print("  KG_INCLUDE_SOURCE_TEXTS - Include source texts (default: true)")
        return

    # Check if required database environment variables are set
    required_db_vars = ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"]
    missing_vars = [var for var in required_db_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Missing database environment variables: {missing_vars}")
        print("Please set these in your .env file")
        return

    workspace_id = config["workspace_id"]
    model_name = config["model_name"]
    sample_size = config["sample_size"]
    output_file = config["output_file"]
    include_source_texts = config["include_source_texts"]

    print("Exporting K-Eval knowledge graph data:")
    print(f"  Workspace ID: {workspace_id}")
    print(f"  Model Name: {model_name}")
    print(f"  Sample Size: {sample_size}")
    print(f"  Output File: {output_file}")
    print(f"  Include Source Texts: {include_source_texts}")
    print()

    try:
        # Initialize LightRAG instance
        rag = LightRAG(
            workspace=workspace_id,
            # Use PostgreSQL-based storages as configured in the original script
            kv_storage="PGOpsSyncKVStorage",
            vector_storage="PGOpsSyncVectorStorage",
            graph_storage="PGOpsSyncGraphStorage",
        )

        # Initialize storages
        await rag.initialize_storages()

        try:
            # Export data using the new method
            kg_data = await rag.export_for_kg_eval(sample_size=sample_size, include_source_texts=include_source_texts)

            # Save to JSON file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(kg_data, f, ensure_ascii=False, indent=2)

            print(f"\n✅ Successfully exported data to {output_file}")

            # Print summary
            print("\nSummary:")
            print(f"  📊 Entities: {len(kg_data['entities'])}")
            print(f"  🔗 Relationships: {len(kg_data['relationships'])}")
            if "source_texts" in kg_data:
                print(f"  📝 Source Texts: {len(kg_data['source_texts'])}")

        finally:
            # Clean up storages
            await rag.finalize_storages()

    except Exception as e:
        print(f"❌ Error exporting data: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
