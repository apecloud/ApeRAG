import asyncio
import os

import numpy as np
import pytest

from aperag.llm.embed.embedding_service import EmbeddingService
from tests.unit_test.llm.test_image_data import (
    one_dog_and_one_cat_on_the_bed_image_base64,
    two_cats_on_the_purplish_red_sofa_image_base64,
)


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def test_image_embedding():
    if not os.getenv("JINA_API_KEY"):
        pytest.skip("JINA_API_KEY environment variable not set.")

    async def main():
        """Main function to test image embedding."""
        jina_api_key = os.getenv("JINA_API_KEY")
        if not jina_api_key:
            print("Error: JINA_API_KEY environment variable not set.")
            print("Please create a .env file with JINA_API_KEY='your-key' or set it in your environment.")
            return

        # Use a Jina AI multimodal model that supports image inputs
        # You might need to adjust the model name based on Jina's offerings
        embedding_model = "jina-embeddings-v4"

        service = EmbeddingService(
            embedding_provider="jina_ai",
            embedding_model=embedding_model,
            # For many providers, LiteLLM knows the api_base, but we pass it for consistency
            embedding_service_url="https://api.jina.ai/v1",
            embedding_service_api_key=jina_api_key,
            embedding_max_chunks_in_batch=1,
            multimodal=True,
        )

        # A sample image URL from the COCO dataset (a picture of two cats)
        # image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        print(f"--- Testing Image Embedding with Jina AI ({embedding_model}) ---")
        # print(f"Image URL: {image_url}\n")

        # --- Test Sync Method ---
        print("1. Testing synchronous embedding (embed_documents)...")
        try:
            embeddings = service.embed_documents([two_cats_on_the_purplish_red_sofa_image_base64])
            print(f"   - Success! Embedding vector dimension: {len(embeddings[0])}")
            # print(f"   - Vector (first 5 dims): {embeddings[0][:5]}")
        except Exception as e:
            print(f"   - Failed: {e}")
            raise

        print("\n" + "-" * 20 + "\n")

        # --- Test Async Method ---
        print("2. Testing asynchronous embedding (aembed_query)...")
        try:
            aembedding = await service.aembed_query(two_cats_on_the_purplish_red_sofa_image_base64)
            print(f"   - Success! Embedding vector dimension: {len(aembedding)}")
            # print(f"   - Vector (first 5 dims): {aembedding[:5]}")
        except Exception as e:
            print(f"   - Failed: {e}")
            raise

        print("\n" + "-" * 20 + "\n")

        # --- Test Batch Method ---
        print("3. Testing batch embedding (embed_documents)...")
        images = [
            two_cats_on_the_purplish_red_sofa_image_base64,
            one_dog_and_one_cat_on_the_bed_image_base64,
        ]
        image_embeddings = []
        try:
            image_embeddings = service.embed_documents(images)
            print(f"   - Success! Embedded {len(image_embeddings)} images.")
            for i, emb in enumerate(image_embeddings):
                print(f"     - Image {i + 1} vector dimension: {len(emb)}")
        except Exception as e:
            print(f"   - Failed: {e}")
            raise

        print("\n" + "-" * 20 + "\n")

        # --- Test Text-Image Similarity ---
        print("4. Testing text-image similarity...")
        if not image_embeddings:
            print("   - Skipping test because image embeddings were not generated.")
            return

        text_to_embed = "two cats on the sofa"
        print(f"   - Embedding text: '{text_to_embed}'")
        try:
            text_embedding = service.embed_query(text_to_embed)
            print(f"   - Success! Text embedding vector dimension: {len(text_embedding)}")

            # Calculate similarity
            sim_two_cats = cosine_similarity(text_embedding, image_embeddings[0])
            sim_dog_cat = cosine_similarity(text_embedding, image_embeddings[1])

            print(f"   - Similarity with 'two cats' image: {sim_two_cats:.4f}")
            print(f"   - Similarity with 'one dog, one cat' image: {sim_dog_cat:.4f}")

            if sim_two_cats > sim_dog_cat:
                print("   - Conclusion: The text is more similar to the 'two cats' image, as expected.")
            else:
                print("   - Conclusion: The text is more similar to the 'one dog, one cat' image.")
                pytest.fail("Expected the text to be more similar to the 'two cats' image")

        except Exception as e:
            print(f"   - Failed to embed text or calculate similarity: {e}")
            raise

        print("\n" + "-" * 20 + "\n")

        # --- Test Text-Image Similarity, Case 2 ---
        print("5. Testing text-image similarity, case 2...")
        if not image_embeddings:
            print("   - Skipping test because image embeddings were not generated.")
            return

        text_to_embed = "one dog and one cat"
        print(f"   - Embedding text: '{text_to_embed}'")
        try:
            text_embedding = service.embed_query(text_to_embed)
            print(f"   - Success! Text embedding vector dimension: {len(text_embedding)}")

            # Calculate similarity
            sim_two_cats = cosine_similarity(text_embedding, image_embeddings[0])
            sim_dog_cat = cosine_similarity(text_embedding, image_embeddings[1])

            print(f"   - Similarity with 'two cats' image: {sim_two_cats:.4f}")
            print(f"   - Similarity with 'one dog, one cat' image: {sim_dog_cat:.4f}")

            if sim_two_cats > sim_dog_cat:
                print("   - Conclusion: The text is more similar to the 'two cats' image.")
                pytest.fail("Expected the text to be more similar to the 'one dog, one cat' image")
            else:
                print("   - Conclusion: The text is more similar to the 'one dog, one cat' image, as expected.")

        except Exception as e:
            print(f"   - Failed to embed text or calculate similarity: {e}")
            raise

    # Run the async main function
    asyncio.run(main())
