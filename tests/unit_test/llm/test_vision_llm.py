import asyncio
import os

import pytest

from aperag.llm.completion.completion_service import CompletionService
from tests.unit_test.llm.test_image_data import (
    one_dog_and_one_cat_on_the_bed_image_base64,
    two_cats_on_the_purplish_red_sofa_image_base64,
)


def test_vision_llm_completion():
    # Allow customization of vision model settings via environment variables
    provider = os.getenv("VISION_LLM_PROVIDER", "openai")
    vision_model = os.getenv("VISION_LLM_MODEL", "gpt-4o")
    base_url = os.getenv("VISION_LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("VISION_LLM_API_KEY")

    if not api_key:
        pytest.skip("VISION_LLM_API_KEY environment variable not set.")

    async def main():
        """Main function to test vision LLM completion."""
        if not api_key:
            print("Error: VISION_LLM_API_KEY environment variable not set.")
            return

        service = CompletionService(
            provider=provider,
            model=vision_model,
            base_url=base_url,
            api_key=api_key,
            vision=True,
            caching=False,  # Disable cache for testing
        )

        # --- Test with two cats image ---
        print("\n--- Testing with two cats image ---")
        images_two_cats = [two_cats_on_the_purplish_red_sofa_image_base64]
        prompt_two_cats = "How many cats are in the image? Respond with a single English word."
        try:
            response = await service.agenerate(history=[], prompt=prompt_two_cats, images=images_two_cats)
            print(f"   - Success! Response: '{response}'")
            assert "2" in response or "two" in response.lower(), (
                f"Expected '2' or 'two' in response, but got '{response}'"
            )
        except Exception as e:
            print(f"   - Failed: {e}")
            pytest.fail(f"Two cats test failed: {e}")

        # --- Test with one dog and one cat image ---
        print("\n--- Testing with one dog and one cat image ---")
        images_dog_cat = [one_dog_and_one_cat_on_the_bed_image_base64]
        prompt_dog_cat = "What animals are in the image? Respond with two words separated by a space."
        try:
            response = await service.agenerate(history=[], prompt=prompt_dog_cat, images=images_dog_cat)
            print(f"   - Success! Response: '{response}'")
            response_lower = response.lower()
            assert "dog" in response_lower and "cat" in response_lower, (
                f"Expected 'dog' and 'cat' in response, but got '{response}'"
            )
        except Exception as e:
            print(f"   - Failed: {e}")
            pytest.fail(f"Dog and cat test failed: {e}")

    # Run the async main function
    asyncio.run(main())


if __name__ == "__main__":
    test_vision_llm_completion()
