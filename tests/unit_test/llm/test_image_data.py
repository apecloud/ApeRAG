# -*- coding: utf-8 -*-
import base64
import os

# Base64 encoded image data for testing


def _get_image_base64(image_name: str) -> str:
    image_path = os.path.join(os.path.dirname(__file__), "images", image_name)
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"


# A picture of two cats
two_cats_on_the_purplish_red_sofa_image_base64 = _get_image_base64("two_cats_on_the_purplish_red_sofa.jpg")

# A picture of one dog and one cat
one_dog_and_one_cat_on_the_bed_image_base64 = _get_image_base64("one_dog_and_one_cat_on_the_bed.jpg")
