import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import PIL.Image
import requests
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, ImageDocument
from PIL import Image

import config.settings as settings

OCR_SERVER_URL = settings.PADDLEOCR_HOST


def read_image_meaning(image: PIL.Image.Image, ctx: str) -> (str, str):
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    inputs = processor(image, ctx, return_tensors="pt")
    out = model.generate(**inputs)

    return processor.decode(out[0], skip_special_tokens=True)


def read_image_text(path) -> str:
    def image_to_base64(image_path):
        with Image.open(image_path) as image:
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_byte = buffered.getvalue()
            img_base64 = base64.b64encode(img_byte)
            return img_base64.decode()

    data = {'images': [image_to_base64(str(path))]}
    headers = {"Content-type": "application/json"}
    url = OCR_SERVER_URL + "/predict/ocr_system"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    data = json.loads(r.text)

    texts = [item['text'] for group in data['results'] for item in group if 'text' in item]

    res = ""

    for text in texts:
        res += text

    return res


class ComposeImageReader(BaseReader):
    def load_data(self, file: Path, metadata: Optional[Dict] = None) -> List[Document]:
        from llama_index.img_utils import img_2_b64
        from PIL import Image

        image = Image.open(file).convert("RGB")
        text = read_image_text(file)
        # meaning = read_image_meaning(image, text)
        meaning = ""
        image_bytes = img_2_b64(image)
        return [
            ImageDocument(
                text="%s\n%s" % (text, meaning),
                image=image_bytes,
                metadata=metadata or {},
            )
        ]
