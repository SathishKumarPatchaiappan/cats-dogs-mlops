from PIL import Image
import numpy as np


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32)
    return image_array


def test_preprocess_image():
    image = Image.new("RGB", (300, 300))

    result = preprocess_image(image)

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32