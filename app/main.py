import time
import logging
import numpy as np
import tensorflow as tf

from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response


logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Cats vs Dogs Classifier")

model = tf.keras.models.load_model(
    "models/cat_dog_model.h5"
)

request_count = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Time taken for prediction"
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "cat_dog_model"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    image_bytes = await file.read()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    if probability >= 0.5:
        label = "Dog"
        confidence = probability
    else:
        label = "Cat"
        confidence = 1 - probability

    request_count.inc()

    prediction_latency.observe(
        time.time() - start_time
    )

    logging.info(
        f"Prediction completed: {label}"
    )

    return {
        "prediction": label,
        "confidence": round(
            confidence,
            4
        )
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )