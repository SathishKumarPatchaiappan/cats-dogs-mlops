import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 3


def load_data():
    train_data = tf.keras.utils.image_dataset_from_directory(
        "data/processed/train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        "data/processed/val",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    test_data = tf.keras.utils.image_dataset_from_directory(
        "data/processed/test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    return train_data, val_data, test_data


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(
            1.0 / 255,
            input_shape=(224, 224, 3)
        ),

        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),

        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_accuracy_graph(history):
    os.makedirs("artifacts", exist_ok=True)

    plt.figure()

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.savefig("artifacts/training_accuracy.png")
    plt.close()


def save_confusion_matrix(model, test_data):
    actual = []
    predicted = []

    for images, labels in test_data:
        predictions = model.predict(images, verbose=0)

        predictions = (
            predictions > 0.5
        ).astype(int).flatten()

        actual.extend(
            labels.numpy().astype(int).flatten()
        )

        predicted.extend(predictions)

    matrix = confusion_matrix(actual, predicted)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Cat", "Dog"]
    )

    display.plot()

    plt.title("Cats vs Dogs Confusion Matrix")
    plt.savefig("artifacts/confusion_matrix.png")
    plt.close()


def train():
    train_data, val_data, test_data = load_data()

    model = build_model()

    mlflow.set_experiment("cats-dogs-classification")

    with mlflow.start_run():
        mlflow.log_param("image_size", "224x224")
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("optimizer", "adam")

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS
        )

        test_loss, test_accuracy = model.evaluate(test_data)

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        os.makedirs("models", exist_ok=True)

        model.save("models/cat_dog_model.h5")

        save_accuracy_graph(history)
        save_confusion_matrix(model, test_data)

        mlflow.log_artifact(
            "artifacts/training_accuracy.png"
        )

        mlflow.log_artifact(
            "artifacts/confusion_matrix.png"
        )

        print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    train()