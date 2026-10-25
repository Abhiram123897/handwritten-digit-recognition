"""
train.py
========
Train a Convolutional Neural Network (CNN) on the MNIST dataset to recognize
handwritten digits (0-9).

What this script does:
  1. Loads and preprocesses the MNIST dataset (normalization + one-hot encoding).
  2. Builds and trains a CNN (targets >= 98% test accuracy; typically ~99%).
  3. Evaluates on the test set and prints a classification report.
  4. Saves the trained Keras model as  model/digit_model.h5  (required).
  5. Exports a lightweight  model/digit_model.tflite  for fast, low-memory serving.
  6. Generates training visualizations:
        - static/images/accuracy_loss.png     (accuracy & loss curves)
        - static/images/confusion_matrix.png  (confusion matrix heatmap)
  7. Writes  model/metrics.json  with performance metrics for the dashboard.

Run:
    python train.py
"""

import os
import json
import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

H5_PATH = os.path.join(MODEL_DIR, "digit_model.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "digit_model.tflite")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
ACC_LOSS_PATH = os.path.join(IMAGES_DIR, "accuracy_loss.png")
CM_PATH = os.path.join(IMAGES_DIR, "confusion_matrix.png")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------------------------------- #
# 1. Data loading & preprocessing
# --------------------------------------------------------------------------- #
def load_data():
    """Load MNIST and preprocess: reshape, normalize to [0,1], one-hot encode."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Add channel dimension -> (N, 28, 28, 1) and scale pixels to [0, 1]
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Keep integer labels for the confusion matrix later
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    print(f"Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")
    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat)


# --------------------------------------------------------------------------- #
# 2. Model definition
# --------------------------------------------------------------------------- #
def build_model():
    """Build a compact but strong CNN that reliably exceeds 98% on MNIST."""
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),

            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# 3. Visualizations
# --------------------------------------------------------------------------- #
def plot_history(history):
    """Save accuracy & loss curves (train vs validation)."""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, acc, "o-", label="Train")
    ax1.plot(epochs, val_acc, "s-", label="Validation")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss, "o-", label="Train")
    ax2.plot(epochs, val_loss, "s-", label="Validation")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(ACC_LOSS_PATH, dpi=110)
    plt.close(fig)
    print(f"Saved accuracy/loss plot -> {ACC_LOSS_PATH}")


def plot_confusion_matrix(y_true, y_pred):
    """Save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=110)
    plt.close(fig)
    print(f"Saved confusion matrix -> {CM_PATH}")


# --------------------------------------------------------------------------- #
# 4. TFLite export
# --------------------------------------------------------------------------- #
def export_tflite(model):
    """Convert the trained Keras model to a lightweight TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # size/latency optimization
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"Saved TFLite model -> {TFLITE_PATH} ({size_kb:.1f} KB)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 60)
    print("MNIST CNN Training")
    print("=" * 60)

    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = load_data()

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    history = model.fit(
        x_train,
        y_train_cat,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- Evaluation ----
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    if test_acc < 0.98:
        print("WARNING: Test accuracy below 98% target.")
    else:
        print("Target met: test accuracy >= 98%.")

    # Predictions for confusion matrix + report
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification report:")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    # ---- Save artifacts ----
    model.save(H5_PATH)
    print(f"Saved Keras model -> {H5_PATH}")

    export_tflite(model)

    plot_history(history)
    plot_confusion_matrix(y_test, y_pred)

    # ---- Metrics JSON (for dashboard) ----
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "epochs_trained": len(history.history["accuracy"]),
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "per_class": {
            str(d): {
                "precision": round(report_dict[str(d)]["precision"], 4),
                "recall": round(report_dict[str(d)]["recall"], 4),
                "f1": round(report_dict[str(d)]["f1-score"], 4),
            }
            for d in range(10)
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {METRICS_PATH}")

    print("\nDone. All artifacts generated successfully.")


if __name__ == "__main__":
    main()
