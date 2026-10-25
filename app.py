"""
app.py
======
Flask backend for the Handwritten Digit Recognition app.

Features
--------
- Loads the trained model ONCE at startup (prefers the lightweight TFLite model,
  falls back to the Keras .h5 model).
- Accepts images from a drawing canvas (base64 data URL) or an uploaded file.
- Preprocesses images the "MNIST way" (grayscale, invert, crop to digit, center
  on a 28x28 canvas) using OpenCV for robust predictions.
- Returns predicted digit, confidence, and the full 0-9 probability distribution.
- Stores every prediction in a SQLite database and exposes history endpoints
  (list, clear, export CSV).
- Serves the landing page and dashboard.

Run (dev):   python app.py
Run (prod):  gunicorn app:app
"""

import os
import io
import csv
import json
import base64
import sqlite3
import datetime

import numpy as np
import cv2
from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    send_file,
    g,
)
from flask_cors import CORS

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_DIR = os.path.join(BASE_DIR, "database")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

H5_PATH = os.path.join(MODEL_DIR, "digit_model.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "digit_model.tflite")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
DB_PATH = os.path.join(DB_DIR, "history.db")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB upload cap

for d in (MODEL_DIR, DB_DIR, UPLOAD_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)


# --------------------------------------------------------------------------- #
# Model loading (once, at startup)
# --------------------------------------------------------------------------- #
class ModelWrapper:
    """
    Wraps either a TFLite interpreter or a Keras model behind a single
    `predict(batch)` interface returning a (N, 10) probability array.
    """

    def __init__(self):
        self.backend = None          # "tflite" or "keras"
        self._tflite = None
        self._in = None
        self._out = None
        self._keras = None
        self._load()

    def _load(self):
        # Prefer a standalone TFLite interpreter (small, fast, low memory).
        # Order of preference:
        #   1. ai-edge-litert   (current standalone LiteRT runtime)
        #   2. tflite_runtime   (older standalone runtime)
        #   3. tensorflow.lite  (full TF — heavy; last resort)
        # Using a standalone runtime keeps memory well under Render's 512 MB
        # free-tier limit; importing full TensorFlow here would exceed it.
        if os.path.exists(TFLITE_PATH):
            try:
                interpreter = None
                for loader in ("ai_edge_litert", "tflite_runtime", "tensorflow"):
                    try:
                        if loader == "ai_edge_litert":
                            from ai_edge_litert.interpreter import Interpreter  # type: ignore
                            interpreter = Interpreter(model_path=TFLITE_PATH)
                        elif loader == "tflite_runtime":
                            import tflite_runtime.interpreter as tflite  # type: ignore
                            interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
                        else:
                            import tensorflow as tf  # heavy fallback
                            interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
                        print(f"[model] TFLite interpreter via '{loader}'")
                        break
                    except ImportError:
                        continue

                if interpreter is None:
                    raise ImportError("No TFLite interpreter runtime available.")

                interpreter.allocate_tensors()
                self._tflite = interpreter
                self._in = interpreter.get_input_details()[0]
                self._out = interpreter.get_output_details()[0]
                self.backend = "tflite"
                print(f"[model] Loaded TFLite model: {TFLITE_PATH}")
                return
            except Exception as exc:  # pragma: no cover
                print(f"[model] TFLite load failed ({exc}); trying Keras .h5")

        # Fallback: Keras .h5 ---------------------------------------------- #
        if os.path.exists(H5_PATH):
            import tensorflow as tf
            self._keras = tf.keras.models.load_model(H5_PATH)
            self.backend = "keras"
            print(f"[model] Loaded Keras model: {H5_PATH}")
            return

        raise FileNotFoundError(
            "No model found. Run `python train.py` to create "
            "model/digit_model.h5 (and .tflite)."
        )

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """batch shape: (N, 28, 28, 1) float32 in [0,1] -> (N, 10) probs."""
        batch = batch.astype(np.float32)
        if self.backend == "tflite":
            # TFLite interpreters process one sample at a time by default.
            outputs = []
            for i in range(batch.shape[0]):
                self._tflite.set_tensor(self._in["index"], batch[i : i + 1])
                self._tflite.invoke()
                outputs.append(self._tflite.get_tensor(self._out["index"])[0])
            return np.array(outputs)
        return self._keras.predict(batch, verbose=0)


# Loaded lazily on first use so the app can import even before training.
MODEL = None


def get_model() -> ModelWrapper:
    global MODEL
    if MODEL is None:
        MODEL = ModelWrapper()
    return MODEL


# --------------------------------------------------------------------------- #
# Database helpers
# --------------------------------------------------------------------------- #
def get_db():
    """Per-request SQLite connection."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create the history table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_digit INTEGER NOT NULL,
            confidence      REAL    NOT NULL,
            image_path      TEXT    NOT NULL,
            source          TEXT    NOT NULL,
            timestamp       TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# --------------------------------------------------------------------------- #
# Image preprocessing (shared by canvas + upload)
# --------------------------------------------------------------------------- #
def _center_digit(gray: np.ndarray) -> np.ndarray:
    """
    Given a grayscale image where the digit is bright on a dark background,
    crop to the digit's bounding box and center it on a 28x28 canvas,
    mimicking how MNIST digits are framed.
    """
    # Threshold to find the digit
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # Empty image -> return a blank 28x28
        return np.zeros((28, 28), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    digit = gray[y : y + h, x : x + w]

    # Scale the digit so its longest side is 20px (MNIST convention), then pad.
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * 20.0 / w)))
    else:
        new_h = 20
        new_w = max(1, int(round(w * 20.0 / h)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = digit
    return canvas


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert an arbitrary BGR/BGRA/gray image into a (1, 28, 28, 1) float32
    tensor normalized to [0, 1], with a white digit on a black background.
    """
    # To grayscale
    if img_bgr.ndim == 3:
        if img_bgr.shape[2] == 4:  # BGRA (canvas export) -> composite on white
            alpha = img_bgr[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img_bgr[:, :, :3].astype(np.float32)
            img_bgr = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    # MNIST digits are white-on-black. Most inputs are black-on-white, so invert
    # when the image is predominantly light.
    if gray.mean() > 127:
        gray = cv2.bitwise_not(gray)

    centered = _center_digit(gray)
    norm = centered.astype(np.float32) / 255.0
    return norm.reshape(1, 28, 28, 1)


def decode_canvas(data_url: str) -> np.ndarray:
    """Decode a base64 'data:image/png;base64,...' string into a BGRA array."""
    header, _, encoded = data_url.partition(",")
    raw = base64.b64decode(encoded)
    pil = Image.open(io.BytesIO(raw)).convert("RGBA")
    arr = np.array(pil)  # RGBA
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------------------------------- #
# Routes: pages
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# --------------------------------------------------------------------------- #
# Routes: prediction
# --------------------------------------------------------------------------- #
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts EITHER:
      - multipart/form-data with a file field named 'image', OR
      - JSON { "image": "data:image/png;base64,..." } from the canvas.
    Optional form/json field 'save' (default true) controls history logging.
    """
    try:
        source = "upload"
        img_bgr = None

        # 1) File upload -------------------------------------------------- #
        if "image" in request.files and request.files["image"].filename:
            file = request.files["image"]
            if not allowed_file(file.filename):
                return jsonify({"error": "Unsupported file type."}), 400
            data = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            source = "upload"

        # 2) Canvas / JSON data URL -------------------------------------- #
        else:
            payload = request.get_json(silent=True) or {}
            data_url = payload.get("image") or request.form.get("image")
            if not data_url:
                return jsonify({"error": "No image provided."}), 400
            img_bgr = decode_canvas(data_url)
            source = payload.get("source", "canvas")

        if img_bgr is None:
            return jsonify({"error": "Could not decode image."}), 400

        # Predict --------------------------------------------------------- #
        tensor = preprocess(img_bgr)
        if float(tensor.sum()) == 0.0:
            return jsonify({"error": "Canvas/image appears to be empty."}), 400

        probs = get_model().predict(tensor)[0]
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs))

        # Persist the (preprocessed) image for the dashboard thumbnail ---- #
        ts = datetime.datetime.now()
        fname = f"pred_{ts.strftime('%Y%m%d_%H%M%S_%f')}.png"
        fpath = os.path.join(UPLOAD_DIR, fname)
        cv2.imwrite(fpath, (tensor.reshape(28, 28) * 255).astype(np.uint8))

        save = True
        body = request.get_json(silent=True) or {}
        if "save" in body:
            save = bool(body["save"])
        elif "save" in request.form:
            save = request.form.get("save") != "false"

        if save:
            db = get_db()
            db.execute(
                "INSERT INTO history (predicted_digit, confidence, image_path, source, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (digit, confidence, f"/uploads/{fname}", source,
                 ts.strftime("%Y-%m-%d %H:%M:%S")),
            )
            db.commit()

        return jsonify(
            {
                "digit": digit,
                "confidence": round(confidence, 4),
                "probabilities": [round(float(p), 6) for p in probs],
                "image_url": f"/uploads/{fname}",
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


# --------------------------------------------------------------------------- #
# Routes: history
# --------------------------------------------------------------------------- #
@app.route("/history")
def history():
    rows = get_db().execute(
        "SELECT id, predicted_digit, confidence, image_path, source, timestamp "
        "FROM history ORDER BY id DESC"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/history/clear", methods=["POST"])
def clear_history():
    db = get_db()
    # Remove stored thumbnail files too
    rows = db.execute("SELECT image_path FROM history").fetchall()
    for r in rows:
        fp = os.path.join(BASE_DIR, r["image_path"].lstrip("/"))
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except OSError:
                pass
    db.execute("DELETE FROM history")
    db.commit()
    return jsonify({"status": "cleared"})


@app.route("/history/export")
def export_history():
    rows = get_db().execute(
        "SELECT id, predicted_digit, confidence, source, timestamp "
        "FROM history ORDER BY id DESC"
    ).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "predicted_digit", "confidence", "source", "timestamp"])
    for r in rows:
        writer.writerow(
            [r["id"], r["predicted_digit"], r["confidence"], r["source"], r["timestamp"]]
        )

    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv",
    )


# --------------------------------------------------------------------------- #
# Routes: metrics
# --------------------------------------------------------------------------- #
@app.route("/metrics")
def metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "metrics.json not found. Run train.py."}), 404


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False keeps the model from loading twice (reloader) in dev.
    app.run(host="0.0.0.0", port=port, debug=False)
