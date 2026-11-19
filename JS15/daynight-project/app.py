import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import base64
import pickle

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template_string, request
from skimage.feature import hog

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # batas upload 10 MB

# Load Model & Scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "day_night_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
print("System Loaded Successfully")


def preprocess_image(image_bytes):
    """Ubah bytes gambar menjadi fitur HOG ter-normalisasi (sama persis dengan training)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File bukan gambar yang valid.")

    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys",
                   visualize=False, feature_vector=True)

    return scaler.transform(hog_feat.reshape(1, -1))


def predict_bytes(image_bytes):
    data = preprocess_image(image_bytes)
    prob_day = float(model.predict(data, verbose=0)[0][0])
    is_day = prob_day > 0.5
    return {
        "label": "Day (Siang)" if is_day else "Night (Malam)",
        "class": "day" if is_day else "night",
        "prob_day": round(prob_day, 4),
        "confidence": round(prob_day if is_day else 1 - prob_day, 4),
    }


PAGE = """
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Day vs Night Classifier</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #f4f5f7; margin: 0; color: #1f2328; }
    .card { max-width: 520px; margin: 40px auto; background: #fff; padding: 32px;
            border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08); text-align: center; }
    h1 { margin-top: 0; font-size: 1.6rem; }
    p.sub { color: #57606a; margin-top: -8px; }
    button { background: #2f6feb; color: #fff; border: 0; padding: 10px 28px;
             border-radius: 8px; font-size: 1rem; cursor: pointer; }
    img { max-width: 100%; border-radius: 8px; margin-top: 16px; }
    .result { margin-top: 20px; padding: 16px; border-radius: 8px; font-size: 1.3rem; font-weight: 600; }
    .day { background: #fff4d6; color: #7a5200; }
    .night { background: #dfe6ff; color: #1b2a6b; }
    .error { background: #ffe1e1; color: #8a1f1f; font-size: 1rem; }
    small { color: #57606a; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Day vs Night Classifier</h1>
    <p class="sub">HOG + Neural Network &middot; Mohammad Al Hafis Hidayatulloh (254107023005)</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required><br><br>
      <button type="submit">Prediksi</button>
    </form>
    {% if error %}
      <div class="result error">Error: {{ error }}</div>
    {% endif %}
    {% if result %}
      <img src="data:image/jpeg;base64,{{ image }}" alt="gambar yang diupload">
      <div class="result {{ result['class'] }}">Hasil: {{ result['label'] }}</div>
      <small>Confidence {{ '%.1f' % (result['confidence'] * 100) }}% &middot; P(day) = {{ result['prob_day'] }}</small>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_template_string(PAGE, error="Tidak ada file yang diupload."), 400
    image_bytes = file.read()
    try:
        result = predict_bytes(image_bytes)
    except Exception as e:
        return render_template_string(PAGE, error=str(e)), 400
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return render_template_string(PAGE, result=result, image=image_b64)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint JSON untuk dipanggil aplikasi lain (pola microservice)."""
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "field 'file' wajib diisi"}), 400
    try:
        return jsonify(predict_bytes(file.read()))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Port 7860 wajib untuk Hugging Face Spaces
    app.run(host="0.0.0.0", port=7860)
