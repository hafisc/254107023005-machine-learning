import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import base64
import json
import pickle

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template_string, request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # batas upload 10 MB

# Load Model, Scaler, Label, dan Hand Landmarker
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bisindo_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
LANDMARKER_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
with open(LABELS_PATH) as f:
    LABELS = json.load(f)

# Konfigurasi detektor sama dengan saat ekstraksi dataset (extract_features.py)
detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.15,
        min_hand_presence_confidence=0.15,
        min_tracking_confidence=0.15,
    )
)
print("System Loaded Successfully")


def extract_features(rgb, is_flip=False):
    """
    Ekstraksi 127 fitur landmark, identik dengan extract_features.py proyek BISINDO:
    urutan tangan primer/sekunder, translasi ke wrist primer, skala jarak wrist -> middle MCP,
    zero-padding tangan sekunder, dan flag is_two_handed.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None, []

    raw_hands = []
    for idx, landmarks in enumerate(result.hand_landmarks):
        handedness = "Right"
        if result.handedness and idx < len(result.handedness):
            handedness = result.handedness[idx][0].category_name
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        if is_flip:
            coords[:, 0] = 1.0 - coords[:, 0]
            handedness = "Right" if handedness == "Left" else "Left"
        raw_hands.append({"coords": coords, "handedness": handedness, "wrist_x": coords[0, 0]})

    if len(raw_hands) == 1:
        primary, secondary, is_two = raw_hands[0]["coords"], None, 0.0
    else:
        h0, h1 = raw_hands[0], raw_hands[1]
        if h0["handedness"] == "Right" and h1["handedness"] == "Left":
            primary, secondary = h0["coords"], h1["coords"]
        elif h0["handedness"] == "Left" and h1["handedness"] == "Right":
            primary, secondary = h1["coords"], h0["coords"]
        elif h0["wrist_x"] >= h1["wrist_x"]:
            primary, secondary = h0["coords"], h1["coords"]
        else:
            primary, secondary = h1["coords"], h0["coords"]
        is_two = 1.0

    origin = primary[0].copy()
    primary_norm = primary - origin
    palm_scale = np.linalg.norm(primary_norm[9])
    if palm_scale < 1e-4:
        palm_scale = 1.0
    primary_norm = primary_norm / palm_scale
    if secondary is not None:
        secondary_norm = (secondary - origin) / palm_scale
    else:
        secondary_norm = np.zeros((21, 3), dtype=np.float32)

    features = np.concatenate([primary_norm.flatten(), secondary_norm.flatten(), [is_two]])
    # koordinat piksel untuk visualisasi (koordinat asli gambar, bukan hasil un-mirror)
    points = [[(lm.x, lm.y) for lm in hand] for hand in result.hand_landmarks]
    return features.astype(np.float32), points


def decode_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("File bukan gambar yang valid.")
    return bgr


def predict_image(bgr, is_flip=False):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    features, points = extract_features(rgb, is_flip=is_flip)
    if features is None:
        return None, points
    probs = model.predict(scaler.transform(features.reshape(1, -1)), verbose=0)[0]
    top = np.argsort(probs)[::-1][:3]
    return {
        "huruf": LABELS[int(top[0])],
        "confidence": round(float(probs[top[0]]), 4),
        "top3": [{"huruf": LABELS[int(i)], "prob": round(float(probs[i]), 4)} for i in top],
        "jumlah_tangan": len(points),
    }, points


HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                    (17, 18), (18, 19), (19, 20)]


def draw_landmarks(bgr, points):
    img = bgr.copy()
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]
    thick = max(2, w // 200)
    for hand, color in zip(points, [(0, 200, 0), (0, 140, 255)]):
        px = [(int(x * w), int(y * h)) for x, y in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, px[a], px[b], color, thick)
        for p in px:
            cv2.circle(img, p, thick + 1, (255, 255, 255), -1)
    ok, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("ascii")


PAGE = """
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BISINDO Classifier</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #f4f5f7; margin: 0; color: #1f2328; }
    .card { max-width: 560px; margin: 40px auto; background: #fff; padding: 32px;
            border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08); text-align: center; }
    h1 { margin-top: 0; font-size: 1.6rem; }
    p.sub { color: #57606a; margin-top: -8px; }
    button { background: #2f6feb; color: #fff; border: 0; padding: 10px 28px;
             border-radius: 8px; font-size: 1rem; cursor: pointer; }
    img { max-width: 100%; border-radius: 8px; margin-top: 16px; }
    .result { margin-top: 20px; padding: 16px; border-radius: 8px; background: #e6f4ea; color: #14532d; }
    .huruf { font-size: 3rem; font-weight: 700; line-height: 1; }
    .error { background: #ffe1e1; color: #8a1f1f; }
    table { margin: 12px auto 0; border-collapse: collapse; }
    td { padding: 2px 12px; }
    label.opt { color: #57606a; font-size: .9rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Klasifikasi Huruf BISINDO</h1>
    <p class="sub">MediaPipe Hand Landmark + Neural Network &middot; Mohammad Al Hafis Hidayatulloh (254107023005)</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required><br><br>
      <label class="opt"><input type="checkbox" name="mirror" value="1"> Gambar mirror (kamera depan / selfie)</label><br><br>
      <button type="submit">Prediksi</button>
    </form>
    {% if error %}
      <div class="result error">{{ error }}</div>
    {% endif %}
    {% if image %}
      <img src="data:image/jpeg;base64,{{ image }}" alt="gambar dengan landmark tangan">
    {% endif %}
    {% if result %}
      <div class="result">
        <div>Huruf terprediksi</div>
        <div class="huruf">{{ result['huruf'] }}</div>
        <div>Confidence {{ '%.1f' % (result['confidence'] * 100) }}% &middot; {{ result['jumlah_tangan'] }} tangan terdeteksi</div>
        <table>
          {% for t in result['top3'] %}
            <tr><td>{{ loop.index }}. {{ t['huruf'] }}</td><td>{{ '%.1f' % (t['prob'] * 100) }}%</td></tr>
          {% endfor %}
        </table>
      </div>
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
    try:
        bgr = decode_image(file.read())
        result, points = predict_image(bgr, is_flip=request.form.get("mirror") == "1")
    except Exception as e:
        return render_template_string(PAGE, error=f"Error: {e}"), 400
    image = draw_landmarks(bgr, points)
    if result is None:
        return render_template_string(PAGE, image=image,
                                      error="Tangan tidak terdeteksi. Coba gambar lain dengan tangan terlihat jelas.")
    return render_template_string(PAGE, result=result, image=image)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint JSON untuk dipanggil aplikasi lain (pola microservice)."""
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "field 'file' wajib diisi"}), 400
    try:
        bgr = decode_image(file.read())
        result, _ = predict_image(bgr, is_flip=request.form.get("mirror") == "1")
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    if result is None:
        return jsonify({"error": "tangan tidak terdeteksi"}), 422
    return jsonify(result)


if __name__ == "__main__":
    # Port 7860 wajib untuk Hugging Face Spaces
    app.run(host="0.0.0.0", port=7860)
