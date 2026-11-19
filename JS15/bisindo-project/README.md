---
title: BISINDO Classifier
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Klasifikasi Huruf BISINDO

Aplikasi Flask yang mengenali 26 huruf alfabet Bahasa Isyarat Indonesia (BISINDO) dari gambar tangan
(Tugas JS15, Pembelajaran Mesin).

Pipeline inferensi:

1. Gambar upload di-decode dengan OpenCV.
2. MediaPipe Hand Landmarker (`hand_landmarker.task`) mendeteksi hingga 2 tangan.
3. Landmark dinormalisasi menjadi 127 fitur (sama persis dengan saat training).
4. Fitur dinormalisasi dengan `scaler.pkl` (StandardScaler).
5. Neural Network `bisindo_model.h5` memprediksi huruf (softmax 26 kelas).

Endpoint:

- `GET /` : halaman upload gambar
- `POST /predict` : hasil prediksi dalam bentuk HTML
- `POST /api/predict` : hasil prediksi dalam bentuk JSON (field form `file`, opsional `mirror=1`)
