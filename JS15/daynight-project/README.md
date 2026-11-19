---
title: Day Night Classifier
emoji: 🌗
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Day vs Night Classifier

Aplikasi Flask yang mengklasifikasikan gambar pemandangan menjadi **siang (day)** atau **malam (night)**
menggunakan fitur HOG dan Neural Network (Praktikum JS15, Pembelajaran Mesin).

- `GET /` : halaman upload gambar
- `POST /predict` : hasil prediksi dalam bentuk HTML
- `POST /api/predict` : hasil prediksi dalam bentuk JSON (field form `file`)

Contoh pemanggilan API:

```bash
curl -F "file=@gambar.jpg" https://<username>-daynight-classifier-254107023005.hf.space/api/predict
```
