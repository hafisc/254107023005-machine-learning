"""
Script Ekstraksi Fitur Landmark Citra BISINDO Menggunakan MediaPipe HandLandmarker.
Mendukung deteksi hingga 2 tangan (max_num_hands=2), penanganan citra flipped (un-mirroring),
normalisasi posisi wrist & skala telapak tangan, serta pembuatan representasi 127 fitur:
- Fitur [0..62]   : 21 koordinat 3D Tangan Primer (x, y, z)
- Fitur [63..125] : 21 koordinat 3D Tangan Sekunder (x, y, z) dengan zero-padding jika 1 tangan
- Fitur [126]     : Flag biner is_two_handed (1 jika 2 tangan, 0 jika 1 tangan)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def setup_landmarker(model_path):
    """Menginisialisasi MediaPipe HandLandmarker Task."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} tidak ditemukan.")
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.15,
        min_hand_presence_confidence=0.15,
        min_tracking_confidence=0.15
    )
    return vision.HandLandmarker.create_from_options(options)

def extract_landmarks_from_image(detector, image_path, is_flip=False):
    """
    Mengekstraksi landmark tangan dari citra tunggal.
    Melakukan normalisasi handedness, koreksi flip (un-mirroring),
    translasi relatif terhadap wrist primer, dan skala ukuran tangan.
    
    Returns:
        feature_vector: array 127 dimensi atau None jika tidak ada tangan terdeteksi.
        num_hands: jumlah tangan terdeteksi (1 atau 2).
    """
    try:
        mp_image = mp.Image.create_from_file(image_path)
        detection_result = detector.detect(mp_image)
    except Exception as e:
        return None, 0

    if not detection_result.hand_landmarks:
        return None, 0

    num_hands_detected = len(detection_result.hand_landmarks)
    raw_hands = []

    # Ekstraksi koordinat mentah dan label handedness
    for idx, landmarks in enumerate(detection_result.hand_landmarks):
        # Kategori handedness: 'Left' atau 'Right'
        handedness_cat = "Right"
        if detection_result.handedness and idx < len(detection_result.handedness):
            handedness_cat = detection_result.handedness[idx][0].category_name

        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        # Koreksi citra flip (un-mirroring horizontal)
        # Jika citra adalah hasil flip, x dibalik kembali (1.0 - x)
        # dan handedness Left/Right dibalik untuk mengembalikan ke bentuk asli
        if is_flip:
            coords[:, 0] = 1.0 - coords[:, 0]
            handedness_cat = "Right" if handedness_cat == "Left" else "Left"

        raw_hands.append({
            "coords": coords,
            "handedness": handedness_cat,
            "wrist_x": coords[0, 0]
        })

    # Pengurutan tangan kanonikal:
    # Tangan Primer (Hand 1): Diutamakan tangan 'Right' (tangan dominan umum BISINDO)
    # Jika dua tangan sama labelnya atau tidak terlabel, urutkan berdasarkan posisi horizontal x
    if len(raw_hands) == 1:
        primary_hand = raw_hands[0]["coords"]
        secondary_hand = None
        is_two_handed = 0.0
    else:
        # Jika ada 2 tangan, tentukan mana primer dan mana sekunder
        if raw_hands[0]["handedness"] == "Right" and raw_hands[1]["handedness"] == "Left":
            primary_hand = raw_hands[0]["coords"]
            secondary_hand = raw_hands[1]["coords"]
        elif raw_hands[0]["handedness"] == "Left" and raw_hands[1]["handedness"] == "Right":
            primary_hand = raw_hands[1]["coords"]
            secondary_hand = raw_hands[0]["coords"]
        else:
            # Fallback jika kedua tangan memiliki label sama: urutkan dari x lebih besar (biasanya kanan pengguna)
            if raw_hands[0]["wrist_x"] >= raw_hands[1]["wrist_x"]:
                primary_hand = raw_hands[0]["coords"]
                secondary_hand = raw_hands[1]["coords"]
            else:
                primary_hand = raw_hands[1]["coords"]
                secondary_hand = raw_hands[0]["coords"]
        is_two_handed = 1.0

    # Normalisasi Geometris:
    # 1. Translasi: Jadikan wrist primer (titik 0) sebagai titik acuan origin (0, 0, 0)
    origin = primary_hand[0].copy()
    primary_norm = primary_hand - origin

    # 2. Skala: Gunakan jarak wrist primer ke middle MCP (titik 9) sebagai skala telapak
    palm_scale = np.linalg.norm(primary_norm[9])
    if palm_scale < 1e-4:
        palm_scale = 1.0

    primary_norm = primary_norm / palm_scale

    # 3. Normalisasi Tangan Sekunder (jika ada) relatif terhadap wrist primer
    if secondary_hand is not None:
        secondary_norm = (secondary_hand - origin) / palm_scale
    else:
        secondary_norm = np.zeros((21, 3), dtype=np.float32)

    # Ratakan koordinat menjadi vektor 1D
    # 21 * 3 = 63 fitur untuk tangan primer
    # 21 * 3 = 63 fitur untuk tangan sekunder (zero-padded jika 1 tangan)
    feat_primary = primary_norm.flatten()
    feat_secondary = secondary_norm.flatten()

    # Gabungkan menjadi 127 fitur
    feature_vector = np.concatenate([feat_primary, feat_secondary, [is_two_handed]])
    return feature_vector, num_hands_detected

def run_feature_extraction(dataset_dir, output_dir, model_path):
    """Menjalankan ekstraksi seluruh dataset BISINDO train."""
    print("Memulai proses ekstraksi fitur dataset BISINDO...")
    print(f"Direktori dataset: {dataset_dir}")
    print(f"Direktori output : {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    detector = setup_landmarker(model_path)
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    print(f"Ditemukan {len(classes)} kelas: {classes}")

    all_features = []
    metadata_records = []
    
    total_images = 0
    total_detected = 0
    two_hand_count = 0
    one_hand_count = 0
    start_time = time.time()

    for cls_idx, cls_name in enumerate(classes):
        cls_folder = os.path.join(dataset_dir, cls_name)
        file_list = sorted([f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        cls_detected = 0
        cls_start = time.time()

        for fname in file_list:
            total_images += 1
            fpath = os.path.join(cls_folder, fname)
            is_flip = "flip" in fname.lower()

            feat, num_hands = extract_landmarks_from_image(detector, fpath, is_flip=is_flip)

            if feat is not None:
                all_features.append(feat)
                metadata_records.append({
                    "label": cls_name,
                    "label_idx": cls_idx,
                    "filename": fname,
                    "num_hands": num_hands,
                    "is_flip": int(is_flip)
                })
                total_detected += 1
                cls_detected += 1
                if num_hands >= 2:
                    two_hand_count += 1
                else:
                    one_hand_count += 1

        cls_elapsed = time.time() - cls_start
        print(f"Kelas {cls_name}: {cls_detected}/{len(file_list)} terdeteksi ({cls_detected/len(file_list)*100:.1f}%) dalam {cls_elapsed:.1f}s")

    detector.close()
    elapsed_total = time.time() - start_time

    X = np.array(all_features, dtype=np.float32)
    df_meta = pd.DataFrame(metadata_records)

    print("\n" + "="*70)
    print("RINGKASAN EKSTRAKSI FITUR:")
    print(f"Total citra diproses      : {total_images}")
    print(f"Total citra terdeteksi    : {total_detected} ({total_detected/total_images*100:.2f}%)")
    print(f"Citra 2 tangan            : {two_hand_count} ({two_hand_count/total_detected*100:.1f}%)")
    print(f"Citra 1 tangan (zero-pad) : {one_hand_count} ({one_hand_count/total_detected*100:.1f}%)")
    print(f"Waktu ekstraksi total     : {elapsed_total:.1f} detik ({total_images/elapsed_total:.1f} citra/detik)")
    print(f"Dimensi matriks fitur (X) : {X.shape}")
    print("="*70)

    # Konstruksi nama kolom
    col_names = []
    for h in [1, 2]:
        for pt in range(21):
            for axis in ['x', 'y', 'z']:
                col_names.append(f"h{h}_pt{pt}_{axis}")
    col_names.append("is_two_handed")

    # Simpan ke CSV
    df_features = pd.DataFrame(X, columns=col_names)
    df_full = pd.concat([df_meta, df_features], axis=1)

    csv_path = os.path.join(output_dir, "dataset_bisindo_landmarks.csv")
    df_full.to_csv(csv_path, index=False)
    print(f"Dataset lengkap disimpan ke: '{csv_path}'")

    # Simpan versi numpy terkompresi untuk pemuatan kilat di notebook
    npz_path = os.path.join(output_dir, "dataset_bisindo_landmarks.npz")
    np.savez_compressed(
        npz_path,
        X=X,
        labels=df_meta["label"].values,
        label_indices=df_meta["label_idx"].values,
        is_two_handed=X[:, -1],
        feature_names=col_names
    )
    print(f"Dataset terkompresi disimpan ke: '{npz_path}'")
    return csv_path, npz_path

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_train_dir = r"E:\Dokumen\Kuliah\Semester 5\Pembelajaran Mesin\UTS_Bisindo_Klasterisasi_Kelompok-1\train"
    model_asset_path = os.path.join(current_dir, "hand_landmarker.task")
    
    run_feature_extraction(dataset_train_dir, current_dir, model_asset_path)
