# Laporan Teknis dan Analisis Komparasi Klasterisasi Bahasa Isyarat Indonesia (BISINDO)

Dokumen ini menyajikan dokumentasi komprehensif, formulasi matematis, dan analisis komparatif eksperimen unsupervised learning pada dataset alfabet Bahasa Isyarat Indonesia (BISINDO). Lima paradigma klasterisasi utama dievaluasi:
1. **Partitioning Clustering:** K-Means
2. **Density-Based Clustering:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
3. **Hierarchical Agglomerative Clustering (HAC):** Bottom-Up Complete Linkage
4. **Hierarchical Divisive Clustering:** Top-Down Bipartitioning (Bisecting 2-Means / 2-Medians)
5. **Hierarchical Density-Based Clustering (HDBSCAN):** Multi-Scale Density Persistence & Excess of Mass (EOM)

Masing-masing algoritma dievaluasi secara sistematis menggunakan 3 fungsi jarak: **Euclidean Distance**, **Manhattan Distance**, dan **Cosine Similarity/Distance** dengan jumlah target klaster $K \approx 26$ (merepresentasikan 26 huruf alfabet A–Z).

---

## Daftar Isi
1. [Ringkasan Eksekutif dan Latar Belakang Proyek](#1-ringkasan-eksekutif-dan-latar-belakang-proyek)
2. [Arsitektur Rekayasa Fitur 127 Dimensi](#2-arsitektur-rekayasa-fitur-127-dimensi)
3. [Formulasi Matematis Fungsi Jarak](#3-formulasi-matematis-fungsi-jarak)
4. [Eksperimen 1: K-Means Clustering](#4-eksperimen-1-k-means-clustering)
5. [Eksperimen 2: DBSCAN (Density-Based Spatial Clustering)](#5-eksperimen-2-dbscan-density-based-spatial-clustering)
6. [Eksperimen 3: Hierarchical Agglomerative Clustering (HAC)](#6-eksperimen-3-hierarchical-agglomerative-clustering-hac)
7. [Eksperimen 4: Divisive Hierarchical Clustering (Top-Down Bipartitioning)](#7-eksperimen-4-divisive-hierarchical-clustering-top-down-bipartitioning)
8. [Eksperimen 5: HDBSCAN (Hierarchical Density-Based Clustering)](#8-eksperimen-5-hdbscan-hierarchical-density-based-clustering)
9. [Matriks Komparasi Lintas Paradigma (15 Konfigurasi Algoritma)](#9-matriks-komparasi-lintas-paradigma-15-konfigurasi-algoritma)
10. [Sintesis Final Comparison & Analisis 11 Diagram Visualisasi](#10-sintesis-final-comparison--analisis-11-diagram-visualisasi)
11. [Analisis Kritis, Temuan Kontraintuitif & Diskusi Teoretis](#11-analisis-kritis-temuan-kontraintuitif--diskusi-teoretis)
12. [Rekomendasi Terapan Rekayasa untuk PBL BisindoVision](#12-rekomendasi-terapan-rekayasa-untuk-pbl-bisindovision)
13. [Struktur Direktori Repositori](#13-struktur-direktori-repositori)
14. [Panduan Menjalankan Eksperimen](#14-panduan-menjalankan-eksperimen)

---

## 1. Ringkasan Eksekutif dan Latar Belakang Proyek

Proyek ini merupakan bagian dari pengembangan sistem pengenalan Bahasa Isyarat Indonesia (BISINDO) berbasis visi komputer (BisindoVision). Berbeda dengan American Sign Language (ASL) yang mayoritas menggunakan satu tangan, BISINDO memiliki karakteristik unik:
- Sebagian huruf dibentuk menggunakan **1 tangan** (misalnya huruf C, E, I, J, L, O, R, U, V, Z).
- Sebagian huruf dibentuk menggunakan **2 tangan secara simultan** (misalnya huruf A, B, D, F, G, H, K, M, N, P, Q, S, T, W, X, Y).

Tantangan utama dalam eksplorasi unsupervised learning untuk BISINDO adalah menangani heterogenitas jumlah tangan, variasi posisi kamera, skala fisik tangan subjek, serta orientasi gambar akibat augmentasi (seperti horizontal flip).

### Ringkasan Dataset Mentah
- Sumber dataset: Direktori [train/](train)
- Total citra mentah: 9.169 gambar RGB yang terbagi dalam 26 folder kelas alfabet (A–Z).
- Total sampel valid terekstraksi oleh MediaPipe HandLandmarker: **7.893 sampel** (tingkat deteksi $86,08\%$).
- Total sampel gagal deteksi (discarded): 1.276 gambar ($13,92\%$, disebabkan oklusi parah, blur, atau pencahayaan ekstrem).
- Distribusi kelas valid:
  - Dua tangan (Two-handed): 3.599 sampel ($45,60\%$).
  - Satu tangan (One-handed): 4.294 sampel ($54,40\%$).
- Penanganan citra augmentasi flip: Sebanyak 3.063 citra yang memiliki nama file mengandung kata `flip` dilakukan *un-mirroring* geometris sebelum ekstraksi fitur.

---

## 2. Arsitektur Rekayasa Fitur 127 Dimensi

Untuk memungkinkan algoritma klasterisasi memproses sampel 1-tangan dan 2-tangan secara adil dalam satu ruang vektor berdimensi tetap, dirancang arsitektur vektor fitur berdimensi 127:

```
[--- Tangan Primer (63) ---] [--- Tangan Sekunder (63) ---] [Flag (1)]
(21 landmark x 3: x, y, z)    (21 landmark x 3: x, y, z)    (is_two_handed)
```

```
+-----------------------------------------------------------------------------------+
|                            Vektor Fitur 127-Dimensi                               |
+-----------------------------------+-----------------------------------+-----------+
| Tangan Primer (Landmark 0 s.d. 20)| Tangan Sekunder (Landmark 0 s.d 20| Binary    |
| Koordinat (x, y, z) -> 63 Fitur   | Koordinat (x, y, z) -> 63 Fitur   | Flag (1)  |
|                                   | (Zero-Padding jika 1 tangan)      | [0 atau 1]|
+-----------------------------------+-----------------------------------+-----------+
| Fitur [0 ... 62]                  | Fitur [63 ... 125]                | Fitur 126 |
+-----------------------------------+-----------------------------------+-----------+
```

### 2.1 Spesifikasi Komponen Fitur
1. **Tangan Primer (Fitur 0–62):**
   - 21 landmark MediaPipe, masing-masing memiliki koordinat teranotasi $(x_i, y_i, z_i)$.
   - $21 \times 3 = 63$ nilai numerik kontinu.
   - Jika citra memiliki 2 tangan, tangan kiri ditetapkan sebagai tangan primer secara konsisten berdasarkan koordinat lateral atau urutan deteksi tangan.
2. **Tangan Sekunder dengan Zero-Padding (Fitur 63–125):**
   - Untuk gestur 2-tangan: Koordinat $(x_j, y_j, z_j)$ tangan kedua ($21 \times 3 = 63$ fitur).
   - Untuk gestur 1-tangan: Dilakukan **Zero-Padding** dengan mengisi seluruh 63 slot menggunakan nilai $0.0$.
3. **Binary Flag `is_two_handed` (Fitur 126):**
   - Nilai $1.0$ jika terdeteksi 2 tangan.
   - Nilai $0.0$ jika hanya terdeteksi 1 tangan.
   - Menjadi pemisah topologis eksplisit antara klaster gestur 1-tangan dan 2-tangan.

### 2.2 Penanganan Geometri Citra Flip (Un-mirroring)
Pada huruf BISINDO yang asimetris (seperti B, D, dan P), tangan dominan memegang peranan semantik yang berbeda dengan tangan penopang. Citra yang dihasilkan dari horizontal flip dibalikkan koordinatnya agar merepresentasikan gestur aslinya:
- $x_{\text{unflip}} = 1.0 - x_{\text{flip}}$
- $y$ dan $z$ dipertahankan.
- Handedness (label tangan kanan dan tangan kiri) ditukar secara otomatis.

Data fitur tersimpan dalam:
- Format CSV: `dataset_bisindo_landmarks.csv` (9,2 MB, 7.893 baris $\times$ 128 kolom termasuk label).
- Format NumPy Compressed: `dataset_bisindo_landmarks.npz` (2,2 MB, arrays: `X`, `labels`, `image_paths`).

---

## 3. Formulasi Matematis Fungsi Jarak

Seluruh eksperimen mengevaluasi 3 fungsi jarak berbeda terhadap matriks data $\mathbf{X} \in \mathbb{R}^{n \times d}$, di mana $n = 7.893$ dan $d = 127$:

### 3.1 Euclidean Distance ($L_2$ Norm)
Mengukur jarak geometris garis lurus (straight-line distance) antar dua vektor fitur $\mathbf{u}$ dan $\mathbf{v}$:
$$d_{\text{Euclidean}}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2}$$
- **Karakteristik:** Sensitif terhadap magnitudo absolut fitur, variasi jarak tangan ke kamera, dan skala ukuran fisik tangan pengguna.

### 3.2 Manhattan Distance ($L_1$ Norm / City Block)
Mengukur jumlah selisih absolut proyeksi pada setiap sumbu koordinat:
$$d_{\text{Manhattan}}(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_1 = \sum_{i=1}^{d} |u_i - v_i|$$
- **Karakteristik:** Lebih tahan (robust) terhadap outlier ekstrem dan pergeseran koordinat tunggal dibandingkan Euclidean karena tidak mengkuadratkan deviasi jarak.

### 3.3 Cosine Distance ($1 - \text{Cosine Similarity}$)
Mengukur sudut deviasi arah (angular divergence) antar dua vektor, mengabaikan magnitudo absolut:
$$\text{Sim}_{\text{Cosine}}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$
$$d_{\text{Cosine}}(\mathbf{u}, \mathbf{v}) = 1 - \text{Sim}_{\text{Cosine}}(\mathbf{u}, \mathbf{v})$$
- **Karakteristik:** Sangat ideal untuk data landmark koordinat tangan karena kebal terhadap skala jarak kamera ke tangan (scale-invariant) dan fokus pada orientasi relatif sendi-sendi jari. Pada sampel 1-tangan ber-zero padding, bagian tangan kedua menghasilkan dot product 0, sehingga perbandingan sudut murni berfokus pada sendi tangan pertama.

---

## 4. Eksperimen 1: K-Means Clustering

K-Means mempartisi $n$ sampel ke dalam $K=26$ klaster dengan meminimalkan fungsi inersia (Within-Cluster Sum of Squares).

### 4.1 Metodologi Implementasi
- **Euclidean:** Algoritma K-Means standar Scikit-Learn (`n_init=10`, `max_iter=300`, `init='k-means++'`).
- **Manhattan:** K-Medians kustom teroptimasi vektor, di mana posisi representasi klaster diperbarui menggunakan nilai median dari anggota klaster untuk meminimalkan $L_1$ norm.
- **Cosine (Spherical K-Means):** Data dinormalisasi ke unit sphere $\|\mathbf{x}\|_2 = 1$ menggunakan $L_2$ normalizer. Dengan normalisasi ini, minimisasi Euclidean pada unit hypersphere secara matematis setara dengan maksimisasi Cosine Similarity:
  $$\|\mathbf{u} - \mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 + \|\mathbf{v}\|_2^2 - 2(\mathbf{u} \cdot \mathbf{v}) = 2 - 2 \cos(\theta)$$
  Centroid diperbarui sebagai rata-rata anggota yang dinormalisasi kembali ke unit sphere pada setiap iterasi.

### 4.2 Hasil Empiris K-Means ($K = 26$)

| Metrik Evaluasi | Euclidean Distance | Manhattan Distance | Cosine Distance | Konfigurasi Terbaik |
| :--- | :---: | :---: | :---: | :---: |
| **Silhouette Score** ($\uparrow$) | 0,2060 | 0,2178 | **0,3334** | **Cosine (+61,8% vs Euclidean)** |
| **Davies–Bouldin Index** ($\downarrow$) | 1,4273 | **1,4271** | 1,4964 | **Manhattan (Paling Kompak)** |
| **Calinski–Harabasz Index** ($\uparrow$) | 941,27 | 863,90 | **1.120,34** | **Cosine (+19,0% vs Euclidean)** |
| **Execution Time** ($\downarrow$) | 0,4273 s | 0,5096 s | **0,3373 s** | **Cosine (Paling Cepat)** |
| **Jumlah Iterasi Konvergensi** ($\downarrow$) | 49 | **19** | 25 | **Manhattan (Paling Sedikit)** |
| **Nilai Objektif Akhir ($J$)** | 245.223,37 | 14.796.547,52 | 154,32 | N/A (Skala Berbeda) |

### 4.3 Analisis Mendalam K-Means
1. **Keunggulan Mutlak Cosine Distance pada Silhouette & Calinski-Harabasz:**
   - Silhouette Score Cosine mencapai **0,3334**, melonjak $+61,8\%$ dibanding Euclidean (0,2060).
   - Calinski-Harabasz Index Cosine mencapai **1.120,34**, unggul $+19,0\%$ dibanding Euclidean (941,27).
   - **Penyebab Fisis:** Dalam gestur tangan, panjang jari dan jarak subjek ke kamera menyebabkan magnitudo koordinat $L_2$ bervariasi luas meski bentuk gestur sama. Proyeksi spherical mengeliminasi variasi magnitudo ini, sehingga sampel dengan konfigurasi sudut sendi yang serupa berkumpul jauh lebih rapat.
2. **Efisiensi Konvergensi Manhattan:**
   - K-Medians Manhattan berhasil mencapai konvergensi hanya dalam **19 iterasi**, dibandingkan Euclidean yang membutuhkan 49 iterasi.
   - Hal ini membuktikan bahwa pembaruan berbasis median koordinat per-dimensi lebih tahan terhadap tarikan sampel outlier yang berada di batas luar klaster.
3. **Mengapa Davies-Bouldin Euclidean Sedikit Lebih Rendah:**
   - Formulasi Davies-Bouldin standar Scikit-Learn menghitung dispersi klaster menggunakan rata-rata jarak Euclidean ke centroid. Karena algoritma Euclidean K-Means secara eksplisit meminimalkan dispersi kuadratik Euclidean, metrik DBI secara inheren sedikit menguntungkan Euclidean dan Manhattan. Namun selisihnya sangat tipis ($1,427$ vs $1,496$).

Artifacts tersimpan di: [KMeans/outputKmeans/](KMeans/outputKmeans)
- Visualisasi: `cluster_comparison_pca2d.png`, `metrics_comparison_barchart.png`, `elbow_curve.png`
- Data komparasi: `metrics_comparison_summary.csv`, `evaluasi_kmeans_bisindo.xlsx`

---

## 5. Eksperimen 2: DBSCAN (Density-Based Spatial Clustering)

DBSCAN mengelompokkan data berdasarkan kepadatan spasial titik dalam radius $\varepsilon$ (Epsilon) dengan jumlah tetangga minimum $\text{MinPts}$ (Min Samples). Titik yang tidak memenuhi ambang batas kepadatan diklasifikasikan sebagai **Noise / Outlier** (label `-1`).

### 5.1 Metodologi Implementasi dan Kalibrasi Epsilon
- Parameter $\text{MinPts} = 10$ ditetapkan secara konsisten untuk meredam noise pada ruang berdimensi 127.
- Nilai Epsilon ($\varepsilon$) dikalibrasi secara presisi menggunakan **k-NN Distance Graph (Elbow Curve)** pada $k = 10$ dan binary search sehingga ketiga metrik jarak menghasilkan tepat **$K = 26$ klaster** untuk perbandingan yang valid.
- Nilai $\varepsilon$ hasil kalibrasi:
  - Euclidean: $\varepsilon = 4,015$
  - Manhattan: $\varepsilon = 28,350$
  - Cosine: $\varepsilon = 0,067$

### 5.2 Hasil Empiris DBSCAN ($K = 26, \text{MinPts} = 10$)

| Metrik Evaluasi | Euclidean Distance | Manhattan Distance | Cosine Distance | Konfigurasi Terbaik |
| :--- | :---: | :---: | :---: | :---: |
| **Epsilon ($\varepsilon$)** | 4,015 | 28,350 | 0,067 | N/A |
| **Jumlah Klaster Dihasilkan** | 26 | 26 | 26 | Seimbang ($K=26$) |
| **Jumlah Sampel Noise** ($\downarrow$) | 892 | 1.190 | **525** | **Cosine (Paling Bersih)** |
| **Rasio Noise (%)** ($\downarrow$) | $11,30\%$ | $15,08\%$ | **6,65%** | **Cosine (Retensi Data 93,35%)** |
| **Jumlah Core Samples** ($\uparrow$) | 6.499 | 6.241 | **6.787** | **Cosine (Kepadatan Terbanyak)** |
| **Silhouette Score** ($\uparrow$) | -0,0027 | **0,2339** | -0,1727 | **Manhattan (Kohesi Tertinggi)** |
| **Davies–Bouldin Index** ($\downarrow$) | **0,9884** | 1,1751 | 1,0795 | **Euclidean (Pemisahan Terbaik)** |
| **Calinski–Harabasz Index** ($\uparrow$) | **255,36** | 233,62 | 234,06 | **Euclidean** |
| **Execution Time** ($\downarrow$) | **0,1071 s** | 0,6524 s | 0,4648 s | **Euclidean (Paling Cepat)** |

### 5.3 Analisis Mendalam DBSCAN
1. **Keunggulan Retensi Data Cosine DBSCAN (Noise Hanya 6,65%):**
   - Cosine DBSCAN hanya membuang **525 sampel** sebagai noise, berhasil mengelompokkan **93,35% sampel** (7.368 sampel) ke dalam 26 klaster.
   - Sebaliknya, Manhattan membuang 1.190 sampel ($15,08\%$) dan Euclidean membuang 892 sampel ($11,30\%$).
   - **Alasan Ilmiah:** Pada ruang berdimensi 127 (*curse of dimensionality*), jarak metrik berbasis koordinat ($L_1$ dan $L_2$) mengalami degradasi volume hiperbola (jarak antar titik menjadi seragam). Sebaliknya, Cosine Distance mengelompokkan sampel berdasarkan kesamaan arah vektor, sehingga densitas angular jauh lebih terjaga pada data gestur.
2. **Mengapa Manhattan DBSCAN Memiliki Silhouette Tertinggi (0,2339):**
   - Manhattan DBSCAN membuang sampel-sampel perbatasan (border points) dalam jumlah terbesar ($15,08\%$ noise). Klaster yang tersisa murni terdiri dari inti (core) gestur yang sangat padat dan tidak memiliki tumpang tindih. Dengan membuang sampel ambigu, skor kohesi internal melonjak drastis.
3. **Mengapa Silhouette Cosine Bernilai Negatif (-0,1727):**
   - Silhouette dihitung menggunakan metrik Euclidean pada ruang fitur ter-assign. Pada Cosine DBSCAN, klaster terbentuk di permukaan bola (hypersphere) dengan topologi non-konveks angular. Ketika dievaluasi menggunakan metrik Euclidean linier, bentuk klaster yang melengkung tampak tumpang tindih secara spasial linier, meski secara angular terpisah sempurna.

Artifacts tersimpan di: [DBScan/outputDBScan/](DBScan/outputDBScan)
- Visualisasi: `cluster_comparison_pca2d.png`, `metrics_comparison_barchart.png`, `elbow_curve.png`
- Data komparasi: `metrics_comparison_summary.csv`, `evaluasi_dbscan_bisindo.xlsx`

---

## 6. Eksperimen 3: Hierarchical Agglomerative Clustering (HAC)

HAC membangun struktur pohon hierarki klaster secara *bottom-up*, menggabungkan pasangan klaster yang memiliki jarak terkecil pada setiap iterasi hingga seluruh data tergabung dalam satu pohon (dendrogram).

### 6.1 Metodologi Implementasi dan Pemilihan Linkage
- **Linkage Method: Complete Linkage (Farthest Neighbor)**
  $$D(A, B) = \max \{ d(\mathbf{u}, \mathbf{v}) : \mathbf{u} \in A, \mathbf{v} \in B \}$$
  Complete Linkage dipilih karena mencegah fenomena *chaining effect* (rantai memanjang), memaksa terbentuknya klaster-klaster kompak dengan diameter maksimum terbatas, dan kompatibel dengan jarak non-Euclidean (Manhattan dan Cosine).
- **Pemotongan Dendrogram (Tree Cut-off):**
  Tinggi pemotongan ($h$) dikalibrasi secara eksak agar memotong pohon tepat pada $K = 26$ cabang:
  - Euclidean: $h = 25,795$
  - Manhattan: $h = 233,217$
  - Cosine: $h = 1,049$

### 6.2 Hasil Empiris HAC ($K = 26$, Complete Linkage)

| Metrik Evaluasi | Euclidean Distance | Manhattan Distance | Cosine Distance | Konfigurasi Terbaik |
| :--- | :---: | :---: | :---: | :---: |
| **Cut-off Height ($h$)** | 25,795 | 233,217 | 1,049 | N/A |
| **Ukuran Klaster Minimum** | 1 sampel (Singleton) | 1 sampel (Singleton) | **11 sampel** | **Cosine (Nol Singleton)** |
| **Ukuran Klaster Maksimum** | 2.281 sampel | 3.176 sampel | **1.179 sampel** | **Cosine (Paling Merata)** |
| **Keseimbangan Distribusi** | Buruk (Ada klaster isi 1) | Sangat Buruk (Ada klaster isi 1) | **Sangat Baik & Stabil** | **Cosine** |
| **Silhouette Score** ($\uparrow$) | 0,1182 | 0,1430 | **0,1839** | **Cosine (+55,6% vs Euclidean)** |
| **Davies–Bouldin Index** ($\downarrow$) | **1,1485** | 1,2737 | 1,8357 | **Euclidean** |
| **Calinski–Harabasz Index** ($\uparrow$) | 422,47 | 349,85 | **818,14** | **Cosine (+93,7% vs Euclidean)** |
| **Execution Time** ($\downarrow$) | 1,7010 s | 1,7011 s | **1,7010 s** | Setara ($O(n^2)$) |

### 6.3 Analisis Mendalam HAC
1. **Kegagalan Fatal Euclidean dan Manhattan HAC (Fenomena Klaster Singleton):**
   - Pada pemotongan $K=26$, Euclidean dan Manhattan HAC menghasilkan beberapa klaster yang hanya berisi **1 sampel tunggal (singleton)**, sementara satu klaster raksasa menampung hingga 2.281 s.d. 3.176 sampel.
   - **Penyebab Fisis:** Jarak $L_1$ dan $L_2$ sangat rentan terhadap sampel yang sedikit keluar dari distribusi normal (misalnya tangan yang lebih dekat ke kamera). Sampel outlier ini menolak bergabung dengan klaster manapun hingga level hierarki teratas, sehingga saat pohon dipotong pada level 26 klaster, outlier tersebut terisolasi menjadi klaster individual beranggotakan 1.
2. **Kestabilan dan Keseimbangan Luar Biasa Cosine HAC:**
   - Cosine Complete Linkage sama sekali **tidak memiliki klaster singleton**. Klaster terkecil berisi 11 sampel, dan klaster terbesar hanya 1.179 sampel.
   - Skor Silhouette Cosine mencapai **0,1839** ($+55,6\%$ dibanding Euclidean 0,1182).
   - Skor Calinski-Harabasz melonjak hingga **818,14** ($+93,7\%$ dibanding Euclidean 422,47).
   - **Interpretasi Ilmiah:** Karena sudut antar sendi tangan memiliki batasan anatomis yang teratur, pengelompokan berbasis jarak Cosine Complete Linkage berhasil merefleksikan 26 konfigurasi gestur alfabet secara harmonis tanpa terdistorsi oleh variasi skala.

Artifacts tersimpan di: [HAC/outputHAC/](HAC/outputHAC)
- Visualisasi: `cluster_comparison_pca2d.png`, `dendrogram_comparison.png`, `metrics_comparison_barchart.png`
- Data komparasi: `metrics_comparison_summary.csv`, `evaluasi_hac_bisindo.xlsx`

---

## 7. Eksperimen 4: Divisive Hierarchical Clustering (Top-Down Bipartitioning)

Berlawanan dengan pendekatan *agglomerative* (bottom-up), **Divisive Hierarchical Clustering** memulai proses dengan seluruh 7.893 observasi tergabung dalam satu klaster tunggal (root). Pohon hierarki dibangun secara rekursif dari atas ke bawah (*top-down*) dengan memilih klaster yang memiliki dispersi/inersia internal tertinggi untuk dibelah (*bisection*) menjadi dua sub-klaster hingga terbentuk target $K = 26$ klaster.

### 7.1 Metodologi Implementasi & Konstruksi Linkage Matrix
- **Euclidean:** Algoritma Bisecting 2-Means (`BisectingKMeans`, `bisecting_strategy='biggest_inertia'`), meminimalkan Within-Cluster Sum of Squares pada setiap pembelahan.
- **Manhattan:** Algoritma Bisecting 2-Medians kustom teroptimasi, memilih klaster dengan dispersi mutlak $\sum \|x - \text{median}(C)\|_1$ terbesar dan membelahnya menggunakan pembaruan median koordinat.
- **Cosine (Spherical Bisecting 2-Means):** Data dinormalisasi ke permukaan bola satuan $\|\mathbf{x}\|_2 = 1$, memaksimalkan perataan sudut arah (*cosine alignment*) terhadap centroid arah pada setiap tahap pembelahan biner.
- **Konstruksi Dendrogram:** Setiap node pembelahan pada pohon biner divisive dipetakan kembali secara *post-order traversal* menjadi matriks linkage SciPy $Z \in \mathbb{R}^{(K-1) \times 4}$, memungkinkan visualisasi dendrogram hierarkis top-down yang sepenuhnya kompatibel dengan SciPy.

### 7.2 Hasil Empiris Divisive Clustering ($K = 26$)

| Metrik Evaluasi | Euclidean Distance | Manhattan Distance | Cosine Distance | Konfigurasi Terbaik |
| :--- | :---: | :---: | :---: | :---: |
| **Strategi Pembelahan** | Bisecting 2-Means (Inertia) | Bisecting 2-Medians ($L_1$ Disp) | Spherical Bisecting 2-Means | Sesuai Karakteristik Metrik |
| **Ukuran Klaster Minimum** | 5 sampel | 1 sampel | **19 sampel** | **Cosine (Nol Singleton)** |
| **Ukuran Klaster Maksimum** | 817 sampel | 1.101 sampel | **652 sampel** | **Cosine (Paling Seimbang)** |
| **Klaster Singleton (size=1)** | **0 klaster** | 2 klaster | **0 klaster** | **Euclidean & Cosine (Nol Singleton)** |
| **Silhouette Score** ($\uparrow$) | 0,1557 | 0,1407 | **0,2744** | **Cosine (+76,2% vs Euclidean)** |
| **Davies–Bouldin Index** ($\downarrow$) | 1,6729 | **1,5631** | 1,6262 | **Manhattan (Paling Kompak)** |
| **Calinski–Harabasz Index** ($\uparrow$) | 768,85 | 660,59 | **1.408,53** | **Cosine (+83,2% vs Euclidean)** |
| **Execution Time** ($\downarrow$) | 2,2368 s | 0,8067 s | **0,0979 s** | **Cosine (Super Cepat, < 0,1 s)** |

### 7.3 Analisis Mendalam Divisive Clustering
1. **Keunggulan Spektakuler Cosine Spherical Bisecting 2-Means:**
   - Silhouette Score mencapai **0,2744**, melonjak $+76,2\%$ di atas Euclidean (0,1557).
   - Calinski-Harabasz Index mencapai **1.408,53**, mengungguli Euclidean (768,85) sebesar $+83,2\%$.
   - Waktu eksekusi tercepat di kelasnya: hanya **0,0979 detik** untuk 7.893 sampel!
   - Rentang ukuran klaster sangat seimbang (19 hingga 652 sampel), bebas klaster singleton.
2. **Mengapa Divisive Mengatasi Masalah Singleton pada HAC Euclidean:**
   - Karena pembelahan top-down dipandu oleh inersia global terbesar, sampel outlier tidak pernah dibiarkan terisolasi menjadi klaster individu berukuran 1 sampel, melainkan diabsorpsi ke dalam partisi yang representatif.

Artifacts tersimpan di: [Divisive/outputDivisive/](Divisive/outputDivisive)
- Visualisasi: `cluster_comparison_pca2d.png`, `dendrogram_comparison.png`, `metrics_comparison_barchart.png`, `cluster_cosine.png`, `dendrogram_cosine.png`
- Data komparasi: `metrics_comparison_summary.csv`, `evaluasi_divisive_bisindo.xlsx`

---

## 8. Eksperimen 5: HDBSCAN (Hierarchical Density-Based Clustering)

**HDBSCAN** memadukan prinsip klasterisasi hierarkis dan berbasis kepadatan. Berbeda dengan DBSCAN yang bergantung pada satu ambang batas $\varepsilon$ kaku, HDBSCAN membangun hierarki kepadatan spasial menggunakan metrik **Mutual Reachability Distance**, memangkas hierarki menjadi pohon ringkas (*Condensed Tree*), dan mengekstraksi klaster yang paling persisten secara stabilitas densitas menggunakan metode **Excess of Mass (EOM)**.

### 8.1 Metodologi Implementasi & Kalibrasi Parameter
- **Parameter Kunci:** Parameter `min_cluster_size` (mcs) dikalibrasi secara sistematis agar klaster yang diekstraksi beroperasi pada resolusi yang setara dengan target 26 alfabet BISINDO:
  - Euclidean: `min_cluster_size = 46` $\rightarrow$ Menghasilkan **25 klaster stabil**.
  - Manhattan: `min_cluster_size = 48` $\rightarrow$ Menghasilkan **30 klaster stabil**.
  - Cosine (Spherical): `min_cluster_size = 53` $\rightarrow$ Menghasilkan **28 klaster stabil**.
- **Pohon Hierarki (Single Linkage Tree):** Matriks hierarki keterjangkauan timbal balik diekstraksi dari atribut internal model `_single_linkage_tree_` dan divisualisasikan dalam bentuk dendrogram.
- **Probabilitas Keanggotaan:** HDBSCAN menghitung nilai probabilitas keanggotaan setiap titik (`probabilities_` $\in [0, 1]$), menunjukkan tingkat keyakinan bahwa suatu sampel benar-benar berada di inti klaster.

### 8.2 Hasil Empiris HDBSCAN

| Metrik Evaluasi | Euclidean Distance | Manhattan Distance | Cosine Distance | Konfigurasi Terbaik |
| :--- | :---: | :---: | :---: | :---: |
| **Parameter min_cluster_size (mcs)** | 46 | 48 | 53 | Dikalibrasi ke $\approx 26$ |
| **Jumlah Klaster Ditemukan ($K$)** | **25 klaster** | 30 klaster | **28 klaster** | **Euclidean & Cosine ($\approx 26$)** |
| **Sampel Noise (Outlier)** ($\downarrow$) | **3.937 sampel** | 4.905 sampel | 4.210 sampel | **Euclidean (Noise Terendah)** |
| **Rasio Noise (%)** ($\downarrow$) | **49,88%** | 62,14% | 53,34% | **Euclidean** |
| **Sampel Terklaster (Core Exemplars)** | 3.956 sampel | 2.988 sampel | **3.683 sampel** | **Euclidean & Cosine** |
| **Silhouette Score** ($\uparrow$) | 0,3486 | 0,5479 | **0,5760** | **Cosine (Rekor Tertinggi Absolut)** |
| **Davies–Bouldin Index** ($\downarrow$) | 0,7467 | **0,7441** | 0,8203 | **Manhattan (Paling Kompak)** |
| **Calinski–Harabasz Index** ($\uparrow$) | 513,99 | 1.265,49 | **1.539,26** | **Cosine (Rekor Tertinggi Absolut)** |
| **Execution Time** ($\downarrow$) | 5,4956 s | 5,6686 s | **3.9883 s** | **Cosine (Paling Cepat)** |

### 8.3 Analisis Mendalam HDBSCAN
1. **Mengapa Silhouette Score HDBSCAN Cosine Mencapai Rekor Tertinggi (0,5760):**
   - Skor Silhouette Cosine melonjak drastis hingga **0,5760** dan Calinski-Harabasz mencapai **1.539,26**—keduanya merupakan nilai tertinggi di antara seluruh eksperimen yang pernah dilakukan pada dataset ini.
   - **Penyebab Fisis:** HDBSCAN tidak memaksakan titik-titik transisi yang ambigu untuk masuk ke dalam klaster manapun. Sekitar $50\%\text{--}53\%$ data yang memiliki variasi anatomi atau oklusi parsial secara objektif dialokasikan ke dalam kolam *noise*. Hasilnya, klaster yang diekstrak murni merupakan **inti gestur terpadat (core exemplars)** dengan batas pemisah yang sangat tegas.
2. **DBSCAN vs HDBSCAN (Fixed Epsilon vs Multi-Scale Persistence):**
   - DBSCAN membatasi diri pada satu skala densitas global $\varepsilon$. Jika $\varepsilon$ diperbesar, klaster gestur padat akan melebur menjadi satu raksasa. Jika $\varepsilon$ diperkecil, klaster gestur 2-tangan yang lebih longgar akan hancur menjadi noise.
   - HDBSCAN menyelesaikan paradoks ini dengan mengevaluasi persistensi klaster di sepanjang kontinum densitas, sehingga klaster alfabet yang padat (seperti gestur 1-tangan C dan O) dapat hidup berdampingan dengan klaster gestur 2-tangan yang lebih dinamis.
3. **Pemanfaatan Nilai Probabilitas Keanggotaan:**
   - Berbeda dengan algoritma lain yang hanya menghasilkan label biner, HDBSCAN menyediakan atribut `probabilities_`. Sampel dengan probabilitas $\ge 0,8$ dapat diekstraksi sebagai *golden reference samples* untuk melatih model supervised.

Artifacts tersimpan di: [HDBScan/outputHDBScan/](HDBScan/outputHDBScan)
- Visualisasi: `cluster_comparison_pca2d.png`, `dendrogram_comparison.png`, `metrics_comparison_barchart.png`, `cluster_cosine.png`, `dendrogram_cosine.png`
- Data komparasi: `metrics_comparison_summary.csv`, `evaluasi_hdbscan_bisindo.xlsx`

---

## 9. Matriks Komparasi Lintas Paradigma (15 Konfigurasi Algoritma)

Berikut adalah rangkuman matriks komparasi dari seluruh **15 konfigurasi eksperimen** (5 paradigma $\times$ 3 metrik jarak) yang telah dieksekusi pada dataset 7.893 sampel BISINDO, mencakup evaluasi internal tanpa supervisi (*unsupervised metrics*) dan validasi eksternal terhadap label ground-truth 26 alfabet BISINDO:

| No | Paradigma Algoritma | Metrik Jarak | Waktu Eksekusi (s) | Silhouette ($\uparrow$) | Davies–Bouldin ($\downarrow$) | Calinski–Harabasz ($\uparrow$) | Rasio Noise | Singleton | ARI (All) | NMI (All) | NMI (Core) | Purity (Core) | Borda Score | Peringkat Komposit |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | **K-Means** | Euclidean | 0,4273 | 0,2060 | 1,4273 | 941,27 | 0,00% | 0 | 0,2136 | 0,4933 | 0,4933 | 36,26% | 81,0 | **Peringkat 2** |
| 2 | **K-Means** | Manhattan | 0,5096 | 0,1956 | 1,4271 | 863,90 | 0,00% | 1 | 0,2190 | 0,5022 | 0,5022 | 36,75% | 80,0 | **Peringkat 3** |
| 3 | **K-Means** | **Cosine** | **0,3373** | **0,1847** | **1,6798** | **1.409,01** | **0,00%** | **0** | **0,2491** | **0,5237** | **0,5237** | **40,91%** | **83,0** | **Peringkat 1 (Best All-Rounder)** |
| 4 | **DBSCAN** | Euclidean | **0,1071** | -0,0027 | **0,9884** | 255,36 | 11,30% | 0 | 0,0655 | 0,3603 | 0,3994 | 22,27% | 44,0 | Peringkat 13 |
| 5 | **DBSCAN** | Manhattan | 0,6524 | 0,0673 | 1,1751 | 233,62 | 15,08% | 0 | 0,0636 | 0,3842 | 0,4373 | 27,39% | 37,0 | Peringkat 14 |
| 6 | **DBSCAN** | **Cosine** | 0,4648 | -0,2085 | 1,3389 | 143,09 | **6,65%** | 0 | 0,0661 | 0,3654 | 0,3903 | 23,38% | 37,0 | Peringkat 14 |
| 7 | **HAC (Bottom-Up)** | Euclidean | 1,7010 | 0,1182 | 1,1485 | 422,47 | 0,00% | 3 | 0,0870 | 0,3374 | 0,3374 | 21,35% | 53,0 | Peringkat 9 |
| 8 | **HAC (Bottom-Up)** | Manhattan | 1,7011 | 0,1282 | 1,2737 | 349,85 | 0,00% | 4 | 0,0803 | 0,3322 | 0,3322 | 17,84% | 47,0 | Peringkat 12 |
| 9 | **HAC (Bottom-Up)** | **Cosine** | 1,7010 | 0,0891 | 2,0123 | 1.003,78 | 0,00% | **0** | 0,1637 | 0,4375 | 0,4375 | 28,75% | 57,0 | Peringkat 7 |
| 10 | **Divisive (Top-Down)** | Euclidean | 2,2368 | 0,1557 | 1,6729 | 768,85 | 0,00% | 0 | 0,1996 | 0,4698 | 0,4698 | 34,09% | 62,0 | Peringkat 5 |
| 11 | **Divisive (Top-Down)** | Manhattan | 0,8067 | 0,1407 | 1,5631 | 660,59 | 0,00% | 2 | 0,1636 | 0,4254 | 0,4254 | 28,20% | 57,0 | Peringkat 7 |
| 12 | **Divisive (Top-Down)** | **Cosine** | **0,0979** | **0,1868** | **1,6262** | **1.408,53** | **0,00%** | **0** | **0,1845** | **0,4532** | **0,4532** | **33,41%** | **78,0** | **Peringkat 4 (Fastest Execution)** |
| 13 | **HDBSCAN (Density)** | Euclidean | 5,4956 | 0,3486 | 0,7467 | 513,99 | 49,88% | 0 | 0,0489 | 0,4353 | 0,7274 | 59,45% | 48,0 | Peringkat 11 |
| 14 | **HDBSCAN (Density)** | Manhattan | 5,6686 | 0,5479 | **0,7441** | 1.265,49 | 62,14% | 0 | 0,0234 | 0,4358 | **0,8593** | **83,90%** | 53,0 | Peringkat 9 |
| 15 | **HDBSCAN (Density)** | **Cosine** | **3,9883** | **0,4653** | **0,8203** | **1.539,26** | **53,34%** | **0** | **0,0412** | **0,4686** | **0,8003** | **71,84%** | **60,0** | **Peringkat 6 (Highest Core Purity)** |

---

## 10. Sintesis Final Comparison & Analisis 11 Diagram Visualisasi


### 10.1 `01_grouped_barchart_metrics.png` (Grouped Bar Charts Metrik Internal)
- **Deskripsi:** Menampilkan panel komparasi $2 \times 2$ untuk metrik Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index, dan Waktu Eksekusi (skala logaritmik) yang dikelompokkan berdasarkan 5 paradigma dengan batang warna fungsi jarak (Biru = Euclidean, Oranye = Manhattan, Hijau = Cosine).
- **Temuan Kunci:** Mengonfirmasi lonjakan dramatis indeks Calinski-Harabasz pada fungsi jarak Cosine (mencapai $1.408$ hingga $1.539$) dibandingkan Euclidean ($255$ s.d. $941$) dan Manhattan ($233$ s.d. $863$).

### 10.2 `02_radar_chart_distance_comparison.png` (Radar Profile Tiga Metrik Jarak)
- **Deskripsi:** Profil radar 6 dimensi yang membandingkan rata-rata teragregasi Euclidean, Manhattan, dan Cosine lintas seluruh paradigma pada dimensi: *Kohesi Internal (Silhouette)*, *Separasi Klaster (1/DBI)*, *Rasio Dispersi (CH)*, *Kecepatan Komputasi (1/Latency)*, *Retensi Data (100 - Noise%)*, dan *Kesesuaian Alfabet (NMI Ground-Truth)*.
- **Temuan Kunci:** Poligon Cosine Distance (Hijau) melingkupi area terbesar di hampir seluruh dimensi, membuktikan keunggulan invarian sudut (*angular alignment*) terhadap struktur koordinat kartesian.

### 10.3 `03_radar_chart_paradigm_comparison.png` (Radar Profile 5 Paradigma pada Jarak Cosine)
- **Deskripsi:** Membandingkan 5 paradigma algoritma (K-Means, DBSCAN, HAC, Divisive, HDBSCAN) yang diposisikan pada konfigurasi jarak terbaiknya (Cosine).
- **Temuan Kunci:** Menunjukkan profil trade-off komputasi: **Divisive** unggul mutlak pada efisiensi waktu, **K-Means** mendominasi keseimbangan partisi data penuh, sedangkan **HDBSCAN** memuncaki dimensi kohesi dan separasi spasial.

### 10.4 `04_heatmap_performance_matrix.png` (Heatmap Matriks Kinerja Ternormalisasi)
- **Deskripsi:** Matriks warna $15 \times 8$ (15 model $\times$ 8 dimensi evaluasi) di mana seluruh skor dinormalisasi min-max ke skala seragam $0\text{--}100$ (100 merepresentasikan performa terbaik).
- **Temuan Kunci:** Memberikan pemetaan komparatif instan: model density-based (HDBSCAN) mencetak skor mendekati 100 pada Silhouette dan Separasi namun rendah pada retensi data, sedangkan model partisi (K-Means dan Divisive) mencetak skor 100 pada retensi data dan kecepatan komputasi.

### 10.5 `05_heatmap_ranking_matrix.png` (Heatmap Peringkat dan Total Borda Count)
- **Deskripsi:** Memetakan peringkat numerik mutlak (Rank 1 s.d. 15) untuk setiap model di 7 metrik penilaian, diurutkan secara hierarkis berdasarkan **Total Skor Borda**.
- **Hasil Agregat:** **K-Means Cosine (Skor 83,0)** menempati Peringkat 1 Umum, disusul oleh **K-Means Euclidean (Skor 81,0)** di Peringkat 2, **K-Means Manhattan (Skor 80,0)** di Peringkat 3, dan **Divisive Cosine (Skor 78,0)** di Peringkat 4.

### 10.6 `06_tradeoff_silhouette_vs_time.png` (Pareto Efficiency Frontier: Kualitas vs Kecepatan)
- **Deskripsi:** Scatter plot logaritmik yang memetakan sumbu $X$ = Waktu Eksekusi (detik, skala log) terhadap sumbu $Y$ = Kualitas Kohesi (Silhouette Score).
- **Garis Batas Efisiensi Pareto (Pareto Frontier):**
  - *Titik Optimal Kecepatan:* **Divisive Cosine** $\rightarrow$ Waktu: **0,0979 s**, Silhouette: **0,1868** (Inference latency $< 0,02$ ms).
  - *Titik Optimal Keseimbangan Partisi:* **K-Means Cosine** $\rightarrow$ Waktu: **0,3373 s**, Silhouette: **0,1847**, NMI: **0,5237**.
  - *Titik Optimal Kemurnian Eksemplar:* **HDBSCAN Cosine** $\rightarrow$ Waktu: **3,9883 s**, Silhouette: **0,4653**, NMI Core: **0,8003**.

### 10.7 `07_tradeoff_noise_vs_quality.png` (Dampak Penyaringan Noise terhadap Silhouette)
- **Deskripsi:** Analisis korelasi antara persentase sampel yang dialokasikan sebagai derau (*noise*) terhadap kenaikan kohesi internal klaster, dilengkapi garis regresi tren linear dan 3 zona klasifikasi operasional:
  - *Zona Partisi Penuh (0% Noise):* K-Means, Divisive, HAC.
  - *Zona Pembersihan Ringan (6% – 15% Noise):* DBSCAN (membersihkan blur gerak dan oklusi).
  - *Zona Ekstraksi Eksemplar Inti (50% – 62% Noise):* HDBSCAN (memisahkan pose transisi non-standar).

### 10.8 `08_supervised_metrics_comparison.png` (Validasi Eksternal Ground Truth 26 Alfabet)
- **Deskripsi:** Menampilkan perbandingan 4 metrik validasi eksternal terhadap label alfabet A–Z: ARI (All), NMI (All), NMI (Core Exemplars), dan Purity (Core Exemplars).
- **Temuan Kunci:** Pada sampel inti non-noise, **HDBSCAN Manhattan** dan **HDBSCAN Cosine** mencatatkan **NMI sebesar 0,8003 s.d. 0,8593** dan **Purity 71,84% s.d. 83,90%**, membuktikan bahwa struktur klaster unsupervised HDBSCAN memiliki korespondensi yang sangat kuat dengan 26 huruf alfabet asli.

### 10.9 `09_pca2d_cluster_gallery.png` (Galeri Proyeksi PCA 2D Komparatif)
- **Deskripsi:** Panel visual matriks $2 \times 3$ berskala besar (5,6 MB) yang menyandingkan proyeksi 2D Principal Component Analysis (PC1 vs PC2) antara Ground Truth 26 Alfabet dengan 5 model terbaik masing-masing paradigma (K-Means Cosine, DBSCAN Cosine, HAC Cosine, Divisive Cosine, dan HDBSCAN Cosine).
- **Temuan Kunci:** Menunjukkan secara visual bagaimana proyeksi spherical memisahkan manifold gestur 1-tangan dan 2-tangan menjadi gugusan teratur yang menyerupai sebaran kelas ground truth.

### 10.10 `10_cluster_size_distribution.png` (Sebaran Ukuran Klaster & Deteksi Singleton)
- **Deskripsi:**
  - *Subplot Kiri:* Grafik batang deteksi klaster singleton (berisi hanya 1 sampel). Mengonfirmasi bahwa HAC Euclidean menghasilkan 3 singleton dan HAC Manhattan menghasilkan 4 singleton, sementara **HAC Cosine menghasilkan 0 singleton**.
  - *Subplot Kanan:* Boxplot dan stripplot sebaran ukuran sampel per klaster pada 5 model terbaik, memperlihatkan rentang partisi yang paling harmonis pada Divisive Cosine dan K-Means Cosine.

### 10.11 `11_master_infographic_dashboard.png` (Master Dashboard Poster Sintesis)
- **Deskripsi:** Kompilasi infografis ilmiah berskala besar (24 $\times$ 16 inci, 300 DPI) yang menggabungkan dalam 1 kanvas poster: Bar Chart Metrik Utama, Kurva Pareto Frontier Kualitas vs Waktu, Tabel Rekomendasi Terapan Arsitektural, Scatter Plot NMI Ground Truth vs Silhouette, dan Bar Chart Peringkat Borda Count agregat.

---

## 11. Analisis Kritis, Temuan Kontraintuitif & Diskusi Teoretis

Eksperimen komparatif ini menghasilkan beberapa temuan ilmiah mendalam yang relevan bagi literatur machine learning dan rekayasa visi komputer:

### 11.1 Dominasi Kinematika Cosine Distance pada Data Landmark Tangan
Mengapa Cosine Distance mengungguli Euclidean dan Manhattan di hampir seluruh paradigma?
- **Penyebab Fisis:** Koordinat landmark tangan 3D merupakan representasi kinematika kerangka tulang (*skeletal kinematics*). Ketika pengguna memperagakan suatu huruf (misalnya huruf "V" atau "L"), orientasi sudut antar-ruas sendi jari bersifat konstan. Sebaliknya, posisi absolut $(x, y, z)$ sangat bervariasi bergantung pada jarak subjek ke kamera (skala perspektif) dan ukuran fisik tangan pengguna.
- **Penyebab Geometris:** Dengan memproyeksikan data ke permukaan bola hiperdimensi (*unit hypersphere* $\|\mathbf{x}\|_2 = 1$), Cosine Distance meniadakan pengaruh magnitudo panjang vektor dan murni mengukur sudut deviasi arah ($\theta$):
  $$\|\mathbf{u} - \mathbf{v}\|_2^2 = 2 - 2 \cos(\theta)$$
  Hasilnya, gestur tangan dari berbagai subjek yang memperagakan huruf yang sama berkumpul rapat dalam manifold angular yang harmonis.

### 11.2 Dekonstruksi Patologi Klaster Singleton pada Hierarchical Agglomerative (Bottom-Up)
Mengapa HAC Complete Linkage pada metrik Euclidean dan Manhattan menghasilkan klaster singleton (berisi 1 sampel)?
- **Mekanisme Kegagalan:** Complete Linkage mendefinisikan jarak antar-klaster berdasarkan jarak terjauh (*farthest neighbor*):
  $$D(A, B) = \max \{ d(\mathbf{u}, \mathbf{v}) : \mathbf{u} \in A, \mathbf{v} \in B \}$$
  Pada ruang dimensi 127 (*curse of dimensionality*), sampel tangan yang memiliki orientasi sedikit miring atau outlier minor memiliki jarak $L_2$ dan $L_1$ terjauh yang sangat besar terhadap seluruh sampel lain. Sampel tersebut menolak bergabung dengan klaster manapun hingga level hierarki teratas. Ketika dendrogram dipotong pada tingkat $K=26$, outlier tersebut terisolasi menjadi klaster individu beranggotakan 1 sampel, sementara sampel lainnya menumpuk ke dalam klaster raksasa (hingga 3.176 sampel).
- **Solusi Cosine:** Cosine Distance mengeliminasi distorsi jarak akibat skala, sehingga Complete Linkage mampu membentuk 26 klaster seimbang tanpa satu pun singleton (klaster terkecil 11 sampel, terbesar 1.179 sampel).

### 11.3 Ketahanan Superior Divisive Hierarchical Clustering (Top-Down Bipartitioning)
Mengapa pendekatan Divisive (Top-Down) berhasil mengatasi kelemahan Bottom-Up (HAC)?
- **Mekanisme Regulasi Inersia:** Berbeda dengan HAC yang menggabungkan pasangan titik terdekat dari bawah, Divisive memulai dari 1 klaster global dan secara rekursif memilih klaster dengan **inersia/dispersi terbesar** untuk dibelah (*bisection*).
- Klaster yang hanya berisi sedikit sampel outlier memiliki inersia mendekati 0, sehingga algoritma biseksi tidak akan pernah memprioritaskan pembelahan pada klaster kecil tersebut. Outlier secara alami terserap ke dalam partisi mayoritas tanpa merusak struktur hierarki.
- Selain itu, kompleksitas waktu Divisive Bisecting K-Means hanya $O(K \cdot N \cdot d)$, jauh lebih efisien dibanding HAC yang membutuhkan komputasi matriks jarak penuh $O(N^2 \cdot d)$. Hal ini terbukti dari waktu eksekusi Divisive Cosine yang hanya **0,0979 detik** untuk 7.893 sampel.

### 11.4 Dekonstruksi Paradoks Silhouette Negatif pada Cosine DBSCAN
Mengapa Cosine DBSCAN memiliki Silhouette Score negatif (-0,2085) padahal berhasil mengelompokkan 93,35% sampel dengan Calinski-Harabasz yang baik?
- **Penyebab Metrik Formula:** Implementasi standar `silhouette_score` Scikit-Learn menghitung jarak Euclidean garis lurus antar-sampel pada ruang fitur ter-assign.
- Pada Cosine DBSCAN, klaster terbentuk di sepanjang kelengkungan permukaan bola (*hypersphere surface*). Klaster yang melengkung ini, jika diukur menggunakan jarak garis lurus Euclidean yang memotong bagian dalam bola (*chordal distance*), tampak bertumpukan secara spasial linier, padahal secara angular terpisah tegas. Oleh karena itu, Silhouette Euclidean memberikan penalti artifisial terhadap klaster non-konveks angular.

### 11.5 Lonjakan Spektakuler NMI Core HDBSCAN (0,8003 / 80%)
Mengapa HDBSCAN Cosine mencapai NMI 0,8003 dan Kemurnian (Purity) 71,84% pada sampel inti, padahal mengalokasikan 53,34% sampel sebagai noise?
- **Resolusi Heterogenitas Kerapatan:** Alfabet BISINDO memiliki perbedaan densitas alami: huruf 1-tangan (seperti C, E, O) memiliki variasi pose yang sangat kecil dan sangat padat, sedangkan huruf 2-tangan (seperti B, D, W) memiliki variasi spasial yang lebih luas. Algoritma dengan radius tetap $\varepsilon$ kaku (seperti DBSCAN) tidak dapat menangani kedua kelompok ini secara optimal.
- HDBSCAN menggunakan konsep **Mutual Reachability Distance** dan memangkas hierarki melalui metode **Excess of Mass (EOM)**. Titik-titik transisi yang ambigu (misalnya saat jari tangan sedikit bergeser atau mengalami blur) dipisahkan ke kolam noise. Hasilnya, 3.683 sampel yang tersisa murni merepresentasikan **eksemplar emas (golden exemplars)** dengan batas pemisah yang sangat tegas, menghasilkan kesesuaian hampir 1:1 terhadap label ground truth alfabet A–Z.

### 11.6 Garis Batas Efisiensi Pareto (Pareto Frontier) dalam Rekayasa
Analisis trade-off kualitas vs kecepatan membuktikan bahwa tidak ada satu algoritma tunggal yang terbaik untuk seluruh kebutuhan. Tiga model membentuk **Pareto Efficiency Frontier**:
1. **Divisive Cosine (0,098 s, Silhouette 0,1868):** Model tercepat untuk inferensi real-time edge device.
2. **K-Means Cosine (0,337 s, Silhouette 0,1847, NMI 0,5237):** Model partisi penuh paling seimbang dengan skor Borda tertinggi (Rank 1).
3. **HDBSCAN Cosine (3,988 s, Silhouette 0,4653, Core NMI 0,8003):** Model kemurnian tertinggi untuk kurasi data latih.

---

## 12. Rekomendasi Terapan Rekayasa untuk PBL BisindoVision

Berdasarkan sintesis komparatif di atas, dirancang arsitektur implementasi multi-tahap terpadu untuk sistem pengenalan Bahasa Isyarat Indonesia (BisindoVision):

```
+---------------------------------------------------------------------------------------------------+
|                            PIPELINE ARSITEKTUR TERPADU BISINDOVISION                              |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Video Kamera 30-60 FPS]                                                                         |
|            │                                                                                      |
|            ▼                                                                                      |
|  [MediaPipe HandLandmarker Tasks] ──> Ekstraksi Vektor Fitur 127 Dimensi                          |
|            │                                                                                      |
|            ▼                                                                                      |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │ MODUL 1: INFERENSI WAKTU NYATA PADA PERANGKAT MOBILE (ON-DEVICE REAL-TIME ENGINE)           │  |
|  │ - Pilihan: Spherical Divisive Cosine Tree atau Spherical K-Means Cosine                     │  |
|  │ - Kinerja: Penelusuran pohon biner sedalam 5 level (< 0,02 ms per frame)                   │  |
|  │ - Memori : Hanya 13,2 KB float32 (26 Centroid x 127 dimensi), sangat hemat baterai          │  |
|  └─────────────────────────────────────────────────────────────────────────────────────────────┘  |
|            │                                                                                      |
|            ▼                                                                                      |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │ MODUL 2: PIPELINE PEMBERSIHAN DATASET OTOMATIS (DATA CLEANSING PIPELINE)                    │  |
|  │ - Pilihan: DBSCAN Cosine (Eps = 0,067, MinPts = 10)                                         │  |
|  │ - Peran  : Memfilter 6,65% frame video anomali (tangan blur, oklusi, deteksi parsial)       │  |
|  │            tanpa membuang variasi gestur normal subjek                                      │  |
|  └─────────────────────────────────────────────────────────────────────────────────────────────┘  |
|            │                                                                                      |
|            ▼                                                                                      |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │ MODUL 3: KURASI DATASET EMAS & CURRICULUM LEARNING (GOLDEN EXEMPLAR EXTRACTION)             │  |
|  │ - Pilihan: HDBSCAN Cosine (mcs = 53, Ambang Probabilitas Membership >= 0.8)                 │  |
|  │ - Peran  : Mengekstrak 3.683 sampel murni (NMI 0,8003 terhadap alfabet ground-truth)       │  |
|  │            sebagai data latih prioritas tinggi untuk model Supervised Deep Learning         │  |
|  └─────────────────────────────────────────────────────────────────────────────────────────────┘  |
|            │                                                                                      |
|            ▼                                                                                      |
|  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  |
|  │ MODUL 4: TAKSONOMI KLASIFIKASI DUA TAHAP (TWO-STAGE HIERARCHICAL CLASSIFIER)                │  |
|  │ - Pilihan: Pohon Hierarki HAC Cosine & Divisive Tree Linkage Matrix                         │  |
|  │ - Tahap 1: Klasifikasi Makro (Gestur 1-Tangan vs 2-Tangan) via Flag is_two_handed           │  |
|  │ - Tahap 2: Klasifikasi Mikro Huruf Spesifik pada kelompok bentuk jari serupa (U vs V, C/O)  │  |
|  └─────────────────────────────────────────────────────────────────────────────────────────────┘  |
+---------------------------------------------------------------------------------------------------+
```

---

## 13. Struktur Direktori Repositori

```
E:/Dokumen/Kuliah/Semester 5/Pembelajaran Mesin/UTS_Bisindo_Klasterisasi_Kelompok-1/
├── train/                               # Dataset mentah 26 kelas alfabet BISINDO (9.169 citra)
├── readme.md                            # Laporan teknis komprehensif 5 paradigma
│
├── KMeans/                              # Modul Klasterisasi K-Means
│   ├── extract_features.py              # Script ekstraktor fitur 127D MediaPipe
│   ├── hand_landmarker.task             # Model binary MediaPipe HandLandmarker Tasks
│   ├── dataset_bisindo_landmarks.csv    # Fitur format CSV
│   ├── dataset_bisindo_landmarks.npz    # Fitur format NumPy NPZ
│   ├── kmeans_distance_comparison.ipynb # Notebook komparasi K-Means
│   ├── kmeans_euclidean.ipynb           # Notebook individual Euclidean
│   ├── kmeans_manhattan.ipynb           # Notebook individual Manhattan
│   ├── kmeans_cosine.ipynb              # Notebook individual Cosine
│   └── outputKmeans/                    # Visualisasi, CSV, dan Excel evaluasi
│
├── DBScan/                              # Modul Klasterisasi DBSCAN
│   ├── extract_features.py              # Script ekstraktor fitur
│   ├── hand_landmarker.task             # Model MediaPipe
│   ├── dataset_bisindo_landmarks.csv    # Fitur CSV
│   ├── dataset_bisindo_landmarks.npz    # Fitur NPZ
│   ├── dbscan_distance_comparison.ipynb # Notebook komparasi DBSCAN
│   ├── dbscan_euclidean.ipynb           # Notebook individual Euclidean
│   ├── dbscan_manhattan.ipynb           # Notebook individual Manhattan
│   ├── dbscan_cosine.ipynb              # Notebook individual Cosine
│   └── outputDBScan/                    # Visualisasi, CSV, dan Excel evaluasi
│
├── HAC/                                 # Modul Klasterisasi Hirarki Aglomeratif (Bottom-Up)
│   ├── extract_features.py              # Script ekstraktor fitur
│   ├── hand_landmarker.task             # Model MediaPipe
│   ├── dataset_bisindo_landmarks.csv    # Fitur CSV
│   ├── dataset_bisindo_landmarks.npz    # Fitur NPZ
│   ├── hac_distance_comparison.ipynb    # Notebook komparasi HAC Complete Linkage
│   ├── hac_euclidean.ipynb              # Notebook individual Euclidean
│   ├── hac_manhattan.ipynb              # Notebook individual Manhattan
│   ├── hac_cosine.ipynb                 # Notebook individual Cosine
│   └── outputHAC/                       # Visualisasi, CSV, dan Excel evaluasi
│
├── Divisive/                            # Modul Klasterisasi Hirarki Divisive (Top-Down)
│   ├── extract_features.py              # Script ekstraktor fitur
│   ├── hand_landmarker.task             # Model MediaPipe
│   ├── dataset_bisindo_landmarks.csv    # Fitur CSV
│   ├── dataset_bisindo_landmarks.npz    # Fitur NPZ
│   ├── divisive_distance_comparison.ipynb # Notebook komparasi Divisive Bipartitioning
│   ├── divisive_euclidean.ipynb         # Notebook individual Euclidean
│   ├── divisive_manhattan.ipynb         # Notebook individual Manhattan
│   ├── divisive_cosine.ipynb            # Notebook individual Cosine
│   └── outputDivisive/                  # Visualisasi, CSV, dan Excel evaluasi
│
├── HDBScan/                             # Modul Klasterisasi Hirarki Densitas HDBSCAN
│   ├── extract_features.py              # Script ekstraktor fitur
│   ├── hand_landmarker.task             # Model MediaPipe
│   ├── dataset_bisindo_landmarks.csv    # Fitur CSV
│   ├── dataset_bisindo_landmarks.npz    # Fitur NPZ
│   ├── hdbscan_distance_comparison.ipynb # Notebook komparasi HDBSCAN Multi-Scale
│   ├── hdbscan_euclidean.ipynb          # Notebook individual Euclidean HDBSCAN
│   ├── hdbscan_manhattan.ipynb          # Notebook individual Manhattan HDBSCAN
│   ├── hdbscan_cosine.ipynb             # Notebook individual Cosine HDBSCAN
│   └── outputHDBScan/                   # Visualisasi, CSV, dan Excel evaluasi
│
└── Final Comparison/                    # Modul Sintesis Komparasi Final Lintas Paradigma (15 Model)
    ├── dataset_bisindo_landmarks.csv    # Fitur format CSV
    ├── dataset_bisindo_landmarks.npz    # Fitur format NumPy NPZ
    ├── final_comparison.ipynb           # Master Notebook komparasi terpadu 15 model
    └── outputFinalComparison/           # 11 Diagram visualisasi HD (300 DPI), CSV, Excel multi-sheet
```

---

## 14. Panduan Menjalankan Eksperimen

### 14.1 Persyaratan Lingkungan (Environment Prerequisites)
Pastikan lingkungan Python 3.10+ telah terpasang dengan pustaka-pustaka berikut:
```bash
pip install mediapipe opencv-python numpy pandas scikit-learn scipy matplotlib seaborn openpyxl nbconvert nbformat
```

### 14.2 Ekstraksi Fitur (Opsional jika file NPZ/CSV sudah ada)
Untuk mengekstrak ulang koordinat landmark dari dataset mentah:
```bash
cd "E:\Dokumen\Kuliah\Semester 5\Pembelajaran Mesin\UTS_Bisindo_Klasterisasi_Kelompok-1\KMeans"
python extract_features.py
```

### 14.3 Menjalankan Notebook Eksperimen Per Algoritma
Masing-masing notebook komparasi dapat dieksekusi secara otomatis melalui command line:

1. **Eksperimen K-Means:**
   ```bash
   jupyter nbconvert --to notebook --execute "E:\Dokumen\Kuliah\Semester 5\Pembelajaran Mesin\UTS_Bisindo_Klasterisasi_Kelompok-1\KMeans\kmeans_distance_comparison.ipynb"
   ```
2. **Eksperimen DBSCAN:**
   ```bash
   jupyter nbconvert --to notebook --execute "E:\Dokumen\Kuliah\Semester 5\Pembelajaran Mesin\UTS_Bisindo_Klasterisasi_Kelompok-1\DBScan\dbscan_distance_comparison.ipynb"
   ```
3. **Eksperimen HAC (Agglomerative):**
   ```bash
   jupyter nbconvert --to notebook --execute "E:\Dokumen\Kuliah\Semester 5\Pembelajaran Mesin\UTS_Bisindo_Klasterisasi_Kelompok-1\HAC\hac_distance_comparison.ipynb"
   ```
4. **Eksperimen Divisive (Top-Down Bipartitioning):**
   ```bash
   jupyter nbconvert --to notebook --execute "E:\Dokumen\Kuliah\Semester 5\Pembelajaran Mesin\UTS_Bisindo_Klasterisasi_Kelompok-1\Divisive\divisive_distance_comparison.ipynb"
   ```
5. **Eksperimen HDBSCAN (Density-Based Hierarchy):**
   ```bash
   jupyter nbconvert --to notebook --execute "E:\Dokumen\Kuliah\Semester 5\Pembelajaran Mesin\UTS_Bisindo_Klasterisasi_Kelompok-1\HDBScan\hdbscan_distance_comparison.ipynb"
   ```

### 14.4 Menjalankan Master Final Comparison (Sintesis 15 Konfigurasi)
Untuk menjalankan ulang seluruh analisis komparatif terpadu, menghitung metrik ground truth alignment, serta meregenerasi ke-11 diagram visualisasi HD:
```bash
cd "E:\Dokumen\Kuliah\Semester 5\Pembelajaran Mesin\UTS_Bisindo_Klasterisasi_Kelompok-1\Final Comparison"
```
