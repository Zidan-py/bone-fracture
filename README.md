# 🦴 Deteksi Fraktur Tulang (Bone Fracture Detection) dengan JST (CNN)

Proyek ini adalah implementasi **Jaringan Saraf Tiruan (JST)** menggunakan arsitektur *Convolutional Neural Network* (CNN) untuk mendeteksi apakah sebuah citra X-Ray tulang mengalami patah (fraktur) atau dalam kondisi normal. 

Sistem ini dilengkapi dengan antarmuka web interaktif menggunakan **Streamlit** dan dibangun untuk memenuhi kriteria proyek akhir yang mencakup: Pemilihan Dataset, Preprocessing Data, Pembagian Data Train-Test, Implementasi JST (Deep Learning), Evaluasi Model, dan Pembuatan GUI.

---

## 📂 Struktur Repositori

```text
proyek_bone_fracture/
│
├── .venv311/                   # Virtual Environment Python
├── dataset/                    # Direktori dataset citra X-Ray
│   ├── fractured/              # Data citra patah tulang
│   ├── normal/                 # Data citra tulang sehat
│   ├── testing/                # Data uji (Test Split)
│   └── training/               # Data latih (Train Split)
│
├── hasil_training/             # Output hasil evaluasi dan model
│   ├── bone_fracture_model.h5  # Model AI CNN yang telah dilatih
│   ├── classification_report.txt # Laporan presisi, recall, f1-score
│   ├── confusion_matrix.png    # Visualisasi evaluasi model
│   └── grafik_akurasi_loss.png # Grafik performa saat training
│
├── .gitignore                  # File pengecualian Git
├── app_gui.py                  # Skrip Antarmuka/GUI Streamlit (Web)
├── cek_header.py               # Skrip pengecekan metadata/header gambar
├── test_model.py               # Skrip pengujian model berbasis CLI
└── train_model.py              # Skrip utama untuk melatih model CNN
graph TD
    A[Mulai: Buka Aplikasi Web] --> B[Upload Gambar X-Ray]
    B --> C{Cek Grayscale/Warna?}
    
    C -- Ada Warna Dominan --> D[Muncul Peringatan: Gambar Bukan X-Ray Murni]
    D --> E{User Centang Bypass?}
    E -- Tidak --> F[Proses Dihentikan]
    E -- Ya --> G[Preprocessing Data]
    
    C -- Hitam Putih (Valid) --> G
    
    G --> G1[Ubah ke RGB]
    G1 --> G2[Resize ke 224x224 px]
    G2 --> G3[Normalisasi Piksel / 255.0]
    G3 --> G4[Expand Dimensions Array]
    
    G4 --> H[Proses Prediksi oleh Model CNN .h5]
    H --> I[Ekstrak Nilai Mentah & Confidence Score]
    
    I --> J{Confidence >= 55%?}
    
    J -- Tidak --> K[Peringatan: AI Ragu-ragu / Gambar Tidak Valid]
    J -- Ya --> L{Kelas Prediksi?}
    
    L -- Kelas 0 --> M[Tampilkan Hasil: FRAKTUR PATAH TULANG]
    L -- Kelas 1 --> N[Tampilkan Hasil: NORMAL TIDAK PATAH]
    
    K --> O[Selesai]
    M --> O
    N --> O






