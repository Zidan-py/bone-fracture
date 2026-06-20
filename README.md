# 🦴 Deteksi Fraktur Tulang (Bone Fracture Detection) dengan JST (CNN)

Proyek ini adalah implementasi **Jaringan Saraf Tiruan (JST)** menggunakan arsitektur *Convolutional Neural Network* (CNN) untuk mendeteksi apakah sebuah citra X-Ray tulang mengalami patah (fraktur) atau dalam kondisi normal. 

Sistem ini dilengkapi dengan antarmuka web interaktif menggunakan **Streamlit** dan dibangun untuk memenuhi kriteria evaluasi model AI yang mencakup: Pemilihan Dataset, Preprocessing Data, Pembagian Data Train-Test, Implementasi JST (Deep Learning), Evaluasi Model, dan Pembuatan GUI.

---

## 📂 Struktur Repositori

```text
proyek_bone_fracture/
│
├── assets/                     # Folder penyimpanan gambar
│   └── licensed-images.jpg     # Gambar ilustrasi AI
├── dataset/                    # Direktori dataset citra X-Ray
├── hasil_training/             # Output hasil evaluasi dan model
├── app_gui.py                  # Skrip Antarmuka/GUI Streamlit
└── train_model.py              # Skrip utama untuk melatih model CNN

```

📊 Flowchart Sistem Aplikasi (GUI)
Cuplikan kode
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
    J -- Tidak --> K[Peringatan: AI Ragu-ragu]
    J -- Ya --> L{Kelas Prediksi?}
    L -- Kelas 0 --> M[Hasil: FRAKTUR]
    L -- Kelas 1 --> N[Hasil: NORMAL]
    K --> O[Selesai]
    M --> O
    N --> O
🧠 Bagaimana AI Mendeteksi Fraktur?
Proyek ini menggunakan arsitektur Convolutional Neural Network (CNN). Berikut adalah ilustrasi cara AI memproses citra tulang:

Input Layer: Citra X-Ray (224x224 piksel) dimasukkan ke sistem.

Feature Extraction: AI mendeteksi pola tepi tulang, tekstur, dan garis retakan.

Classification: Data dikonversi menjadi probabilitas untuk menentukan kelas Fraktur atau Normal.

⚙️ Fitur Utama Aplikasi
Model CNN Kustom: Mengenali pola diskontinuitas pada tulang.

Validasi Heuristik Warna: Mencegah input gambar non-X-Ray dengan filter RGB.

Sistem Threshold: Memblokir hasil jika confidence AI di bawah 55%.

GUI Interaktif: Antarmuka web yang rapi dan mudah digunakan.

🚀 Cara Menjalankan Proyek
Persiapan: Aktifkan environment: .venv311\Scripts\activate

Instalasi: pip install tensorflow keras numpy pillow streamlit matplotlib scikit-learn

Training: python train_model.py

Jalankan Web: streamlit run app_gui.py

👨‍💻 Pengembang
M Zidan Al Hafiz - Universitas Bina Bangsa Getsempena
Proyek ini dikembangkan sebagai bentuk penyelesaian tugas akhir pengembangan sistem kecerdasan buatan.