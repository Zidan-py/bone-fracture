# 🦴 Bone Fracture Detection Using Deep Learning

Aplikasi deteksi fraktur tulang berbasis Deep Learning menggunakan Convolutional Neural Network (CNN) dan Streamlit.

## 📌 Deskripsi

Proyek ini bertujuan untuk mendeteksi apakah citra X-Ray tulang mengalami fraktur (patah tulang) atau tidak menggunakan model CNN. Pengguna dapat mengunggah gambar X-Ray melalui antarmuka web yang dibuat menggunakan Streamlit.

---

## 🚀 Fitur

* Upload gambar X-Ray tulang.
* Prediksi kondisi tulang (Fracture / Normal).
* Tampilan antarmuka menggunakan Streamlit.
* Model CNN berbasis TensorFlow dan Keras.
* Visualisasi hasil prediksi.

---

## 🛠️ Teknologi yang Digunakan

* Python 3.11
* TensorFlow
* Keras
* Streamlit
* NumPy
* Matplotlib
* Scikit-Learn
* Pillow

---

## 📂 Struktur Folder

```text
proyek_bone_fracture/
│
├── app_gui.py
├── train_model.py
├── test_model.py
├── dataset/
├── hasil_training/
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalasi

Clone repository:

```bash
git clone https://github.com/Zidan-py/bone-fracture.git
cd bone-fracture
```

Buat virtual environment:

```bash
python -m venv .venv311
```

Aktifkan environment:

```bash
.venv311\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Menjalankan Aplikasi

```bash
streamlit run app_gui.py
```

Setelah itu buka browser pada:

```text
http://localhost:8501
```

---

## 🧠 Model Deep Learning

Model menggunakan Convolutional Neural Network (CNN) dengan:

* Conv2D
* MaxPooling2D
* Flatten Layer
* Dense Layer
* Dropout

Output model:

* Fracture
* Normal

---

## 📈 Hasil

Model mampu melakukan klasifikasi citra X-Ray tulang berdasarkan data pelatihan yang digunakan.

---

## 👨‍💻 Developer

**Zidan Alhafiz**

GitHub:
https://github.com/Zidan-py
