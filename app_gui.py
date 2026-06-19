import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import traceback

# =========================
# CONFIG UI
# =========================
st.set_page_config(page_title="Deteksi Fraktur Tulang (AI)", layout="centered")

st.title("🦴 Sistem AI: Deteksi Fraktur Tulang")
st.write("""
Upload gambar X-Ray tulang, sistem akan mendeteksi apakah **FRAKTUR atau NORMAL** menggunakan CNN.
""")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    try:
        # Mengambil model dari dalam folder hasil_training
        return tf.keras.models.load_model("hasil_training/bone_fracture_model.h5")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.write(traceback.format_exc())
        return None

model = load_model()

# Cek apakah model berhasil dimuat
if model is None:
    st.stop() # Hentikan aplikasi jika model tidak ada

# =========================
# AMBIL CLASS INDEX (Untuk Fallback saja)
# =========================
try:
    # Beberapa model h5 menyimpan metadata nama kelas
    class_names = list(model.class_names)
except:
    # Fallback manual berdasarkan asumsi urutan alfabet folder:
    # 0 = fractured, 1 = not_fractured
    class_names = ["fractured", "not_fractured"] 

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader(
    "Unggah gambar X-Ray (PNG/JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar X-Ray yang diunggah", use_container_width=True)

    # =========================
    # PREPROCESSING GAMBAR
    # =========================
    # 1. Paksa ke RGB (menghindari error jika input grayscale atau RGBA)
    img = image.convert("RGB")
    
    # 2. Resize sesuai input model saat training
    img = img.resize((224, 224))

    # 3. Ubah ke array dan normalisasi (harus sama dengan ImageDataGenerator training)
    img_array = np.array(img).astype("float32") / 255.0
    
    # 4. Tambah dimensi batch (Model expect shape (1, 224, 224, 3))
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PROSES PREDIKSI
    # =========================
    with st.spinner("AI sedang menganalisis gambar..."):
        try:
            # Output model biner sigmoid adalah nilai tunggal probabilitas kelas 1
            prediction_score = model.predict(img_array)[0][0]
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
            st.write(traceback.format_exc())
            st.stop()

    # =========================
    # LOGIKA HASIL (FINAL - MAPPING ASLI)
    # =========================
    # Mapping Keras berdasarkan abjad folder:
    # 0 = fractured
    # 1 = not_fractured
    
    st.subheader("📋 Hasil Analisis AI")

    if prediction_score >= 0.5:
        # Nilai mendekati 1 -> not_fractured (Normal)
        label = "NORMAL (TIDAK PATAH)"
        confidence = prediction_score * 100

        st.success(f"🟢 HASIL: {label}")
        st.write(f"Tingkat Kepercayaan (Confidence): **{confidence:.2f}%**")
        st.info("Tidak ditemukan indikasi fraktur yang jelas pada gambar X-Ray ini.")

    else:
        # Nilai mendekati 0 -> fractured (Patah)
        label = "FRAKTUR (PATAH TULANG)"
        confidence = (1 - prediction_score) * 100

        st.error(f"🔴 HASIL: {label}")
        st.write(f"Tingkat Kepercayaan (Confidence): **{confidence:.2f}%**")
        st.warning("Terindikasi adanya retakan atau kerusakan serius pada struktur tulang.")