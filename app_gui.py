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
# FUNGSI VALIDASI
# =========================
def is_valid_xray_heuristic(img_pil):
    """
    Fungsi sederhana untuk mengecek apakah gambar adalah hitam-putih (grayscale).
    X-Ray umumnya memiliki nilai channel R, G, B yang sama atau sangat mirip.
    """
    img_np = np.array(img_pil.convert("RGB")).astype(float)
    r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
    
    # Menghitung perbedaan rata-rata antar channel warna
    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    
    # Jika perbedaan channel warna tinggi, berarti gambar berwarna (bukan X-Ray)
    if diff_rg > 30.0 or diff_rb > 30.0:
        return False
    return True

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
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    
    # Validasi apakah gambar kemungkinan besar X-Ray (Hitam-Putih)
    if not is_valid_xray_heuristic(image):
        st.warning("⚠️ **Peringatan:** Gambar terdeteksi memiliki warna (bukan X-Ray hitam-putih murni). Sistem menghentikan proses untuk mencegah hasil yang tidak akurat.")
        
        # Tambahkan tombol centang untuk memaksa AI memproses gambar
        force_process = st.checkbox("Saya yakin ini adalah gambar X-Ray digital, lanjutkan analisis.")
        
        # Jika belum dicentang, hentikan aplikasi di sini
        if not force_process:
            st.stop()

    # =========================
    # PREPROCESSING GAMBAR
    # =========================
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # =========================
    # PROSES PREDIKSI YANG DIPERBAIKI
    # =========================
    with st.spinner("AI sedang menganalisis gambar..."):
        try:
            # Ambil seluruh array output dari AI (bukan cuma index [0][0])
            raw_prediction = model.predict(img_array)[0]
            
            # --- TAMPILKAN NILAI MENTAH (DEBUG) ---
            st.code(f"Nilai mentah dari AI: {raw_prediction}", language="python")

            # Cek tipe output model (Sigmoid vs Softmax)
            if len(raw_prediction) == 2:
                # Jika AI mengeluarkan 2 angka (Categorical/Softmax)
                class_index = np.argmax(raw_prediction) # Pilih index dengan nilai tertinggi
                confidence = raw_prediction[class_index] * 100
            else:
                # Jika AI mengeluarkan 1 angka (Binary/Sigmoid)
                score = raw_prediction[0]
                if score >= 0.5:
                    class_index = 1
                    confidence = score * 100
                else:
                    class_index = 0
                    confidence = (1 - score) * 100

        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
            st.write(traceback.format_exc())
            st.stop()

    # =========================
    # LOGIKA HASIL (DENGAN THRESHOLD)
    # =========================
    st.subheader("📋 Hasil Analisis AI")

    # Mapping Keras berdasarkan urutan folder saat training
    if class_index == 1:
        label = "NORMAL (TIDAK PATAH)"
        is_fractured = False
    else:
        label = "FRAKTUR (PATAH TULANG)"
        is_fractured = True

    # --- FILTER THRESHOLD UNTUK GAMBAR ANEH ---
    if confidence < 55.0:
        st.warning(f"⚠️ **AI Ragu-ragu!** (Confidence: {confidence:.2f}%)")
        st.info("Gambar ini kemungkinan besar **BUKAN gambar X-Ray tulang** yang valid, atau model AI gagal mengenali pola tulang dengan baik.")
    else:
        if is_fractured:
            st.error(f"🔴 HASIL: {label}")
            st.write(f"Tingkat Kepercayaan (Confidence): **{confidence:.2f}%**")
            st.warning("Terindikasi adanya retakan atau kerusakan serius pada struktur tulang.")
        else:
            st.success(f"🟢 HASIL: {label}")
            st.write(f"Tingkat Kepercayaan (Confidence): **{confidence:.2f}%**")
            st.info("Tidak ditemukan indikasi fraktur yang jelas pada gambar X-Ray ini.")