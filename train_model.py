import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# ==============================
# 1. DATASET & PREPROCESSING
# ==============================
# Menetapkan laluan (path) ke folder training dan testing
train_dir = os.path.join('dataset', 'training')
test_dir = os.path.join('dataset', 'testing')

# Augmentasi data HANYA untuk folder training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Untuk testing/validasi, imej hanya di-rescale (tiada augmentasi agar hasil ujian tulen)
test_datagen = ImageDataGenerator(rescale=1./255)

# Mengambil data latihan
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Mengambil data validasi/testing
val_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Khas untuk Confusion Matrix, fungsi shuffle WAJIB dimatikan (False)
val_generator_eval = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False 
)

print(f"\n[INFO] Pemetaan Kelas dari Keras: {train_generator.class_indices}\n")

# ==============================
# 2. MODEL CNN
# ==============================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==============================
# 3. TRAINING MODEL
# ==============================
print("Memulakan latihan model CNN...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100
)

# ==============================
# BUAT FOLDER PENYIMPANAN HASIL
# ==============================
output_dir = 'hasil_training'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n[INFO] Folder '{output_dir}' berhasil dibuat.")

# ==============================
# 4. SAVE MODEL
# ==============================
model_path = os.path.join(output_dir, 'bone_fracture_model.h5')
model.save(model_path)
print(f"\nModel berjaya disimpan ke: {model_path}")

# ==============================
# 5. EVALUASI HASIL & CONFUSION MATRIX
# ==============================
# Plot Accuracy dan Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Train')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Akurasi Model')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Train')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Loss Model')
plt.legend()

# Simpan Grafik Akurasi & Loss ke folder
acc_loss_path = os.path.join(output_dir, 'grafik_akurasi_loss.png')
plt.savefig(acc_loss_path, bbox_inches='tight')
print(f"[SUKSES] Grafik Akurasi & Loss disimpan di: {acc_loss_path}")
plt.close() # Menutup background proses agar tidak ada pop-up yang nyangkut

# Membuat Prediksi untuk Confusion Matrix
print("\nMengira Confusion Matrix...")
Y_pred = model.predict(val_generator_eval)
y_pred_classes = np.where(Y_pred > 0.5, 1, 0) # Nilai ambang 0.5

true_classes = val_generator_eval.classes
class_labels = list(val_generator_eval.class_indices.keys())

# Memaparkan Laporan Klasifikasi
report = classification_report(true_classes, y_pred_classes, target_names=class_labels)
print("\nLaporan Klasifikasi (Classification Report):")
print(report)

# Simpan laporan teks ke dalam file TXT
report_path = os.path.join(output_dir, 'classification_report.txt')
with open(report_path, 'w') as f:
    f.write(report)
print(f"[SUKSES] Laporan teks disimpan di: {report_path}")

# Memplot Confusion Matrix
cm = confusion_matrix(true_classes, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Pengesanan Fraktur Tulang')
plt.ylabel('Label Sebenar')
plt.xlabel('Prediksi AI')

# Simpan Confusion Matrix ke folder
cm_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(cm_path, bbox_inches='tight')
print(f"[SUKSES] Confusion Matrix disimpan di: {cm_path}")
plt.close() 

# ==============================
# TAMPILKAN PERSENTASE AKURASI
# ==============================
# Mengambil nilai akurasi validasi pada epoch terakhir
final_accuracy = history.history['val_accuracy'][-1] * 100

print("\n" + "="*40)
print(f"🎯 AKURASI AKHIR MODEL: {final_accuracy:.2f}%")
print("="*40 + "\n")# Selesai