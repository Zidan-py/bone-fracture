# 🦴 Bone Fracture Detection Using Deep Learning

An AI-based web application for detecting bone fractures from X-Ray images using a Convolutional Neural Network (CNN). This project provides an easy-to-use interface that allows users to upload X-Ray images and obtain fracture predictions along with confidence scores.

---

## 📌 Overview

Bone fractures are common injuries that require accurate diagnosis. This project applies Deep Learning techniques to assist in detecting fractures from X-Ray images.

The system performs image preprocessing, analyzes the X-Ray using a trained CNN model, and predicts whether the bone is fractured or normal.

---

## ✨ Features

* Upload X-Ray images through a web interface.
* Automatic image validation.
* Grayscale and color detection.
* Image preprocessing pipeline.
* CNN-based fracture classification.
* Confidence score calculation.
* Warning for low-confidence predictions.
* User-friendly interface.
* Visualization of training results.

---

## 🛠️ Technologies Used

* Python 3.11
* TensorFlow / Keras
* NumPy
* OpenCV
* Pillow (PIL)
* Streamlit / Flask
* Matplotlib
* Scikit-learn

---

## 🧠 Deep Learning Model

The model uses a Convolutional Neural Network (CNN) architecture trained on bone X-Ray images.

### Classification Classes

| Class | Description |
| ----- | ----------- |
| 0     | Fracture    |
| 1     | Normal      |

---

## 🔄 System Workflow

1. User uploads an X-Ray image.
2. The system checks whether the image is a valid X-Ray.
3. Image preprocessing:

   * Convert to RGB
   * Resize to 224×224 pixels
   * Normalize pixel values
   * Expand image dimensions
4. CNN model performs prediction.
5. Confidence score is calculated.
6. Final result is displayed.

---

## 📊 Training Results

The project includes:

* Accuracy and loss graphs
* Confusion matrix
* Classification report

These results can be found inside:

```text
hasil_training/
```

---

## 📂 Project Structure

```text
bone-fracture/
│
├── app_gui.py
├── model.h5
├── flowchart.mmd
├── cek_header.py
│
├── hasil_training/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── grafik_akurasi_loss.png
│
└── README.md
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/Zidan-py/bone-fracture.git
cd bone-fracture
```

Create a virtual environment:

```bash
python -m venv .venv311
```

Activate the environment:

```bash
.venv311\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app_gui.py
```

---

## 📈 Future Improvements

* Grad-CAM visualization
* Multiple fracture categories
* Mobile application support
* Cloud deployment
* Real-time prediction

---

## 👨‍💻 Developer

**Zidan**

GitHub: https://github.com/Zidan-py

---

## 📄 License

This project is developed for educational and research purposes.
