# 🐾 Image Classification Project - Animals-10

Proyek ini merupakan implementasi klasifikasi gambar hewan menggunakan deep learning berbasis CNN dengan arsitektur EfficientNetV2S. Dataset yang digunakan adalah Animal Faces-HQ (AFHQ) yang terdiri dari 16.130 gambar resolusi tinggi (512×512 piksel) dalam 3 kelas hewan.

---

## 📂 Dataset

Proyek ini merupakan implementasi klasifikasi gambar hewan menggunakan deep learning berbasis CNN dengan arsitektur EfficientNetV2S. Dataset yang digunakan adalah Animal Faces-HQ (AFHQ) yang terdiri dari 16.130 gambar resolusi tinggi (512×512 piksel) dalam 3 kelas hewan.

- 🐶 **Anjing (Dog)**  
- 🐱 **Kucing (Cat)**  
- 🐯 **Margasatwa (Wildlife)** 

📸 **Total gambar:** ~16.130
🗂️ **Resolusi:** 512×512 piksel
📁 **Dataset tersedia di:** https://www.kaggle.com/datasets/sashs/animal-faces

---

## 🧠 Model Architecture

Model dikembangkan menggunakan **EfficientNetV2S** (pre-trained dari ImageNet), kemudian ditambahkan beberapa layer untuk klasifikasi:

```python
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import layers, models

base_model = EfficientNetV2S(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True  # Fine-tune seluruh base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(3, activation="softmax")  # 3 kelas: Kucing, Anjing, Margasatwa
])
```
---
## 🔧 Compile & Training

Model dikompilasi menggunakan optimizer Adam dan fungsi loss categorical crossentropy:
```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", "precision", "recall"]
)
```
---
## 📊 Hasil Evaluasi

Model mencapai **akurasi keseluruhan: 97.57%** pada data uji, dengan performa per kelas sebagai berikut:

| Kelas       |Precision|Recall|F1-score|
|-------------|---------|------|--------|
| Anjing      | 0.97    | 0.99 | 0.98   |
| Kucing      | 1.00    | 1.00 | 1.00   |
| Margasatwa  | 1.00    | 0.98 | 0.99   |

- **Akurasi total**: 98.837%  
- **Rata-rata F1-score**: 99%
  
---
## 💡 Kesimpulan

Dengan memanfaatkan arsitektur EfficientNetV2S dan transfer learning, model ini mampu mengklasifikasikan gambar kucing, anjing, dan margasatwa secara akurat meskipun gambar memiliki variasi ras, pose, dan latar belakang. Model ini sangat cocok untuk digunakan dalam aplikasi identifikasi hewan berbasis gambar.

---
## 🚀 Teknologi yang Digunakan

- Python
- TensorFlow / Keras
- EfficientNetV2S (pre-trained model)
- Google Colab / Jupyter Notebook
