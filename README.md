# Deepfake-Detection-System-EfficientnetB4
Got it 👍 Here’s a **simpler README.md** for your project:

---

# DeepFake Detection

This project trains a **DeepFake vs Real image classifier** using **EfficientNetB4** on the [DeepFake Detection 6k](https://www.kaggle.com/datasets/examstebi/deep-fake-detection-6k-800-800) dataset.

---

## 📌 What it does

* Uses **EfficientNetB4 (pretrained on ImageNet)**
* Splits dataset into **Train (80%) / Validation (20%)**
* Applies **data augmentation** to improve generalization
* Trains in **two stages**: frozen base → fine-tuning
* Saves model as `efficientnetB4_deepfake.h5`
* Can predict if an image is **Real** or **Fake**

---

## ⚙️ Setup

Run in **Google Colab**.

```bash
!pip install -q kaggle
```

1. Upload `kaggle.json` (API key).
2. Download dataset:

   ```bash
   !kaggle datasets download -d examstebi/deep-fake-detection-6k-800-800
   !unzip -q deep-fake-detection-6k-800-800.zip -d data/
   ```
3. Run the training notebook.

---

## 🧠 Training

* Stage 1: Train with frozen EfficientNetB4 base.
* Stage 2: Fine-tune half the layers.

Final Validation Accuracy: **\~71%** ✅

---

## 🔍 Prediction

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("efficientnetB4_deepfake.h5")

img = image.load_img("test.jpg", target_size=(224,224))
x = np.expand_dims(image.img_to_array(img), axis=0) / 255.0

pred = model.predict(x)[0][0]
print("Real" if pred >= 0.5 else "Fake")
```

---

## 📦 Requirements

* TensorFlow 2.x
* NumPy
* scikit-learn
* Matplotlib
* Kaggle API

