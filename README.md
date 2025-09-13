# DeepFake Detection

A deep learning project to classify images as **Real** or **Fake** using **EfficientNetB4** with transfer learning and fine-tuning.

## 📌 Overview

* Dataset: [DeepFake Detection 6k (800x800)](https://www.kaggle.com/datasets/examstebi/deep-fake-detection-6k-800-800)
* Model: EfficientNetB4 (pretrained on ImageNet)
* Training:

  * Stage 1: Train with frozen base
  * Stage 2: Fine-tune half the layers
* Final Validation Accuracy: **\~71%**
* Output model: `efficientnetB4_deepfake.h5`


## 📦 Requirements

* TensorFlow 2.x
* NumPy
* scikit-learn
* Matplotlib
* Kaggle API

---

Do you want me to make this **super short (one-page)** like a GitHub project description, or keep it like this with a little detail?
