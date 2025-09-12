# MNIST Handwritten Digit Classification

## 🧱 Project Overview
This project is a simple image classification task where a neural network is trained to recognize handwritten digits (0–9) from the MNIST dataset.
It demonstrates key machine learning concepts including data preprocessing, model building, training, evaluation, and model saving.

---

## ✅ Dataset
- Dataset: [MNIST Handwritten Digits](https://keras.io/api/datasets/mnist/)
- 60,000 training images (28x28 grayscale)
- 10,000 testing images (28x28 grayscale)
- Labels: digits from 0 to 9

---

## ⚡ Project Steps

1. **Data Loading**
   Load the MNIST dataset directly from `tensorflow.keras.datasets`.

2. **Data Preprocessing**
   - Normalize pixel values (scale from 0–255 to 0–1).
   - Flatten images (28×28 → 784 features).
   - One-hot encode labels.

3. **Model Building**
   A simple baseline neural network with:
   - Input layer (784 features)
   - Dense layer (128 neurons, ReLU activation)
   - Output layer (10 neurons, softmax activation)

4. **Model Training**
   Train the model for 10 epochs with a batch size of 32, monitoring validation accuracy.

5. **Model Evaluation**
   Visualize training accuracy & loss curves to understand model performance.

6. **Model Saving & Inference Example**
   Save the trained model and demonstrate prediction on a sample image.

---

## 🚀 Technologies Used
- Python
- TensorFlow & Keras
- Matplotlib (for visualization)
- NumPy

---

## 🎯 Results
- Achieved high accuracy (~98%) on validation data.
- Model generalizes well without overfitting.

---

## ✅ How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
