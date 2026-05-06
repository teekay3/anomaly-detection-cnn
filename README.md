# 🔍 Anomaly Detection using Convolutional Neural Networks

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** from scratch using **TensorFlow/Keras** to detect anomalies in images. The model achieves **~90% accuracy** and **95% AUC** on test data, making it suitable for real-world anomaly detection tasks.

### Key Achievements
- ✅ **89.78% Test Accuracy**
- ✅ **95.20% AUC** (Excellent class separation)
- ✅ **90.36% Precision** (Low false alarms)
- ✅ **83.33% Recall** (Catches most anomalies)

---

## 📊 Dataset Description

| Property | Value |
|----------|-------|
| Total Images | 3,000 |
| Image Size | 256×256 PNG |
| Classes | Normal (60%), Anomaly (40%) |
| Pattern Types | noise, dots, waves, grid, stripes |
| Difficulty Levels | easy, medium, hard |
| Train/Val/Test Split | 70%/15%/15% |

---

## 🏗️ Model Architecture

```python
Model: Sequential
┌─────────────────────────────────────────────────────────┐
│ Conv2D (32, 3×3) → BatchNorm → ReLU → MaxPool (2×2)    │
├─────────────────────────────────────────────────────────┤
│ Conv2D (64, 3×3) → BatchNorm → ReLU → MaxPool (2×2)    │
├─────────────────────────────────────────────────────────┤
│ Conv2D (128, 3×3) → BatchNorm → ReLU → MaxPool (2×2)   │
├─────────────────────────────────────────────────────────┤
│ Conv2D (256, 3×3) → BatchNorm → ReLU → MaxPool (2×2)   │
├─────────────────────────────────────────────────────────┤
│ GlobalAveragePooling2D()                                │
├─────────────────────────────────────────────────────────┤
│ Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3)  │
├─────────────────────────────────────────────────────────┤
│ Dense(1, activation='sigmoid')  ← Binary output        │
└─────────────────────────────────────────────────────────┘
