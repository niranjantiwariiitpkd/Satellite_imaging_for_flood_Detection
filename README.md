# 🌊 Satellite Flood Risk Assessment System

## 📌 Overview

This project presents a Satellite-Based Flood Risk Assessment System using Deep Learning and Transfer Learning.  
The system analyzes satellite imagery to detect surface water presence and estimates potential flood risk levels.

It uses a ResNet18 model trained on satellite image data to classify regions as Water or Non-Water and converts the probability into actionable flood risk categories.

---

## 🚀 Live Demo

🔗 Hugging Face Deployment:  
(Add your Hugging Face Space link here)

---

## 🛰 Dataset

Dataset Used: **EuroSAT – Sentinel-2 Satellite Images**

- 27,000 RGB satellite images
- 10 land-use classes
- Water classes used:
  - River
  - Sea/Lake

For this project:
- River + SeaLake → Water (Flood-prone)
- Other classes → Non-Water

Dataset Size: ~1.3GB

---

## 🧠 Model Architecture

We use **Transfer Learning with ResNet18**:

- Pretrained on ImageNet
- Final fully-connected layer modified for binary classification
- Input Size: 224 × 224
- Loss Function: BCEWithLogitsLoss
- Optimizer: Adam

---

## 📊 Training Strategy

- 80% Training Split
- 20% Validation Split
- Best model saved using validation accuracy
- Early stopping via best validation checkpoint

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Validation Accuracy Achieved:  
(Add your final validation accuracy here)

---

## 🌊 Flood Risk Logic

The system converts water detection probability into flood risk levels:

| Probability Range | Risk Level |
|------------------|------------|
| 0.0 – 0.3 | Low Risk |
| 0.3 – 0.6 | Moderate Risk |
| 0.6 – 0.8 | High Risk |
| 0.8 – 1.0 | Severe Flood Risk |

This allows interpretable decision-making rather than raw classification output.

---

## 🖥 Application Interface

The project includes a Streamlit web application:

- Upload satellite image
- Model predicts water probability
- Displays risk level
- Visual probability bar
- Clean and professional UI

---

## 🏗 Project Structure
