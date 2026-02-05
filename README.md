# Multi-Label Environmental Audio Tagging using CNN and CRNN

A deep learning–based study on **multi-label sound event classification** for real-world environmental audio using the **FSD50K dataset**.

## 📌 Project Overview

Real-world environmental audio is inherently **polyphonic**, containing multiple overlapping sound events such as speech, music, traffic, animals, and ambient noise.  
This makes traditional single-label audio classification unrealistic for real-world applications.

This project formulates environmental sound recognition as a **multi-label classification problem** and evaluates deep learning models capable of capturing complex **spectral and temporal patterns** in noisy audio data.

## 🎯 Motivation

### Complexity of Environmental Audio
- Real-world audio contains **multiple overlapping sound sources**
- Background noise and acoustic variability make interpretation difficult

### Limitations of Traditional Methods
- Conventional classifiers assume **one sound per clip**
- Single-label models fail to represent overlapping sound events

### Importance of Sound Event Recognition
Accurate audio understanding is critical for:
- Smart cities and surveillance
- Multimedia indexing and retrieval
- Robotics and autonomous systems
- Assistive technologies

## 🧠 Problem Statement

- Environmental audio often contains **multiple sound events simultaneously**
- Traditional single-label classification performs poorly on such data
- The **FSD50K dataset** provides weakly labeled, real-world audio clips with multiple sound labels
- The task is to **predict one or more sound event labels per audio clip**
- This requires models that can handle **overlapping sounds**, **weak labels**, and **class imbalance**

## 📂 Dataset

### Dataset Used: **Freesound Dataset 50K (FSD50K)**
- Source: Released via **Zenodo**
- Large-scale, real-world environmental audio dataset
- **200 sound classes** (human sounds, animals, music, environmental noise, etc.)
- Audio clips may contain **single or multiple overlapping sound events**

### Actual Dataset Statistics
- **Development set:** 40,966 clips (80.4 hours)
- **Evaluation set:** 10,231 clips (27.9 hours)

### Subset Used in This Project
- Selected **15,000 clips** from the development set
  - **Training:** 12,000 clips
  - **Evaluation:** 3,000 clips
- Same split used across all models to ensure **fair comparison**

## 🔊 Data Preprocessing

- All audio clips:
  - Resampled and normalized
  - Fixed to a **10-second duration**
- Converted raw audio into **Log-Mel spectrograms**
- Sound event labels encoded as **multi-label binary vectors**
  - Each vector represents the presence or absence of multiple sound classes in a clip

## 🛠️ Proposed Solution Strategy

### 1. Baseline Modeling – CRNN
- Trained a **CRNN (Convolutional Recurrent Neural Network)** from scratch
- CNN layers capture local spectral patterns
- RNN layers model temporal dependencies
- Serves as a baseline to understand dataset complexity

### 2. Enhanced Modeling – CNN14 (PANNs)
- Fine-tuned a **pre-trained CNN14 model** from the PANNs framework
- Leverages transfer learning from large-scale audio data
- Enables faster convergence and improved tagging performance

### 3. Evaluation and Comparison
- Compared baseline and pre-trained models using:
  - Micro-F1
  - Macro-F1
  - Training loss curves
- Analyzed convergence behavior and robustness

## 📊 Results

| Metric      | CNN14 | CRNN |
|------------|-------|------|
| Accuracy   | **89.3%** | 82.7% |
| Precision  | 0.73  | 0.52 |
| Recall     | 0.70  | 0.47 |
| F1-score   | **0.71** | 0.49 |

### Key Observations
- **CNN14 significantly outperforms CRNN** across all metrics
- Pre-training improves:
  - Learning efficiency
  - Generalization
  - Robustness to overlapping sounds
- CRNN provides a reasonable baseline but struggles with complex polyphonic audio

## 📈 Focus of the Study

- Performance comparison between baseline and pre-trained models
- Analysis of training stability and convergence behavior
- Identification of challenges in multi-label environmental audio tagging
- Insights into building robust systems for real-world audio applications

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **Librosa**
- **NumPy, Pandas**
- **scikit-learn**

## 📂 Project Structure

```text
├── data/
│   ├── raw/
│   └── processed/
├── preprocessing/
│   └── logmel_extraction.py
├── models/
│   ├── crnn.py
│   └── cnn14.py
├── training/
│   └── train.py
├── evaluation/
│   └── metrics.py
├── results/
│   └── performance.json
├── requirements.txt
└── README.md
