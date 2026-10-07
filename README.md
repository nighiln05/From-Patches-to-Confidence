# From Patches to Confidence

### A Dual-Attention and Confidence-Guided Approach for Domain-Adaptive Anomalous Sound Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository contains the implementation of **From Patches to Confidence: A Dual-Attention and Confidence-Guided Approach for Domain-Adaptive Anomalous Sound Detection**.

The proposed approach combines:

- Patch-level spatial attention
- Temporal attention
- Confidence-guided contrastive learning
- Domain adaptation
- Multi-centroid anomaly scoring

The method is evaluated on the **DCASE 2023 Task 2** and **DCASE 2025 Task 2** anomalous sound detection benchmarks.

---

## Inference

The proposed model converts audio recordings into Mel-spectrogram representations and performs patch-level feature extraction, attention-based aggregation, and anomaly scoring.

<div align="center">

<img src="assets/test_pipeline.jpeg" width="900">

</div>

---

## DCASE 2025 Results

Performance of the proposed approach across the seven machine types in **DCASE 2025 Task 2**.

<div align="center">

<img src="assets/dcase2025_results.png" width="900">

</div>

### Target-Domain AUC

| Machine Type | Target AUC |
|---|---:|
| ToyCar | 71.72 |
| ToyTrain | 64.52 |
| Fan | 60.08 |
| Gearbox | 65.76 |
| Bearing | 59.48 |
| Slider | 54.32 |
| Valve | 77.36 |

### Target-Domain pAUC

| Machine Type | pAUC |
|---|---:|
| ToyCar | 54.05 |
| ToyTrain | 54.58 |
| Fan | 53.63 |
| Gearbox | 55.89 |
| Bearing | 50.68 |
| Slider | 57.74 |
| Valve | 69.74 |

---

## Method

The framework operates at both **spatial/patch** and **temporal** levels.

### 1. Mel-Spectrogram Representation

Audio recordings are converted into normalized Mel-spectrogram representations and processed as three-channel inputs.

### 2. Patch-Level Feature Extraction

Overlapping spectrogram patches are extracted and encoded using an ImageNet-pretrained **ResNet-34** backbone.

### 3. Dual Attention

Spatial patch representations are aggregated using attribute-conditioned attention, followed by temporal attention to capture sequential dependencies.

### 4. Confidence-Guided Learning

Contrastive learning and latent-space regularization are used to improve the separation between normal and anomalous representations.

### 5. Domain Adaptation

The model incorporates domain alignment using:

- CORAL
- MMD
- Domain-Adversarial Training

### 6. Anomaly Scoring

Anomalies are detected using a combination of:

- Mahalanobis distance
- Cosine distance
- Multi-centroid representations

---

## Dataset

Experiments are conducted on the **DCASE Task 2** anomalous sound detection benchmark covering:

- ToyCar
- ToyTrain
- Fan
- Gearbox
- Bearing
- Slider
- Valve

Audio recordings are converted into **128-bin Mel-spectrograms** using STFT-based processing.

---

## Implementation Details

| Parameter | Value |
|---|---|
| Backbone | ResNet-34 |
| Input | 224 × 224 |
| Mel bins | 128 |
| Patch size | 32 × 32 |
| Patch stride | 16 |
| Embedding dimension | 128 |
| Optimizer | Adam |
| Initial learning rate | 2 × 10⁻⁴ |
| Batch size | 96 |
| Epochs | 150 |
| NT-Xent temperature | 0.05 |
| EMA momentum | 0.9 |
| GPU | NVIDIA RTX 5060 Laptop GPU |

---

## Repository Structure

```text
From-Patches-to-Confidence/
│
├── assets/
│   ├── test_pipeline.jpeg
│   └── dcase2025_results.png
│
├── astra_attn_patch_dataset.py
├── attention_pooling.py
├── convert_rgb.py
├── evaluation4.py
├── evaluation4_k_ablation.py
├── patch_attn_model.py
├── pauc.py
├── train_1.py
├── requirements.txt
├── LICENSE
└── README.md
