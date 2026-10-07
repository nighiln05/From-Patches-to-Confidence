# From Patches to Confidence

### A Dual-Attention and Confidence-Guided Approach for Domain-Adaptive Anomalous Sound Detection

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of:

> **From Patches to Confidence: A Dual-Attention and Confidence-Guided Approach for Domain-Adaptive Anomalous Sound Detection**

This repository contains the implementation of a domain-adaptive anomalous sound detection framework for industrial machine condition monitoring under domain shift.

The proposed framework combines:

- Patch-based acoustic representation learning
- Spatial attention pooling
- Temporal attention and temporal encoding
- Confidence-guided contrastive learning
- CORAL and MMD-based domain alignment
- Domain-Adversarial Neural Network (DANN) with Gradient Reversal
- Multi-centroid K-Means modeling
- Hybrid Mahalanobis + cosine anomaly scoring

---

## 📌 Overview

Anomalous Sound Detection (ASD) aims to identify abnormal machine sounds using predominantly normal training data.

A major challenge in real-world industrial environments is **domain shift**, where the acoustic characteristics of a machine change because of differences in operating conditions, background noise, machine states, or recording environments.

Our framework addresses these challenges by jointly learning:

1. **Localized acoustic representations** through overlapping spectrogram patches.
2. **Spatially informative features** using attention pooling.
3. **Temporal dependencies** using temporal attention and temporal encoding.
4. **Robust representations** using confidence-weighted contrastive learning.
5. **Domain-invariant embeddings** using CORAL, MMD, and DANN.
6. **Multi-modal normal distributions** using K-Means clustering.
7. **Robust anomaly scores** using Mahalanobis and cosine distances.

---

# 🏗️ Architecture

## Training Pipeline

The training pipeline converts raw audio into RGB log-Mel spectrograms and extracts overlapping patches.

Two augmented views are generated for contrastive learning. Patch embeddings are processed through parallel spatial and temporal pathways.

The resulting representations are optimized using confidence-weighted contrastive learning together with domain alignment and adversarial objectives.

<p align="center">
  <img src="assets/train_pipeline.jpeg" alt="Training Pipeline" width="100%">
</p>

### Training Flow

```text
Raw Audio
    ↓
RGB Log-Mel Spectrogram
    ↓
224 × 224 Spectrogram
    ↓
32 × 32 Overlapping Patches
    ↓
ResNet-34 Encoder
    ↓
Patch Embeddings
    ↓
 ┌─────────────────────┐
 │                     │
 ↓                     ↓
Spatial Attention   Temporal Attention
 │                     │
 ↓                     ↓
Spatial Feature     Temporal Feature
 │                     │
 └──────────┬──────────┘
            ↓
      Feature Fusion
            ↓
      128-D Embedding
            ↓
 ┌──────────┼───────────────┐
 ↓          ↓               ↓
NT-Xent   CORAL + MMD    DANN + GRL
 │          │               │
 └──────────┴───────────────┘
            ↓
       Total Loss
