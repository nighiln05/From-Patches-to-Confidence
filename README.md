# From Patches to Confidence

### A Dual-Attention and Confidence-Guided Approach for Domain-Adaptive Anomalous Sound Detection

This repository contains the implementation of **From Patches to Confidence**, a domain-adaptive anomalous sound detection framework that learns robust audio representations using patch-level and temporal information.

## Datasets

The framework is evaluated on:

- **DCASE 2023 Task 2**
- **DCASE 2025 Task 2**

The experiments consider the following machine types:

`ToyCar · ToyTrain · Fan · Gearbox · Bearing · Slider · Valve`

## Architecture

The proposed framework operates on Mel-spectrogram representations and uses a **ResNet-34** backbone with spatial and temporal attention to learn discriminative representations. Confidence-guided learning and domain adaptation are incorporated to improve robustness across domains, followed by distance-based anomaly scoring.
