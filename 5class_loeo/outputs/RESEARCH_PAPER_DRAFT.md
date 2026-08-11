# SHARP++: Environment-Agnostic Human Activity Recognition via Residual CNN–CBAM–BiLSTM Fusion on Wi-Fi Doppler Spectrograms

---

## 1. Abstract

Device-free Human Activity Recognition (HAR) using Wi-Fi Channel State Information (CSI) offers a non-intrusive, privacy-preserving alternative to camera- and wearable-based systems. A persistent challenge is **cross-environment generalization**: models trained in one physical space (e.g., a laboratory) frequently fail when deployed in a different room due to varying multipath propagation profiles. In this work we present **SHARP++**, a deep learning architecture that combines a three-stage **Residual Convolutional Neural Network (ResCNN)** with a **Convolutional Block Attention Module (CBAM)** and a two-layer **Bidirectional LSTM (Bi-LSTM)** to jointly capture spatial–spectral and temporal patterns in Doppler spectrograms derived from CSI phase data. We evaluate SHARP++ on the publicly available SHARP dataset using a strict **three-fold Leave-One-Environment-Out (LOEO)** protocol across three physically distinct environments — Bedroom, Living Room, and Laboratory. During inference a **four-antenna probability-level fusion** strategy combined with **temporal smoothing** is applied to produce robust final predictions. The 8-class experiment yields a mean cross-environment accuracy of **64.39 ± 18.9%** and a mean Macro-F1 of **54.60 ± 21.3%**, with individual fold accuracies ranging from 38.63% to 83.58%, underscoring both the promise and the remaining difficulty of environment-agnostic Wi-Fi HAR.

---

## 2. Introduction

### 2.1. Motivation

Non-line-of-sight (NLOS) sensing using commodity Wi-Fi access points is an attractive HAR modality for smart homes, elderly care, and healthcare monitoring. Unlike cameras, CSI-based systems do not capture visual data, thus preserving occupant privacy. Unlike wearables, they require no user cooperation.

### 2.2. Problem Statement

Wi-Fi CSI signals encode the channel frequency response between a transmitter–receiver pair. Human motion modulates this response through Doppler shifts and multipath reflections. However, the **static component** of the channel — determined by room geometry, furniture layout, wall materials, and transceiver placement — dominates the signal. Consequently, a model trained in one room learns environment-specific features that do not transfer to a new room, a phenomenon widely documented in the CSI-HAR literature.

### 2.3. Contributions

1. **SHARP++ Architecture**: We propose an enhanced architecture that stacks Residual CNN blocks for hierarchical spatial feature extraction, a full CBAM module (channel + spatial attention) to suppress environment-dependent static reflections and highlight human Doppler signatures, and a two-layer Bi-LSTM for temporal modelling.
2. **Four-Antenna Probability Fusion**: Instead of concatenating raw antenna data at the input, each antenna's spectrogram is classified independently and the softmax probability vectors are averaged, reducing over-reliance on any single antenna's spatial perspective.
3. **Rigorous Cross-Environment Evaluation**: We adopt a strict LOEO protocol where the held-out environment is excluded from training, validation, and normalization, providing a realistic assessment of deployment-time performance.

---

## 3. Dataset

### 3.1. The SHARP Dataset

The SHARP dataset consists of Wi-Fi CSI recordings collected with a 4-antenna receiver across three environments:

| Environment | Sessions | Description |
| :--- | :--- | :--- |
| **Bedroom** | S1a, S1b, S1c, S2a, S2b, S3a, S4a, S4b, S5a | Residential bedroom |
| **Living Room** | S6a, S6b | Residential living room |
| **Laboratory** | S7a | University laboratory |

Data was collected from **three volunteers** (one male, two females) performing eight activities:

| Activity | Code | Description |
| :--- | :---: | :--- |
| Empty | `E` | No human presence in the room |
| Walking | `W` | Walking around the room |
| Running | `R` | Running around the room |
| Jumping | `J` | Jumping in place |
| Sitting | `L` | Seated somewhere in the room |
| Standing | `S` | Standing still somewhere in the room |
| Continuous sit/stand | `C` | Repeatedly sitting down and standing up |
| Arm gym | `H` | Upper-body arm exercises |

### 3.2. Signal Preprocessing

1. **CSI Phase Sanitization**: Raw phase values are unwrapped and sanitized following the SHARP reference pipeline.
2. **Doppler Spectrum Extraction**: A Short-Time Fourier Transform is applied to produce time–frequency Doppler spectrograms.
3. **Mean Subtraction**: Each raw stream is mean-subtracted along the time axis (`data - mean(data, axis=0)`) to remove the DC (static) component, implemented in `load_raw_stream()`.
4. **Windowing**: Continuous spectrograms are divided into fixed-size windows of **100 time-steps × 100 frequency bins** with a stride of **5 samples**, yielding overlapping windows.
5. **Z-Score Normalization**: Per-fold, the mean ($\mu$) and standard deviation ($\sigma$) are computed from a random subset (max 5,000 samples) of the **training set only**. Each window is normalized as: $x' = \frac{x - \mu}{\sigma}$. No validation or test data influences normalization statistics.

### 3.3. Data Augmentation

During training, two online augmentation strategies are applied:

- **Gaussian Noise Injection**: Additive noise $\mathcal{N}(0, 0.05)$ is added to every training window to improve robustness against signal variations.
- **Time Masking** (applied with 30% probability): A random contiguous block of 1–30 time-steps is zeroed out, simulating brief signal dropouts and encouraging the model to learn from partial temporal context.

---

## 4. SHARP++ Architecture

The SHARP++ model follows a **Spatial Feature Extractor → Attention → Temporal Modeller → Classifier** pipeline. The full architecture is detailed below.

### 4.1. Architecture Diagram

```
Input: (T, F, C) = (100, 100, 4)
        │
        ▼
┌─────────────────────────┐
│  Stem Conv2D(64, 3×3)   │  padding='same', bias=False
│  BatchNormalization      │
│  ReLU                    │
└─────────────────────────┘
        │ Output: (100, 100, 64)
        ▼
┌─────────────────────────┐
│  Residual Block (64)     │  2× Conv2D(64, 3×3) + skip connection
│  MaxPooling2D(2×2)       │
└─────────────────────────┘
        │ Output: (50, 50, 64)
        ▼
┌─────────────────────────┐
│  Residual Block (128)    │  2× Conv2D(128, 3×3) + 1×1 projection skip
│  MaxPooling2D(2×2)       │
└─────────────────────────┘
        │ Output: (25, 25, 128)
        ▼
┌─────────────────────────┐
│  Residual Block (256)    │  2× Conv2D(256, 3×3) + 1×1 projection skip
└─────────────────────────┘
        │ Output: (25, 25, 256)
        ▼
┌─────────────────────────┐
│  CBAM Attention Module   │
│  ├─ Channel Attention    │  GlobalAvgPool + GlobalMaxPool → shared MLP (256→32→256) → sigmoid
│  └─ Spatial Attention    │  ChannelAvg ∥ ChannelMax → Conv2D(1, 7×7) → sigmoid
└─────────────────────────┘
        │ Output: (25, 25, 256)
        ▼
┌─────────────────────────┐
│  Reshape                 │  (25, 25×256) = (25, 6400)
└─────────────────────────┘
        │ Output: (25, 6400)
        ▼
┌─────────────────────────┐
│  Bi-LSTM Layer 1         │  128 units per direction → 256 total, dropout=0.2
│  (return_sequences=True) │
└─────────────────────────┘
        │ Output: (25, 256)
        ▼
┌─────────────────────────┐
│  Bi-LSTM Layer 2         │  128 units per direction → 256 total, dropout=0.2
│  (return_sequences=True) │
└─────────────────────────┘
        │ Output: (25, 256)
        ▼
┌─────────────────────────┐
│  GlobalAveragePooling1D  │
└─────────────────────────┘
        │ Output: (256,)
        ▼
┌─────────────────────────┐
│  Dense(256, ReLU)        │
│  Dropout(0.5)            │
└─────────────────────────┘
        │ Output: (256,)
        ▼
┌─────────────────────────┐
│  Dense(8, Softmax)       │  Classification head
└─────────────────────────┘
        │ Output: (8,)
```

### 4.2. Component Details

#### 4.2.1. Stem Convolution

A single `Conv2D(64, 3×3, padding='same', use_bias=False)` layer followed by Batch Normalization and ReLU activation. This projects the 4-channel antenna input into a 64-channel feature space.

#### 4.2.2. Residual CNN Backbone (3 Stages)

Each residual block consists of:
1. `Conv2D(filters, 3×3, padding='same', bias=False)` → `BatchNorm` → `ReLU`
2. `Conv2D(filters, 3×3, padding='same', bias=False)` → `BatchNorm`
3. **Skip Connection**: If the input and output channel dimensions differ, a `Conv2D(filters, 1×1)` projection with BatchNorm is applied to the shortcut before element-wise addition.
4. Final `ReLU` after the addition.

| Stage | Filters | MaxPool | Output Shape |
| :---: | :---: | :---: | :--- |
| 1 | 64 | 2×2 | (T/2, F/2, 64) |
| 2 | 128 | 2×2 | (T/4, F/4, 128) |
| 3 | 256 | None | (T/4, F/4, 256) |

The residual connections mitigate the vanishing gradient problem and allow the network to learn incremental feature refinements.

#### 4.2.3. CBAM (Convolutional Block Attention Module)

CBAM is applied sequentially as **Channel Attention → Spatial Attention**:

**Channel Attention** recalibrates *which* feature channels are informative:
1. The input feature map $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$ is independently reduced via Global Average Pooling and Global Max Pooling to two $1 \times 1 \times C$ descriptors.
2. Both are passed through a shared two-layer MLP: `Dense(C/r, ReLU) → Dense(C)` where the reduction ratio $r = 8$.
3. The two MLP outputs are summed element-wise and passed through a sigmoid gate: $\mathbf{M}_c = \sigma(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F})))$.
4. The input is scaled: $\mathbf{F}' = \mathbf{M}_c \odot \mathbf{F}$.

**Spatial Attention** recalibrates *where* to focus in the time–frequency plane:
1. Channel-wise average and max pooling are computed: two $H \times W \times 1$ maps.
2. The two maps are concatenated and convolved: `Conv2D(1, 7×7, padding='same') → sigmoid`.
3. The channel-attended features are scaled spatially: $\mathbf{F}'' = \mathbf{M}_s \odot \mathbf{F}'$.

This dual attention mechanism is critical for CSI-HAR: channel attention suppresses static (environment-dependent) feature channels while spatial attention highlights the time–frequency regions that correspond to human Doppler motion.

#### 4.2.4. Temporal Modelling (Bi-LSTM)

After CBAM, the feature maps are reshaped from $(T/4, F/4, 256)$ to $(T/4, F/4 \times 256)$, treating the reduced time dimension as sequence steps and the flattened frequency–channel features as the per-step feature vector.

Two stacked **Bidirectional LSTM** layers (each with 128 units per direction, 256 total) with `dropout=0.2` process the sequence in both forward and backward directions. Both layers use `return_sequences=True` to preserve the full temporal resolution for downstream pooling.

#### 4.2.5. Classification Head

- **GlobalAveragePooling1D**: Aggregates temporal features into a single 256-d vector, providing translation invariance along the time axis.
- **Dense(256, ReLU)** + **Dropout(0.5)**: A fully connected embedding layer with strong regularization.
- **Dense(num_classes, Softmax)**: Final classification layer producing class probability distributions. The softmax is computed in `float32` precision regardless of the global precision policy to prevent numerical instability.

---

## 5. Experimental Protocol

### 5.1. Three-Fold Leave-One-Environment-Out (LOEO)

The LOEO protocol ensures that the model is evaluated on a **completely unseen physical environment**. In each fold, one of the three environments is held out entirely — it is excluded from training, validation, and normalization statistic computation.

| Fold | Train Environments | Test Environment | Train Sessions | Test Sessions |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Living Room, Laboratory | **Bedroom** | S6a, S6b, S7a | S1a–S1c, S2a–S2b, S3a, S4a–S4b, S5a |
| 2 | Bedroom, Laboratory | **Living Room** | S1a–S1c, S2a–S2b, S3a, S4a–S4b, S5a, S7a | S6a, S6b |
| 3 | Bedroom, Living Room | **Laboratory** | S1a–S1c, S2a–S2b, S3a, S4a–S4b, S5a, S6a, S6b | S7a |

Within each fold, the training sessions are split 75% for training and 25% for validation (time-based split within each activity recording).

### 5.2. Training Configuration

| Hyperparameter | Value |
| :--- | :--- |
| Optimizer | AdamW |
| Learning Rate | $1 \times 10^{-4}$ |
| Weight Decay | $1 \times 10^{-4}$ |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 32 |
| Maximum Epochs | 25 |
| Early Stopping Patience | 10 epochs (monitors `val_loss`, restores best weights) |
| LR Reduction | Factor 0.5 when `val_loss` plateaus for 3 epochs (min LR: $10^{-6}$) |
| Model Checkpointing | Saves model with best `val_accuracy` |

### 5.3. Inference Pipeline

1. **Independent Antenna Processing**: Each test window is a 4-antenna group $(4, 100, 100, 1)$. The trained model processes all 4 antennas through `model.predict()` in a single batch. The softmax outputs $\mathbf{P}_1, \mathbf{P}_2, \mathbf{P}_3, \mathbf{P}_4 \in \mathbb{R}^{8}$ are obtained per antenna.

2. **Probability Fusion**:
$$\mathbf{P}_{\text{fused}}(c) = \frac{1}{4} \sum_{a=1}^{4} \mathbf{P}_a(c), \quad c = 1, \ldots, 8$$

3. **Temporal Smoothing**: A 1-D uniform filter of size 11 is applied along the time axis of the fused probability sequence using `scipy.ndimage.uniform_filter1d`. This removes short-lived spurious predictions caused by transient signal noise.

4. **Final Prediction**: $\hat{y} = \arg\max_c \; \mathbf{P}_{\text{smoothed}}(c)$

---

## 6. Results and Discussion

### 6.1. Dataset Statistics per Fold

| Fold | Test Env | Train Windows | Val Windows | Test Windows |
| :---: | :--- | ---: | ---: | ---: |
| 1 | Bedroom | 50,795 | 16,662 | 223,590 |
| 2 | Living Room | 182,931 | 60,050 | 47,078 |
| 3 | Laboratory | 202,616 | 66,522 | 20,786 |

> **Note**: Fold 1 has the fewest training samples because only Living Room (2 sessions) and Laboratory (1 session) are available for training. Fold 3 has the most training data but the fewest test samples since the Laboratory consists of a single session.

### 6.2. Overall Cross-Environment Performance

| Metric | Fold 1 (Bedroom) | Fold 2 (Living Room) | Fold 3 (Laboratory) | **Mean ± Std** |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 70.95% | 83.58% | 38.63% | **64.39% ± 18.9%** |
| **Macro-Precision** | 64.21% | 78.73% | 27.19% | **56.71% ± 21.6%** |
| **Macro-Recall** | 65.96% | 80.60% | 32.95% | **59.84% ± 19.9%** |
| **Macro-F1** | 59.67% | 77.77% | 26.35% | **54.60% ± 21.3%** |

### 6.3. Per-Class Performance (All 8 Classes)

#### Fold 1 — Test: Bedroom

| Activity | Code | Precision | Recall | F1-Score | Support |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Empty | E | 99.91% | 99.21% | 99.56% | 40,286 |
| Walking | W | 94.48% | 21.12% | 34.53% | 33,941 |
| Running | R | 72.76% | 76.61% | 74.64% | 34,349 |
| Jumping | J | 52.43% | 93.82% | 67.27% | 28,950 |
| Sitting | L | 90.36% | 90.29% | 90.32% | 40,931 |
| Standing | S | 68.25% | 60.59% | 64.19% | 18,896 |
| Continuous sit/stand | C | 32.04% | 85.97% | 46.69% | 11,162 |
| Arm gym | H | 3.45% | 0.09% | 0.18% | 15,075 |

#### Fold 2 — Test: Living Room

| Activity | Code | Precision | Recall | F1-Score | Support |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Empty | E | 100.00% | 99.21% | 99.61% | 7,510 |
| Walking | W | 99.36% | 99.09% | 99.22% | 7,653 |
| Running | R | 98.26% | 99.95% | 99.09% | 7,664 |
| Jumping | J | 90.95% | 99.67% | 95.11% | 5,225 |
| Sitting | L | 97.96% | 54.07% | 69.68% | 7,650 |
| Standing | S | 99.16% | 100.00% | 99.58% | 3,789 |
| Continuous sit/stand | C | 44.16% | 92.80% | 59.85% | 3,794 |
| Arm gym | H | 0.00% | 0.00% | 0.00% | 3,793 |

#### Fold 3 — Test: Laboratory

| Activity | Code | Precision | Recall | F1-Score | Support |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Empty | E | 97.57% | 100.00% | 98.77% | 3,649 |
| Walking | W | 99.42% | 64.27% | 78.07% | 3,725 |
| Running | R | 0.00% | 0.00% | 0.00% | 0 |
| Jumping | J | 20.50% | 99.30% | 33.98% | 2,001 |
| Sitting | L | 0.00% | 0.00% | 0.00% | 0 |
| Standing | S | 0.00% | 0.00% | 0.00% | 3,805 |
| Continuous sit/stand | C | 0.00% | 0.00% | 0.00% | 3,802 |
| Arm gym | H | 0.00% | 0.00% | 0.00% | 3,804 |

> **Important**: The Laboratory session (`S7a`) contains **zero samples** for Running and Sitting. Their 0% scores are artefacts of missing test data, not model failure. Excluding these two absent classes would raise the Laboratory Macro-F1 substantially.

### 6.4. Confusion Matrix Analysis

#### Key Observations

1. **Empty State is Near-Perfect**: The `E` class achieved F1 scores of 99.56%, 99.61%, and 98.77% across the three folds. The static channel signature when no human is present is highly consistent and transfers trivially across environments.

2. **High-Energy Activities Generalize Well (Fold 2)**: In the Living Room fold, Walking (99.22% F1), Running (99.09% F1), and Standing (99.58% F1) all achieved excellent cross-environment performance, suggesting that the Bedroom + Laboratory training data provided sufficient diversity for these gross-motor activities.

3. **Walking–Running Confusion (Fold 1)**: In the Bedroom test, 9,767 Walking windows were misclassified as Running and 17,002 as Jumping. The model's severe underperformance on Walking (21.12% recall) in this fold reveals that the periodic gait Doppler signature learned from only 3 training sessions (S6a, S6b, S7a) was insufficient to generalize.

4. **Arm Gym (H) is Consistently Problematic**: The model achieved near-zero F1 for Arm gym across all folds (0.18%, 0.00%, 0.00%). This suggests that upper-body-only movements produce Doppler signatures too subtle and too variable across environments for the current architecture to distinguish.

5. **Standing vs. Continuous sit/stand Confusion**: In the Bedroom test, 3,682 Standing windows were misclassified as Continuous sit/stand. These two stationary-dominant activities produce similar low-Doppler signatures, making them inherently difficult to separate from spectrograms alone.

### 6.5. Cross-Environment Variance Discussion

The **18.9% standard deviation** in accuracy across folds highlights the fundamental challenge:

- **Fold 2 (Living Room)** benefits from having the largest and most diverse training pool (Bedroom: 9 sessions + Lab: 1 session = 10 sessions), resulting in the highest accuracy.
- **Fold 1 (Bedroom)** has only 3 training sessions, severely limiting the model's exposure to activity variations. Despite this, the model achieves 70.95% accuracy, driven largely by the strong performance on Empty and Sitting.
- **Fold 3 (Laboratory)** illustrates the worst case: the Lab environment has distinct physical characteristics (metallic surfaces, different room dimensions) from the residential training environments, and two activity classes are entirely absent from the test set, collapsing the macro average.

---

## 7. Conclusion

We presented **SHARP++**, a Residual CNN–CBAM–BiLSTM architecture for device-free human activity recognition using Wi-Fi CSI Doppler spectrograms. Through a strict 3-fold LOEO protocol, we demonstrated that:

1. **The architecture generalizes well for high-energy activities** (Walking, Running, Jumping) and the Empty state, achieving near-perfect F1 scores in favorable conditions (Fold 2).
2. **Cross-environment generalization remains a significant open challenge**, with a mean accuracy of 64.39% ± 18.9% and a mean Macro-F1 of 54.60% ± 21.3%.
3. **Training set size critically influences performance**: folds with fewer training sessions show marked degradation.
4. **Subtle activities** (Arm gym, Continuous sit/stand) are inherently difficult to distinguish from spectrograms alone and require further architectural innovation or additional signal modalities.

### Future Work

- **Domain Adaptation**: Applying unsupervised or few-shot domain adaptation techniques to bridge the gap between training and deployment environments.
- **Larger Doppler Windows**: Increasing the window size (e.g., 340 × 100 as explored in our 5-class variant) to capture longer temporal context for subtle activities.
- **Multi-Person Scenarios**: Extending the framework to handle simultaneous activities by multiple occupants.
- **Edge Deployment**: Optimising the model via quantisation and pruning for real-time inference on embedded Wi-Fi routers.

---

## References

1. SHARP Dataset reference paper (original dataset description and activity definitions).
2. S. Woo, J. Park, J.-Y. Lee, and I.-S. Kweon, "CBAM: Convolutional Block Attention Module," *ECCV 2018*.
3. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR 2016*.
4. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," *ICCV 2017*.
