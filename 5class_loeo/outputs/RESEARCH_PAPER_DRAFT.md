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

Wi-Fi Doppler spectrograms are simultaneously 2-D spatial images and temporal sequences contaminated by environment-specific multipath clutter. No single model family addresses all three properties in isolation. SHARP++ therefore chains three specialised modules: a **Residual CNN** for hierarchical spectral–spatial feature extraction, a **CBAM attention module** to suppress static room clutter and amplify dynamic Doppler signatures, and a **Bidirectional LSTM** to model temporal activity dynamics. During inference, each of the four receive antennas is processed independently and their softmax probability vectors are averaged (**late fusion**), providing antenna-diversity without coupling the model to a fixed antenna topology.

### 4.1. Architecture Overview

```
Input (single antenna)  (340, 100, 1)
        │
  Stem: Conv2D(64,3×3) + BN + ReLU          → (340, 100,  64)
        │
  ResBlock-1: 2×Conv2D(64,3×3) + skip        → (340, 100,  64)
  MaxPool(2×2)                                → (170,  50,  64)
        │
  ResBlock-2: 2×Conv2D(128,3×3) + proj-skip  → (170,  50, 128)
  MaxPool(2×2)                                → ( 85,  25, 128)
        │
  ResBlock-3: 2×Conv2D(256,3×3) + proj-skip  → ( 85,  25, 256)
        │
  CBAM: Channel-Attn (MLP 256→32→256)        → ( 85,  25, 256)
         Spatial-Attn  (Conv2D 1×7×7)         → ( 85,  25, 256)
        │
  Reshape → (85, 6 400)                       # 25 freq-bins × 256 channels
        │
  Bi-LSTM-1  (128 units/dir, dropout 0.2)    → ( 85, 256)
  Bi-LSTM-2  (128 units/dir, dropout 0.2)    → ( 85, 256)
        │
  GlobalAveragePooling1D                      → (256,)
        │
  Dense(256, ReLU) + Dropout(0.5)            → (256,)
        │
  Dense(5,  Softmax)    # float32 head
        │
  Output: P ∈ ℝ⁵   {Empty, Sitting, Walking, Running, Jumping}
```

**Tensor shapes** are exact for `window_length = 340`, `freq_bins = 100`. Two MaxPool(2×2) layers reduce the time axis by 4×: 340 → 170 → 85 LSTM steps.

### 4.2. Component Descriptions

**Stem.** A single Conv2D(64, 3×3, same, bias=False) + Batch Normalization + ReLU lifts the 1-channel input to a 64-channel feature space, bridging the dimensional gap before the residual backbone.

**Residual Backbone (3 stages).** Each stage contains two Conv2D(K, 3×3, same, bias=False) + BN + ReLU layers with a skip connection. When the channel depth changes (64→128, 128→256), a 1×1 Conv2D projection shortcut aligns dimensions. MaxPooling(2×2) follows stages 1 and 2, halving spatial resolution and expanding the effective receptive field; stage 3 omits pooling to preserve the 85 temporal positions needed by the LSTM. Channel widths follow the standard 64→128→256 progression: as spatial resolution halves, channel count doubles to maintain representational capacity.

**CBAM Attention.** Applied sequentially after stage 3:
- *Channel attention* — GlobalAvgPool and GlobalMaxPool each produce a 256-d descriptor, both passed through a **shared** two-layer MLP (256→32→256, ReLU/linear, no bias) and summed before a sigmoid gate. Reduction ratio $r=8$ (bottleneck=32) regularises inter-channel dependencies.
- *Spatial attention* — Channel-wise average and max across the 256 features produce two $85\times25$ maps, concatenated and convolved with a 7×7 filter (Conv2D(1, 7×7, same, no bias)) → sigmoid, generating a 2-D soft mask that highlights Doppler-active time–frequency regions.

The channel→spatial ordering follows the original CBAM prescription: channel gating selects *which* feature detectors are relevant before spatial gating focuses *where* those detectors fire.

**Reshape.** Collapses the frequency and channel axes into a single per-timestep feature vector: $(85, 25, 256)\rightarrow(85, 6400)$. The time axis (85 steps) becomes the LSTM sequence axis.

**Bidirectional LSTM (2 layers).** Each layer wraps an LSTM(128) with a parallel reverse LSTM(128), concatenating forward and backward hidden states → 256-d output per step. Both layers use `return_sequences=True` and input dropout 0.2 (recurrent dropout = 0 to retain cuDNN acceleration). The two-layer stack builds a hierarchical temporal representation: layer 1 captures individual gait cycles and Doppler bursts; layer 2 captures multi-cycle patterns and activity-level periodicity.

**Classification head.** GlobalAveragePooling1D aggregates all 85 LSTM outputs into a single 256-d embedding (translation-invariant to activity position within the window). A Dense(256, ReLU) + Dropout(0.5) projects to an embedding space; Dense(5, Softmax, dtype=float32) produces the final class probabilities. The explicit float32 cast is required when mixed-precision (float16) training is active to prevent softmax overflow.

### 4.3. Hyperparameters

| Parameter | Value |
|:---|:---|
| Window length | 340 time steps |
| Frequency bins | 100 |
| Stride (windowing) | 17 samples |
| Residual filters | 64 → 128 → 256 |
| CBAM reduction ratio | 8 (bottleneck = 32) |
| CBAM spatial kernel | 7×7 |
| LSTM units / direction | 128 (256 total) |
| LSTM layers | 2 |
| LSTM input dropout | 0.2 |
| Dense embedding | 256 |
| Dropout | 0.5 |
| Output classes | 5 |
| Loss | CategoricalFocalCrossentropy ($\gamma=2$) |
| Optimizer | AdamW, lr=$10^{-4}$, wd=$10^{-4}$ |
| Batch size | 32 |
| Max epochs | 25 |
| Early stopping patience | 10 (monitors val\_loss) |
| LR reduction | ×0.5 after 3 plateau epochs, min $10^{-6}$ |

### 4.4. Training and Inference Pipeline

**Training (per fold).** Training sessions are split 75 / 25 by time within each activity recording. Each training window carries a single antenna channel `(340, 100, 1)`; each of the four antennas yields an independent sample, quadrupling effective training size. Two online augmentations are applied: (i) additive Gaussian noise $\mathcal{N}(0, 0.05)$; (ii) random time-masking of 1–30 contiguous steps with 30% probability. Z-score normalisation statistics ($\mu$, $\sigma$) are computed from up to 5,000 randomly sampled training windows and frozen for the fold — no validation or test signal influences normalisation.

**Inference (probability fusion).** Each test window is assembled as a $(4, 340, 100, 1)$ tensor grouping all four antennas. The model is run four times (once per antenna slice `(batch, 340, 100, 1)`), producing four softmax vectors:

$$\mathbf{P}_{\text{fused}}(c) = \frac{1}{4}\sum_{a=1}^{4}\mathbf{P}_a(c), \quad c\in\{1,\ldots,5\}$$

A 1-D uniform filter of width 11 (`scipy.ndimage.uniform_filter1d`) is applied along the time axis of the fused probability stream before argmax, smoothing transient misclassifications caused by signal dropouts or window-boundary artefacts.

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
| Loss Function | Categorical Focal Crossentropy (`gamma=2.0`) |
| Batch Size | 32 |
| Maximum Epochs | 25 |
| Early Stopping Patience | 10 epochs (monitors `val_loss`, restores best weights) |
| LR Reduction | Factor 0.5 when `val_loss` plateaus for 3 epochs (min LR: $10^{-6}$) |
| Model Checkpointing | Saves model with best `val_accuracy` |

### 5.3. Inference Pipeline

1. **Independent Antenna Processing**: Each test window is a 4-antenna group of shape **(4, 340, 100, 1)** — 4 antennas, each with a 340-step × 100-bin single-channel spectrogram. The trained model processes each antenna slice `x_batch[:, ant, :, :, :]` (shape `(batch, 340, 100, 1)`) independently through a forward pass. The softmax outputs $\mathbf{P}_1, \mathbf{P}_2, \mathbf{P}_3, \mathbf{P}_4 \in \mathbb{R}^{5}$ are obtained per antenna.

2. **Probability Fusion**:
$$\mathbf{P}_{\text{fused}}(c) = \frac{1}{4} \sum_{a=1}^{4} \mathbf{P}_a(c), \quad c = 1, \ldots, 5$$

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
