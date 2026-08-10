# Complete Details: Fold 3 (SHARP++ LOEO Protocol)

Fold 3 of the Leave-One-Environment-Out (LOEO) evaluation measures how well the model generalizes to the **Laboratory** environment when trained exclusively on data from the Bedroom and Living Room. 

## Dataset Splitting & Configuration

The protocol enforces strict separation between environments to ensure no data leakage.

### Testing Dataset (Held-out Environment)
*   **Environment:** Laboratory
*   **Sessions Used:** `S7a`
*   **Total Test Samples (Windows):** `20,786`

### Training & Validation Dataset
*   **Environments:** Bedroom + Living Room
*   **Sessions Used:**
    *   *Bedroom:* `S1a`, `S1b`, `S1c`, `S2a`, `S2b`, `S3a`, `S4a`, `S4b`, `S5a`
    *   *Living Room:* `S6a`, `S6b`
*   **Total Training Samples:** `202,616` (75% of the train+val split)
*   **Total Validation Samples:** `66,522` (25% of the train+val split)

### Preprocessing Rules Enforced
*   **Window Size:** `100` samples per window
*   **Overlap/Stride:** `5` samples 
*   **Normalization:** Z-score normalization was applied. The `mean` and `std` statistics were computed **exclusively** on a random subset (max 5,000 samples) of the 202,616 training windows. No validation or test data influenced the normalization.
*   **Input Shape:** `(100, 100, 4)`

---

## Model Architecture

The SHARP++ model uses a **Residual CNN + CBAM Attention + Bi-LSTM** architecture designed to extract spatial-temporal features from WiFi Doppler spectrograms.

1.  **Stem (Conv2D):** A 64-filter 3x3 Conv2D layer + BatchNormalization + ReLU.
2.  **Residual CNN Backbone:**
    *   Stage 1: Residual Block (64 filters) + MaxPooling2D (2x2)
    *   Stage 2: Residual Block (128 filters) + MaxPooling2D (2x2)
    *   Stage 3: Residual Block (256 filters)
3.  **CBAM Attention:** A Convolutional Block Attention Module consisting of Channel Attention (reduction ratio=8) followed by Spatial Attention (7x7 kernel). This suppresses background noise/static reflections and highlights human Doppler motion.
4.  **Reshape:** Flattens the frequency and channel dimensions to feed into the temporal layers.
5.  **Temporal Modeling (Bi-LSTM):** Two stacked Bidirectional LSTM layers, each with 128 hidden units (256 total per layer), with 20% dropout.
6.  **Global Average Pooling (1D):** Aggregates temporal features into a single vector.
7.  **Dense Embedding:** A 256-unit Dense layer with ReLU activation and 50% dropout.
8.  **Classification Head:** An 8-unit Dense layer with Softmax activation.

---

## Training Methodology

*   **Optimizer:** AdamW (Learning Rate = `1e-4`, Weight Decay = `1e-4`)
*   **Loss Function:** Sparse Categorical Crossentropy
*   **Batch Size:** `32`
*   **Max Epochs:** `25`
*   **Data Augmentation:** Applied during training.
*   **Callbacks:**
    *   *Model Checkpoint:* Saves the model with the best `val_accuracy`.
    *   *Early Stopping:* Monitors `val_loss` with a patience of `10` epochs (restores best weights).
    *   *ReduceLROnPlateau:* Reduces the learning rate by half (factor=0.5) if `val_loss` plateaus for `3` epochs.
*   **Decision Fusion (Testing):** During evaluation on the test set, predictions are smoothed across consecutive windows using a uniform 1D filter (size=11) to remove short-term noise.

---

## Overall Performance Metrics

These metrics represent the model's performance on the completely unseen Laboratory test set (`S7a`).

> [!WARNING]
> Note that the Laboratory dataset (`S7a`) does not contain any `R` (Running) or `L` (Sitting) samples. The zero scores for these missing classes negatively impact the unweighted macro averages.

- **Overall Accuracy:** `38.63%`
- **Macro Precision:** `27.19%`
- **Macro Recall:** `32.95%`
- **Macro F1-Score:** `26.35%`

---

## Per-Class Metrics

A detailed breakdown of performance across all 8 activity classes.

| Activity | Symbol | Precision | Recall | F1-Score | Total Support (N) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Empty** | `E` | 97.57% | 100.00%| 98.77% | 3,649 |
| **Walking** | `W` | 99.42% | 64.27% | 78.07% | 3,725 |
| **Running** | `R` | 0.00% | 0.00% | 0.00% | 0 |
| **Jumping** | `J` | 20.50% | 99.30% | 33.98% | 2,001 |
| **Sitting Down** | `L` | 0.00% | 0.00% | 0.00% | 0 |
| **Standing Up** | `S` | 0.00% | 0.00% | 0.00% | 3,805 |
| **Falling** | `C` | 0.00% | 0.00% | 0.00% | 3,802 |
| **Other/Unknown**| `H` | 0.00% | 0.00% | 0.00% | 3,804 |

---

## Confusion Matrix

The confusion matrix shows the predicted vs. actual classifications. 
*Rows represent the **Actual** (True) classes.*
*Columns represent the **Predicted** classes.*

| Actual \ Predicted | E | W | R | J | L | S | C | H |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E (Empty)** | **3,649** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **W (Walking)** | 1 | **2,394** | 112 | 1,218 | 0 | 0 | 0 | 0 |
| **R (Running)** | 0 | 0 | **0** | 0 | 0 | 0 | 0 | 0 |
| **J (Jumping)** | 0 | 14 | 0 | **1,987** | 0 | 0 | 0 | 0 |
| **L (Sitting)** | 0 | 0 | 0 | 0 | **0** | 0 | 0 | 0 |
| **S (Standing)** | 0 | 0 | 0 | 0 | 3,805 | **0** | 0 | 0 |
| **C (Falling)** | 0 | 0 | 0 | 2,955 | 15 | 0 | **0** | 832 |
| **H (Other)** | 90 | 0 | 171 | 3,533 | 10 | 0 | 0 | **0** |
