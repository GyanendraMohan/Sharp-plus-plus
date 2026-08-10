# Complete Details: Fold 2 (SHARP++ LOEO Protocol)

Fold 2 of the Leave-One-Environment-Out (LOEO) evaluation measures how well the model generalizes to the **Living Room** environment when trained exclusively on data from other rooms. 

## Dataset Splitting & Configuration

The protocol enforces strict separation between environments to ensure no data leakage.

### Testing Dataset (Held-out Environment)
*   **Environment:** Living Room
*   **Sessions Used:** `S6a`, `S6b`
*   **Total Test Samples (Windows):** `47,078`

### Training & Validation Dataset
*   **Environments:** Bedroom + Laboratory
*   **Sessions Used:**
    *   *Bedroom:* `S1a`, `S1b`, `S1c`, `S2a`, `S2b`, `S3a`, `S4a`, `S4b`, `S5a`
    *   *Laboratory:* `S7a`
*   **Total Training Samples:** `182,931` (75% of the train+val split)
*   **Total Validation Samples:** `60,050` (25% of the train+val split)

### Preprocessing Rules Enforced
*   **Window Size:** `100` samples per window
*   **Overlap/Stride:** `5` samples 
*   **Normalization:** Z-score normalization was applied. The `mean` and `std` statistics were computed **exclusively** on a random subset (max 5,000 samples) of the 182,931 training windows. No validation or test data influenced the normalization.
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

These metrics represent the model's performance on the completely unseen Living Room test set (`S6a`, `S6b`).

- **Overall Accuracy:** `83.58%`
- **Macro Precision:** `78.73%`
- **Macro Recall:** `80.60%`
- **Macro F1-Score:** `77.77%`

---

## Per-Class Metrics

A detailed breakdown of performance across all 8 activity classes.

| Activity | Symbol | Precision | Recall | F1-Score | Total Support (N) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Empty** | `E` | 100.00% | 99.21% | 99.61% | 7,510 |
| **Walking** | `W` | 99.36% | 99.09% | 99.22% | 7,653 |
| **Running** | `R` | 98.26% | 99.95% | 99.09% | 7,664 |
| **Jumping** | `J` | 90.95% | 99.67% | 95.11% | 5,225 |
| **Sitting Down** | `L` | 97.96% | 54.07% | 69.68% | 7,650 |
| **Standing Up** | `S` | 99.16% | 100.00%| 99.58% | 3,789 |
| **Falling** | `C` | 44.16% | 92.80% | 59.85% | 3,794 |
| **Other/Unknown**| `H` | 0.00% | 0.00% | 0.00% | 3,793 |

---

## Confusion Matrix

The confusion matrix shows the predicted vs. actual classifications. 
*Rows represent the **Actual** (True) classes.*
*Columns represent the **Predicted** classes.*

| Actual \ Predicted | E | W | R | J | L | S | C | H |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E (Empty)** | **7,453** | 28 | 17 | 0 | 12 | 0 | 2 | 0 |
| **W (Walking)** | 0 | **7,583** | 70 | 0 | 0 | 0 | 0 | 0 |
| **R (Running)** | 0 | 4 | **7,660** | 0 | 0 | 0 | 0 | 0 |
| **J (Jumping)** | 0 | 17 | 0 | **5,208** | 0 | 0 | 0 | 0 |
| **L (Sitting)** | 0 | 0 | 0 | 23 | **4,136** | 32 | 1,277 | 2,182 |
| **S (Standing)** | 0 | 0 | 0 | 0 | 0 | **3,789** | 0 | 0 |
| **C (Falling)** | 0 | 0 | 0 | 0 | 0 | 0 | **3,521** | 273 |
| **H (Other)** | 0 | 0 | 49 | 495 | 74 | 0 | 3,173 | **0** |
